# %%
import numpy as np 
import cv2 
import functools
from os.path import getctime, splitext
from keras.models import model_from_json
import glob
from app import ALLOWED_EXTENSIONS, IMAGE_EXTENSIONS, _MEDIA_TYPE, recognition_result

#%%
def get_extension(filename): 
    if '.' in filename:
        return filename.rsplit('.', 1)[1].lower()
    return None

def get_filename_without_extension(filename):
    if '.' in filename: 
        return filename.rsplit('.', 1)[0].lower()
    return filename

def allowed_file(filename): 
    return get_extension(filename) in ALLOWED_EXTENSIONS


# %%
class Label:
    def __init__(self, cl=-1, tl=np.array([0., 0.]), br=np.array([0., 0.]), prob=None):
        self.__tl = tl
        self.__br = br
        self.__cl = cl
        self.__prob = prob

    def __str__(self):
        return 'Class: %d, top left(x: %f, y: %f), bottom right(x: %f, y: %f)' % (
        self.__cl, self.__tl[0], self.__tl[1], self.__br[0], self.__br[1])

    # Make a copy
    def copy(self):
        return Label(self.__cl, self.__tl, self.__br)

    # Find width and height
    def wh(self): return self.__br - self.__tl

    # Find center point
    def cc(self): return self.__tl + self.wh() / 2

    # Get coordinate of top left
    def tl(self): return self.__tl

    # Get coordinate of bottom right
    def br(self): return self.__br

    # Get coordinate of top right
    def tr(self): return np.array([self.__br[0], self.__tl[1]])

    # Get coordinate of bottom left
    def bl(self): return np.array([self.__tl[0], self.__br[1]])
    
    # Return class
    def cl(self): return self.__cl

    # Calculate area 
    def area(self): return np.prod(self.wh())

    # Return probability
    def prob(self): return self.__prob

    def set_class(self, cl):
        self.__cl = cl

    def set_tl(self, tl):
        self.__tl = tl

    def set_br(self, br):
        self.__br = br

    def set_wh(self, wh):
        cc = self.cc()
        self.__tl = cc - .5 * wh
        self.__br = cc + .5 * wh

    def set_prob(self, prob):
        self.__prob = prob

# %%
#Normalize picture
def imnormalize(Image):
    return Image.astype('float32') / 255

#Width and height
def getWH(image):
    return np.array(image[1::-1]).astype(float)

#Find intersection over union area
def IOU(tl1, br1, tl2, br2):
    """Find intersection over union area

    Args:
        tl1 (tuple(int, int)): Top-left coordinates of the first rectangle
        br1 (tuple(int, int)): Bottom-right coordinates of the first rectangle
        tl2 (tuple(int, int)): Top-left coordinates of the second rectangle
        br2 (tuple(int, int)): Bottom-right coordinates of the second rectangle

    Returns:
        float: Area of intersection
    """
    wh1 = br1-tl1
    wh2 = br2-tl2
    assert((wh1 >= 0).all() and (wh2 >= 0).all())
    
    intersect_wh = np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0)
    intersect_area = np.prod(intersect_wh)
    area1 = np.prod(wh1)
    area2 = np.prod(wh2)
    union_area = area1 + area2 - intersect_area
    return intersect_area/union_area

def IOU_labels(l1, l2):
    return IOU(l1.tl(), l1.br(), l2.tl(), l2.br())

def nms(Labels, iou_threshold=0.5):
    SelectedLabels = []
    Labels.sort(key=lambda l: l.prob(), reverse=True)
    
    for label in Labels:
        non_overlap = True
        for selected in SelectedLabels:
            if IOU_labels(label, selected) > iou_threshold:
                non_overlap = False
                break

        if non_overlap:
            SelectedLabels.append(label)
    return SelectedLabels

def load_model(path):
    path = splitext(path)[0]
    with open('%s.json' % path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json, custom_objects={})
    model.load_weights('%s.h5' % path)
    return model

def find_T_matrix(pts, t_pts):
    A = np.zeros((8, 9))
    for i in range(0, 4):
        xi = pts[:, i]
        xil = t_pts[:, i]
        xi = xi.T
        
        A[i*2, 3:6] = -xil[2]*xi
        A[i*2, 6:] = xil[1]*xi
        A[i*2+1, :3] = xil[2]*xi
        A[i*2+1, 6:] = -xil[0]*xi

    [U, S, V] = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))
    return H

# Get 4 edge of rectangle
def getRectPts(tlx, tly, brx, bry):
    return np.matrix([[tlx, brx, brx, tlx], [tly, tly, bry, bry], [1, 1, 1, 1]], dtype=float)

def normal(pts, side, mn, MN):
    pts_MN_center_mn = pts * side
    pts_MN = pts_MN_center_mn + mn.reshape((2, 1))
    pts_prop = pts_MN / MN.reshape((2, 1))
    return pts_prop

class DLabel(Label):
    def __init__(self, cl, pts, prob):
        self.pts = pts
        tl = np.amin(pts, axis=1)
        br = np.amax(pts, axis=1)
        Label.__init__(self, cl, tl, br, prob)

# %%
# Reconstruct function to cut license plate from original image
def reconstruct(Im, Imresized, Yr, lp_threshold):
    # 4 max-pooling layers, stride = 2
    stride = 2**4
    side = ((208 + 40)/2)/stride

    # one line and two lines license plate size
    one_line = (470, 110)
    two_lines = (280, 200)

    Probs = Yr[..., 0]
    Affines = Yr[..., 2:]

    xx, yy = np.where(Probs > lp_threshold)
    # CNN input image size 
    WH = getWH(Imresized.shape)
    # Output feature map size
    MN = WH/stride

    vxx = vyy = 0.5 #alpha
    filt = lambda vx, vy: np.matrix([[-vx, -vy, 1], [vx, -vy, 1], [vx, vy, 1], [-vx, vy, 1]]).T
    labels = []
    labels_frontal = []

    for i in range(len(xx)):
        x, y = xx[i], yy[i]
        affine = Affines[x, y]
        prob = Probs[x, y]

        mn = np.array([float(y) + 0.5, float(x) + 0.5])

        # Affine transformation matrix
        A = np.reshape(affine, (2, 3))
        A[0, 0] = max(A[0, 0], 0)
        A[1, 1] = max(A[1, 1], 0)
        # Identity transformation
        B = np.zeros((2, 3))
        B[0, 0] = max(A[0, 0], 0)
        B[1, 1] = max(A[1, 1], 0)

        pts = np.array(A*filt(vxx, vyy))
        pts_frontal = np.array(B*filt(vxx, vyy))

        pts_prop = normal(pts, side, mn, MN)
        frontal = normal(pts_frontal, side, mn, MN)

        labels.append(DLabel(0, pts_prop, prob))
        labels_frontal.append(DLabel(0, frontal, prob))

    final_labels = nms(labels, 0.1)
    final_labels_frontal = nms(labels_frontal, 0.1)

    print(final_labels_frontal)

    # LP size and type
    out_size, lp_type = (two_lines, 2) if ((final_labels_frontal[0].wh()[0] / final_labels_frontal[0].wh()[1]) < 1.7) else (one_line, 1)

    license_plates = []
    if len(final_labels):
        final_labels.sort(key=lambda x: x.prob(), reverse=True)
        for _, label in enumerate(final_labels):
            t_ptsh = getRectPts(0, 0, out_size[0], out_size[1])
            ptsh = np.concatenate((label.pts * getWH(Im.shape).reshape((2, 1)), np.ones((1, 4))))
            H = find_T_matrix(ptsh, t_ptsh)
            # Applies perspective transformation
            lp = cv2.warpPerspective(Im, H, out_size, borderValue=0)
            license_plates.append(lp)
    print(final_labels)
    return final_labels, license_plates, lp_type

# %%
def detect_lp(model, Im, max_dim, lp_threshold):

    # Calculate factor to resize  the image
    min_dim_img = min(Im.shape[:2])
    factor = float(max_dim) / min_dim_img

    # Calculate new weight and height
    w, h = (np.array(Im.shape[1::-1], dtype=float) * factor).astype(int).tolist()

    # Resize image
    Imresized = cv2.resize(Im, (w, h))

    T = Imresized.copy()

    # Convert to Tensor
    T = T.reshape((1, T.shape[0], T.shape[1], T.shape[2]))

    # Use Wpod-net pretrain to detect license plate
    pred = model.predict(T)

    # Remove axes of length one from pred
    pred = np.squeeze(pred)

    print(pred.shape)

    # Reconstruct and return license plate (1: long, 2: square)
    L, TLp, lp_type = reconstruct(Im, Imresized, pred, lp_threshold)

    return L, TLp, lp_type

wpod_net_path = "wpod-net_update1.json" 
wpod_net = load_model(wpod_net_path)

def get_bounding_boxes(img, value = 255):
    size = img.shape 
    visited = np.zeros(size, dtype=np.bool_) 
    i = 0 
    boxes = []
    while i < size[0]: 
        j = 0
        while j < size[1]:
            # print (i, j, img[i, j])
            if img[i][j] == value and not visited[i][j]: 
                qi = [i] 
                qj = [j]
                visited[i][j] = True
                ihigh = i
                ilow = i
                jhigh = j
                jlow = j
                while len(qi) > 0: 
                    fronti = qi.pop()
                    frontj = qj.pop()
                    ihigh = max(ihigh, fronti)
                    ilow = min(ilow, fronti)
                    jhigh = max(jhigh, frontj)
                    jlow = min(jlow, frontj)
                    for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]: 
                        nexti = fronti + dx
                        nextj = frontj + dy
                        if 0 <= nexti < size[0] and 0 <= nextj < size[1]:
                            if not visited[nexti][nextj]:
                                visited[nexti][nextj] = True
                                qi.append(nexti) 
                                qj.append(nextj)
                boxes.append(((ilow, jlow),(ihigh, jhigh)))
            j += 1
        i += 1
    return boxes

def solve_image(pic):
    """From loaded cv2 image, solve for the license plate(s)

    Args:
        pic (cv2 loaded): image

    Returns:
        List[(cv2image, str)]: List of tuple(plate image, plate number)
    """
    Dmax = 608
    Dmin = 288
    ratio = float(max(pic.shape[:2])) / min(pic.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    _ , license_plates, lp_type = detect_lp(wpod_net, imnormalize(pic), bound_dim, lp_threshold=0.5)
    for _plate in license_plates: 
        plate = cv2.convertScaleAbs(_plate, alpha = 255.0)
        #cv2.imshow("Plate", plate)
        #cv2.waitKey()
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
        cv2.imshow("Binary", binary)
        cv2.waitKey()
        boxes = get_bounding_boxes(binary)
        print(boxes)
    pass


#%%
def recognition(filepath):
    """Called by webservice. 

    Args: 
        media_file (~werkzeug.datastructures.FileStorage): File from frontend form submission

    Returns: 
        Either: 
        (_MEDIA_TYPE.IMAGE, (List[str, cv.Mat])) 
        or: 
        (_MEDIA_TYPE.VIDEO, (List[str, cv.Mat, str, str]))
    """ 

    file_path = filepath
    extension = get_extension(file_path)    

    if extension in IMAGE_EXTENSIONS:

        pic = cv2.imread(file_path) 
        solve_image(pic)

        # Calculate ratio of weight/height and find the smallest
        


