# %%
import numpy as np 
import cv2 
import functools
import matplotlib.pyplot as plt
from os.path import getctime, splitext, join
import tensorflow
import tensorflow.keras
from tensorflow.keras.models import model_from_json
import glob
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import app
from app import ALLOWED_EXTENSIONS, IMAGE_EXTENSIONS
chars = ['0','1','A','B','C','D','E','F','G','H','K','L','2','M','N','P','R','S','T','U','V','X','Y','3','Z','4','5','6','7','8','9']

#%%
class MultiLayerCNN(nn.Module):
    def __init__(self):
        super(MultiLayerCNN, self).__init__()
        # Call a convolutional layer with input_channels = 1, output_channels = 10, kernel_size = 5
        self.conv1 = nn.Conv2d(1, 10, 5)
        # Max Pooling takes a sliding window (2*2) and replaces the window with the maximum value. 
        self.pool = nn.MaxPool2d(2, 2)
        # Call a convolutional layer with input_channels = 10, output_channels = 20, kernel_size = 5
        self.conv2 = nn.Conv2d(10, 20, 5)
        # 320 inputs, 31 outputs
        self.fc = nn.Linear(320,31)

    def forward(self, x):
        # We use relu activation function in between layers
        # Layer 1
        x = self.pool(nn.functional.relu(self.conv1(x)))
        # Layer 2
        x = self.pool(nn.functional.relu(self.conv2(x)))
        # Flatten tensor shape to 320
        x = x.view(-1, 320)
        # Calculate softmax which return probabilities for 31 catogories
        x = nn.functional.log_softmax(self.fc(x),dim=1)
        return x

cnn_model = MultiLayerCNN()
cnn_model.load_state_dict(torch.load("model.pth"))
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

    print("Final labels: ", final_labels_frontal)
    if len(final_labels_frontal) == 0 or len(final_labels) != len (final_labels_frontal):
        return [], [], None

    # LP size and type
    out_size, lp_type = (two_lines, 2) if ((final_labels_frontal[0].wh()[0] / final_labels_frontal[0].wh()[1]) < 1.7) else (one_line, 1)

    license_plates = []
    if len(final_labels):
        final_labels.sort(key=lambda x: x.prob(), reverse=True)
        for label in final_labels:
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

def get_bounding_boxes(img, value = 255, lower_bound = 1/80, upper_bound = 1/10):
    size = img.shape 
    visited = np.zeros(size, dtype=np.bool_) 
    i = 0 
    boxes = []
    lower = int(size[0] * size[1] * lower_bound) 
    upper = int(size[0] * size[1] * upper_bound)
    print (" low = {}, up = {}, total = {}".format(lower, upper, size[0] * size[1]))
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
                            if not visited[nexti][nextj] and img[nexti][nextj] == value:
                                visited[nexti][nextj] = True
                                qi.append(nexti) 
                                qj.append(nextj)
                width = jhigh - jlow + 1
                height = ihigh - ilow + 1
                area = width * height
                if lower <= area <= upper and 6 <= width and 10 <= height:
                    print ("({}, {}) -> ({}, {}), width = {}, height = {}, area = {}".format(ilow, jlow, ihigh, jhigh, width, height, area))
                    boxes.append(((ilow, jlow),(ihigh, jhigh)))
            j += 1
        i += 1
    def compare(h1, h2):
        if abs(h1[0][0] - h2[0][0]) <= 8:
            return h1[0][1] - h2[0][1]
        else:
            return h1[0][0] - h2[0][0]
    boxes = sorted(boxes, key=functools.cmp_to_key(compare))
    return boxes

def get_character_from_cropped_image(crop):
    crop = cv2.resize(crop, dsize=(28,28))
    convert = np.array(crop,dtype=np.float32)/255
    convert = convert.reshape(1, 1, 28, 28)
    convert = torch.from_numpy(convert)
    std_normalize = transforms.Normalize(mean=[0.456],
                          std=[0.224])
    final = std_normalize(convert)
    cnn_model.eval()
    with torch.no_grad():
        pred = cnn_model(final)
        #result = pred.argmax(1).item()
        result = torch.max(pred,1)[1].item()
        #print(pred)
    return chars[result]


def solve_image(pic):
    """From loaded cv2 image, solve for the license plate(s)

    Args:
        pic (cv2 loaded): image

    Returns:
        List[(cv2image, str)]: List of tuple(plate image, plate number)
    """
    print ("Enter solve_image function")
    Dmax = 608
    Dmin = 288
    ratio = float(max(pic.shape[:2])) / min(pic.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    result = []

    _ , license_plates, lp_type = detect_lp(wpod_net, imnormalize(pic), bound_dim, lp_threshold=0.5)
    for _plate in license_plates: 
        plate = cv2.convertScaleAbs(_plate, alpha = 255.0)
        #cv2.imshow("Plate", plate)
        #cv2.waitKey()
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
        #cv2.imshow("Binary", binary)
        #cv2.waitKey()
        boxes = get_bounding_boxes(binary)
        print(boxes)
        plate_number = ""
        for box in boxes:
            #crop = binary[box[0][1]:box[1][1] + 1][box[0][0]:box[1][0] + 1]
            crop = binary[box[0][0]:box[1][0] + 1, box[0][1]:box[1][1]]
            plate_number += get_character_from_cropped_image(crop)
        
        #cv2.imshow("Plate: {}".format(plate_number), plate)
        result.append((plate, plate_number))

    return result


def int_to_string_filename(int_value, extension=".jpg"):
    return join("uploads", "{:06d}".format(int_value) + extension)

#%%
def recognition(filepath):
    """Called by webservice. 

    Args: 
        media_file (~werkzeug.datastructures.FileStorage): File from frontend form submission

    Returns: 
        (List[(str, str, str, str)])): image path, plate number, start time, end time
    """ 

    file_path = filepath
    extension = get_extension(file_path)    
    recognition_result = []
    if extension in IMAGE_EXTENSIONS:

        pic = cv2.imread(file_path) 
        recognition_result = solve_image(pic)
        for i, x in enumerate(recognition_result):
            cv2.imwrite(int_to_string_filename(i), x[0]) 
        recognition_result = [(int_to_string_filename(i), x[1], 0, 0) for (i, x) in enumerate(recognition_result)]

    else: # if extension is a video extension
        frame_count = 0
        cap = cv2.VideoCapture(filepath)
        app.fps = cap.get(cv2.CAP_PROP_FPS)

        def generate_frame_filename(filename):
            return 'f' + filename

        plateid = 0
        plate_time_of_occurences = {}
        plate_number_to_framename = {}
        while True:
            is_read, frame = cap.read()
            if not is_read:
                # break out of the loop if there are no frames to read
                break
            frame_count += 1
            frame_result = solve_image(frame) 
            for (i, (plate_frame, plate_number))  in enumerate(frame_result):

                if plate_number not in plate_time_of_occurences:
                    plate_time_of_occurences[plate_number] = [frame_count]
                    plate_name = generate_frame_filename(int_to_string_filename(plateid))
                    plateid += 1
                    plate_number_to_framename[plate_number] = plate_name
                    cv2.imwrite(plate_name, plate_frame)
                else:
                    plate_time_of_occurences[plate_number].append(frame_count)

        print ("Plate occurrences: ", (plate_time_of_occurences))
        print ("Plate to file: ", (plate_number_to_framename))
        
        most_likely_frame_count = 0
        for plate_number, timestamps in plate_time_of_occurences.items():
            if len(plate_number) < 7 or len(plate_number) > 9: continue

            plate_name = plate_number_to_framename[plate_number]
            this_plate_frame_count = len(timestamps) 

            if this_plate_frame_count > most_likely_frame_count:
                most_likely_frame_count = this_plate_frame_count
                app.most_likely_plate_number = plate_number 
                app.most_likely_plate_certainty = most_likely_frame_count / (timestamps[-1] - timestamps[0] + 1)
                app.most_likely_plate_path = plate_name

            begin, end = None, None
            for timepoint in timestamps:
                if end != None and timepoint == end + 1:
                    end = timepoint
                else:
                    if begin != None: 
                        recognition_result.append((plate_name, plate_number, begin, end))
                    begin = timepoint
                    end = timepoint
            if begin != None:
                recognition_result.append((plate_name, plate_number, (begin), (end)))
    print(recognition_result)
    return recognition_result

# %%
