import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from enum import Enum, unique
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = {'jpg', 'png', 'mp4'}
IMAGE_EXTENSIONS = {'jpg', 'png'}
filepath = ""
recognition_result = None 
class _MEDIA_TYPE(Enum): 
    NONE = 0
    IMAGE = 1
    VIDEO = 2

def get_media_type(file_extension):
    if file_extension in IMAGE_EXTENSIONS:
        return _MEDIA_TYPE.IMAGE
    elif file_extension in ALLOWED_EXTENSIONS:
        return _MEDIA_TYPE.VIDEO
    return _MEDIA_TYPE.NONE

uploaded_mediatype = _MEDIA_TYPE.NONE

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024

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


import pr_model 

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename = "media." + get_extension(filename)
            print ("Filename in main.py: %s" % filename)
            global filepath 
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print ("Filepath in main.py: %s" % filepath)
            file.save(filepath)
            url = (url_for('display_result', name=filename))
            global recognition_result
            recognition_result = pr_model.recognition(filepath)
            return redirect(url)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

from flask import send_from_directory

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

app.add_url_rule(
    "/uploads/<name>", endpoint="download_file", build_only=True
)

@app.route('/result/<name>')
def display_result(name):
    return render_template("result_display.html", results=recognition_result, name=name)

app.add_url_rule(
    "/result/<name>", endpoint="display_result", build_only=True
)

