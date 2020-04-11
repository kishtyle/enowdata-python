import os
import subprocess
import sys
import logging
import shutil
import face_recognition
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)
app.config['PROPAGATE_EXCEPTIONS'] = True
app.config['TEMP_FOLDER'] = '/tmp'
app.config['OCR_OUTPUT_FILE'] = 'ocr'
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024

def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1] in set(['png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff'])

@app.errorhandler(404)
def not_found(error):
  resp = jsonify( {
    u'status': 404,
    u'message': u'Resource not found'
  } )
  resp.status_code = 404
  return resp

@app.route('/')
def api_root():
  resp = jsonify( {
    u'status': 200,
    u'message': u'Welcome to our secret APIs'
  } )
  resp.status_code = 200
  return resp

@app.route('/test', methods = ['GET'])
def test():
  return render_template('upload_form.html', landing_page = 'process')

@app.route('/facerec', methods = ['GET'])
def facerec():
  return render_template('facerec_form.html', landing_page = 'facedetect')

@app.route('/facedetect', methods = ['GET','POST'])
def facedetect():
    if request.method == 'POST':

      vsource = request.form.get('vsource')
      video_capture = cv2.VideoCapture(vsource)
      # Load a sample picture and learn how to recognize it.
      # Load a sample picture and learn how to recognize it.
      akhil_image = face_recognition.load_image_file("Akhil.jpg")
      akhil_face_encoding = face_recognition.face_encodings(akhil_image)[0]
      ravi_image = face_recognition.load_image_file("Ravikishore.jpg")
      ravi_face_encoding = face_recognition.face_encodings(ravi_image)[0]
      # Create arrays of known face encodings and their names
      known_face_encodings = [
          akhil_face_encoding,
          ravi_face_encoding
          ]
      known_face_names = [
          "Akhil",
          "ravi"
          ]
      # Initialize some variables
      face_locations = []
      face_encodings = []
      face_names = []
      process_this_frame = True

      while True:
          # Grab a single frame of video
          ret, frame = video_capture.read()

          # Resize frame of video to 1/4 size for faster face recognition processing
          small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

          # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
          rgb_small_frame = small_frame[:, :, ::-1]

          # Only process every other frame of video to save time
          if process_this_frame:
              # Find all the faces and face encodings in the current frame of video
              face_locations = face_recognition.face_locations(rgb_small_frame)
              face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

              face_names = []
              for face_encoding in face_encodings:
                  # See if the face is a match for the known face(s)
                  matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.7)
                  # name = "Unknown"

                  # # If a match was found in known_face_encodings, just use the first one.
                  #if True in matches:
                  #	first_match_index = matches.index(True)
                  #	name = known_face_names[first_match_index]

                  # Or instead, use the known face with the smallest distance to the new face
                  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                  best_match_index = np.argmin(face_distances)
                  if matches[best_match_index]:
                      if best_match_index == 0:
                          name = known_face_names[best_match_index]
                          print('first')
                      else:
                          name = known_face_names[best_match_index-1]
                          print('Not first')
                  else:
                  	name = "Unknown"
                  # if matches[best_match_index]:
                  #     if best_match_index > 0:
                  #         name = known_face_names[best_match_index-1]
                  resp = jsonify( {
                    u'status': 200,
                    u'Name': name
                  } )

                  face_names.append(name)

          process_this_frame = not process_this_frame

  else:
    resp = jsonify( {
      u'status': 405,
      u'message': u'The method is not allowed for the requested URL'
    } )
    resp.status_code = 405
    return resp


@app.route('/process', methods = ['GET','POST'])
def process():
  if request.method == 'POST':
    file = request.files['file']
    hocr = request.form.get('hocr') or ''
    ext = '.hocr' if hocr else '.txt'
    if file and allowed_file(file.filename):
      folder = os.path.join(app.config['TEMP_FOLDER'], str(os.getpid()))
      os.mkdir(folder)
      input_file = os.path.join(folder, secure_filename(file.filename))
      output_file = os.path.join(folder, app.config['OCR_OUTPUT_FILE'])
      file.save(input_file)

      command = ['tesseract', input_file, output_file, '-l', request.form['lang'], hocr]
      proc = subprocess.Popen(command, stderr=subprocess.PIPE)
      proc.wait()

      output_file += ext

      if os.path.isfile(output_file):
        f = open(output_file)
        resp = jsonify( {
          u'status': 200,
          u'ocr':{k:v.decode('utf-8') for k,v in enumerate(f.read().splitlines())}
        } )
      else:
        resp = jsonify( {
          u'status': 422,
          u'message': u'Unprocessable Entity'
        } )
        resp.status_code = 422

      shutil.rmtree(folder)
      return resp
    else:
      resp = jsonify( {
        u'status': 415,
        u'message': u'Unsupported Media Type'
      } )
      resp.status_code = 415
      return resp
  else:
    resp = jsonify( {
      u'status': 405,
      u'message': u'The method is not allowed for the requested URL'
    } )
    resp.status_code = 405
    return resp

if __name__ == '__main__':
  app.run(debug=True)
  
