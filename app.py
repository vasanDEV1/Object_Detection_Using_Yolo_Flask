#import neccessary packages
from flask import Flask, request, render_template, send_file, Response
from werkzeug.utils import secure_filename
import io
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import os

#create a instance for flask app
app = Flask(__name__)

#flask's config system for file handling, 
#when we upload a image, it is temporarly stored in 'uploads/'
app.config['UPLOAD_FOLDER'] = 'uploads/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
#detection class is 
class Detection:
    #intialize the class with model path and store the model in instance variable self.model
    #its a good practice to load the model in class intialization itself 
    def __init__(self,model_path):
        self.model = YOLO(model_path)
    
    #method for running prediction on the image
    def detect_from_image(self, image,conf=0.5):
        
        #use model.predict fxn for preforming object_detection
        # returns a list of predicted objects
        results = self.model.predict(image,conf=conf)

        #for multiclass image multiple bounding boxes should be drawn
        #results is a list of predicted objects
        for result in results:
            #result.boxes contains all detected object 
            #each box has xyxy: coordinates[x1,y1,x2,y2] , cls: Class index , conf: confidence score
            for box in result.boxes:
                cv2.rectangle(image, (int(box.xyxy[0][0]),int(box.xyxy[0][1])),
                                         (int(box.xyxy[0][2]),int(box.xyxy[0][3])),
                                         (0,255,0),2)
                
                #result.names is dictionary that map class indices to their actual name.
                cv2.putText(image, result.names[int(box.cls[0])],
                            (int(box.xyxy[0][0]),int(box.xyxy[0][1] - 10)),
                            cv2.FONT_HERSHEY_PLAIN , 1, (0,255,0),2)
        
        return image

detection = Detection(r"object_detection/yolov8n.pt")


#api routing

#home route
#returns rendered index.html
@app.route('/')
def index():
    return render_template('index.html')

#object detection route, handles POST request for object detection
@app.route('/object-detection/', methods=['POST'])
def apply_detection():
    file = request.files.get('image')
    if not file or not file.filename:
        return 'No file provided or invalid file', 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(file_path)

    img = cv2.resize(np.array(Image.open(file_path).convert("RGB")), (512, 512))
    os.remove(file_path)

    detected_img = detection.detect_from_image(img)
    output = Image.fromarray(detected_img)

    buf = io.BytesIO()
    output.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)


                








