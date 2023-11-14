from flask import Flask, request, render_template, jsonify
from PIL import Image
from io import BytesIO
from flask_cors import CORS

import tensorflow as tf
import cv2 
import os
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
CORS(app)


def make_prediction(image):
    
    # Load the model
    new_model = tf.keras.models.load_model("/path/to/trained436epoch200.h5")
    classes = ["angry","disgust","fear","happy","neutral","sad","surprise"]

    # Reading the image 
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Converting image to grayscale 
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # Loading the required haar-cascade xml classifier file 
    haar_cascade = cv2.CascadeClassifier('/path/to/haarcascade_frontalface_default.xml') 
    
    # Applying the face detection method on the grayscale image 
    faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)
    
    # Zooming in on face
    for x,y,w,h in faces_rect:
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        facess = haar_cascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("Face not detected")
        else:
            for(ex,ey,ew,eh) in facess:
                face_roi = roi_color[ey: ey+eh, ex: ex+ew]
            
    # Resizing image before making a prediction
    final_image = cv2.resize(face_roi, (224,224))
    final_image = np.expand_dims(final_image, axis=0)
    final_image = final_image/255.0
    
    # Making the Prediction
    Predictions = new_model.predict(final_image)
    
    for (x, y, w, h) in faces_rect: 
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) 
    
    prediction = classes[np.argmax(Predictions)]
    
    return img, prediction




@app.route('/process_image', methods=['POST'])
def process_image():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Read the image file
        img = Image.open(file)
        image, prediction = make_prediction(img)

        # Process the image (you can replace this with your own logic)
        result_string = prediction

        return jsonify({'result': result_string})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()