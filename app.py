# -*- coding: utf-8 -*-
"""
Created on Fri May 29 00:31:59 2020

@author: dotan
"""

import numpy as np
from flask import Flask,request,jsonify,render_template
from keras.models import load_model
import cv2
from keras.preprocessing.image import img_to_array

app=Flask(__name__)
model = load_model('Emotion_little_vgg.h5')
class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    image = request.files['input_image'][0]
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    # rect,face,image = face_detector(frame)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class

            preds = model.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position = (x,y)
    return render_template('index.html',predict_text='Result: ${}'.format(label))
    
if __name__=='__main__':
    app.run(debug=True)
    
    

