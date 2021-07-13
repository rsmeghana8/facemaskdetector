import streamlit as st
from streamlit import caching
import io
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

# Environment configuration: Cuda 10.0 and CuDNN 7.6.5, tensorflow 1.15
# If you get cublas and cudnn error delete ~/.nv folder
# Configuring tensorflow to avoid cudnn import error

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

model = keras.models.load_model('./Models/mobilenet.h5')

img_width, img_hight = 300, 300
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img_count_full= 0
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
org = (1,1)
class_label = ' '
fontscale = 0.8
thickness = 1
img_width, img_hight = 224, 224

scaling_factorx = 0.5
scaling_factory = 0.5

video_capture = cv2.VideoCapture(0)

while (True):

    ret, frame = video_capture.read()
    if ret:
        image = cv2.resize(image, None, fx=scaling_factorx, fy=scaling_factory, interpolation=cv2.INTER_AREA)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_img, 1.1,5, minSize=(1,1))
        count = 0
        for (x,y,a,b) in faces:
            org = (x-10, y-10)
            count += 1
            color_face = image[y:y+b, x:x+a]
            cv2.imwrite('faces/input/%d%dface.jpg'%(img_count_full,count),color_face)
            img = load_img('faces/input/%d%dface.jpg'%(img_count_full,count), target_size=(img_width,img_hight))
            img = img_to_array(img)/255
            img = np.expand_dims(img,axis = 0)
            pred_prob = model.predict(img)
            pred = np.argmax(pred_prob)
            if pred == 0:
                print('User with mask - {} % sure '.format(pred_prob[0][0] * 100) )
                class_lable = 'Mask'
                color = (255,255,255)
                cv2.imwrite('faces/with_mask/%d%dface.jpg'%(img_count_full,count),color_face)
            else:
                print('User with out mask - {} % sure'.format(pred_prob[0][1] * 100))
                class_lable = 'No Mask'
                color = (0, 0, 255)
                cv2.imwrite('faces/without_mask/%d%dface.jpg'%(img_count_full,count),color_face)
            cv2.rectangle(image,(x,y), (x+a, y+b),(255,255,255),1)
            cv2.putText(image, class_lable,org,font, fontscale, color,
                    thickness,cv2.LINE_AA)
        # cv2.imshow('Live Mask detection', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        print("there was a problem or video was finished")
        cv2.destroyAllWindows()
        video_capture.release()
        break
        # check if image is None
    if image is None:
        print("there was a problem None")
        # if True break the infinite loop
        break
video_capture.release()
cv2.destroyAllWindows()