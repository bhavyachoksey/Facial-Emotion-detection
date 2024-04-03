import numpy as np
import pickle
import pandas as pd
import streamlit as st
from skimage.feature import hog
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from streamlit_webrtc import webrtc_streamer

pickle_in_svm = open(
    r"C:\Users\bhavy\OneDrive\Desktop\facial emotion recognition\svm_model.pkl", "rb")
svm_model = pickle.load(pickle_in_svm)

pickle_in_logistic = open(
    r"C:\Users\bhavy\OneDrive\Desktop\facial emotion recognition\logistic_model.pkl", "rb")
logistic_model = pickle.load(pickle_in_logistic)

pickle_in_gridsearch = open(
    r"C:\Users\bhavy\OneDrive\Desktop\facial emotion recognition\grid_search.pkl", "rb")
gridsearch_model = pickle.load(pickle_in_gridsearch)

pickle_in_randomforest = open(
    r"C:\Users\bhavy\OneDrive\Desktop\facial emotion recognition\rf_classifier.pkl", "rb")
randomforest_model = pickle.load(pickle_in_randomforest)

pickle_in_pca1 = open(
    r"C:\Users\bhavy\OneDrive\Desktop\facial emotion recognition\pca1.pkl", "rb")
pca1 = pickle.load(pickle_in_pca1)

pickle_in_pca = open(
    r"C:\Users\bhavy\OneDrive\Desktop\facial emotion recognition\pca.pkl", "rb")
pca = pickle.load(pickle_in_pca)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def calculate_hog1(image):
    hog_features = []          

    
        # Convert the image to grayscale if necessary
    if len(image.shape) > 2:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image

    # Ensure the image has the correct data type
    if gray_img.dtype != np.uint8:
        gray_img = gray_img.astype(np.uint8)

    # Compute HOG features for each image
    hog_features.append(hog(gray_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys'))

    # Convert hog_features to a numpy array
    hog_features = np.array(hog_features)
    #print(hog_features.shape)
    return hog_features



def image_pca(df):
   
    
    pca3 = PCA()
    test=pca3.fit(df)
    var_cumu = np.cumsum(pca3.explained_variance_ratio_)*100
    q = np.argmax(var_cumu>95)
    
   
    df = df.reshape(1, -1)

    df=pca1.transform(df)
    
    
    return df




def predict_svm(image):
    x_image=calculate_hog1(image)
    x_image=image_pca(x_image)
    y_image=logistic_model.predict(x_image)
    return y_image
       


def live_cam():
    cap = cv2.VideoCapture(0)


    while True:
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't capture frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)


        for (x, y, w, h) in faces:
            # Predict emotion for the face region
            face_roi = gray[y:y+h, x:x+w]
            resized_frame = cv2.resize(frame, (48, 48))
            emotion=predict_svm(resized_frame)

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Annotate with predicted emotion
            cv2.putText(frame, str(emotion), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        cv2.imshow('live cam',frame)
        #cv2.imshow('Emotion Detection', gray_frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the windows
    cap.release()
    cv2.destroyAllWindows()


      

#logo_image=r"C:\Users\bhavy\Downloads\emotion detection logo.png"


def main():
    st.title("Facial Emotion Recognition")
    #st.image(logo_image, use_column_width=True)
    cap = cv2.VideoCapture(0)
    frame_placeholder=st.empty()
    stop_button=st.button('STOP')

    if st.button('START CAM'):
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't capture frame")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)


            for (x, y, w, h) in faces:
                # Predict emotion for the face region
                face_roi = gray[y:y+h, x:x+w]
                resized_frame = cv2.resize(frame, (48, 48))
                emotion=predict_svm(resized_frame)

                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Annotate with predicted emotion
                cv2.putText(frame, str(emotion), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            cv2.imshow('live cam',frame)
            frame_placeholder.image(frame,channels='RGB')

            #st.text('Current Emotion: {}'.format(str(emotion)))

    if cv2.waitKey(1) & 0xFF == ord('q') or stop_button:
        cap.release()
        cv2.destroyAllWindows()
      



if __name__ == "__main__":
    main()
