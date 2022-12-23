#importation of libraries
import cv2 as cv
import numpy as np
import face_recognition as face
import threading
import os
import time
from tqdm import tqdm

#video feeds
capture = cv.VideoCapture(0)

#fatch the working dir for known faces
path = "encoded_faces/"
path_files = os.listdir(path)
known_faces_encoding = []
known_faces_names = []

#loading known people from dir
print("Loading known faces...")
for file in tqdm(path_files):
    current_encoding = np.load(f"{path}{file}")
    known_faces_encoding.append(current_encoding)
    known_faces_names.append(os.path.splitext(file)[0])

# rscaling the size of the image for faster processing
def rescale(img, scale=.75):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dimension = (width, height)
    return cv.resize(img, dimension, interpolation=cv.INTER_AREA)

#face encoding function 

#draw a tag on the face with the name
def faceTag(img, locS, nameS):
    # pre-allocate memory for the output image
    img_out = np.empty_like(img)
    # use integer division to avoid floating point calculations
    rescale = 2
    for name, loc in zip(nameS, locS):
        y1, x2, y2, x1 = loc 
        y1, x2, y2, x1 = y1//rescale, x2//rescale, y2//rescale, x1//rescale
        cv.rectangle(img_out, (x1, y1), (x2, y2), (255, 0, 255), thickness=1)
        cv.rectangle(img_out, (x1, y2), (x2, y2+20), (255, 0, 255), cv.FILLED)
        cv.putText(img_out, name, (x1, y2+10), cv.FONT_HERSHEY_COMPLEX_SMALL, .5, (255, 255, 255), thickness=1)
    return img_out

process_this_frame = True
draw_loc = []
draw_name = []
while True:
    success, img = capture.read()
    imgS = rescale(img, scale=.5)
    if process_this_frame:
        draw_loc = []
        draw_name = []
        imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)
        #encode the Live Web Cam frame
        imgS_loc = face.face_locations(imgS)
        if len(imgS_loc)> 0:
            imgS_encoding = face.face_encodings(imgS, imgS_loc)
            #Loop through each face from the live cam 
            for encoded_face, face_loc in zip(imgS_encoding, imgS_loc):
                
                face_distance = face.face_distance(known_faces_encoding, encoded_face)
                matchedIndex = np.argmin(face_distance)
                today = time.localtime()
                if face_distance[matchedIndex]:
                    name = known_faces_names[matchedIndex].upper()
                    draw_loc.append(face_loc)
                    draw_name.append(name)
                    print(f"[INFO]: {today.tm_hour}:{today.tm_min}:{today.tm_sec} ----> ", name)
                else: 
                    draw_loc.append(face_loc)
                    draw_name.append("Unknown")
                    print(f"[INFO]: {today.tm_hour}:{today.tm_min}:{today.tm_sec} ----> ", name)                    
                    
    process_this_frame = not process_this_frame
    if len(draw_name)>0:
        img = faceTag(img, draw_loc, draw_name)
    cv.imshow("Live Webcam", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv.destroyAllWindows()
