import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime



# Path to the directory with images
path = "AttendanceImages"
images = []
classNames = []

# Load images and class names
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)


# Function to mark attendance
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtString}')

def findEncodings(imageList):
    encodeList = []
    for img in imageList:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img_rgb)[0]
        encodeList.append(encode)
    return encodeList

# Find encodings for known images
encodeListKnown = findEncodings(images)
print("Encoding Complete")

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    # Resize the frame to 1/4 size for faster face recognition processing
    imgs = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    # Find all the face locations and face encodings in the current frame
    facesCurFrame = face_recognition.face_locations(imgs)
    encodesCurFrame = face_recognition.face_encodings(imgs, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)  # Find the best match

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            markAttendance(name)

            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            # Draw rectangle around face and label it
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Webcam', img)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
