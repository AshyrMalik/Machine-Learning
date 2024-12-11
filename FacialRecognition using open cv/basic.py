import cv2
import numpy as np
import face_recognition

imgShahrukh= face_recognition.load_image_file(r"MainImages\shahrukh.JPG")
imgShahrukh=cv2.cvtColor(imgShahrukh,cv2.COLOR_BGR2RGB)
test_img = face_recognition.load_image_file(r"MainImages\shahrukh2.JPG")
test_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)

faceLoc=face_recognition.face_locations(imgShahrukh)[0]
encodeShahrukh= face_recognition.face_encodings(imgShahrukh)[0]
cv2.rectangle(imgShahrukh,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,200,0),2)
faceLocTest=face_recognition.face_locations(test_img)[0]
encodeTest= face_recognition.face_encodings(test_img)[0]
cv2.rectangle(test_img,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,200,0),2)

results=face_recognition.compare_faces([encodeShahrukh],encodeTest)
faceDis=face_recognition.face_distance([encodeShahrukh],encodeTest)
print(results,faceDis)

text = f"{results[0]}, {faceDis[0]:.2f}"
cv2.putText(test_img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
cv2.imshow("Shahrukhimage",imgShahrukh)
cv2.imshow("TestImage",test_img)
cv2.waitKey(0)

