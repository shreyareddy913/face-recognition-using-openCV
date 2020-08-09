import cv2
import face_recognition


imgdamon = face_recognition.load_image_file('ImagesBasic/damon.jpg')
imgdamon = cv2.cvtColor(imgdamon, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/wc1759318.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgdamon)[0]
encodedamon = face_recognition.face_encodings(imgdamon)[0]
cv2.rectangle(imgdamon, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (255, 0, 255), 2)

facelocTest = face_recognition.face_locations(imgTest)[0]
encodedamonTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (facelocTest[3], facelocTest[0]), (facelocTest[1], facelocTest[2]), (255, 0, 255), 2)
results = face_recognition.compare_faces([encodedamon],encodedamonTest)
facedis= face_recognition.face_distance([encodedamon],encodedamonTest)
print(results,facedis)
cv2.putText(imgTest, f'{results} {round(facedis[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
cv2.imshow('Damon', imgdamon)
cv2.imshow('damon test', imgTest)
cv2.waitKey(0)
