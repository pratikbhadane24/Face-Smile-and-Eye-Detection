import cv2

face_cascade = cv2.CascadeClassifier('./HaarcascadeXMLs/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./HaarcascadeXMLs/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('./HaarcascadeXMLs/haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (182, 25, 25), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 235, 161), 2)

            smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

            for (sx, sy, sw, sh) in smiles:
                font = cv2.FONT_HERSHEY_DUPLEX
                img = cv2.putText(img, 'Smiling hehe', (70, 450), font, 2, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
