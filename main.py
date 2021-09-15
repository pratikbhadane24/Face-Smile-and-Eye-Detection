import cv2

# HaarCascade Importing
face_cascade = cv2.CascadeClassifier(
    './HaarcascadeXMLs/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./HaarcascadeXMLs/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(
    './HaarcascadeXMLs/haarcascade_smile.xml')


def RealtimeCoverage():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        # Reading Coverage
        ret, img = cap.read()
        # Inverting Video
        img = cv2.flip(img, 1)
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Face Detection
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (182, 25, 25), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

        # Eyes Detection
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey),
                          (ex+ew, ey+eh), (255, 235, 161), 2)

        # Smile Detection
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        for (sx, sy, sw, sh) in smiles:
            font = cv2.FONT_HERSHEY_DUPLEX
            img = cv2.putText(img, 'Smiling hehe', (70, 450),
                              font, 2, (255, 255, 255), 3, cv2.LINE_AA)

        # Displaying the output
        cv2.imshow('Realtime Coverage', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def readImg():
    img = cv2.imread('pain.jpg')
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (182, 25, 25), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 235, 161), 2)

    # Displaying the output
    cv2.imshow('Local Image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


cont = True
while cont:
    my_choice = input(
        "Select from 1 or 2 or 3:\n1. Realtime Coverage\n2. Local Image\n3. Exit\nYour Choice: ")

    if my_choice == "1":
        RealtimeCoverage()
    elif my_choice == "2":
        readImg()
    elif my_choice == "3":
        print("Exiting.....")
        break
    else:
        print("Wrong Input")
