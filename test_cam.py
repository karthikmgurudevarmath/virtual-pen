import cv2
print("Testing webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
else:
    print("Webcam opened successfully")
    ret, frame = cap.read()
    if ret:
        print("Read frame successfully")
    else:
        print("Cannot read frame")
cap.release()
