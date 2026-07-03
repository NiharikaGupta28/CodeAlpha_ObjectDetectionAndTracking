import cv2

# 0 usually refers to the default/built-in webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()   # ret = True/False if frame was read successfully
    if not ret:
        break

    cv2.imshow("Webcam Test", frame)  # display the frame in a window

    if cv2.waitKey(1) & 0xFF == ord('q'):  # exit when 'q' is pressed
        break

cap.release()             # release the webcam
cv2.destroyAllWindows()   # close the display window