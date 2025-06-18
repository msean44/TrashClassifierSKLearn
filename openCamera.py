#Import the library to open camera
import cv2

#Command to open default (0) webcam
cap = cv2.VideoCapture(0)

#If the webcam is not opened, then error and reboot.
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

#Set the frame dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#Loop to ensure webcam is always open
while True:

    #ret is the boolean success flag, frame is the actual image array
    ret, frame = cap.read()

    #If ret is False (e.g. camera disconnected), then break
    if not ret:
        print("Error: Could not capture frame.")
        break

    #Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    #To display the current frame named "Camera Feed"
    cv2.imshow("Camera Feed", frame)
    
    #To quit, 'q' has to be pressed so that the ASCII code matched.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Releasing sources (camera and display windows)
cap.release()
cv2.destroyAllWindows()
