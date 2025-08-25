import cv2
import mediapipe as mp
from math import hypot
import numpy as np
import subprocess

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2
)
Draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(1)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)  
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    process = hands.process(frameRGB)
    
    landmarklist = []
    

    if process.multi_hand_landmarks:
        for handlm in process.multi_hand_landmarks:
            for _id, landmarks in enumerate(handlm.landmark):
                height, width, color_channels = frame.shape
                x, y = int(landmarks.x * width), int(landmarks.y * height)
                landmarklist.append([_id, x, y])
            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)
    
    if landmarklist != []:
        x1, y1 = landmarklist[4][1], landmarklist[4][2]
        x2, y2 = landmarklist[8][1], landmarklist[8][2]
        
        # Draw circles on fingertips
        cv2.circle(frame, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 7, (255, 0, 255), cv2.FILLED)
        
        # Draw line between fingertips
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        
        # Calculate distance between fingertips
        length = hypot(x2 - x1, y2 - y1)
        
        # Map distance to volume level (0-100 range)
        volume_level = int(np.interp(length, [15, 220], [0, 100]))
        
        # Display volume level on screen
        cv2.putText(frame, f'Volume: {volume_level}%', 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Distance: {int(length)}px', 
                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Set volume using AppleScript (more reliable than brightness)
        try:
            script = f'set volume output volume {volume_level}'
            subprocess.run(['osascript', '-e', script], check=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            print(f"Volume level: {volume_level}%")
    cv2.putText(frame, 'Pinch fingers to control volume', 
               (50, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, 'Press Q to quit', 
               (50, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow("Volume Control", frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()