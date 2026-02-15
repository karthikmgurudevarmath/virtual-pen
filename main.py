import cv2
import numpy as np
import mediapipe as mp

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=1
    )
    mp_draw = mp.solutions.drawing_utils

    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280) # Width
    cap.set(4, 720)  # Height

    # Canvas to draw on
    canvas = None
    
    # Previous point for drawing smooth lines
    px, py = 0, 0
    
    # Drawing color (BGR) - Red
    draw_color = (0, 0, 255)
    brush_thickness = 5
    eraser_thickness = 50

    print("Virtual Pen Started")
    print("-------------------")
    print("Controls:")
    print(" - Index Finger UP ONLY: Draw")
    print(" - Two Fingers (Index + Middle) UP: Hover (Move without drawing)")
    print(" - All Fingers UP: Erase Mode")
    print(" - 'c' key: Clear Canvas")
    print(" - 'q' key: Quit")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from webcam.")
            break

        # Flip the image for mirror effect
        img = cv2.flip(img, 1)
        
        if canvas is None:
            canvas = np.zeros_like(img)

        # Convert BGR to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        
        # Get frame dimensions
        h, w, c = img.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                # mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get landmark positions
                lmList = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                if len(lmList) != 0:
                    # Tip of Index Finger
                    x1, y1 = lmList[8][1], lmList[8][2]
                    # Tip of Middle Finger
                    x2, y2 = lmList[12][1], lmList[12][2]

                    # Check which fingers are up
                    fingers = []
                    
                    # Thumb (Tip x > IP x for right hand in mirror mode? SImple check relative to x)
                    # For simplicity, we'll focus on Index and Middle for modes.
                    
                    # Index Finger Up
                    if lmList[8][2] < lmList[6][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                    
                    # Middle Finger Up
                    if lmList[12][2] < lmList[10][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                        
                    # Ring Finger Up
                    if lmList[16][2] < lmList[14][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                        
                    # Pinky Up
                    if lmList[20][2] < lmList[18][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    # Logic
                    # 1. Selection/Hover Mode: Two fingers are up (Index and Middle)
                    if fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
                        px, py = 0, 0 # Reset previous point so we don't draw a line from where we left off
                        cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
                        # print("Hover Mode")

                    # 2. Drawing Mode: Index finger is up
                    elif fingers[0] and not fingers[1]:
                        cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
                        # print("Drawing Mode")
                        
                        if px == 0 and py == 0:
                            px, py = x1, y1
                        
                        cv2.line(canvas, (px, py), (x1, y1), draw_color, brush_thickness)
                        px, py = x1, y1
                    
                    # 3. Eraser Mode: All fingers (4 main ones) up - simple eraser or specific gesture?
                    # Let's simpler: if fingers[0] and fingers[1] and fingers[2] and fingers[3]:
                    #    # Erase
                    #    cv2.circle(img, (x1, y1), 30, (0,0,0), -1) 
                    #    cv2.circle(canvas, (x1, y1), eraser_thickness, (0, 0, 0), -1)
                    #    px, py = 0, 0
                        
                    else:
                        px, py = 0, 0 # Reset if other gestures

        # Combine logic
        # Convert canvas to gray to create a mask
        imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        
        # Black-out the area of the drawing in the original image (optional, or just add weighted)
        # Using addWeighted is simpler and looks good for light drawing
        # But to make it solid:
        
        # 1. Mask the original image
        img = cv2.bitwise_and(img, imgInv) 
        # 2. Add the canvas (colors)
        img = cv2.bitwise_or(img, canvas)

        # UI Header (optional)
        cv2.putText(img, "Index: Draw | Index+Middle: Hover | q: Quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("Virtual Pen", img)
        # cv2.imshow("Canvas", canvas) # Debug view

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros_like(img)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
