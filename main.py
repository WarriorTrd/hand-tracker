import cv2
import mediapipe as mp
import logging

# Optional: Suppress MediaPipe log noise
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# Initialize MediaPipe drawing and hand modules
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Start webcam (try 0 or 1 depending on your system)
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

with mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5,
    max_num_hands=2
) as hands:
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=3, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(20, 180, 90), thickness=2, circle_radius=2)
                )

                # Optional: Label the hand (Left/Right)
                if results.multi_handedness:
                    label = results.multi_handedness[idx].classification[0].label
                    coord = hand_landmarks.landmark[0]
                    x_label = int(coord.x * image.shape[1])
                    y_label = int(coord.y * image.shape[0]) - 20
                    cv2.putText(image, label, (x_label, y_label),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Get thumb and index fingertip landmarks
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                # Convert to pixel coordinates
                x1, y1 = int(thumb_tip.x * image.shape[1]), int(thumb_tip.y * image.shape[0])
                x2, y2 = int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0])

                # Draw line between thumb and index
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Calculate distance
                distance = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)

                # Display the distance
                cv2.putText(image, f'Dist: {distance}px', (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)

        # Show the output
        cv2.imshow('Hand Distance Tracker', image)

        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
capture.release()
cv2.destroyAllWindows()
