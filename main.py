import time
from collections import deque, Counter
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision.drawing_utils import draw_landmarks, DrawingSpec
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions, HandLandmarkerResult, HandLandmarksConnections
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizer, GestureRecognizerOptions, GestureRecognizerResult
from mediapipe.tasks.python.vision import RunningMode

gesture_recoginition_model_task_path = Path(__file__).parent / 'gesture_recognizer.task'          

latest_result: GestureRecognizerResult = None
gesture_history = deque(maxlen=8)

def on_results(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=str(gesture_recoginition_model_task_path)),
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=on_results,
    num_hands=2,
    min_tracking_confidence=0.6,
)

with GestureRecognizer.create_from_options(options) as gesture_recognizer:
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Error: Cannot open webcam")
        exit()
        
    frame_width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    file_encoding = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    out = cv2.VideoWriter('output.mp4', file_encoding, fps, (frame_width, frame_height))

    last_stable_gesture = None
    
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # 1. MediaPipe expects RGB format, OpenCV defaults to BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 2. Generate a unique, increasing timestamp in milliseconds
        frame_timestamp_ms = int(time.time() * 1000)
        
        # 3. detect_async does not return anything; data goes to your callback
        gesture_recognizer.recognize_async(mp_image, frame_timestamp_ms)
        
        # 4. stabilize and debounce gestures to prevent printing duplicate gestures
        if latest_result and latest_result.gestures:
            gesture_name = latest_result.gestures[0][0].category_name
            gesture_history.append(gesture_name)
            
            counts = Counter(gesture_history)
            most_common_label, count = counts.most_common(1)[0]

            if count / len(gesture_history) >= 0.75:  # e.g. 75% agreement
                stable_gesture = most_common_label
            else:
                stable_gesture = None  # not stable yet, don't act
            
            if stable_gesture != last_stable_gesture:
                if stable_gesture:
                    print(f"Stable gesture: {stable_gesture}")
                last_stable_gesture = stable_gesture
                    
        #5. draw on landmarks
        if latest_result and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:                 
                draw_landmarks(                                                 
                    frame, 
                    hand_landmarks,                      
                    HandLandmarksConnections.HAND_CONNECTIONS, 
                    DrawingSpec((35, 25, 66), 8, 10),
                    DrawingSpec((0, 255, 0), 2)                                                      
                )
                
        # 6. Write the original raw BGR frame to your file
        out.write(frame)
        
        # 7. Display the webcam frame (not the result object)
        cv2.imshow('Camera', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
webcam.release()
out.release()
cv2.destroyAllWindows()
