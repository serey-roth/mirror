import argparse
import time
from collections import deque, Counter
from multiprocessing import shared_memory
from pathlib import Path
from typing import Literal

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision.drawing_utils import draw_landmarks, DrawingSpec
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions, HandLandmarkerResult, HandLandmarksConnections
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizer, GestureRecognizerOptions, GestureRecognizerResult
from mediapipe.tasks.python.vision import RunningMode

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig

import hand_sim

gesture_recoginition_model_task_path = Path(__file__).parent / 'gesture_recognizer.task'

latest_result: GestureRecognizerResult = None
gesture_history = deque(maxlen=8)

# dex_retargeting's configs are built against the standard "MANO" hand-pose
# convention (landmarks expressed in the hand's own local frame), not raw
# camera-space coordinates. This is ported directly from dex_retargeting's
# own reference SingleHandDetector (example/vector_retargeting/single_hand_detector.py)
# -- feeding raw hand_world_landmarks straight into retarget() looks
# geometrically sane in camera space but is in the wrong reference frame,
# which is what was causing several joints to saturate at their limits
# regardless of the real hand's actual pose.
OPERATOR2MANO_RIGHT = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
])
OPERATOR2MANO_LEFT = np.array([
    [0, 0, -1],
    [1, 0, 0],
    [0, -1, 0],
])

# https://raw.githubusercontent.com/dexsuite/dex-retargeting/main/example/vector_retargeting/single_hand_detector.py
def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """Estimate the hand's own local coordinate frame from wrist + index-MCP + middle-MCP."""
    points = keypoint_3d_array[[0, 5, 9], :]
    x_vector = points[0] - points[2]
    points = points - np.mean(points, axis=0, keepdims=True)
    _, _, v = np.linalg.svd(points)
    normal = v[2, :]
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)
    if np.sum(z * (points[1] - points[2])) < 0:
        normal = -normal
        z = -z
    return np.stack([x, normal, z], axis=1)


def mediapipe_to_mano(world_landmarks, hand_type: str) -> np.ndarray:
    """21x3 raw camera-frame world landmarks -> 21x3 in MANO hand-local convention."""
    keypoint_3d_array = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks])
    keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
    wrist_rot = estimate_frame_from_hand_points(keypoint_3d_array)
    operator2mano = OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
    return keypoint_3d_array @ wrist_rot @ operator2mano


def build_retargeting(side: str):
    RetargetingConfig.set_default_urdf_dir(str(Path(__file__).parent))
    hand_type = HandType.right if side == "right" else HandType.left
    config_path = get_default_config_path(RobotName.svh, RetargetingType.vector, hand_type)
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    # retarget() outputs qpos ordered per retargeting.joint_names, which is
    # dex_retargeting/pinocchio's own joint order -- not necessarily the same
    # order hand_sim.py's shared-memory array uses. Precompute the remap once.
    sim_joint_names = hand_sim.get_joint_names(side)
    remap = np.array([retargeting.joint_names.index(name) for name in sim_joint_names])
    return retargeting, remap


def attach_target_shm(nu: int, name: str = hand_sim.TARGET_SHM_NAME):
    while True:
        try:
            shm = shared_memory.SharedMemory(name=name, create=False)
            return shm, np.ndarray((nu,), dtype=np.float64, buffer=shm.buf)
        except FileNotFoundError:
            print(f"Waiting for hand_sim.py to create shared memory '{name}'...")
            time.sleep(1)
            

def on_results(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result


def run(side: Literal["left", "right"] = "right"):
    print(f"main.py: retargeting for {side.upper()} hand -- hand_sim.py must use --side {side} too")
    retargeting, retarget_to_sim = build_retargeting(side)
    target_shm, targets = attach_target_shm(len(retargeting.joint_names))
    origin_indices, task_indices = retargeting.optimizer.target_link_human_indices
    target_handedness = "Right" if side == "right" else "Left"

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

            # 5b. retarget the hand matching --side's world landmarks onto the
            # Schunk SVH hand and write the result into hand_sim.py's shared
            # target buffer. hand_world_landmarks (unlike hand_landmarks) are in
            # real-world meters, hand-relative -- what dex_retargeting expects.
            # The retargeting config is handedness-specific, so we must use
            # the landmark set MediaPipe actually labeled as the matching
            # hand, not just index 0 (which can be either hand once
            # num_hands=2 detects two).
            if latest_result and latest_result.hand_world_landmarks:
                hand_idx = next(
                    (i for i, h in enumerate(latest_result.handedness)
                     if h[0].category_name == target_handedness),
                    None,
                )
                if hand_idx is not None:
                    world_landmarks = latest_result.hand_world_landmarks[hand_idx]
                    joint_pos = mediapipe_to_mano(world_landmarks, target_handedness)
                    ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                    qpos = retargeting.retarget(ref_value)
                    targets[:] = qpos[retarget_to_sim]

            # 6. Write the original raw BGR frame to your file
            out.write(frame)
            
            # 7. Display the webcam frame (not the result object)
            cv2.imshow('Camera', frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
        
    webcam.release()
    out.release()
    cv2.destroyAllWindows()
    target_shm.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand gesture recognition -> Schunk SVH retargeting")
    parser.add_argument("--side", choices=["left", "right"], default="right")
    args = parser.parse_args()
    
    run(args.side)