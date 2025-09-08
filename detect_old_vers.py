# Pose Landmarker with Picamera2 on Raspberry Pi
# Based on Mediapipe example, modified for Pi Camera

import argparse
import sys
import time
import cv2
import mediapipe as mp
import numpy as np
from picamera2 import Picamera2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables for FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None


def run(model: str, num_poses: int,
        min_pose_detection_confidence: float,
        min_pose_presence_confidence: float, min_tracking_confidence: float,
        output_segmentation_masks: bool,
        width: int, height: int) -> None:
    """Run pose landmarker on frames from the Pi Camera."""

    # ---- Setup Picamera2 ----
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (width, height)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    # Visualization parameters
    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10
    overlay_alpha = 0.5
    mask_color = (100, 100, 0)  # cyan

    def save_result(result: vision.PoseLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, DETECTION_RESULT

        # Calculate FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        DETECTION_RESULT = result
        COUNTER += 1

    # ---- Setup Mediapipe Pose Landmarker ----
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=output_segmentation_masks,
        result_callback=save_result)
    detector = vision.PoseLandmarker.create_from_options(options)

    # ---- Main Loop ----
    while True:
        frame = picam2.capture_array()

        # Flip like webcam (optional)
        frame = cv2.flip(frame, 1)

        # Convert to Mediapipe Image
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run detection
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Show FPS
        fps_text = f'FPS = {FPS:.1f}'
        text_location = (left_margin, row_size)
        cv2.putText(frame, fps_text, text_location,
                    cv2.FONT_HERSHEY_DUPLEX,
                    font_size, text_color, font_thickness, cv2.LINE_AA)

        # Draw landmarks
        if DETECTION_RESULT:
            for pose_landmarks in DETECTION_RESULT.pose_landmarks:
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                                    z=landmark.z) for landmark
                    in pose_landmarks
                ])
                mp_drawing.draw_landmarks(
                    frame,
                    pose_landmarks_proto,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing_styles.get_default_pose_landmarks_style())

        # Segmentation mask (if enabled)
        if (output_segmentation_masks and DETECTION_RESULT):
            if DETECTION_RESULT.segmentation_masks is not None:
                segmentation_mask = DETECTION_RESULT.segmentation_masks[0].numpy_view()
                mask_image = np.zeros(frame.shape, dtype=np.uint8)
                mask_image[:] = mask_color
                condition = np.stack((segmentation_mask,) * 3, axis=-1) > 0.1
                visualized_mask = np.where(condition, mask_image, frame)
                frame = cv2.addWeighted(frame, overlay_alpha,
                                        visualized_mask, overlay_alpha, 0)

        cv2.imshow('pose_landmarker', frame)

        # Quit on ESC
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Name of the pose landmarker model bundle.',
        required=False,
        default='pose_landmarker.task')
    parser.add_argument(
        '--numPoses',
        help='Max number of poses that can be detected by the landmarker.',
        required=False,
        default=1)
    parser.add_argument(
        '--minPoseDetectionConfidence',
        help='The minimum confidence score for pose detection.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minPosePresenceConfidence',
        help='The minimum confidence score for pose presence.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minTrackingConfidence',
        help='The minimum confidence score for pose tracking.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--outputSegmentationMasks',
        help='Visualize the segmentation mask.',
        required=False,
        action='store_true')
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture.',
        required=False,
        default=800)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture.',
        required=False,
        default=800)
    args = parser.parse_args()

    run(args.model, int(args.numPoses), float(args.minPoseDetectionConfidence),
        float(args.minPosePresenceConfidence), float(args.minTrackingConfidence),
        args.outputSegmentationMasks,
        int(args.frameWidth), int(args.frameHeight))


if __name__ == '__main__':
    main()
