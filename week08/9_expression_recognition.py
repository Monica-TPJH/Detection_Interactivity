#!/usr/bin/env python3
"""
Simple expression (smile) recognition using MediaPipe Face Mesh.

Computes a smile score based on the relative vertical position of the
mouth corners compared to the mouth center. Positive score -> smile,
negative -> sad. Includes a --test mode that captures one frame, draws
landmarks and the computed score, and writes `/tmp/expression_test.jpg`.
"""

import cv2
import mediapipe as mp
import argparse
from camera_utils import setup_camera


def compute_smile_score(landmarks, image_w, image_h):
    """Compute a simple smile score from face mesh landmarks.

    We use the landmark indices from MediaPipe Face Mesh:
      - 61: mouth_left_corner
      - 291: mouth_right_corner
      - 0.. basic center use: 13 (upper lip), 14 (lower lip) approx

    Score = (avg(corner_y) - mouth_center_y) normalized by face height
    Positive -> corners are higher (smile), Negative -> corners lower (sad)
    """
    # Landmark indices (MediaPipe Face Mesh)
    LEFT_CORNER = 61
    RIGHT_CORNER = 291
    UPPER_LIP = 13
    LOWER_LIP = 14

    def to_xy(lm):
        return int(lm.x * image_w), int(lm.y * image_h)

    try:
        left = landmarks[LEFT_CORNER]
        right = landmarks[RIGHT_CORNER]
        upper = landmarks[UPPER_LIP]
        lower = landmarks[LOWER_LIP]
    except Exception:
        return None

    lx, ly = to_xy(left)
    rx, ry = to_xy(right)
    ux, uy = to_xy(upper)
    lx2, ly2 = to_xy(lower)

    mouth_center_y = (uy + ly2) / 2.0
    corners_avg_y = (ly + ry) / 2.0

    # Normalize by face height (distance between forehead-ish and chin-ish landmarks)
    # Use landmarks 10 (forehead) and 152 (chin) as rough vertical range
    TOP = 10
    BOTTOM = 152
    top = landmarks[TOP]
    bottom = landmarks[BOTTOM]
    _, top_y = to_xy(top)
    _, bottom_y = to_xy(bottom)
    face_height = max(1.0, bottom_y - top_y)

    # Smaller value -> more smile (corners higher than mouth center)
    raw = (mouth_center_y - corners_avg_y) / face_height

    # Scale to a friendly range roughly [-1..1]
    score = raw * 5.0
    return float(score)


def annotate_image(image, landmarks, score, image_w, image_h):
    out = image.copy()
    overlay = out.copy()
    alpha = 0.6
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    # Draw subtle face mesh contours for clarity
    try:
        mp_drawing.draw_landmarks(
            overlay,
            landmarks,
            mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
        )
    except Exception:
        pass

    # Compute mouth keypoints
    LEFT_CORNER = 61
    RIGHT_CORNER = 291
    UPPER_LIP = 13
    LOWER_LIP = 14

    lx = int(landmarks.landmark[LEFT_CORNER].x * image_w)
    ly = int(landmarks.landmark[LEFT_CORNER].y * image_h)
    rx = int(landmarks.landmark[RIGHT_CORNER].x * image_w)
    ry = int(landmarks.landmark[RIGHT_CORNER].y * image_h)
    ux = int(landmarks.landmark[UPPER_LIP].x * image_w)
    uy = int(landmarks.landmark[UPPER_LIP].y * image_h)
    lx2 = int(landmarks.landmark[LOWER_LIP].x * image_w)
    ly2 = int(landmarks.landmark[LOWER_LIP].y * image_h)

    # Draw mouth corner markers and connecting line
    cv2.line(overlay, (lx, ly), (rx, ry), (50, 230, 50), 2, cv2.LINE_AA)
    cv2.circle(overlay, (lx, ly), 6, (0, 200, 0), -1, cv2.LINE_AA)
    cv2.circle(overlay, (rx, ry), 6, (0, 200, 0), -1, cv2.LINE_AA)

    # Draw mouth center marker
    mouth_cx = int((ux + lx2) / 2)
    mouth_cy = int((uy + ly2) / 2)
    cv2.circle(overlay, (mouth_cx, mouth_cy), 5, (200, 180, 0), -1, cv2.LINE_AA)

    # Merge overlay with original image
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

    # Bottom score bar background
    bar_h = 28
    bar_w = int(image_w * 0.6)
    bar_x = 10
    bar_y = image_h - bar_h - 10
    cv2.rectangle(out, (bar_x - 2, bar_y - 2), (bar_x + bar_w + 2, bar_y + bar_h + 2), (30, 30, 30), -1)

    # Draw score indicator (map expected score roughly to [-1..1])
    if score is None:
        fill_w = 0
        color = (150, 150, 150)
        label = 'No face'
    else:
        s = max(-1.0, min(1.0, score))
        # center of bar is neutral; positive -> right, negative -> left
        center = bar_x + bar_w // 2
        if s >= 0:
            fill_w = int((bar_w // 2) * s)
            cv2.rectangle(out, (center, bar_y), (center + fill_w, bar_y + bar_h), (50, 180, 50), -1)
            color = (50, 180, 50)
        else:
            fill_w = int((bar_w // 2) * (-s))
            cv2.rectangle(out, (center - fill_w, bar_y), (center, bar_y + bar_h), (180, 50, 50), -1)
            color = (180, 50, 50)
        label = f'Score: {score:+.2f}'

    # Draw bar frame and label
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
    cv2.putText(out, label, (bar_x, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # Emotion label top-left with semi-transparent background
    emotion = 'Neutral'
    if score is not None:
        if score > 0.2:
            emotion = 'Happy'
        elif score < -0.2:
            emotion = 'Sad'
        else:
            emotion = 'Neutral'
    info = f'{emotion}'
    (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    pad = 8
    cv2.rectangle(out, (10, 10), (10 + tw + pad * 2, 10 + th + pad), (30, 30, 30), -1)
    cv2.putText(out, info, (10 + pad, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def run_once_save(image_path='/tmp/expression_test.jpg'):
    cap, camera_id = setup_camera()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print('❌ Could not read frame')
        return 2

    h, w, _ = frame.shape

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as fm:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            print('No face detected')
            cv2.imwrite(image_path, frame)
            print('Wrote', image_path)
            return 3

        landmarks = res.multi_face_landmarks[0]
        score = compute_smile_score(landmarks.landmark, w, h)
        out = annotate_image(frame, landmarks, score, w, h)
        cv2.imwrite(image_path, out)
        print('OK', image_path, 'score=', score)
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Capture one frame, annotate and save to /tmp')
    args = parser.parse_args()

    if args.test:
        return run_once_save()

    # Interactive mode
    cap, camera_id = setup_camera()
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as fm:
        print('Press q to quit')
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)

            score = None
            if res.multi_face_landmarks:
                score = compute_smile_score(res.multi_face_landmarks[0].landmark, w, h)
                frame = annotate_image(frame, res.multi_face_landmarks[0], score, w, h)

            cv2.imshow('Expression Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
