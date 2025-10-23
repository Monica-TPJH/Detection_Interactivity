#!/usr/bin/env python3
"""
Simple interactive crown game:
 - Uses MediaPipe Face Detection to find faces in the camera frame
 - When exactly 3 faces are visible, randomly choose one to receive a clown overlay
 - Draws a crown image over the chosen face and displays the result
 - Supports `--test` mode that captures one frame, runs detection, selects (if 3 faces), saves to /tmp/crown_test.jpg
"""

import cv2
import mediapipe as mp
import random
import argparse
import numpy as np
from camera_utils import setup_camera

# Optional external clown/nose image (BGRA) loaded from --clown-image
external_clown = None
external_nose = None


def make_rgba_strip_white(img, white_thresh=240):
    """Return BGRA image where near-white pixels become transparent.
    white_thresh: any pixel with all channels >= white_thresh is treated as background.
    """
    if img is None:
        return None
    # ensure BGR or BGRA
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        bgr = img
        alpha = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
    else:
        bgr = img[..., :3]
        alpha = img[..., 3].copy()

    # compute mask of near-white pixels
    white_mask = np.all(bgr >= white_thresh, axis=2)
    alpha[white_mask] = 0
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = alpha
    return rgba


def load_crown_image(scale=1.0):
    # Create a cartoon-style clown overlay (procedural) to avoid external files
    clown = 255 * np.ones((160, 160, 4), dtype=np.uint8)
    clown[..., 3] = 0
    # Bold clown hat (bright red) with thick black outline
    hat_pts = np.array([[20,110], [80,20], [140,110]], np.int32).reshape((-1,1,2))
    cv2.fillPoly(clown, [hat_pts], (0,0,230,255))
    cv2.polylines(clown, [hat_pts], isClosed=True, color=(0,0,0,255), thickness=4)
    cv2.circle(clown, (80,30), 12, (0,200,200,255), -1)  # pompom

    # Big cartoon face (white) with bold black outline
    cv2.circle(clown, (80,95), 45, (255,255,255,255), -1)
    cv2.circle(clown, (80,95), 45, (0,0,0,255), 4)

    # Eyes with thick lashes
    cv2.ellipse(clown, (62,85), (8,6), 0, 0, 360, (0,0,0,255), -1)
    cv2.ellipse(clown, (98,85), (8,6), 0, 0, 360, (0,0,0,255), -1)
    cv2.circle(clown, (62,85), 3, (255,255,255,255), -1)
    cv2.circle(clown, (98,85), 3, (255,255,255,255), -1)

    # Big red nose with outline
    cv2.circle(clown, (80,100), 12, (0,0,255,255), -1)
    cv2.circle(clown, (80,100), 12, (0,0,0,255), 3)

    # Big smiling mouth (thick)
    cv2.ellipse(clown, (80,115), (24,12), 0, 0, 180, (0,0,0,255), 6)
    cv2.ellipse(clown, (80,115), (20,10), 0, 0, 180, (0,0,200,255), -1)

    # Colorful cheek spots
    cv2.circle(clown, (60,100), 8, (0,150,255,255), -1)
    cv2.circle(clown, (100,100), 8, (0,150,255,255), -1)

    return clown


def overlay_image_alpha(img, img_overlay, x, y, alpha_mask=None):
    """Overlay img_overlay on top of img at position (x, y) and blend using alpha_mask."""
    # Image ranges
    h, w = img_overlay.shape[:2]
    if x >= img.shape[1] or y >= img.shape[0]:
        return img
    # Clip overlay region to image bounds
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    overlay_x1 = x1 - x
    overlay_y1 = y1 - y
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    # Blend
    if img_overlay.shape[2] == 4:
        alpha = img_overlay[..., 3:4].astype(float) / 255.0
        foreground = img_overlay[..., :3].astype(float)
    else:
        alpha = np.ones((h, w, 1), dtype=float)
        foreground = img_overlay.astype(float)

    alpha = alpha[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    foreground = foreground[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    background = img[y1:y2, x1:x2].astype(float)

    out = background * (1 - alpha) + foreground * alpha
    img[y1:y2, x1:x2] = out.astype(np.uint8)
    return img


def run_once_save(image_path='/tmp/crown_test.jpg'):
    cap, camera_id = setup_camera()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print('❌ Could not read frame')
        return 2

    h, w, _ = frame.shape
    mp_face = mp.solutions.face_detection
    # Use the same model selection and confidence settings as 1_face_detection.py
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detector.process(rgb)
        count = 0 if not res.detections else len(res.detections)
        if count < 3:
            print('Need 3 faces; found', count)
            # draw any found detections for feedback
            if res.detections:
                h, w, _ = frame.shape
                for detection in res.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    confidence = detection.score[0] if detection.score else 0.0
                    cv2.putText(frame, f'{confidence:.2f}', (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Camera {camera_id} | Faces: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imwrite(image_path, frame)
            print('Wrote', image_path)
            return 3

    # pick 3 first detections
    dets = res.detections[:3]
    chosen_idx = random.randrange(3)

    for i, det in enumerate(dets):
        bbox = det.location_data.relative_bounding_box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        if i == chosen_idx:
            if external_clown is not None:
                target_w = int(bw * 1.4)
                target_h = int(bh * 1.6)
                clown_img = cv2.resize(external_clown, (target_w, target_h), interpolation=cv2.INTER_AREA)
                cx = x + bw // 2 - target_w // 2
                cy = y + bh // 2 - target_h // 2
                frame = overlay_image_alpha(frame, clown_img, cx, cy)
            else:
                # attempt to place a red clown nose at the detection's nose keypoint
                nose_x = None
                nose_y = None
                if hasattr(det.location_data, 'relative_keypoints') and det.location_data.relative_keypoints:
                    try:
                        kp = det.location_data.relative_keypoints[2]
                        nose_x = int(kp.x * w)
                        nose_y = int(kp.y * h)
                    except Exception:
                        nose_x = None
                if nose_x is None:
                    nose_x = x + bw // 2
                    nose_y = y + bh // 2
                radius = max(6, int(max(bw, bh) * 0.12))
                cv2.circle(frame, (nose_x, nose_y), radius, (0,0,255), -1)
                cv2.circle(frame, (nose_x, nose_y), radius, (0,0,0), 2)

    cv2.putText(frame, f'Chosen: {chosen_idx+1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.imwrite(image_path, frame)
    print('OK', image_path, 'chosen=', chosen_idx)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clown-image', help='Path to external clown PNG with alpha to overlay on selected face')
    parser.add_argument('--test', action='store_true', help='Capture one frame and save annotated result')
    args = parser.parse_args()

    # Load external clown image if provided
    global external_clown
    global external_nose
    if getattr(args, 'clown_image', None):
        path = args.clown_image
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"⚠️ Could not load clown image: {path}; continuing with procedural/red-nose fallback")
            external_clown = None
        else:
            # ensure 4 channels (BGRA)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                b, g, r = cv2.split(img)
                alpha = np.full_like(b, 255)
                img = cv2.merge([b, g, r, alpha])
            external_clown = img
            # create a nose asset by stripping white background
            external_nose = make_rgba_strip_white(img)
            # try to crop the nose region if the image is larger: center-crop to a square around center
            h_img, w_img = external_nose.shape[:2]
            size = min(h_img, w_img)
            cx = w_img // 2
            cy = h_img // 2
            x1 = max(0, cx - size // 2)
            y1 = max(0, cy - size // 2)
            external_nose = external_nose[y1:y1+size, x1:x1+size]

    if args.test:
        return run_once_save()

    cap, camera_id = setup_camera()
    mp_face = mp.solutions.face_detection
    chosen_idx = None
    # selection state persists until user clears (press 'r') or quits
    selection = {
        'active': False,
        'center': None,        # (x, y) in image coords of chosen face center
        'bbox': None,          # last bbox (x, y, w, h)
        'asset': None,         # BGRA image to overlay (full clown) or nose asset
        'asset_type': None     # 'full' or 'nose' or 'draw_nose'
    }
    # match main demo model selection and confidence
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
        print('Waiting for 3 faces...')
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = detector.process(rgb)
            count = 0 if not res.detections else len(res.detections)

            # draw boxes for any detections for user feedback
            if res.detections:
                for detection in res.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    confidence = detection.score[0] if detection.score else 0.0
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(frame, f'{confidence:.2f}', (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # trigger only when exactly 3 faces detected
            if count == 3 and chosen_idx is None:
                # prepare top-3 detections and positions
                dets = res.detections[:3]
                positions = []
                for det in dets:
                    bbox = det.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)
                    # store bbox coords so we can recolor the original detection box during animation
                    positions.append((x, y, bw, bh))

                # roulette-style animation: cycle through candidates, slowing down
                cycles = random.randint(6, 12)
                for i in range(cycles):
                    idx = i % 3
                    frame_anim = frame.copy()
                    bx, by, bw, bh = positions[idx]
                    # draw a highlight by recoloring the detected bounding box (no extra offset)
                    glow_color = (0, 200, 255) if (i % 2 == 0) else (0, 255, 0)
                    cv2.rectangle(frame_anim, (bx, by), (bx + bw, by + bh), glow_color, 6)
                    cv2.putText(frame_anim, 'Choosing...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                    cv2.imshow('Crown Game', frame_anim)
                    # delay increases as it goes, creating a slow-down effect
                    delay = int(40 + (i / float(cycles)) * random.randint(120, 500))
                    if cv2.waitKey(delay) & 0xFF == ord('q'):
                        chosen_idx = None
                        break

                # final selection is the remainder
                chosen_idx = cycles % 3
                # prepare persistent selection state
                det = dets[chosen_idx]
                bbox = det.location_data.relative_bounding_box
                bx = int(bbox.xmin * w)
                by = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                # prefer external full-clown overlay; if not available, use procedural clown; if nose-only asset exists, use nose
                if 'external_clown' in globals() and external_clown is not None:
                    target_w = int(bw * 1.4)
                    target_h = int(bh * 1.6)
                    clown_img = cv2.resize(external_clown, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    selection['asset'] = clown_img
                    selection['asset_type'] = 'full'
                    selection['bbox'] = (bx, by, bw, bh)
                    selection['center'] = (bx + bw // 2, by + bh // 2)
                    selection['active'] = True
                else:
                    # if external_nose available, overlay it centered on nose; otherwise use procedural clown face
                    nose_x = None
                    nose_y = None
                    if hasattr(det.location_data, 'relative_keypoints') and det.location_data.relative_keypoints:
                        try:
                            kp = det.location_data.relative_keypoints[2]
                            nose_x = int(kp.x * w)
                            nose_y = int(kp.y * h)
                        except Exception:
                            nose_x = None
                    if nose_x is None:
                        nose_x = bx + bw // 2
                        nose_y = by + bh // 2

                    if 'external_nose' in globals() and external_nose is not None:
                        nose_w = int(bw * 0.35)
                        nose_h = int(bh * 0.35)
                        nose_img = cv2.resize(external_nose, (nose_w, nose_h), interpolation=cv2.INTER_AREA)
                        selection['asset'] = nose_img
                        selection['asset_type'] = 'nose'
                        selection['bbox'] = (bx, by, bw, bh)
                        selection['center'] = (nose_x, nose_y)
                        selection['active'] = True
                    else:
                        # fallback to procedural full clown face so the user's request "clown face on chosen person" is honored
                        proc = load_crown_image()
                        target_w = int(bw * 1.4)
                        target_h = int(bh * 1.6)
                        clown_img = cv2.resize(proc, (target_w, target_h), interpolation=cv2.INTER_AREA)
                        selection['asset'] = clown_img
                        selection['asset_type'] = 'full'
                        selection['bbox'] = (bx, by, bw, bh)
                        selection['center'] = (bx + bw // 2, by + bh // 2)
                        selection['active'] = True

                # show one final frame with the chosen overlay before continuing; do not clear selection
                if selection['asset'] is not None:
                    if selection['asset_type'] == 'full':
                        aw = selection['asset'].shape[1]
                        ah = selection['asset'].shape[0]
                        cx = int(selection['center'][0] - aw // 2)
                        cy = int(selection['center'][1] - ah // 2)
                        frame = overlay_image_alpha(frame, selection['asset'], cx, cy)
                    elif selection['asset_type'] == 'nose':
                        aw = selection['asset'].shape[1]
                        ah = selection['asset'].shape[0]
                        nx = int(selection['center'][0] - aw // 2)
                        ny = int(selection['center'][1] - ah // 2)
                        frame = overlay_image_alpha(frame, selection['asset'], nx, ny)

                cv2.putText(frame, f'Chosen: {chosen_idx+1} (press r to clear)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                cv2.imshow('Crown Game', frame)

                # start a short reveal timer (in ms) after which selection will auto-clear and game continues
                reveal_time_ms = 2000
                t0 = cv2.getTickCount()
                reveal_done = False
                while True:
                    # wait small intervals but process camera frames so UI remains responsive
                    if cv2.waitKey(50) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    # break early if user pressed 'r' to clear
                    if cv2.waitKey(1) & 0xFF == ord('r'):
                        selection['active'] = False
                        selection['center'] = None
                        selection['bbox'] = None
                        selection['asset'] = None
                        selection['asset_type'] = None
                        reveal_done = True
                        break
                    # check elapsed
                    t1 = cv2.getTickCount()
                    elapsed_ms = (t1 - t0) / cv2.getTickFrequency() * 1000.0
                    if elapsed_ms >= reveal_time_ms:
                        reveal_done = True
                        break
                # after reveal timeout, clear selection and continue main loop
                if reveal_done:
                    selection['active'] = False
                    selection['center'] = None
                    selection['bbox'] = None
                    selection['asset'] = None
                    selection['asset_type'] = None
                    chosen_idx = None

            cv2.putText(frame, f'Camera {camera_id} | Faces: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, "Press 'r' to clear selection", (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # If a persistent selection is active, try to find the corresponding face each frame and overlay
            if selection['active']:
                # If detections available, pick the detection whose center is closest to selection['center']
                target_x, target_y = selection['center'] if selection['center'] is not None else (None, None)
                overlay_drawn = False
                if res.detections:
                    best_det = None
                    best_dist = None
                    for det in res.detections:
                        bbox = det.location_data.relative_bounding_box
                        cx = int((bbox.xmin + bbox.width/2.0) * w)
                        cy = int((bbox.ymin + bbox.height/2.0) * h)
                        if target_x is None:
                            best_det = det
                            break
                        d = (cx - target_x) ** 2 + (cy - target_y) ** 2
                        if best_dist is None or d < best_dist:
                            best_dist = d
                            best_det = det

                    if best_det is not None:
                        bbox = best_det.location_data.relative_bounding_box
                        bx = int(bbox.xmin * w)
                        by = int(bbox.ymin * h)
                        bw = int(bbox.width * w)
                        bh = int(bbox.height * h)
                        # update selection center to follow the face
                        selection['center'] = (bx + bw // 2, by + bh // 2) if selection['asset_type'] == 'full' else selection['center']

                        if selection['asset_type'] == 'full' and selection['asset'] is not None:
                            aw = selection['asset'].shape[1]
                            ah = selection['asset'].shape[0]
                            cx = int(selection['center'][0] - aw // 2)
                            cy = int(selection['center'][1] - ah // 2)
                            frame = overlay_image_alpha(frame, selection['asset'], cx, cy)
                            overlay_drawn = True
                        elif selection['asset_type'] == 'nose' and selection['asset'] is not None:
                            # find nose keypoint if available
                            nose_x = None
                            nose_y = None
                            if hasattr(best_det.location_data, 'relative_keypoints') and best_det.location_data.relative_keypoints:
                                try:
                                    kp = best_det.location_data.relative_keypoints[2]
                                    nose_x = int(kp.x * w)
                                    nose_y = int(kp.y * h)
                                except Exception:
                                    nose_x = None
                            if nose_x is None:
                                nose_x = bx + bw // 2
                                nose_y = by + bh // 2
                            aw = selection['asset'].shape[1]
                            ah = selection['asset'].shape[0]
                            nx = int(nose_x - aw // 2)
                            ny = int(nose_y - ah // 2)
                            frame = overlay_image_alpha(frame, selection['asset'], nx, ny)
                            overlay_drawn = True

                # if no detections or overlay couldn't be drawn, keep last known overlay location
                if not overlay_drawn and selection['asset'] is not None and selection['center'] is not None:
                    if selection['asset_type'] == 'full':
                        aw = selection['asset'].shape[1]
                        ah = selection['asset'].shape[0]
                        cx = int(selection['center'][0] - aw // 2)
                        cy = int(selection['center'][1] - ah // 2)
                        frame = overlay_image_alpha(frame, selection['asset'], cx, cy)
                    elif selection['asset_type'] == 'nose':
                        aw = selection['asset'].shape[1]
                        ah = selection['asset'].shape[0]
                        nx = int(selection['center'][0] - aw // 2)
                        ny = int(selection['center'][1] - ah // 2)
                        frame = overlay_image_alpha(frame, selection['asset'], nx, ny)
            cv2.imshow('Crown Game', frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            if key & 0xFF == ord('r'):
                # clear persistent selection
                selection['active'] = False
                selection['center'] = None
                selection['bbox'] = None
                selection['asset'] = None
                selection['asset_type'] = None
                chosen_idx = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
