# -*- coding: utf-8 -*-
"""
Realtime gesture recognition using hand keypoints from pre-trained handpose model.
Uses the project's `ReXNetV1` (or other backbone) weights in
`handpose_x 预训练模型/` by default. Detection is rule-based on keypoint geometry:
  - `fist`, `one` (index up), `thumb_up`, `open` (all fingers extended)

Run:
  cd "handpose_x-main (1)\handpose_x-main"
  python gesture_realtime.py --model_path "handpose_x 预训练模型/ReXNetV1-size-256-wingloss102-0.122.pth"

This script depends on the project's model definitions and utils.
"""

import os
import argparse
import time
import math
import numpy as np
import torch
import cv2

from models.rexnetv1 import ReXNetV1
from models.resnet import resnet18,resnet34,resnet50,resnet101
from models.squeezenet import squeezenet1_1,squeezenet1_0
from models.shufflenetv2 import ShuffleNetV2
from models.shufflenet import ShuffleNet
from models.mobilenetv2 import MobileNetV2
from torchvision.models import shufflenet_v2_x1_5 ,shufflenet_v2_x1_0 , shufflenet_v2_x2_0

from utils.common_utils import *
from hand_data_iter.datasets import draw_bd_handpose


def build_model(name, num_classes, img_size, device):
    if name == 'ReXNetV1':
        model_ = ReXNetV1(num_classes=num_classes)
    elif name == 'resnet_50':
        model_ = resnet50(num_classes = num_classes,img_size=img_size)
    elif name == 'resnet_18':
        model_ = resnet18(num_classes = num_classes,img_size=img_size)
    elif name == 'resnet_34':
        model_ = resnet34(num_classes = num_classes,img_size=img_size)
    elif name == 'resnet_101':
        model_ = resnet101(num_classes = num_classes,img_size=img_size)
    elif name == "squeezenet1_0":
        model_ = squeezenet1_0(num_classes=num_classes)
    elif name == "squeezenet1_1":
        model_ = squeezenet1_1(num_classes=num_classes)
    elif name == "shufflenetv2":
        model_ = ShuffleNetV2(ratio=1., num_classes=num_classes)
    elif name == "shufflenet":
        model_ = ShuffleNet(num_blocks = [2,4,2], num_classes=num_classes, groups=3)
    elif name == "mobilenetv2":
        model_ = MobileNetV2(num_classes=num_classes)
    else:
        raise ValueError('Unsupported model: %s' % name)

    model_ = model_.to(device)
    model_.eval()
    return model_


def preprocess_frame(frame, img_size):
    img_ = cv2.resize(frame, (img_size[1], img_size[0]), interpolation = cv2.INTER_CUBIC)
    img_ = img_.astype(np.float32)
    img_ = (img_ - 128.) / 256.
    img_ = img_.transpose(2,0,1)
    img_ = torch.from_numpy(img_).unsqueeze(0)
    return img_


def euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def joint_angle(mcp, pip, tip):
    """Return angle at `pip` formed by vectors (mcp->pip) and (tip->pip) in degrees.
    When finger is extended, angle ~ 180; when folded, angle ~ 0.
    """
    try:
        v1x = mcp[0] - pip[0]
        v1y = mcp[1] - pip[1]
        v2x = tip[0] - pip[0]
        v2y = tip[1] - pip[1]
        dot = v1x * v2x + v1y * v2y
        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        denom = n1 * n2
        if denom < 1e-8:
            return 0.0
        cosv = max(-1.0, min(1.0, dot / denom))
        ang = math.degrees(math.acos(cosv))
        return ang
    except Exception:
        return 0.0


def detect_gesture(pts, img_w, img_h):
    """Improved rule-based gesture detection.
    Returns (gesture_name_or_None, finger_states_list)
    finger_states_list: list of 'E' (extended), 'F' (folded), 'U' (unknown)
    """
    try:
        tips_idx = [4,8,12,16,20]
        mcps_idx = [1,5,9,13,17]
        wrist = (pts['0']['x'], pts['0']['y'])
        tip_pts = [(pts[str(i)]['x'], pts[str(i)]['y']) for i in tips_idx]
        mcp_pts = [(pts[str(i)]['x'], pts[str(i)]['y']) for i in mcps_idx]
    except Exception:
        return None, ['U']*5
    # Use pip points to determine finger extension more robustly
    # Indices follow common 21-point layout: pip indices are [2,6,10,14,18]
    pip_idx = [2,6,10,14,18]
    hand_scale = euclid(wrist, tip_pts[2])
    if hand_scale < 1e-6:
        hand_scale = 1.0

    # angle-based test at pip joint (degrees). extended ~ 180, folded ~ 0
    ext_angle_thresh = 150.0
    fold_angle_thresh = 100.0

    states = []
    extended = []
    folded = []

    # For each finger, compute angle at PIP (mcp - pip - tip)
    for i in range(5):
        try:
            tip = tip_pts[i]
            pip = (pts[str(pip_idx[i])]['x'], pts[str(pip_idx[i])]['y'])
            mcp = mcp_pts[i]
            ang = joint_angle(mcp, pip, tip)
        except Exception:
            ang = 0.0

        is_ext = ang >= ext_angle_thresh
        is_fold = ang <= fold_angle_thresh
        extended.append(is_ext)
        folded.append(is_fold)
        states.append('E' if is_ext else ('F' if is_fold else 'U'))

    # Gesture rules (use more tolerant checks)
    # fist: all folded
    if all(folded):
        return 'fist', states

    # one: only index extended
    if extended[1] and all(not extended[i] for i in [0,2,3,4]):
        return 'one', states

    # two: index and middle extended
    if extended[1] and extended[2] and all(not extended[i] for i in [0,3,4]):
        return 'two', states

    # three: index, middle, ring extended
    if extended[1] and extended[2] and extended[3] and not extended[0] and not extended[4]:
        return 'three', states

    # four: index, middle, ring, pinky extended (thumb not)
    if extended[1] and extended[2] and extended[3] and extended[4] and not extended[0]:
        return 'four', states

    # five: all five fingers extended
    if all(extended):
        return 'five', states

    # thumb up: thumb extended and tip above wrist by margin
    if extended[0] and all(not extended[i] for i in [1,2,3,4]):
        thumb_tip = tip_pts[0]
        if thumb_tip[1] < wrist[1] - 0.15 * hand_scale:
            return 'thumb_up', states

    # open: majority extended
    if sum(1 for v in extended if v) >= 4:
        return 'open', states

    return None, states


def main():
    parser = argparse.ArgumentParser(description='Realtime gesture recognition')
    parser.add_argument('--model_path', type=str, default = 'handpose_x 预训练模型/ReXNetV1-size-256-wingloss102-0.122.pth')
    parser.add_argument('--model', type=str, default = 'ReXNetV1')
    parser.add_argument('--num_classes', type=int , default = 42)
    parser.add_argument('--GPUS', type=str, default = '0')
    parser.add_argument('--img_size', type=tuple , default = (256,256))
    parser.add_argument('--camera_id', type=int, default = 0)
    parser.add_argument('--vis', type=bool , default = True)
    parser.add_argument('--min_hand_scale', type=float, default = 0.02, help='min hand scale (fraction of short side) to consider hand present')
    parser.add_argument('--hand_presence_frames', type=int, default = 3, help='number of consecutive frames to confirm hand presence')
    parser.add_argument('--finger_alpha', type=float, default = 0.3, help='EMA alpha for smoothing per-finger angle values')
    parser.add_argument('--ext_thresh', type=float, default = 150.0, help='threshold on smoothed angle (deg) to consider finger extended')
    parser.add_argument('--fold_thresh', type=float, default = 100.0, help='threshold on smoothed angle (deg) to consider finger folded')
    parser.add_argument('--concentration_thresh', type=float, default = 0.03, help='if 21 pts span less than fraction of short side, treat as no hand')
    parser.add_argument('--tip_pip_min', type=float, default = 0.03, help='min tip-pip distance (fraction of short side) to consider finger not collapsed')
    parser.add_argument('--per_finger_conc_thresh', type=float, default = 0.02, help='per-finger concentration threshold (fraction of short side) to consider finger collapsed')
    parser.add_argument('--thumb_collapse_thresh', type=float, default = 0.018, help='thumb tip-to-mcp collapse threshold (fraction of short side) to consider thumb folded')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPUS
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    model_ = build_model(args.model, args.num_classes, args.img_size[0], device)

    if os.access(args.model_path, os.F_OK):
        chkpt = torch.load(args.model_path, map_location=device)
        try:
            model_.load_state_dict(chkpt)
        except Exception:
            # try if checkpoint is a dict with 'state_dict'
            if isinstance(chkpt, dict) and 'state_dict' in chkpt:
                model_.load_state_dict(chkpt['state_dict'])
            else:
                model_.load_state_dict(chkpt)
        print('Loaded model:', args.model_path)
    else:
        print('Warning: model_path not found:', args.model_path)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print('Cannot open camera id', args.camera_id)
        return

    prev_time = time.time()
    fps_smooth = None
    # hand presence smoothing counters
    hand_presence_counter = 0
    hand_absence_counter = 0
    hand_is_present = False
    concentrated_flag = False
    # per-finger smoothed angles (degrees) and states: helps avoid sticky 'E'/'U' and rapid jitter
    finger_angle_avg = [0.0] * 5
    finger_state = ['U'] * 5

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            img_t = preprocess_frame(frame, args.img_size)
            if use_cuda:
                img_t = img_t.cuda()

            out = model_(img_t.float())
            output = out.cpu().numpy()
            output = np.squeeze(output)

            pts_hand = {}
            for i in range(int(output.shape[0]/2)):
                x = (output[i*2+0]*float(w))
                y = (output[i*2+1]*float(h))
                pts_hand[str(i)] = {"x":x, "y":y}

            # 判断画面是否有手：通过手腕到中指尖的像素距离作为 hand_scale
            try:
                wrist = (pts_hand['0']['x'], pts_hand['0']['y'])
                mid_tip = (pts_hand['12']['x'], pts_hand['12']['y'])
                hand_scale = euclid(wrist, mid_tip)
            except Exception:
                hand_scale = 0.0

            # 阈值：取图像较短边的 min_hand_scale（可调整）
            min_hand_scale_px = min(w, h) * args.min_hand_scale
            has_hand = hand_scale >= min_hand_scale_px

            # 检测 21 点是否过度集中（例如误检单点或噪声）——若过于集中则视为无手
            try:
                pts_count = int(output.shape[0] / 2)
                xs = [pts_hand[str(i)]['x'] for i in range(pts_count)]
                ys = [pts_hand[str(i)]['y'] for i in range(pts_count)]
                max_span = max(max(xs) - min(xs), max(ys) - min(ys))
            except Exception:
                max_span = 0.0
            concentration_thresh_px = min(w, h) * args.concentration_thresh
            is_concentrated = (max_span < concentration_thresh_px)

            # 若过度集中则当作无手（但记录以显示 'none'）
            concentrated_flag = is_concentrated
            if concentrated_flag:
                has_hand = False

            # 平滑：需要连续若干帧确认手存在/消失，避免噪声导致闪烁
            if has_hand:
                hand_presence_counter += 1
                hand_absence_counter = 0
            else:
                hand_absence_counter += 1
                hand_presence_counter = 0

            if hand_presence_counter >= args.hand_presence_frames:
                hand_is_present = True
            if hand_absence_counter >= args.hand_presence_frames:
                hand_is_present = False

            if hand_is_present:
                # draw skeleton and points
                try:
                    draw_bd_handpose(frame, pts_hand, 0, 0)
                except Exception:
                    pass
                for i in range(int(output.shape[0]/2)):
                    x = (output[i*2+0]*float(w))
                    y = (output[i*2+1]*float(h))
                    cv2.circle(frame, (int(x),int(y)), 3, (255,50,60), -1)

                # compute per-finger angle at PIP and apply EMA smoothing
                tips_idx = [4,8,12,16,20]
                pip_idx = [2,6,10,14,18]
                mcps_idx = [1,5,9,13,17]
                extended = [False]*5
                tip_pip_min_px = min(w, h) * args.tip_pip_min
                thumb_collapse_px = min(w, h) * args.thumb_collapse_thresh
                for i in range(5):
                    try:
                        tip = (pts_hand[str(tips_idx[i])]['x'], pts_hand[str(tips_idx[i])]['y'])
                        pip = (pts_hand[str(pip_idx[i])]['x'], pts_hand[str(pip_idx[i])]['y'])
                        mcp = (pts_hand[str(mcps_idx[i])]['x'], pts_hand[str(mcps_idx[i])]['y'])
                        ang = joint_angle(mcp, pip, tip)
                    except Exception:
                        ang = 0.0
                    # EMA smoothing on angle (degrees)
                    alpha = max(0.0, min(1.0, args.finger_alpha))
                    finger_angle_avg[i] = finger_angle_avg[i] * (1.0 - alpha) + ang * alpha
                    # hysteresis thresholds to avoid flicker (angles in degrees)
                    # 除了角度外，还要求 tip-pip 距离不能太小（避免点集中时误判为伸展）
                    try:
                        tip_pip_dist = euclid(tip, pip)
                    except Exception:
                        tip_pip_dist = 0.0
                    
                    # per-finger concentration: measure span on this finger (tip-pip, pip-mcp, tip-mcp)
                    try:
                        pip_mcp_dist = euclid(pip, mcp)
                        tip_mcp_dist = euclid(tip, mcp)
                        finger_span = max(tip_pip_dist, pip_mcp_dist, tip_mcp_dist)
                    except Exception:
                        pip_mcp_dist = 0.0
                        tip_mcp_dist = 0.0
                        finger_span = 0.0

                    is_ext = (finger_angle_avg[i] >= args.ext_thresh) and (tip_pip_dist >= tip_pip_min_px)
                    # consider finger folded if angle low OR tip-pip very small OR per-finger span below threshold
                    per_finger_conc_px = min(w, h) * args.per_finger_conc_thresh
                    # special-case thumb: if thumb tip-to-mcp distance becomes very small, mark as folded
                    thumb_collapsed = False
                    try:
                        if i == 0 and tip_mcp_dist < thumb_collapse_px:
                            thumb_collapsed = True
                    except Exception:
                        thumb_collapsed = False
                    is_fold = (finger_angle_avg[i] <= args.fold_thresh) or (tip_pip_dist < (0.6 * tip_pip_min_px)) or (finger_span < per_finger_conc_px) or thumb_collapsed
                    if is_ext:
                        finger_state[i] = 'E'
                        extended[i] = True
                    elif is_fold:
                        finger_state[i] = 'F'
                        extended[i] = False
                    else:
                        # keep previous state if in between, otherwise unknown
                        if finger_state[i] in ('E','F'):
                            extended[i] = (finger_state[i] == 'E')
                        else:
                            finger_state[i] = 'U'
                            extended[i] = False

                # determine gesture based on smoothed discrete states
                gesture = None
                # fist: all folded
                if all(s == 'F' for s in finger_state):
                    gesture = 'fist'
                # one: only index extended
                elif extended[1] and all(not extended[i] for i in [0,2,3,4]):
                    gesture = 'one'
                # two: index and middle
                elif extended[1] and extended[2] and all(not extended[i] for i in [0,3,4]):
                    gesture = 'two'
                # three: index, middle, ring
                elif extended[1] and extended[2] and extended[3] and not extended[0] and not extended[4]:
                    gesture = 'three'
                # four: index, middle, ring, pinky (thumb not)
                elif extended[1] and extended[2] and extended[3] and extended[4] and not extended[0]:
                    gesture = 'four'
                # five: all five
                elif all(extended):
                    gesture = 'five'
                # thumb up: thumb extended and tip above wrist by margin
                elif extended[0] and all(not extended[i] for i in [1,2,3,4]):
                    try:
                        wrist = (pts_hand['0']['x'], pts_hand['0']['y'])
                        thumb_tip = (pts_hand['4']['x'], pts_hand['4']['y'])
                        if thumb_tip[1] < wrist[1] - 0.15 * hand_scale:
                            gesture = 'thumb_up'
                    except Exception:
                        pass
                # open: majority extended
                elif sum(1 for v in extended if v) >= 4:
                    gesture = 'open'
                states = finger_state
            else:
                # 无手：不绘制关键点，手势为空，重置平滑值与状态
                gesture = None
                states = ['U']*5
                finger_angle_avg = [0.0] * 5
                finger_state = ['U'] * 5
            # 显示手势文本：若点集中则显示 'none'，否则仅当手存在且识别出手势时显示
            if concentrated_flag:
                cv2.putText(frame, 'Gesture: none', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,255), 2)
            elif hand_is_present and gesture is not None:
                cv2.putText(frame, 'Gesture: %s' % gesture, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,255), 2)
            # show finger states only when hand is present
            if hand_is_present:
                state_str = ' '.join(states)
                cv2.putText(frame, 'Fingers: %s' % state_str, (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,50), 2)

            # FPS
            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps = 1.0 / dt if dt>0 else 0.0
            fps_smooth = fps if fps_smooth is None else (fps_smooth*0.9 + fps*0.1)
            cv2.putText(frame, 'FPS: %.1f' % fps_smooth, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

            if args.vis:
                cv2.imshow('gesture_realtime', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
