# -*- coding: utf-8 -*-
"""
Realtime gesture recognition using hand keypoints from pre-trained handpose model.
Uses the project's `ReXNetV1` (or other backbone) weights in
`handpose_x 预训练模型/` by default. Detection is rule-based on keypoint geometry:
  - `fist`, `one` (index up), `thumb_up`, `open` (all fingers extended)

Run:
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

    hand_scale = euclid(wrist, tip_pts[2])
    if hand_scale < 1e-6:
        hand_scale = 1.0

    rel = [euclid(tip_pts[i], mcp_pts[i]) / hand_scale for i in range(5)]

    # empirical thresholds, tuned for relative ratios
    folded_thresh = 0.45
    extended_thresh = 0.6

    extended = [r > extended_thresh for r in rel]
    folded = [r < folded_thresh for r in rel]
    states = [ 'E' if extended[i] else ('F' if folded[i] else 'U') for i in range(5) ]

    # Gesture rules
    # fist: all folded
    if all(folded):
        return 'fist', states

    # one: only index extended
    if extended[1] and all(not extended[i] for i in [0,2,3,4]):
        return 'one', states

    # thumb up: thumb extended and thumb tip is above wrist by margin
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

            # draw skeleton and points
            try:
                draw_bd_handpose(frame, pts_hand, 0, 0)
            except Exception:
                pass
            for i in range(int(output.shape[0]/2)):
                x = (output[i*2+0]*float(w))
                y = (output[i*2+1]*float(h))
                cv2.circle(frame, (int(x),int(y)), 3, (255,50,60), -1)

            gesture, states = detect_gesture(pts_hand, w, h)
            if gesture is not None:
                cv2.putText(frame, 'Gesture: %s' % gesture, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,255), 2)
            # show finger states (Thumb,Index,Mid,Ring,Pinky)
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
