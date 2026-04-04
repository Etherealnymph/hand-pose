# -*- coding: utf-8 -*-
"""
Run:
  cd "handpose"
  python gesture_realtime.py --model_path "ReXNetV1-size-256-wingloss102-0.122.pth" --use_skin True --remove_head True

This script depends on the project's model definitions and utils.
"""

import os
import argparse
import time
import math
import numpy as np
import torch
import cv2
from collections import deque

from models.rexnetv1 import ReXNetV1
from models.resnet import resnet18,resnet34,resnet50,resnet101
from models.squeezenet import squeezenet1_1,squeezenet1_0
from models.shufflenetv2 import ShuffleNetV2
from models.shufflenet import ShuffleNet
from models.mobilenetv2 import MobileNetV2
from torchvision.models import shufflenet_v2_x1_5 ,shufflenet_v2_x1_0 , shufflenet_v2_x2_0

from utils.common_utils import *
from hand_data_iter.datasets import draw_bd_handpose

# 修正：拇指专用阈值，更符合生理结构
DEFAULT_THUMB_RATIO_THRESH = 0.3  # 比值低于此值判定为折叠（原0.45太松）
THUMB_ANGLE_THRESH = 120.0         # 拇指与手掌夹角低于此值判定为折叠
THUMB_TIP_Y_THRESH = 0.0           # 拇指tip在手掌下方（y更大）判定为折叠
SMOOTH_WINDOW = 5                   # 每点平滑帧数（可调整）
MAX_DEV_RATIO = 0.5                 # 异常点距离阈值比例，相对于 hand_scale


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


def angle_between(v1, v2):
    try:
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 * n2 < 1e-8:
            return 0.0
        cosv = max(-1.0, min(1.0, dot / (n1 * n2)))
        return math.degrees(math.acos(cosv))
    except Exception:
        return 0.0


def detect_gesture(pts, img_w, img_h):
    """Improved rule-based gesture detection with fixed thumb folding.
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
    pip_idx = [2,6,10,14,18]
    hand_scale = euclid(wrist, tip_pts[2])
    if hand_scale < 1e-6:
        hand_scale = 1.0

    ext_angle_thresh = 150.0
    fold_angle_thresh = 100.0

    states = []
    extended = []
    folded = []

    try:
        palm_center = tuple(np.mean(np.array(mcp_pts), axis=0))
    except Exception:
        palm_center = mcp_pts[0]

    # 计算手掌方向向量（手腕 -> 中指MCP，用于拇指夹角计算）
    palm_vec = (mcp_pts[2][0] - wrist[0], mcp_pts[2][1] - wrist[1])

    for i in range(5):
        try:
            tip = tip_pts[i]
            pip = (pts[str(pip_idx[i])]['x'], pts[str(pip_idx[i])]['y'])
            mcp = mcp_pts[i]
            ang = joint_angle(mcp, pip, tip)
        except Exception:
            ang = 0.0

        if i == 0:
            # 🔴 拇指专用判断逻辑（使用用户指定的比值公式）
            try:
                # 使用拇指上的四个点：TIP(4), DIP(3), PIP(2), MCP(1)，以及Wrist(0)
                tip_pt = (pts['4']['x'], pts['4']['y']) if '4' in pts else tip
                dip_pt = (pts['3']['x'], pts['3']['y']) if '3' in pts else pip
                pip_pt = (pts['2']['x'], pts['2']['y']) if '2' in pts else mcp
                mcp_pt = (pts['1']['x'], pts['1']['y']) if '1' in pts else mcp
                wrist_pt = (pts['0']['x'], pts['0']['y']) if '0' in pts else wrist

                tip_dip = euclid(tip_pt, dip_pt)
                dip_pip = euclid(dip_pt, pip_pt)
                pip_mcp = euclid(pip_pt, mcp_pt)
                mcp_wrist = euclid(mcp_pt, wrist_pt)

                numer = tip_dip + dip_pip
                denom = numer + pip_mcp + mcp_wrist + 1e-8
                ratio = numer / denom
                print('%.3f', ratio)

                thumb_vec = (tip_pt[0] - mcp_pt[0], tip_pt[1] - mcp_pt[1])
                thumb_angle = angle_between(thumb_vec, palm_vec)
                tip_relative_y = (tip_pt[1] - palm_center[1]) / hand_scale
            except Exception:
                ratio = 1.0
                thumb_angle = 180.0
                tip_relative_y = -1.0

            # 三重条件判断：任意满足则判定为折叠
            # is_fold = (ratio < DEFAULT_THUMB_RATIO_THRESH) or (thumb_angle < THUMB_ANGLE_THRESH) or (tip_relative_y > THUMB_TIP_Y_THRESH)
            is_fold = (ratio < DEFAULT_THUMB_RATIO_THRESH) or (tip_relative_y > THUMB_TIP_Y_THRESH)
            is_ext = not is_fold
        else:
            # 其他四指保持原有逻辑
            is_ext = ang >= ext_angle_thresh
            is_fold = ang <= fold_angle_thresh

        extended.append(is_ext)
        folded.append(is_fold)
        states.append('E' if is_ext else ('F' if is_fold else 'U'))

    # 🔴 修正手势判定顺序：先判断非five手势，避免兜底误判
    # fist: all folded
    if all(folded):
        return 'fist', states
    
    if extended[2] and all(not extended[i] for i in [0,1,3,4]):
        return 'fuckyou', states
    
    if extended[4] and all(not extended[i] for i in [0,1,2,3]):
        return 'shit', states

    # one: only index extended
    if extended[1] and all(not extended[i] for i in [0,2,3,4]):
        return '1', states

    # two: index and middle extended
    if extended[1] and extended[2] and all(not extended[i] for i in [0,3,4]):
        return '2', states

    # custom gestures: 6,7,8,fuckyou,shit
    # 6: thumb + pinky
    if extended[0] and extended[4] and all(not extended[i] for i in [1,2,3]):
        return '6', states

    # 7: thumb + index
    if extended[0] and extended[1] and all(not extended[i] for i in [2,3,4]):
        return '7', states

    # OK: 拇指与食指指尖距离小于阈值，且中指/无名指/小拇指伸展
    try:
        thumb_tip = tip_pts[0]
        index_tip = tip_pts[1]
        tip_dist = euclid(thumb_tip, index_tip)
    except Exception:
        tip_dist = float('inf')
    if tip_dist < 0.15 * hand_scale and extended[2] and extended[3] and extended[4]:
        return 'OK', states

    # 8: thumb + index + middle
    if extended[0] and extended[1] and extended[2] and all(not extended[i] for i in [3,4]):
        return '8', states

    # fuckyou: only middle finger
    if extended[2] and all(not extended[i] for i in [0,1,3,4]):
        return 'fuckyou', states

    # shit: only pinky
    if extended[4] and all(not extended[i] for i in [0,1,2,3]):
        return 'shit', states

    # three: index, middle, ring extended
    if extended[1] and extended[2] and extended[3] and not extended[0] and not extended[4]:
        return '3', states

    # four: index, middle, ring, pinky extended (thumb not)
    if extended[1] and extended[2] and extended[3] and extended[4] and not extended[0]:
        return '4', states
    
    if not extended[1] and not extended[2] and not extended[3] and extended[4] and extended[0]:
        return 'six', states
    
    if extended[1] and not extended[2] and not extended[3] and not extended[4] and extended[0]:
        return 'seven', states
    
    if extended[1] and extended[2] and not extended[3] and not extended[4] and extended[0]:
        return 'eight', states

    # five: all five fingers extended
    if all(extended):
        return '5', states

    # thumb up: thumb extended and tip above wrist by margin
    if extended[0] and all(not extended[i] for i in [1,2,3,4]):
        thumb_tip = tip_pts[0]
        if thumb_tip[1] < wrist[1] - 0.15 * hand_scale:
            return 'thumb_up', states

    # open: majority extended（仅当没有其他手势匹配时兜底）
    if sum(1 for v in extended if v) >= 4:
        return 'open', states

    return None, states


def main():
    parser = argparse.ArgumentParser(description='Realtime gesture recognition (fixed thumb)')
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
    parser.add_argument('--thumb_ratio_thresh', type=float, default = DEFAULT_THUMB_RATIO_THRESH, help='ratio threshold for thumb')
    parser.add_argument('--use_skin', type=bool, default = True, help='whether to apply skin-color masking before inference')
    parser.add_argument('--remove_head', type=bool, default = True, help='whether to mask detected face/head region to avoid influence')
    parser.add_argument('--ok_tip_dist_frac', type=float, default = 0.15, help='fraction of hand_scale used as OK thumb-index tip distance threshold')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPUS
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    model_ = build_model(args.model, args.num_classes, args.img_size[0], device)

    # 加载人脸检测器（用于屏蔽头部区域，减少干扰）
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception:
        face_cascade = None

    if os.access(args.model_path, os.F_OK):
        chkpt = torch.load(args.model_path, map_location=device)
        try:
            model_.load_state_dict(chkpt)
        except Exception:
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
    hand_presence_counter = 0
    hand_absence_counter = 0
    hand_is_present = False
    concentrated_flag = False
    finger_angle_avg = [0.0] * 5
    finger_state = ['U'] * 5
    pts_buffer = deque(maxlen=SMOOTH_WINDOW)

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # 可选：先基于肤色检测出手部区域，再屏蔽人脸区域，避免脑袋对手势的影响
            frame_proc = frame.copy()
            try:
                if args.use_skin:
                    hsv = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2HSV)
                    # 常用肤色 HSV 范围 & YCrCb 辅助
                    lower_hsv = np.array([0, 30, 60])
                    upper_hsv = np.array([25, 200, 255])
                    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

                    ycrcb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2YCrCb)
                    lower_ycrcb = np.array((0,133,77))
                    upper_ycrcb = np.array((255,173,127))
                    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

                    skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
                    # 形态学去噪
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                    skin_mask = cv2.medianBlur(skin_mask, 5)
                    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

                    # 若启用 remove_head，则检测人脸并从 mask 中移除人脸区域
                    if args.remove_head and face_cascade is not None:
                        gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
                        for (fx,fy,fw,fh) in faces:
                            cv2.rectangle(skin_mask, (fx,fy), (fx+fw, fy+fh), 0, -1)

                    # 将非皮肤区域设为中灰(128)，以降低对回归模型的误导
                    inv_mask = cv2.bitwise_not(skin_mask)
                    frame_proc[inv_mask>0] = (128,128,128)
                else:
                    frame_proc = frame
            except Exception:
                frame_proc = frame

            img_t = preprocess_frame(frame_proc, args.img_size)
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

            # 将当前帧关键点加入缓冲，并用缓冲中的中位数替换明显离群值
            try:
                # 计算手尺度（用于判断异常点阈值）
                wrist = (pts_hand['0']['x'], pts_hand['0']['y'])
                mid_tip = (pts_hand['12']['x'], pts_hand['12']['y'])
                hand_scale = euclid(wrist, mid_tip)
            except Exception:
                hand_scale = 0.0

            pts_buffer.append(pts_hand)

            # 平滑/过滤：对每个点使用缓冲帧的中位数作为参考，若当前点与中位数偏差过大则替换
            try:
                if len(pts_buffer) > 0:
                    pts_count = int(output.shape[0] / 2)
                    pts_filtered = {}
                    # 当hand_scale very small时使用像素尺的一个保底小值
                    hs = hand_scale if hand_scale > 1e-6 else max(1.0, 0.02 * min(w, h))
                    max_dev = MAX_DEV_RATIO * hs  # 若当前点偏离中位数超过此距离则认为是离群
                    for idx in range(pts_count):
                        key = str(idx)
                        xs = []
                        ys = []
                        for b in pts_buffer:
                            if key in b:
                                xs.append(b[key]['x'])
                                ys.append(b[key]['y'])
                        if len(xs) == 0:
                            continue
                        mx = float(np.median(np.array(xs)))
                        my = float(np.median(np.array(ys)))
                        cur_x = pts_hand[key]['x'] if key in pts_hand else mx
                        cur_y = pts_hand[key]['y'] if key in pts_hand else my
                        if euclid((cur_x, cur_y), (mx, my)) > max_dev:
                            # 用中位数替换异常点
                            pts_filtered[key] = {"x": mx, "y": my}
                        else:
                            pts_filtered[key] = {"x": cur_x, "y": cur_y}
                    # 用过滤后的点替代后续处理使用的pts_hand
                    pts_hand = pts_filtered
            except Exception:
                pass

            try:
                wrist = (pts_hand['0']['x'], pts_hand['0']['y'])
                mid_tip = (pts_hand['12']['x'], pts_hand['12']['y'])
                hand_scale = euclid(wrist, mid_tip)
            except Exception:
                hand_scale = 0.0

            min_hand_scale_px = min(w, h) * args.min_hand_scale
            has_hand = hand_scale >= min_hand_scale_px

            try:
                pts_count = int(output.shape[0] / 2)
                xs = [pts_hand[str(i)]['x'] for i in range(pts_count)]
                ys = [pts_hand[str(i)]['y'] for i in range(pts_count)]
                max_span = max(max(xs) - min(xs), max(ys) - min(ys))
            except Exception:
                max_span = 0.0
            concentration_thresh_px = min(w, h) * args.concentration_thresh
            is_concentrated = (max_span < concentration_thresh_px)

            concentrated_flag = is_concentrated
            if concentrated_flag:
                has_hand = False

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
                try:
                    draw_bd_handpose(frame, pts_hand, 0, 0)
                except Exception:
                    pass
                for i in range(int(output.shape[0]/2)):
                    x = (output[i*2+0]*float(w))
                    y = (output[i*2+1]*float(h))
                    cv2.circle(frame, (int(x),int(y)), 3, (255,50,60), -1)

                tips_idx = [4,8,12,16,20]
                pip_idx = [2,6,10,14,18]
                mcps_idx = [1,5,9,13,17]
                try:
                    mcp_points = [(pts_hand[str(idx)]['x'], pts_hand[str(idx)]['y']) for idx in mcps_idx]
                    palm_center = tuple(np.mean(np.array(mcp_points), axis=0))
                    # 计算手掌方向向量
                    wrist_pt = (pts_hand['0']['x'], pts_hand['0']['y'])
                    palm_vec = (mcp_points[2][0] - wrist_pt[0], mcp_points[2][1] - wrist_pt[1])
                except Exception:
                    mcp_points = [(pts_hand[str(idx)]['x'], pts_hand[str(idx)]['y']) if str(idx) in pts_hand else (0.0,0.0) for idx in mcps_idx]
                    palm_center = mcp_points[0]
                    palm_vec = (0,1)
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
                    alpha = max(0.0, min(1.0, args.finger_alpha))
                    finger_angle_avg[i] = finger_angle_avg[i] * (1.0 - alpha) + ang * alpha
                    try:
                        tip_pip_dist = euclid(tip, pip)
                    except Exception:
                        tip_pip_dist = 0.0
                    
                    try:
                        pip_mcp_dist = euclid(pip, mcp)
                        tip_mcp_dist = euclid(tip, mcp)
                        finger_span = max(tip_pip_dist, pip_mcp_dist, tip_mcp_dist)
                    except Exception:
                        pip_mcp_dist = 0.0
                        tip_mcp_dist = 0.0
                        finger_span = 0.0

                    if i == 0:
                        # 拇指专用判断（使用 TIP(4), DIP(3), PIP(2), MCP(1), Wrist(0)）
                        try:
                            tip_pt = (pts_hand['4']['x'], pts_hand['4']['y']) if '4' in pts_hand else tip
                            dip_pt = (pts_hand['3']['x'], pts_hand['3']['y']) if '3' in pts_hand else pip
                            pip_pt = (pts_hand['2']['x'], pts_hand['2']['y']) if '2' in pts_hand else mcp
                            mcp_pt = (pts_hand['1']['x'], pts_hand['1']['y']) if '1' in pts_hand else mcp
                            wrist_pt = (pts_hand['0']['x'], pts_hand['0']['y']) if '0' in pts_hand else wrist_pt

                            tip_dip = euclid(tip_pt, dip_pt)
                            dip_pip = euclid(dip_pt, pip_pt)
                            pip_mcp = euclid(pip_pt, mcp_pt)
                            mcp_wrist = euclid(mcp_pt, wrist_pt)

                            numer = tip_dip + dip_pip
                            denom = numer + pip_mcp + mcp_wrist + 1e-8
                            ratio = numer / denom

                            thumb_vec = (tip_pt[0] - mcp_pt[0], tip_pt[1] - mcp_pt[1])
                            thumb_angle = angle_between(thumb_vec, palm_vec)
                            tip_relative_y = (tip_pt[1] - palm_center[1]) / hand_scale
                        except Exception:
                            ratio = 1.0
                            thumb_angle = 180.0
                            tip_relative_y = -1.0

                        # is_fold = (ratio < args.thumb_ratio_thresh) or (thumb_angle < THUMB_ANGLE_THRESH) or (tip_relative_y > THUMB_TIP_Y_THRESH)
                        is_fold = (ratio < args.thumb_ratio_thresh) or (tip_relative_y > THUMB_TIP_Y_THRESH)
                        is_ext = not is_fold
                    else:
                        # 其他四指逻辑
                        is_ext = (finger_angle_avg[i] >= args.ext_thresh) and (tip_pip_dist >= tip_pip_min_px)
                        is_fold = (finger_angle_avg[i] <= args.fold_thresh) or (tip_pip_dist < (0.6 * tip_pip_min_px)) or (finger_span < per_finger_conc_px)

                    per_finger_conc_px = min(w, h) * args.per_finger_conc_thresh
                    thumb_collapsed = False
                    try:
                        if i == 0 and tip_mcp_dist < thumb_collapse_px:
                            thumb_collapsed = True
                    except Exception:
                        thumb_collapsed = False
                    is_fold = is_fold or thumb_collapsed

                    if is_ext:
                        finger_state[i] = 'E'
                        extended[i] = True
                    elif is_fold:
                        finger_state[i] = 'F'
                        extended[i] = False
                    else:
                        if finger_state[i] in ('E','F'):
                            extended[i] = (finger_state[i] == 'E')
                        else:
                            finger_state[i] = 'U'
                            extended[i] = False

                gesture = None
                if all(s == 'F' for s in finger_state):
                    gesture = 'fist'
                elif extended[1] and all(not extended[i] for i in [0,2,3,4]):
                    gesture = '1'
                elif extended[1] and extended[2] and all(not extended[i] for i in [0,3,4]):
                    gesture = '2'
                elif extended[0] and extended[4] and all(not extended[i] for i in [1,2,3]):
                    gesture = '6'
                elif extended[0] and extended[1] and all(not extended[i] for i in [2,3,4]):
                    gesture = '7'
                elif extended[0] and extended[1] and extended[2] and all(not extended[i] for i in [3,4]):
                    gesture = '8'
                elif extended[2] and all(not extended[i] for i in [0,1,3,4]):
                    gesture = 'fuckyou'
                elif extended[4] and all(not extended[i] for i in [0,1,2,3]):
                    gesture = 'shit'
                elif extended[1] and extended[2] and extended[3] and not extended[0] and not extended[4]:
                    gesture = '3'
                elif extended[1] and extended[2] and extended[3] and extended[4] and not extended[0]:
                    gesture = '4'
                elif all(extended):
                    gesture = '5'
                elif extended[0] and all(not extended[i] for i in [1,2,3,4]):
                    try:
                        wrist = (pts_hand['0']['x'], pts_hand['0']['y'])
                        thumb_tip = (pts_hand['4']['x'], pts_hand['4']['y'])
                        if thumb_tip[1] < wrist[1] - 0.15 * hand_scale:
                            gesture = 'thumb_up'
                    except Exception:
                        pass
                elif sum(1 for v in extended if v) >= 4:
                    gesture = 'open'
                elif True:
                    # 改为：拇指与食指指尖距离小于阈值，且中指/无名指/小拇指伸展
                    try:
                        thumb_tip = (pts_hand['4']['x'], pts_hand['4']['y']) if '4' in pts_hand else None
                        index_tip = (pts_hand['8']['x'], pts_hand['8']['y']) if '8' in pts_hand else None
                        if thumb_tip is not None and index_tip is not None:
                            tip_dist = euclid(thumb_tip, index_tip)
                        else:
                            tip_dist = float('inf')
                    except Exception:
                        tip_dist = float('inf')
                    if tip_dist < 0.15 * hand_scale and extended[2] and extended[3] and extended[4]:
                        gesture = 'OK'
                states = finger_state
                # 判断左右手：基于拇指尖相对于掌心的横向位置（图像坐标）
                try:
                    thumb_tip_disp = (pts_hand['4']['x'], pts_hand['4']['y']) if '4' in pts_hand else None
                    if thumb_tip_disp is not None:
                        hand_side = 'Left' if thumb_tip_disp[0] < palm_center[0] else 'Right'
                    else:
                        hand_side = 'Unknown'
                except Exception:
                    hand_side = 'Unknown'
            else:
                gesture = None
                states = ['U']*5
                finger_angle_avg = [0.0] * 5
                finger_state = ['U'] * 5

            if concentrated_flag:
                cv2.putText(frame, 'Gesture: none', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,255), 2)
            elif hand_is_present and gesture is not None:
                # 在显示中加入左右手信息
                side_str = (' (%s)' % hand_side) if 'hand_side' in locals() else ''
                cv2.putText(frame, 'Gesture: %s%s' % (gesture, side_str), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,255), 2)
            if hand_is_present:
                state_str = ' '.join(states)
                cv2.putText(frame, 'Fingers: %s' % state_str, (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,50), 2)

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