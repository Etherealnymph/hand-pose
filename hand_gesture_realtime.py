#!/usr/bin/env python3
"""
实时手势识别（基于 MediaPipe Hands + 简单规则分类器）

功能：
- 打开摄像头并实时检测手部关键点
- 基于关键点判断五指伸展状态并映射到常见手势：Open/Fist/Peace/ThumbsUp/Point
- 在视频帧上绘制关键点和识别结果

依赖：mediapipe, opencv-python, numpy

用法：
  python hand_gesture_realtime.py

按 `q` 退出。
"""
import cv2
import numpy as np
# 兼容不同版本/安装方式的 mediapipe 导入，并在失败时给出兼容性建议
import importlib
import sys

mp_hands = None
mp_drawing = None

def _try_import_mediapipe():
    global mp_hands, mp_drawing
    # 常见导入尝试顺序
    candidates = [
        ("import mediapipe as mp; mp.solutions", None),
        ("from mediapipe import solutions as mp_solutions", None),
        ("from mediapipe.python import solutions as mp_solutions", None),
    ]

    for code, _ in candidates:
        try:
            # 动态执行导入表达式并拿到 solutions 名称
            local = {}
            exec(code, globals(), local)
            if 'mp' in local and hasattr(local['mp'], 'solutions'):
                sol = local['mp'].solutions
            elif 'mp_solutions' in local:
                sol = local['mp_solutions']
            else:
                continue

            mp_hands = sol.hands
            mp_drawing = sol.drawing_utils
            return True
        except Exception:
            continue

    # 再尝试通过 importlib 导入可能的模块路径
    for modname in ('mediapipe.solutions', 'mediapipe.python.solutions'):
        try:
            sol = importlib.import_module(modname)
            mp_hands = sol.hands
            mp_drawing = sol.drawing_utils
            return True
        except Exception:
            continue

    return False


if not _try_import_mediapipe():
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    msg = (
        "无法导入 mediapipe。常见原因是 Python 版本与 mediapipe 不兼容。\n"
        f"当前 Python 版本: {py_ver}\n"
        "建议:\n"
        "  1) 使用 Python 3.8/3.9/3.10/3.11 创建虚拟环境（例如 pyenv 或 从 python.org 安装）\n"
        "  2) 在新环境中运行: pip install mediapipe opencv-python numpy\n"
        "快速示例（PowerShell/命令提示符）:\n"
        "  python -m venv .venv && .venv\\Scripts\\activate && pip install --upgrade pip && pip install mediapipe opencv-python numpy\n"
        "如果你必须在当前 Python 版本上运行，请尝试查找适配该版本的 mediapipe 发行版或切换解释器。"
    )
    raise SystemExit(msg)


def fingers_up(landmarks, handedness='Right'):
    """返回五根手指是否伸展的布尔数组 [thumb, index, middle, ring, pinky]"""
    # landmarks: 21 个点，x,y 相对坐标
    tips_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]

    fingers = []

    # 对于拇指，比较 x（水平）方向（取决于左右手）
    thumb_tip = landmarks[tips_ids[0]]
    thumb_ip = landmarks[pip_ids[0]]
    if handedness == 'Right':
        fingers.append(thumb_tip[0] < thumb_ip[0])
    else:
        fingers.append(thumb_tip[0] > thumb_ip[0])

    # 其余手指，通过 tip.y 与 pip.y 比较（图像坐标系：y 越小越上）
    for i in range(1, 5):
        tip = landmarks[tips_ids[i]]
        pip = landmarks[pip_ids[i]]
        fingers.append(tip[1] < pip[1])

    return fingers


def classify_gesture(fingers):
    """基于 fingers 布尔列表返回手势名称"""
    # fingers: [thumb, index, middle, ring, pinky]
    t, i, m, r, p = fingers

    # 常见映射
    if all(x is False for x in fingers):
        return 'Fist'
    if all(x is True for x in fingers):
        return 'Open'
    if (i and m) and (not r and not p):
        return 'Peace'
    if t and not i and not m and not r and not p:
        return 'ThumbsUp'
    if i and not m and not r and not p and not t:
        return 'Point'

    # 默认返回 fingers 状态
    return 'Unknown'


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open camera')
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img = cv2.flip(frame, 1)
            h, w, _ = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)

            gesture_text = ''

            if results.multi_hand_landmarks:
                for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # 提取 (x,y) 列表
                    lm = [(p.x, p.y) for p in hand_landmarks.landmark]
                    handedness_label = hand_handedness.classification[0].label if hand_handedness and hand_handedness.classification else 'Right'

                    fingers = fingers_up(lm, handedness=handedness_label)
                    gesture_text = classify_gesture(fingers)

                    # 在帧上绘制关键点
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # 绘制手指状态小图
                    status_text = ''.join(['1' if x else '0' for x in fingers])
                    cv2.putText(img, f'{handedness_label} {status_text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # 在右上角显示识别结果
            if gesture_text:
                cv2.putText(img, f'Gesture: {gesture_text}', (10, img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,128,255), 2)

            cv2.imshow('Hand Gesture (q to quit)', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
