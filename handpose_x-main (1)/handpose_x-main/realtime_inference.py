# python realtime_inference.py --model_path ./weights/your_model.pth --model ReXNetV1 --camera_id 0 --GPUS 0

# -*- coding: utf-8 -*-
# Realtime webcam inference based on inference.py
import os
import argparse
import time
import numpy as np
import torch
import cv2

from models.resnet import resnet18,resnet34,resnet50,resnet101
from models.squeezenet import squeezenet1_1,squeezenet1_0
from models.shufflenetv2 import ShuffleNetV2
from models.shufflenet import ShuffleNet
from models.mobilenetv2 import MobileNetV2
from torchvision.models import shufflenet_v2_x1_5 ,shufflenet_v2_x1_0 , shufflenet_v2_x2_0
from models.rexnetv1 import ReXNetV1

from utils.common_utils import *
from hand_data_iter.datasets import draw_bd_handpose


def build_model(name, num_classes, img_size, device):
    if name == 'resnet_50':
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
    elif name == "shufflenet_v2_x1_5":
        model_ = shufflenet_v2_x1_5(pretrained=False,num_classes=num_classes)
    elif name == "shufflenet_v2_x1_0":
        model_ = shufflenet_v2_x1_0(pretrained=False,num_classes=num_classes)
    elif name == "shufflenet_v2_x2_0":
        model_ = shufflenet_v2_x2_0(pretrained=False,num_classes=num_classes)
    elif name == "shufflenet":
        model_ = ShuffleNet(num_blocks = [2,4,2], num_classes=num_classes, groups=3)
    elif name == "mobilenetv2":
        model_ = MobileNetV2(num_classes=num_classes)
    elif name == "ReXNetV1":
        model_ = ReXNetV1(num_classes=num_classes)
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


def main():
    parser = argparse.ArgumentParser(description='Realtime HandPose Inference (webcam)')
    parser.add_argument('--model_path', type=str, default = './weights/ReXNetV1-size-256-wingloss102-0.122.pth')
    parser.add_argument('--model', type=str, default = 'ReXNetV1')
    parser.add_argument('--num_classes', type=int , default = 42)
    parser.add_argument('--GPUS', type=str, default = '0')
    parser.add_argument('--img_size', type=tuple , default = (256,256))
    parser.add_argument('--camera_id', type=int, default = 0)
    parser.add_argument('--vis', type=bool , default = True)
    parser.add_argument('--fp16', action='store_true', help='use fp16 (only with CUDA)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPUS
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    model_ = build_model(args.model, args.num_classes, args.img_size[0], device)

    if os.access(args.model_path, os.F_OK):
        chkpt = torch.load(args.model_path, map_location=device)
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
                if args.fp16:
                    img_t = img_t.half()

            out = model_(img_t.float())
            output = out.cpu().numpy()
            output = np.squeeze(output)

            pts_hand = {}
            for i in range(int(output.shape[0]/2)):
                x = (output[i*2+0]*float(w))
                y = (output[i*2+1]*float(h))
                pts_hand[str(i)] = {"x":x, "y":y}

            try:
                draw_bd_handpose(frame, pts_hand, 0, 0)
            except Exception:
                pass

            for i in range(int(output.shape[0]/2)):
                x = (output[i*2+0]*float(w))
                y = (output[i*2+1]*float(h))
                cv2.circle(frame, (int(x),int(y)), 3, (255,50,60), -1)

            # FPS
            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps = 1.0 / dt if dt>0 else 0.0
            fps_smooth = fps if fps_smooth is None else (fps_smooth*0.9 + fps*0.1)
            cv2.putText(frame, 'FPS: %.1f' % fps_smooth, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

            if args.vis:
                cv2.imshow('realtime_handpose', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
