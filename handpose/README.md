主函数在gesture_realtime.py中运行
命令行运行命令：
  cd "handpose"
  python gesture_realtime.py --model_path "ReXNetV1-size-256-wingloss102-0.122.pth" --use_skin True --remove_head True

作者版本依赖：
numpy: 2.2.6
torch: 2.11.0+cpu
opencv/cv2: 4.10.0
thop: 0.1.1
模型地址：ReXNetV1-size-256-wingloss102-0.122.pth（已经在文件中）

代码实现了1-8，OK，shit（小指单举），fuckyou（中指单举），fist，thumb_up，左右手的共计26种手势识别

存在不足：
1.头颜色与手颜色一致，导致识别干扰
2.小指与无名指的伸展收缩识别灵敏度不高