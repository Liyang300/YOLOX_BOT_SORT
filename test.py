import onnxruntime as ort
import numpy as np
import cv2

# 加载 ONNX
session = ort.InferenceSession("yolox_voc_s_no_decode.onnx")
input_name = session.get_inputs()[0].name

# 创建随机输入（或读取一张图片）
# 模拟预处理：resize to (960,544), /255, mean/std, HWC->CHW, 加 batch
dummy = np.random.randn(544, 960, 3).astype(np.float32)  # 随机输入，可改为真实图片
# 如果使用真实图片：
# img = cv2.imread("test.jpg")
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# dummy = cv2.resize(img_rgb, (960, 544)).astype(np.float32) / 255.0
# dummy = (dummy - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

blob = np.transpose(dummy, (2, 0, 1))[None, ...]  # (1,3,544,960)

outputs = session.run(None, {input_name: blob})
print("Output shapes:", [o.shape for o in outputs])
for i, out in enumerate(outputs):
    print(f"Output {i} shape {out.shape}, mean={out.mean():.4f}, std={out.std():.4f}")
    # 打印每个通道的均值
    for c in range(out.shape[1]):
        print(f"  Channel {c} mean: {out[0, c, :, :].mean():.4f}")