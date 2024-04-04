import time
import numpy as np
import onnxruntime as ort

# 指定ONNX模型文件路径
onnx_model_path = "/root/corespace/mmab/workdirs/mmab_test/trt_engine/end2end.onnx"

# 创建ONNX Runtime会话
session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])

# 获取模型的输入和输出节点名称
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]
print(input_name)
print(output_names)

# 创建一个示例输入数据
# 这里假设输入是一个256x256的单通道图像
# 如果你的模型期待不同的输入格式，请相应调整
input_data = np.random.rand(1, 3, 256, 256).astype(np.float32)

# 进行推理
for _ in range(5):
    outputs = session.run(output_names, {input_name: input_data})

for i in range(1000):
    t0 = time.time()
    outputs = session.run(output_names, {input_name: input_data})
    t1 = time.time()
    print(f"ONNX推理耗时为{(t1-t0)*1000:.2f} ms")

# 输出推理结果
# 根据你的模型，输出可能包括得分图、图像得分等
# 这里假设模型有两个输出：score_map和image_score
score_map, image_score = outputs
print("Score Map:", score_map.shape)
print("Image Score:", image_score.shape)