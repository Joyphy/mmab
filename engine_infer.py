import cv2, time
from mmdeploy_runtime import Detector

image = cv2.imread("/root/workspace/datasets/bottle/test/broken_small/001.png")
model = Detector(model_path="/root/corespace/mmab/workdirs/mmab_test/trt_engine", device_name="cuda", device_id=0)

for _ in range(100):
    t0 = time.time()
    dets, labels, masks = model(image)
    t1 = time.time()
    print(f"{(t1-t0)*1000:.2f} ms")

print(dets.shape, labels.shape)
print(dets)
print(masks[0].shape)
print(masks[0].dtype)
print(masks[0].min())
print(masks[0].max())

cv2.imwrite("test_mask.png", masks[0])