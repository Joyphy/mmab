import cv2, time
from mmab.infer import PatchCoreTRT

if __name__ == "__main__":
    model_trt = PatchCoreTRT(model_path="/root/corespace/mmab/workdirs/mmab_test/trt_engine", gaussian_blur=False, norm=False)
    image = cv2.imread("/root/workspace/datasets/bottle/test/broken_small/000.png")
    image = cv2.resize(image, (256, 256))
    for _ in range(100):
        t0 = time.time()
        image_score_trt, score_map_trt = model_trt(image)
        t1 = time.time()
        print(f"trt: {(t1 - t0)*1000:.2f} ms")