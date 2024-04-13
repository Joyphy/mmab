from mmab.infer import PatchCoreTRT, PatchCoreTorch
import cv2, time, os
import numpy as np

if __name__ == "__main__":
    model_trt = PatchCoreTRT(model_path="/root/corespace/mmab/workdirs/mmab_test/trt_engine")
    model_torch = PatchCoreTorch(
        model_config="/root/corespace/mmab/configs/patchcore_config.py",
        checkpoint="/root/corespace/mmab/workdirs/mmab_test/epoch_1.pth")
    
    for group in os.walk("/root/workspace/datasets/bottle"):
        for each_file in group[2]:
            if each_file.endswith("png") and "mask" not in each_file:
                image = cv2.imread(os.path.join(group[0], each_file))
                image = cv2.resize(image, (256, 256))
                for _ in range(1):
                    t0 = time.time()
                    image_score_trt, score_map_trt = model_trt(image)
                    t1 = time.time()
                    image_score_torch, score_map_torch = model_torch(image)
                    t2 = time.time()
                    print(f"trt: {(t1 - t0)*1000:.2f} ms, torch: {(t2 -t1)*1000:.2f} ms")

                image_score_err = abs(image_score_trt-image_score_torch)
                score_map_err = np.abs(score_map_trt-score_map_torch).mean().item()
                if (image_score_err > 1e-5 or score_map_err > 1e-4) or image_score_err == 0.0:
                    print(f"image_score误差: {image_score_err}")
                    print(f"score_map误差: {score_map_err}")
                print("*"*50)