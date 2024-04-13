import cv2
from mmab.infer import PatchCoreTRT
from mmab.engine.hooks.visualization_hook import plot_fig

if __name__ == "__main__":
    model_trt = PatchCoreTRT(model_path="/root/workspace/mmab_patchcore_examples/workdirs/mmab_test/trt_engine", device_id=0)
    image = cv2.imread("/root/workspace/datasets/bottle/test/broken_small/000.png")
    image_score_trt, score_map_trt = model_trt(image)
    plot_fig(
        test_img=image,
        scores=score_map_trt,
        save_path="/root/workspace/mmab_patchcore_examples/test_mask_trt.png",
        threshold=0.5,
        image_score=image_score_trt,
        dpi=100
    )
    print("trt plot success!")