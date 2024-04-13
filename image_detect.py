import cv2
from mmab.infer import PatchCoreTorch
from mmab.engine.hooks.visualization_hook import plot_fig

if __name__ == "__main__":
    model_torch = PatchCoreTorch(
        model_config="/root/corespace/mmab/configs/patchcore_config.py",
        checkpoint="/root/corespace/mmab/workdirs/mmab_test/epoch_1.pth",
        device="cuda:0"
    )
    image = cv2.imread("/root/workspace/datasets/bottle/test/broken_small/000.png")
    image_score_torch, score_map_torch = model_torch(image)
    plot_fig(
        test_img=image,
        scores=score_map_torch,
        save_path="./test_mask_torch.png",
        threshold=0.5,
        image_score=image_score_torch,
        dpi=100
    )
    print("torch plot success!")