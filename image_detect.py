from mmdet.apis import inference_detector, init_detector

image_path = "/root/workspace/datasets/bottle/test/broken_large/000.png"
config = "/root/corespace/mmab/configs/patchcore_config.py"
checkpoint = "/root/corespace/mmab/workdirs/mmab_test/epoch_1.pth"

model = init_detector(config, checkpoint, device="cuda:0", palette=[])
result = inference_detector(model, image_path)
print(result.shape)
# print(result)