"""
mmvis trt engine模型导出
"""

import os

# 用哪张卡导出, 只修改这里就行, --device固定为0, 不要动
device_id = 0
# deploy cfg 修改
deploy_cfg_path = "/root/corespace/mmab/configs/patchcore_tensorrt_config.py"
# model cfg 修改
model_cfg_path = "/root/corespace/mmab/configs/patchcore_config.py"
# 模型路径修改
model_checkpoint_path = "/root/corespace/mmab/workdirs/mmab_test/epoch_1.pth"
# 转换时样图路径修改
demo_jpg = "/root/corespace/mmdetection/demo/demo.jpg"
# trt engine存放位置修改
trt_engine_dir = "/root/corespace/mmab/workdirs/mmab_test/trt_engine"

if __name__ == "__main__":
    os.system(" ".join([
        f"export CUDA_VISIBLE_DEVICES={device_id} &&",
        "python",
        "/root/corespace/mmdeploy/tools/deploy.py",
        deploy_cfg_path,
        model_cfg_path,
        model_checkpoint_path,
        demo_jpg,
        "--work-dir",
        trt_engine_dir,
        "--device cuda:0", 
        "--dump-info"
    ]))