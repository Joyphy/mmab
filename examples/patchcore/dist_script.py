"""
单机多卡训练启动命令
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
ROOTDIR = "/root/workspace/mmab_patchcore_examples"

os.chdir(ROOTDIR)
os.system(" ".join([
    "python",
    "-m",
    "torch.distributed.launch",
    "--nproc_per_node=4",
    "--master_port=29500",
    "/root/corespace/mmab/tools/train.py",
    f"{ROOTDIR}/configs/patchcore_config.py",
    "--work-dir",
    f"{ROOTDIR}/workdirs/mmab_test",
    "--launcher",
    "pytorch"
]))