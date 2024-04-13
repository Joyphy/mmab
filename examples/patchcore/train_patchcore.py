"""
单卡训练启动命令
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ROOTDIR = "/root/workspace/mmab_patchcore_examples"

os.chdir(ROOTDIR)
os.system(" ".join([
    "python",
    "/root/corespace/mmab/tools/train.py",
    f"{ROOTDIR}/configs/patchcore_config.py",
    "--work-dir",
    f"{ROOTDIR}/workdirs/mmab_test"
]))