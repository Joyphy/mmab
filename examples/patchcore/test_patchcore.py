"""
单卡测试启动命令,  可评估指标和画图
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ROOTDIR = "/root/workspace/mmab_patchcore_examples"

os.chdir(ROOTDIR)
os.system(" ".join([
    "python",
    "/root/corespace/mmab/tools/test.py",
    f"{ROOTDIR}/configs/patchcore_config.py",
    f"{ROOTDIR}/workdirs/mmab_test/epoch_1.pth",
    "--work-dir",
    f"{ROOTDIR}/workdirs/mmab_test",
    "--show-dir test_show"
]))