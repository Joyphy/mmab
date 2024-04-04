import torch
model_dict = torch.load("/root/corespace/mmab/workdirs/mmab_test/epoch_1.pth")
print(model_dict["state_dict"]["memory_bank"].shape)