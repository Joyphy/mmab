# import torch
# model_dict = torch.load("/root/corespace/mmab/workdirs/mmab_test/epoch_1.pth")
# print(model_dict["state_dict"]["memory_bank"].shape)
import torch
x = torch.zeros([1, 3, 256, 256])
score_map = torch.zeros([1, 256, 256])

dets = torch.tensor(data=[[0, 0, x.shape[2], x.shape[3], 0.3]], dtype=torch.float32, device=x.device)
labels = torch.zeros([x.shape[0], 1], dtype=torch.int32, device=x.device)
print(dets.shape, labels.shape, score_map.unsqueeze(0).shape)