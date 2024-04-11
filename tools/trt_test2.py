import torch, time
import numpy as np
from mmdeploy.backend.tensorrt import TRTWrapper
inputs = torch.from_numpy(np.load("/root/corespace/mmab/x.npy").reshape((1, 3, 256, 256))).to("cuda:0")

engine_file = '/root/corespace/mmab/workdirs/mmab_test/trt_engine/end2end.engine'
model1 = TRTWrapper(engine_file, device_id=0)
inputs = dict(input=inputs)
for _ in range(10):
    t0 = time.time()
    outputs = model1(inputs)
    t1 = time.time()
    print(f"{(t1-t0)*1000:.2f}ms")
print(outputs["dets"].cpu().numpy().tolist())
a = outputs["masks"].cpu().numpy()
print(a.min(), a.max())