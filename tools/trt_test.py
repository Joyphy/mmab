import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # This is needed for initializing CUDA driver

def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

def allocate_buffers(engine):
    inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = engine.get_tensor_dtype(binding).itemsize
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, np.float32)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream

def infer(context, bindings, inputs, outputs, stream, input_data):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(inputs[0]['device'], input_data, stream)
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    for output in outputs:
        cuda.memcpy_dtoh_async(output['host'], output['device'], stream)
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out['host'] for out in outputs]

# Initialize TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

# Load the TensorRT engine
engine_path = '/root/corespace/mmab/workdirs/mmab_test/trt_engine/end2end.engine'
engine = load_engine(trt_runtime, engine_path)

# Allocate buffers for input and output
inputs, outputs, bindings, stream = allocate_buffers(engine)

# Create an execution context
context = engine.create_execution_context()

# Prepare input data
input_data = np.random.rand(1, 3, 256, 256).astype(np.float32).ravel()

# Perform inference and measure time
for i in range(10):
    t0 = time.time()
    trt_outputs = infer(context, bindings, inputs, outputs, stream, input_data)
    t1 = time.time()
    print(f"TensorRT推理耗时为{(t1-t0)*1000:.2f} ms")

# Assuming model has two outputs: score_map and image_score
print(trt_outputs)
# score_map, image_score = trt_outputs
# print("Score Map:", score_map.shape)
# print("Image Score:", image_score.shape)