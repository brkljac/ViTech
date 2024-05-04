# Copyright (c) [2023] [Branko Brkljač, Faculty of Technical Sciences, University of Novi Sad]
#
# Licensed under the Lesser General Public License (LGPL)
# You may obtain a copy of the License at: https://www.gnu.org/licenses/lgpl-3.0.txt
#
# This software is provided "as is," without warranty of any kind. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability.
#

# Conversion of PyTorch ONNX model to TensorRT and measurement of inference speed up
# ONNX2TensorRT_example.py


USE_FP16 = True				# False, use FP32
skip_conversion = True			# True, just load the existing TensorRT model

BATCH_SIZE = 16				# size of the input batch, needs to match ONNX model batch size

model_base_name = "resnet50_pytorch"		# model name
model_relative_path = "models/torch/resnet50/"	# relative path to the .onnx file

# clear gpu mem
import torch
torch.cuda.empty_cache()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#           Create a test batch of image data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt
import numpy as np
import os

img_file = os.path.join(os.getcwd(),"test_image/machine.jpg")
img = resize(io.imread(img_file), (224, 224))
img = np.expand_dims(np.array(img, dtype=np.float32), axis=0) # Expand image to have a batch dimension
input_batch = np.array(np.repeat(img, BATCH_SIZE, axis=0), dtype=np.float32) # Repeat across the batch dimension
# print("Size of the input batch: " + str(input_batch.shape))
# e.g. (32, 224, 224, 3)
f1 = plt.figure(1)
plt.imshow(input_batch[0,:].astype(np.float32), aspect='auto')
plt.axis('off')
f1.show()

#  By default TensorRT will use the input precision of the runtime as the default precision for the rest of the network
target_dtype = np.float16 if USE_FP16 else np.float32


# Preprocess image to match normalization performed by PyTorch during model training
# Since the model was trained on ImageNet, for more details please check the following code:
# https://github.com/pytorch/examples/blob/c0b889d5f43150f288ecdd5dbd16c146d79e5bdf/imagenet/main.py#L233
#

from torchvision.transforms import Normalize
def preprocess_image(img):
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    result = norm(torch.from_numpy(img).transpose(0,2).transpose(1,2))
    return np.array(result, dtype=target_dtype)
preprocessed_images = np.array([preprocess_image(image) for image in input_batch])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#           Conversion of PyTorch ONNX model to TensorRT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

if USE_FP16:
    fp_precision = 16
else:
    fp_precision = 32

input_model_name =  os.path.join(os.getcwd(), model_relative_path, "{model_base_name}_bsize{batch_size}.onnx".format(model_base_name=model_base_name, batch_size=str(BATCH_SIZE)))

trt_model_name = input_model_name.rsplit(".",1)[0]+"__fp{fp_precision}_engine.trt".format(fp_precision = str(fp_precision))

# Create TensorRT engine file (use fixed/explicit batch size)
if not skip_conversion:
    import subprocess
    trtexec_path = "/usr/src/tensorrt/bin/trtexec"	# convertor
    if USE_FP16:
        subprocess.run([trtexec_path, '--onnx={onnx_model}'.format(onnx_model=input_model_name), '--saveEngine={tensorrt_model}'.format(tensorrt_model=trt_model_name), '--explicitBatch', '--inputIOFormats=fp16:chw', '--outputIOFormats=fp16:chw', '--fp16'])
    else:
        subprocess.run([trtexec_path, '--onnx={onnx_model}'.format(onnx_model=input_model_name), '--saveEngine={tensorrt_model}'.format(tensorrt_model=trt_model_name), '--explicitBatch'])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#           GPU execution with TensorRT acceleration (FP32 or FP16)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# In order to get better speedup, set input and output precisions to FP16

# Open generated .trt file and create runtime object
print(f'\nTensorRT model running on GPU: {trt_model_name[trt_model_name.rfind("/")+1:]}')
print("Size of the input batch: " + str(preprocessed_images.shape))
f = open(trt_model_name, "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

# Read model, create inference engine and an execution context
engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# Define the shape and the type of output data
output = np.empty([BATCH_SIZE, 1000], dtype = target_dtype) 

# Allocate CUDA device memory for input and output data
d_input = cuda.mem_alloc(1 * input_batch.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

# Convert allocated memory addresses to integer values and put them in the list
# Values represent pointers to the device memory and are used to initializie Bindings object withing the TensorRT engine execution context
bindings = [int(d_input), int(d_output)]

# Linear sequence of execution that belongs to a specific CUDA device
stream = cuda.Stream()

def predict(batch):
    # transfer input data to device using asynchronous host-to-device CUDA copy
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute TensorRT inference pipeline (model) using the asynchronous execute API
    context.execute_async_v2(bindings, stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()
    return output


predictions = predict(preprocessed_images)


import timeit
code_snippet = '''
pred = predict(preprocessed_images)
'''
t = timeit.Timer(stmt=code_snippet, globals=globals())
execution_times = t.repeat(repeat=5, number=1)
average_time = sum(execution_times) / len(execution_times)
print(f'Average TensorRT execution time on GPU: {average_time:.6f} seconds')


# Print top 5 scores of single image predictions (e.g. tractor id is 866)
print("**** Top 5 prediction scores ****")
indices = (-predictions[0]).argsort()[:5]

import json
class_index_path = os.path.join(os.getcwd(),"models/torch/imagenet_class_index.json")
with open(class_index_path, 'r') as f:
    class_index = json.load(f)
for i in indices:
    class_name = class_index[str(i)][1]
    print(f'Class ID: {i}, Class name: {class_name}')

print("Class ID | Likelihood score")
print(list(zip(indices, predictions[0][indices])))



