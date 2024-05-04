# Copyright (c) [2023] [Branko Brkljač, Faculty of Technical Sciences, University of Novi Sad]
#
# Licensed under the Lesser General Public License (LGPL)
# You may obtain a copy of the License at: https://www.gnu.org/licenses/lgpl-3.0.txt
#
# This software is provided "as is," without warranty of any kind. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability.
#

# Processing template
# processing_template.py


import numpy

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                        PyCUDA 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Pythonic access to Nvidia’s CUDA parallel computation API
import pycuda.autoinit
import pycuda.driver as cuda
print("pycuda.version: "+ str(cuda.get_version()))
print("pycuda.driver_version: "+ str(cuda.get_driver_version()))
cuda.Device.count()

# Get the CUDA device
device = cuda.Device(0)     # Assuming device 0

print("device: "+ device.name())
print("compute capability: "+ str(device.compute_capability()))
print("total memory: "+ str(round(device.total_memory()/(1024*1024*1024), 2))+" GB, 1MB=1024MB")
# print("(free, total) = "+str(cuda.mem_get_info()) + " Bytes")
freeGpuMem, totalGpuMem = cuda.mem_get_info()
print("free GPU memory (1GB=1024MB): "+ str(freeGpuMem) + " Bytes = " + str(round(freeGpuMem/(1024*1024*1024), 2))+" GB")
print("total GPU memory (1GB=1024MB): "+ str(totalGpuMem) + " Bytes = " + str(round(totalGpuMem/(1024*1024*1024), 2))+" GB")

device_attributes = device.get_attributes()
# print(device_attributes)


from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
""")

multiply_them = mod.get_function("multiply_them")

a = numpy.random.randn(400).astype(numpy.float32)
b = numpy.random.randn(400).astype(numpy.float32)

dest = numpy.zeros_like(a)
multiply_them(
        cuda.Out(dest), cuda.In(a), cuda.In(b),
        block=(400,1,1), grid=(1,1))

print(dest-a*b)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                        tensorrt 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import tensorrt as trt
print("tensorrt: "+ trt.__version__)
assert trt.Builder(trt.Logger())

import onnxruntime as ort
print("onnxruntime: "+ ort.__version__)
import numpy as np
import cv2


BATCH_SIZE=64
PRECISION = np.float32



# Load the model and create InferenceSession
model_path = "path/to/your/onnx/model"
session = ort.InferenceSession(model_path)
# Load and preprocess the input image inputTensor
...
# Run inference
outputs = session.run( {"input": inputTensor})
print(outputs)
