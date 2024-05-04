# Copyright (c) [2023] [Branko Brkljač, Faculty of Technical Sciences, University of Novi Sad]
#
# Licensed under the Lesser General Public License (LGPL)
# You may obtain a copy of the License at: https://www.gnu.org/licenses/lgpl-3.0.txt
#
# This software is provided "as is," without warranty of any kind. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability.
#

# Running of PyTorch models on GPU without additional optimization (acceleration)
# PyTorchGPUComparison_example.py


import numpy as np
import torchvision.models as models
import torch

torch.cuda.empty_cache()

# Load the pretrained model, .pth file
# after downloading check the folder /home/blab/.cache/torch/hub/checkpoints for .pth files
# Create nn model and set it to evaluation mode
resnet50 = models.resnet50(pretrained=True, progress=True).eval()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#           Create a test batch of image data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt
import numpy as np
import os

BATCH_SIZE=32
img_file = os.path.join(os.getcwd(),"test_image/machine.jpg")
img = resize(io.imread(img_file), (224, 224))
img = np.expand_dims(np.array(img, dtype=np.float32), axis=0) # Expand image to have a batch dimension
input_batch = np.array(np.repeat(img, BATCH_SIZE, axis=0), dtype=np.float32) # Repeat across the batch dimension
print("Size of the input batch: " + str(input_batch.shape))
# e.g. (32, 224, 224, 3)
f1 = plt.figure(1)
plt.imshow(input_batch[0,:].astype(np.float32), aspect='auto')
plt.axis('off')
f1.show()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#           GPU execution using standard PyTorch 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Create nn model put it on GPU and set it to evaluation mode (requires a lot of RAM)
resnet50_gpu = models.resnet50(pretrained=True, progress=True).to("cuda").eval()

# reshape the input tensor - convert from numpy to PyTorch tensor and reshape
input_batch_chw = torch.from_numpy(input_batch).transpose(1,3).transpose(2,3)
input_batch_gpu = input_batch_chw.to("cuda")
print("Size of the input batch loaded as torch.tensor to GPU: " + str(input_batch_gpu.shape))
# e.g. [32, 3, 224, 224]

# .forward() method of nn.Module type class is used to make predictions on input batch of data; result is then moved back to cpu memory and converted to np.array
with torch.no_grad():
    predictions = np.array(resnet50_gpu(input_batch_gpu).cpu())
print("Dimensions of the prediction vector: " + str(predictions.shape))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#           Execution speed comparison - FP32 vs FP16 precision 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import timeit

# FP32 precision performance
code_snippet = '''
with torch.no_grad():
    preds = np.array(resnet50_gpu(input_batch_gpu).cpu())
'''
t = timeit.Timer(stmt=code_snippet, globals=globals())
execution_times = t.repeat(repeat=5, number=1)
average_time = sum(execution_times) / len(execution_times)
print(f'Average execution time on GPU: {average_time:.6f} seconds')

# FP16 precision performance
# .half() method is used to cast all floating point parameters and buffers of a module to half precision (float16). Modifies module in-place
resnet50_gpu_half = resnet50_gpu.half()
input_half = input_batch_gpu.half()

# FP16 initialization run
with torch.no_grad():
    preds = np.array(resnet50_gpu_half(input_half).cpu())

# execution time measurement
code_snippet = '''
with torch.no_grad():
    preds = np.array(resnet50_gpu_half(input_half).cpu())
'''
t = timeit.Timer(stmt=code_snippet, globals=globals())
execution_times = t.repeat(repeat=5, number=1)
average_time = sum(execution_times) / len(execution_times)
print(f'Average execution time on GPU: {average_time:.6f} seconds')

# Print top 5 scores of single image predictions (tractor id is 866)
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




