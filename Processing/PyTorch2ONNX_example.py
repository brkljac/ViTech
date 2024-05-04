# Copyright (c) [2023] [Branko Brkljač, Faculty of Technical Sciences, University of Novi Sad]
#
# Licensed under the Lesser General Public License (LGPL)
# You may obtain a copy of the License at: https://www.gnu.org/licenses/lgpl-3.0.txt
#
# This software is provided "as is," without warranty of any kind. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability.
#

# Conversion of PyTorch model to ONNX representation
# PyTorch2ONNX_example.py


BATCH_SIZE=16

model_base_name = "resnet50_pytorch"
model_relative_path = "models/torch/resnet50/"


import os
import numpy as np
import torchvision.models as models
import torch
import torch.onnx

torch.cuda.empty_cache()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#           conversion of PyTorch .pth model to .onnx
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Conversion is relatively slow, requires a test batch in proper shape and format

output_model_name =  os.path.join(os.getcwd(), model_relative_path, "{model_base_name}_bsize{batch_size}.onnx".format(model_base_name=model_base_name, batch_size=str(BATCH_SIZE)))

# Load the pretrained model, .pth file
# after downloading check the folder /home/blab/.cache/torch/hub/checkpoints for .pth files
# Create nn model and set it to evaluation mode
resnet50 = models.resnet50(pretrained=True, progress=True).eval()

dummy_input=torch.randn(BATCH_SIZE, 3, 224, 224)	# [32, 3, 224, 224]

torch.onnx.export(resnet50, dummy_input, output_model_name, verbose=True)
