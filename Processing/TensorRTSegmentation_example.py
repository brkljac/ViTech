# Copyright (c) [2023] [Branko Brkljač, Faculty of Technical Sciences, University of Novi Sad]
#
# Licensed under the Lesser General Public License (LGPL)
# You may obtain a copy of the License at: https://www.gnu.org/licenses/lgpl-3.0.txt
#
# This software is provided "as is," without warranty of any kind. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability.
#

# Demonstration of model conversion to ONNX and building of TensorRT engine on the example of PyTorch image segmentation model
# TensorRTSegmentation_example.py


# Note: If both, conversion to ONNX and build of TensorRT engine need to be run on Jetson nano, please consider running the script twice, separately for each of the tasks, since running both tasks in the same script can result in memory fragmentation, which degrades the conversion speed significantly (process can take excesivelly long). However, if each conversion is run in separate process, this does not occur (it can be that this is only related to specific version of JetPack, or some other setting). Therefore, it is a good practice to generte .onnx and .trt files in two iterations.



USE_FP16 = False			# considered model works only in FP32 precision
skip_onnx_conversion = True		# True, skip model conversion to ONNX
skip_conversion = False			# True, just load the existing TensorRT model, or exit (if one is not available)

# Define input image resolution that will dynamically shape tensorRT engine build
target_img_width = 520	
target_img_height = 520

model_base_name = "fcn_resnet101"				# model name
model_relative_path = "models/torch/fcn_resnet101/"		# relative path to the .onnx file


import os
from io import BytesIO
import requests
from PIL import Image
import matplotlib.pyplot as plt


output_image=os.path.join(os.getcwd(), "temp/FCN_input.ppm")

# Read sample image input and save it in ppm format
print("Exporting ppm image {}".format(output_image))
response = requests.get("https://pytorch.org/assets/images/deeplab1.png")
with Image.open(BytesIO(response.content)) as img:
    ppm = Image.new("RGB", img.size, (255, 255, 255))
    ppm.paste(img, mask=img.split()[3])
    ppm.save(output_image)

output_onnx = os.path.join(os.getcwd(), model_relative_path, "{model_name}_pytorch.onnx".format(model_name=model_base_name))

import torch
import torch.nn as nn

if not skip_onnx_conversion:
    # FC-ResNet101 pretrained model from torch-hub extended with argmax layer
    # Model description: https://pytorch.org/hub/pytorch_vision_fcn_resnet101/
    # FCN segmentation model was trained on PASCAL VOC dataset, which has 20 class labels + 1 background class. Class labels are: 0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
    # Model was trained on 520x520 images
    class FCN_ResNet101(nn.Module):
        def __init__(self):
            super(FCN_ResNet101, self).__init__()
            self.model = torch.hub.load('pytorch/vision:v0.6.0', model_base_name, pretrained=True)
        def forward(self, inputs):
            x = self.model(inputs)['out']
            x = x.argmax(1, keepdims=True)
            return x
    
    model = FCN_ResNet101()
    model.eval()

    # Generate input tensor with random values, and (B, C, H, W) = (Batch, Channel, Height, Width) order of dimensions
    input_tensor = torch.rand(4, 3, 224, 224)

    # Export torch model to ONNX
    print("Exporting ONNX model {}".format(output_onnx))
    torch.onnx.export(model, input_tensor, output_onnx,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"}, "output": {0: "batch", 2: "height", 3: "width"}},verbose=True)





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#           Conversion of PyTorch ONNX model to TensorRT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch.cuda.empty_cache()

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


if USE_FP16:
    fp_precision = 16
else:
    fp_precision = 32

input_model_name =  output_onnx
trt_model_name = input_model_name.rsplit(".",1)[0]+"_{tw}x{th}__fp{fp_precision}_engine.trt".format(tw=target_img_width, th=target_img_height, fp_precision = str(fp_precision))


# Create TensorRT engine file
if not skip_conversion:
    import subprocess
    trtexec_path = "/usr/src/tensorrt/bin/trtexec"	# convertor
    if USE_FP16:
        subprocess.run([trtexec_path, '--onnx={onnx_model}'.format(onnx_model=input_model_name), '--saveEngine={tensorrt_model}'.format(tensorrt_model=trt_model_name), '--explicitBatch', '--inputIOFormats=fp16:chw', '--outputIOFormats=fp16:chw', '--fp16', '--shapes=input:1x3x{in_height}x{in_width}, output:1x1x{out_height}x{out_width}'.format(in_height=target_img_height, in_width=target_img_width, out_height=target_img_height, out_width=target_img_width)])
    else:
        subprocess.run([trtexec_path, '--onnx={onnx_model}'.format(onnx_model=input_model_name), '--saveEngine={tensorrt_model}'.format(tensorrt_model=trt_model_name), '--explicitBatch', '--shapes=input:1x3x{in_height}x{in_width}, output:1x1x{out_height}x{out_width}'.format(in_height=target_img_height, in_width=target_img_width, out_height=target_img_height, out_width=target_img_width)])




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#           Run TensorRT model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TRT_LOGGER = trt.Logger()

# Filenames of TensorRT plan file and input/output images.
#engine_file = "fcn-resnet101.engine"

engine_file = trt_model_name

input_file  = output_image
output_file = input_file.replace("input", "output")


# For torchvision models, input images are loaded in to a range of [0, 1] and
# normalized using mean = [0.485, 0.456, 0.406] and stddev = [0.229, 0.224, 0.225].
def preprocess(image):
    # Mean normalization
    mean = np.array([0.485, 0.456, 0.406]).astype('float32')
    stddev = np.array([0.229, 0.224, 0.225]).astype('float32')
    data = (np.asarray(image).astype('float32') / float(255.0) - mean) / stddev
    # Switch from HWC to to CHW order
    return np.moveaxis(data, 2, 0)

def postprocess(data):
    num_classes = 21
    # create a color palette, selecting a color for each class
    palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = np.array([palette*i%255 for i in range(num_classes)]).astype("uint8")
    # plot the segmentation predictions for 21 classes in different colors
    img = Image.fromarray(data.astype('uint8'), mode='P')
    img.putpalette(colors)
    return img


def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def infer(engine, input_file, output_file):
    print("Reading input image from file {}".format(input_file))
    with Image.open(input_file) as img:
        new_size = (target_img_width, target_img_height)
        img = img.resize(new_size)
        input_image = preprocess(img)
        image_width = img.width
        image_height = img.height
    with engine.create_execution_context() as context:
        # Set input shape based on image dimensions for inference
        context.set_binding_shape(engine.get_binding_index("input"), (1, 3, image_height, image_width))
        # Allocate host and device buffers
        bindings = []
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(input_image)
                input_memory = cuda.mem_alloc(input_image.nbytes)
                bindings.append(int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))
        stream = cuda.Stream()
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        # Synchronize the stream
        stream.synchronize()
    with postprocess(np.reshape(output_buffer, (image_height, image_width))) as img:
        print("Writing output image to file {}".format(output_file))
        img.convert('RGB').save(output_file, "PPM")

f1 = plt.figure(1)
img1 = Image.open(input_file)
plt.imshow(img1)
plt.axis('off')
f1.show()


print("Running TensorRT inference for " + model_base_name)
with load_engine(engine_file) as engine:
    infer(engine, input_file, output_file)


f2 = plt.figure(2)
plt.axis('off')
img2 = Image.open(output_file)
img2 = img2.resize(img1.size)
plt.imshow(img2)
f2.show()

# Blend two images by using alpha channel made from the result
new_img1 = Image.new("RGBA", img1.size, (255,255,255,0))
new_img1.paste(img1, (0,0))
new_img2 = Image.new("RGBA", img2.size, (255,255,255,0))
new_img2.paste(img2, (0,0))
r_img2 = img2.getchannel(0)
alpha = r_img2.point(lambda p: 0 if p==0 else 128)
new_img2.putalpha(alpha)
img3 = Image.alpha_composite(new_img1, new_img2)

f3 = plt.figure(3)
plt.axis('off')
plt.imshow(img3)
f3.show()
img3.save(os.path.join(os.getcwd(), "temp/FCN_InputOutput_Blend.ppm"), "PPM")
plt.show()

# press 'q' to close figures














