DIRECTORY TREE

+---Streaming
    |   experiment_gstreamer_camera.py
    |   gstreamer_camera.py
    |   Streaming.code-workspace
    |   
    +---.vscode
    |       launch.json
    |       
    +---imOut
    +---test_image
    |       
    +---test_video
    \---vidOut

+---Processing
|   |   experiment_streamProcessingSegmentation.py
|   |   ONNX2TensorRT_example.py
|   |   Processing.code-workspace
|   |   processing_template.py
|   |   PyTorch2ONNX_example.py
|   |   PyTorchGPUComparison_example.py
|   |   streamProcessingSegmentation.py
|   |   TensorRT engine configurations comparison.txt
|   |   TensorRTSegmentation_example.py
|   |   
|   +---.vscode
|   |       launch.json
|   |       
|   +---imOut
|   +---models
|   |   \---torch
|   |       |   imagenet_class_index.json
|   |       |   
|   |       +---fcn_resnet101
|   |       |       FCN_input.ppm
|   |       |       FCN_InputOutput_Blend.ppm
|   |       |       FCN_output.ppm
|   |       |       fcn_resnet101_pytorch_520x520__fp32_engine.trt
|   |       |       pytorch_vision_v0.6.0.zip
|   |       |       
|   |       \---resnet50
|   |           |   resnet50-19c8e357.pth
|   |           |   resnet50_pytorch_bsize16__fp16_engine.trt
|   |           |   resnet50_pytorch_bsize1__fp16_engine.trt
|   |           |   
|   |           \---resnet50__repo
|   |                   resnet50.tar.gz
|   |                   
|   +---test_image
|   |       FCN_input.ppm
|   |       machine.jpg
|   |       
|   +---test_video
|   \---vidOut

+---Other
|   |   depthPerception.py
|   |   depthPerception_setup.py
|   |   engine_performance_comparison.txt
|   |   experiment_depthPerception.py
|   |   experiment_depthPerception2.py
|   |   other.code-workspace
|   |   
|   \---results
|       \---produced engines
|               nyu_full_GuideDepth-S_FP16.engine
|               nyu_full_GuideDepth-S_FP32.engine
|               nyu_half_GuideDepth-S_FP16.engine
|               nyu_half_GuideDepth-S_FP32.engine
|               