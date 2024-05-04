# Copyright (c) [2023] [Branko Brkljač, Faculty of Technical Sciences, University of Novi Sad]
#
# Licensed under the Lesser General Public License (LGPL)
# You may obtain a copy of the License at: https://www.gnu.org/licenses/lgpl-3.0.txt
#
# This software is provided "as is," without warranty of any kind. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability.
#

# depth perception - test script 2
# experiment_depthPerception2.py

# Note: in order to also show original camera frames on the screen, besides estimated depth maps, please also consider the option to stream unprocessed camera input over RTP port and display the stream by using the following receiver (which can be launched from the command line, in the separate terminal window):
#
#RECEIVER:
# gst-launch-1.0 udpsrc address=127.0.0.1 port=8011 ! application/x-rtp, encoding-name=H265, payload=96 ! rtph265depay ! queue ! h265parse ! nvv4l2decoder ! nv3dsink sync=false -e -v
#
# In order to use the receiver, please consider the RTP output setup similar to the one below:
	# output_video_stream = 1
	# send_to_port = 1
	# host_address = "127.0.0.1"
	# host_port = 8001
	# max_filesize = 2000
	# max_duration = 3600
	# send_unprocessed = 1
	# host_address_unprocessed = host_address
	# host_port_unprocessed = host_port + 10


# Check whether Opencv build with gstreamer support is available:
# echo 'import cv2; print(cv2.getBuildInformation())' | python3 | grep -A5 GStreamer
# if necessary, camera restart can be done by: sudo service nvargus-daemon restart

from copy import deepcopy
import os
import cv2
import numpy as np
import math
from datetime import datetime
import subprocess



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                   Auxiliary functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def get_folder_size(folder, fileformat = ""):
    total_size = 0
    for path, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(fileformat):
                fp = os.path.join(path, f)
                total_size += os.path.getsize(fp)
    return total_size

def get_files_list(folderpath, keyword="", format=".jpg"):
    # returns the list of files with the given keyword and format
    files_list = []
    for path, dirs, files in os.walk(folderpath):
        for filename in files:
            if filename.endswith(format) and keyword in filename:
                files_list.append(os.path.join(path, filename))
    return files_list


def get_video_input_from_rtp(address, port, encoding="h265", play_video=0, bufferSize=1048576, latency=0, dropOnLatency="true", resolution=(), interpolation_method=5, flip_method=0, fps=0, verboseVidOutput=0):
    #   address: IP address of the incoming rtp video stream
    #   port: streaming port on the given address
    #   encoding: Codec used for video streaming ("h265, "h264")
    #   play_video: Reproduce received video stream on screen (play_video=1, test mode). If set to 1, function returns 0 instead of default VideoCapture object. If set to 0, function returns opened VideoCapture object (receiving mode). If set to 2, function plays incoming video stream on screen without any preprocessing, but also returns VideoCapture object for further application level processing (demux mode with 2 independent threads)
    #   bufferSize: Size of the kernel receive buffer in bytes (default: 1MB=1048576)
    #   latency: Amount of ms to buffer (default: 0)
    #   dropOnLatency: Tells the jitterbuffer to never exceed the given latency in size (default: "true")
    #   resolution: New output resolution (resolution = (width, height)). If set, input processing pipeline performs hardware accelerated interpolation, otherwise video with original resolution is received (default: ())
    #   interpolation_method: nvvidconv interpolation type utilized in the case that output video resolution is changed (default: 5)
    #   flip_method: >0, flip input frames (default: 0)
    #   fps: If set to value >0, original fps is changed (default: 0)
    #   verboseVidOutput: if set to 1, processing pipeline is displayed (default: 0)
    #   Return values: 0 (reproduction/test mode), -1 (videocapture initialization failed), VideoCapture object (video handle object successfully initialized); 

    vidout_change = ""
    if resolution:
        vidout_width = resolution[0]
        vidout_height = resolution[1]  
        vidout_change = "! nvvidconv interpolation-method={interpolation_method} flip-method={flip_method} ! video/x-raw(memory:NVMM), width={vidout_width}, height={vidout_height}, format=(string)NV12".format(vidout_width=vidout_width, vidout_height=vidout_height, flip_method=flip_method, interpolation_method=interpolation_method)
    elif flip_method:
        vidout_change = "! nvvidconv flip-method={flip_method} ! video/x-raw(memory:NVMM), format=(string)NV12".format(flip_method=flip_method)
    if fps:
        if vidout_change:
            vidout_change = vidout_change + ", framerate=(fraction){fps}/1".format(fps=round(fps))
        else: 
            vidout_change = "! nvvidconv flip-method=0 ! video/x-raw(memory:NVMM), format=(string)NV12, framerate=(fraction){fps}/1".format(fps=round(fps))
    pipeline_vidinput = "udpsrc buffer-size={bufferSize} address={address} port={port} ! application/x-rtp, media=video, encoding-name={encoding_name}, payload=96 ! rtpjitterbuffer latency={latency} drop-on-latency={dropOnLatency} ! rtp{encoding}depay ! queue ! {encoding}parse ! nvv4l2decoder ! videorate {vidout_change} ! nvvidconv ! video/x-raw, format=BGRx ! queue ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=5 drop=true".format(bufferSize=bufferSize, address=address, port=port, encoding_name=encoding.upper(), latency=latency, dropOnLatency=dropOnLatency, encoding=encoding, vidout_change=vidout_change)
    pipeline_vidinput = " ".join(pipeline_vidinput.split())
    
    if play_video == 2:
        # demultiplex incoming video stream at hardware level into 2 output downstreams: 1) one providing live preview of the received video without any preprocessing, and 2) one ready for further application level processing (videocapture object handle)
        demux_keyword = "nvv4l2decoder !"
        demux_position = pipeline_vidinput.find(demux_keyword)+len(demux_keyword)
        
        demux_branch = " tee name=invidrtp  invidrtp. ! queue ! nv3dsink sync=false invidrtp. ! queue ! "
        pipeline_vidinput = pipeline_vidinput[0:demux_position] + demux_branch + pipeline_vidinput[demux_position+1:]

    output_str = "Incoming {encoding} RTP video stream on {address}:{port}".format(encoding=encoding.upper(), address=address, port=port)
    if verboseVidOutput:
        print("\n" + output_str+ ", processing pipeline:\n"+pipeline_vidinput+"\n")
    
    # Create a VideoCapture object with the pipeline
    cap = cv2.VideoCapture(pipeline_vidinput, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("\n\nFailed to open " + output_str +"\n")
        return -1
    
    if play_video == 1:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        if verboseVidOutput:
            print("\nIncoming " + output_str + ":\n\n"+"\tframe width: {w}\n\tframe height: {h} \n\tfps: {f}".format(w=frame_width, h=frame_height, f=video_fps).expandtabs(6))
        cv2.namedWindow(output_str, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(output_str, frame_width, frame_height)    
        cv2.waitKey(1)

        while True:
            retrieved, frame = cap.read()
            if not retrieved or (cv2.waitKey(1)&0xFF == ord('q')):
                cap.release()
                cv2.destroyWindow(output_str)
                break
            cv2.imshow(output_str, frame)
        return 0
    else:
        return cap



def get_video_input_from_file(filepath="", play_video=0, resolution=(), interpolation_method=5, flip_method=0, fps=0, verboseVidOutput=0):
    #   filepath: Absolute path to video file (filepath = "/path/to/video/file.ext")
    #   play_video: Reproduce retrieved video on screen (play_video = 1, test mode). If set, function returns 0 instead of opened VideoCapture object (default: 0)
    #   resolution: New output resolution (resolution = (width, height)). If set, input processing pipeline performs hardware accelerated interpolation, otherwise original video signal is retrieved (default: ())
    #   interpolation_method: nvvidconv interpolation type utilized in the case that output video resolution is changed (default: 5)
    #   flip_method: >0, flip input frames (default: 0)
    #   fps: If set to value >0, original fps is changed (default: 0)
    #   verboseVidOutput: if set to 1, processing pipeline is displayed (default: 0)
    #   Return values: 0 - reproduction/test mode, -1 - videocapture initialization failed, VideoCapture object - video handle object initialized; 

    output = subprocess.check_output(["gst-discoverer-1.0", filepath])
    output_str = output.decode("utf-8")
    start_index=output_str.find("video codec:")
    end_index = output_str.find("\n", start_index)
    encoderType = output_str[start_index:end_index].strip()
    start_index=output_str.find("container:")
    end_index = output_str.find("\n", start_index)
    containerType = output_str[start_index:end_index].strip()    
    if encoderType.lower().find("265")>=0:
        decoderType = " ! h265parse"
    if encoderType.lower().find("264")>=0:
        decoderType = " ! h264parse"
    demux = ""
    if containerType.lower().find("quicktime")>=0:
        demux = " ! qtdemux"    
    if containerType.lower().find("matroska")>=0:
        demux = " ! matroskademux"
    if containerType.lower().find("avi")>=0:
        demux = " ! avidemux"
    vidout_change = ""
    if resolution:
        vidout_width = resolution[0]
        vidout_height = resolution[1]  
        vidout_change = "! nvvidconv interpolation-method={interpolation_method} flip-method={flip_method} ! video/x-raw(memory:NVMM), width={vidout_width}, height={vidout_height}, format=(string)NV12".format(vidout_width=vidout_width, vidout_height=vidout_height, flip_method=flip_method, interpolation_method=interpolation_method)
    elif flip_method:
        vidout_change = "! nvvidconv flip-method={flip_method} ! video/x-raw(memory:NVMM), format=(string)NV12".format(flip_method=flip_method)
    if fps:
        if vidout_change:
            vidout_change = vidout_change + ", framerate=(fraction){fps}/1".format(fps=round(fps))
        else: 
            vidout_change = "! nvvidconv flip-method=0 ! video/x-raw(memory:NVMM), format=(string)NV12, framerate=(fraction){fps}/1".format(fps=round(fps))

    pipeline_vidinput = "filesrc location={filepath} {demux} ! queue {decoderType} ! nvv4l2decoder ! videorate {vidout_change} ! nvvidconv ! video/x-raw,format=BGRx ! queue ! videoconvert ! queue ! video/x-raw, format=BGR ! appsink".format(filepath= filepath, demux=demux, decoderType=decoderType, vidout_change=vidout_change)
    pipeline_vidinput = " ".join(pipeline_vidinput.split())

    if verboseVidOutput:
        print("\nRetrieved input video from file, processing pipeline:\n"+pipeline_vidinput+"\n")

    # Create a VideoCapture object with the pipeline
    cap = cv2.VideoCapture(pipeline_vidinput, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("\n\nFailed to open: "+ filepath +"\n")
        return -1

    if play_video:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        if verboseVidOutput:
            print("\nVideo file properties:\n\n" + output_str + "\tframe width: {w}\n\tframe height: {h} \n\tfps: {f}".format(w=frame_width, h=frame_height, f=video_fps).expandtabs(6))
        cv2.namedWindow("Retrieved video input", cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("Retrieved video input", frame_width, frame_height)    
        cv2.waitKey(1)
    
        while True:
            retrieved, frame = cap.read()
            if not retrieved or (cv2.waitKey(1)&0xFF == ord('q')):
                cap.release()
                cv2.destroyWindow("Retrieved video input")
                break
            cv2.imshow("Retrieved video input", frame)
        return 0
    else:
        return cap
    


def get_video_input_from_images(folderpath, fileformat="image_%05d.jpg", encoding="jpg", fileindex=1, stopindex=-1, fps=1, play_video=0, resolution=(), interpolation_method=5, flip_method=0,  vidout_location="outvidimg", vidout_encoder="h265", vidout_format="mp4", preset_level=4, iframeinterval=1, control_rate=0, qp_range="qp-range=\"24,36:30,31:-1,-1\"", bitrate="", vbv_size="", idrinterval=-1, peak_bitrate="", ratecontrol_enable="", wait_for_videowriter=0, sync_video=0,demux_on_screen=0, verboseVidOutput=0):
    #   folderpath: Absolute path to folder with sequence of images
    #   fileformat: Name prefix, indexing format and extension of input images (default: "image_%05d.jpg" = image_00001.jpg)
    #   encoding: image encoder type, "jpg", "png" (default: "jpg")
    #   fileindex: start index of the input image files (default: 1)
    #   stopindex: stop index of the imput image files (default: -1, do not stop)
    #   fps: framerate of generated video/videocapture (default: 1)
    #   play_video: If set to 1, reproduce generated video on screen (test mode). If not set (default: 0), return generated VideoCapture object. If equal 2, save video to file (mode 2)
    #   demux_on_screen: If set to 1, demultiplex generated video stream into two, and show one directly on screen (demux mode)  (default: 0, do not split)
    #   sync_video: If set to 1, video writing and live preview using demux mode  will be at the speed of set framerate (usually slower writing than possible, not suitable for video files production from image sequences). Use only in live preview or test mode/demux mode (default: 0)
    #   wait_for_videowriter: If set to 1, function will wait for the subprocess writing to video file to finish - when the play mode is set to 2. Otherwise, execution will continue after launching the videowriter subrpocess (default: 0, do not wait)
    #  resolution: New output resolution (resolution = (width, height)). If set, input processing pipeline performs hardware accelerated interpolation, otherwise generated video signal has the same resolution as input images (default: ())
    #   interpolation_method: nvvidconv interpolation type utilized in the case that output video resolution is changed (default: 5)
    #   flip_method: >0, flip input frames (default: 0)
    #   vidout_location: absolute path and name of video file, if such is generated from input images
    #   video encoding parameters:  vidout_encoder, vidout_format, preset_level, iframeinterval, control_rate, qp_range, bitrate, vbv_size, peak_bitrate, ratecontrol_enable 
    #   verboseVidOutput: if set to 1, processing pipeline is displayed (default: 0)
    #   Return values: 0 (reproduction/test mode), -1 (videocapture initialization failed), VideoCapture object (video handle object successfully initialized); 
    
    if idrinterval<0:
        idrinterval = iframeinterval
    sync_video = bool(sync_video)

    if encoding=="jpg":
        decoderType="nvjpegdec idct-method=2 ! \"video/x-raw(memory:NVMM), format=(string)I420\""
    elif encoding=="png":
        decoderType="pngdec"

    pipeline_vidinput = "multifilesrc location={filepath} index={fileindex} stop-index={stopindex}  ! {decoderType} ! videorate ! nvvidconv ! \"video/x-raw(memory:NVMM), format=(string)NV12,  framerate=(fraction){vidout_fps}/1\"".format(filepath=os.path.join(folderpath, fileformat), fileindex=fileindex, stopindex=stopindex, encoding=encoding, vidout_fps=int(fps), decoderType=decoderType)

    vidout_change = ""
    if resolution:
        vidout_width = resolution[0]
        vidout_height = resolution[1]  
        vidout_change = " ! nvvidconv interpolation-method={interpolation_method} flip-method={flip_method} ! \"video/x-raw(memory:NVMM), width={vidout_width}, height={vidout_height}, format=(string)NV12\"".format(vidout_width=vidout_width, vidout_height=vidout_height, flip_method=flip_method, interpolation_method=interpolation_method)
    elif flip_method:
        vidout_change = " ! nvvidconv flip-method={flip_method} ! \"video/x-raw(memory:NVMM), format=(string)NV12\"".format(flip_method=flip_method)

    pipeline_vidinput = pipeline_vidinput + vidout_change

    if play_video == 2:
        # Write generated video stream to file
        enc_spec = " ! nvv4l2{encoder}enc profile=0 preset-level={preset_level} iframeinterval={iframeinterval} control-rate={control_rate} {ratecontrol_enable} {qp_range} {bitrate} {vbv_size} {peak_bitrate} insert-sps-pps=true insert-vui=true idrinterval={idrinterval} ! \"video/x-{encoder}, stream-format=(string)byte-stream, alignment=(string)au\"".format(fps=int(fps), vidout_change=vidout_change, encoder=vidout_encoder, preset_level=preset_level, iframeinterval=int(fps), control_rate=control_rate, qp_range=qp_range, bitrate=bitrate, vbv_size=vbv_size, idrinterval=idrinterval, peak_bitrate=peak_bitrate, ratecontrol_enable=ratecontrol_enable)
        if vidout_format=="h265" or vidout_format=="h264":
            file_spec = " ! {encoder}parse ! queue ! filesink location={location}.{format} sync={sync}".format(encoder=vidout_encoder, location=vidout_location, format=vidout_format, sync=sync_video)
        if vidout_format=="mp4":
            # use matroskamux instead of qtmux since writing to video file is launched as separate subprocess, and in case of interrupt before stop_index EOS signal will be missing from the video file (potential problem for qtdemux)
            file_spec = " ! {encoder}parse ! queue ! matroskamux ! filesink location={location}.{format} sync={sync}".format(encoder=vidout_encoder, location=vidout_location, format=vidout_format, sync=sync_video)
        app_spec = ""
    else:
        enc_spec = ""
        file_spec = ""
        app_spec = " ! nvvidconv ! \"video/x-raw, format=BGRx\" ! queue ! videoconvert ! queue ! \"video/x-raw, format=BGR\" ! appsink sync={sync} max-buffers=5 drop=true".format(sync=sync_video)

    pipeline_vidinput = pipeline_vidinput  + enc_spec + file_spec + app_spec

    if demux_on_screen:
        # demultiplex generated video stream at hardware level into 2 output downstreams: 1) one auxiliary providing live preview directly on screen, and 2) main stream (going to file or application)
        if enc_spec:
            demux_keyword = "! nvv4l2{encoder}enc".format(encoder=vidout_encoder)
            demux_position = pipeline_vidinput.find(demux_keyword)
            demux_branch = " ! tee name=invidim invidim. ! queue ! nv3dsink sync={sync} invidim. ! queue ! ".format(sync=sync_video)
            pipeline_vidinput = pipeline_vidinput[0:demux_position] + demux_branch + pipeline_vidinput[demux_position+1:]
        if app_spec:
            demux_keyword = "! nvvidconv ! \"video/x-raw, format=BGRx\""
            demux_position = pipeline_vidinput.find(demux_keyword)
            demux_branch = " ! tee name=invidim invidim. ! queue ! nv3dsink sync={sync} invidim. ! queue ! ".format(sync=sync_video)
            pipeline_vidinput = pipeline_vidinput[0:demux_position] + demux_branch + pipeline_vidinput[demux_position+1:]

    pipeline_vidinput = " ".join(pipeline_vidinput.split())
    output_str = "Input video stream from .{encoding} images in {folderpath}".format(encoding=encoding, folderpath=folderpath)

    if play_video == 2:
        # Launch the subprocess writing video file
        try:
            if verboseVidOutput:
                print("\n*** {datetimestr} ***\n".format(datetimestr=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + "\n" + output_str+ ", processing pipeline:\n\n"+pipeline_vidinput+"\n\nWriting video file to:\n{outputfile}\n\n".format(outputfile=vidout_location+"."+vidout_format))
            command = "gst-launch-1.0 -e -v " + pipeline_vidinput
            write_vidinput = subprocess.Popen("exec " + command, shell = True, stdout=subprocess.PIPE)
            if wait_for_videowriter:
                write_vidinput.wait()
        except KeyboardInterrupt:
            print("Writing video file terminated by user ...\n")
            write_vidinput.terminate()
        except Exception:
            print("Failed to write video file" + output_str +"\n")
            return -1
        return 0

    # Create a VideoCapture object with the pipeline
    pipeline_vidinput = pipeline_vidinput.replace('"', '')
    if verboseVidOutput:
        print("\n*** {datetimestr} ***\n".format(datetimestr=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))+ "\n" + output_str+ ", processing pipeline:\n\n"+pipeline_vidinput+"\n\n")
    cap = cv2.VideoCapture(pipeline_vidinput, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("\n\nFailed to open " + output_str +"\n")
        return -1

    if play_video == 1:
        # enter test mode for appsink
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        if verboseVidOutput:
            print("\n" + output_str + ":\n\n"+"\tframe width: {w}\n\tframe height: {h} \n\tfps: {f}".format(w=frame_width, h=frame_height, f=video_fps).expandtabs(6))
        cv2.namedWindow(output_str, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(output_str, frame_width, frame_height)    
        cv2.waitKey(1)
        while True:
            retrieved, frame = cap.read()
            if not retrieved or (cv2.waitKey(1)&0xFF == ord('q')):
                cap.release()
                cv2.destroyWindow(output_str)
                break
            cv2.imshow(output_str, frame)
        return 0
    else:
        # return valid VideoCapture object
        return cap





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                      Main program
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

datetimeNow = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

USE_TensorRT = 1   # run tensorRT optimized model


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Image parameters at the end of input acquistion chain
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# target resolution
# camout_width = 1920
# camout_height= 1080
camout_width = 640          
camout_height = 480


# in case that you need to flip the image (e.g. due to custom camera mount that results in rotated image), consider replacing above mentioned width and height values with the ones given below
#camout_width = 480          
#camout_height = 640


# target fps
target_fps = 30      


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Screen controls
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

show_camout_on_screen = 1       # 1-show (default), 0-turn off
verboseStreamOutput = 1         # 1-show (default), 0-turn off



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Input RTP video stream setup (alternative input device)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Use only if there is no input camera, or specific incoming RTP video stream is needed as video input. See: get_video_input_from_rtp()

inputVideoFromRTP = 0          # 1-RTP input, 0-other input

# Specify incoming rtp video stream address (default: "", camera input or input from video file are used instead)
inVidRtp_address = "127.0.0.1"

inVidRtp_port = 8001           # incoming stream port number

# incoming stream encoding (default: "h265")
inVidRtp_encoding = "h265"     # "h264", "h265"

# rtp decoding parameters
inVidRtp_latency = 0
inVidRtp_bufferSize = 10485760
inVidRtp_dropOnLatency = "true"

# Incoming video stream conversion parameters (default: resolution, orientation and frame rate are not changed)

# inVidRtpTargetResolution (default: (), do not inteprolate)
# (width, height)
inVidRtpInterpolationMethod = 5
inVidRtpTargetResolution = ()
#inVidRtpTargetResolution = (camout_width, camout_height)    

# inVidRtpTargetFps (default: 0, do not change frame rate)
#inVidRtpTargetFps = 0
inVidRtpTargetFps = target_fps

# inVidRtpFlipImage (default: 0, do not flip)
inVidRtpFlipImage = 0

# Test incoming RTP video stream reproduction and exit (default: 0, do not enter test mode)
# 1-yes, test created videocapture handle by playing incoming video stream and exit
# 0-no, return initialized videocapture object
# 2 - demux mode, return initialized videocapture object and at the same time provide live preview of incoming video stream on the screen (demultiplex incoming video stream at hardware level)
testInVidRtpRep = 2      


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Input video file setup (alternative input device)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Use only if there is no input camera, or specific input video file is needed. See: get_video_input_from_file()

inputVideoFromFile = 0         # 1-file input, 0-other input

# Specify input video file location (absolute path)
# default: "", camera input is used instead
inputVideoFilePath = ""
inputVideoFilePath = "/home/blab/ViTech/Processing/test_video/DJI_1170.MP4"
#inputVideoFilePath = "/home/blab/ViTech/Processing/test_video/MVI_3962.MP4"
#inputVideoFilePath = "/home/blab/ViTech/Processing/test_video/testVideoOut.mp4"


# Input video file conversion parameters (default: resolution, orientation and frame rate are not changed)

# inVidTargetResolution (default: (), do not inteprolate)
# (width, height)
inVidInterpolationMethod = 5
#inVidTargetResolution = ()
inVidTargetResolution = (camout_width, camout_height)


# inVidTargetFps (default: 0, do not change frame rate)
inVidTargetFps = 0
#inVidTargetFps = target_fps

# inVidRtpFlipImage (default: 0, do not flip)
inVidFlipImage = 0

# Test input video file reproduction and exit
testInVidRep = 0      # (1-yes, 0-no, do not enter test mode)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Input video from image sequence (alternative input device)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Use only if there is no input camera, or specific input from sequence of image files is needed. See: get_video_input_from_images()

inputVideoFromImageSequence = 0     # 1-set, 0-other input

# Specify location of image sequence (folder absolute path)
# default: "", camera input is used instead
inVidImSeq_folderpath = "/home/blab/ViTech/Streaming/imOut/"
inVidImSeq_fileformat = "image_%05d.jpg"
inVidImSeq_encoding = "jpg"
inVidImSeq_fileindex = 1
inVidImSeq_stopindex = -1
inVidImSeq_fps = 1

# inVidImSeq_play_video (default: 0, return initialized videocapture object)
# 1-yes, test created videocapture handle by playing generated video stream and exit
# 2 - write generated video file by launching separate subprocess
testInVidImSeqRep = 0  
inVidImSeq_play_video = testInVidImSeqRep

inVidImSeq_resolution = ()
inVidImSeq_interpolation_method = 5
inVidImSeq_flip_method = 0

inVidImSeq_vidout_location = "/home/blab/ViTech/Streaming/test_video/outvidimg"
inVidImSeq_vidout_encoder = "h265"
inVidImSeq_vidout_format = "mp4"
inVidImSeq_preset_level = 4
inVidImSeq_iframeinterval = 1
inVidImSeq_control_rate = 0
inVidImSeq_qp_range = "qp-range=\"24,36:30,31:-1,-1\""
"qp-range=\"24,36:30,31:-1,-1\""
inVidImSeq_bitrate = ""
inVidImSeq_vbv_size = ""
inVidImSeq_idrinterval = -1
inVidImSeq_peak_bitrate = ""
inVidImSeq_ratecontrol_enable = ""

# for more description see get_video_input_from_images()  
inVidImSeq_wait_for_videowriter = 0
inVidImSeq_sync_video = 0
inVidImSeq_demux_on_screen = 0




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Camera setup (primary input device)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Acquistion modes: resolution and fps
#
# mode 0:
# GST_ARGUS: 2616 x 1946 FR = 29,999999 fps Duration = 33333334 ; Analog Gain range min 1,000000, max 31,622776; Exposure Range min 20000, max 500000000;
# mode 1:
# GST_ARGUS: 2592 x 1944 FR = 29,999999 fps Duration = 33333334 ; Analog Gain range min 1,000000, max 31,622776; Exposure Range min 20000, max 500000000;
# mode 2 (default):
# GST_ARGUS: 1920 x 1080 FR = 51,999998 fps Duration = 19230770 ; Analog Gain range min 1,000000, max 31,622776; Exposure Range min 20000, max 500000000;
# mode 3:
# GST_ARGUS: 1296 x 972 FR = 57,999998 fps Duration = 17241380 ; Analog Gain range min 1,000000, max 31,622776; Exposure Range min 20000, max 500000000;

# to reset or unblock camera use:
# sudo service nvargus-daemon restart

# mode 0:
camin_width = 2616
camin_height = 1946
camin_FR = 30           # fps produced by camera, before any conversions

set_exposuretimerange = 1   # 0-no,1-yes
set_gainrange = 1           # 0-no,1-yes


# Set camera exposure and gain
if set_exposuretimerange:
    exposuretimerange = "exposuretimerange=\"34000 358733000\""
else:
    exposuretimerange = ""
if set_gainrange:
    gainrange = "gainrange=\"1 16\""
else:
    gainrange = ""

# Set white balance (affects the color temperature)
# (0): off
# (1): auto
# (2): incandescent
# (3): fluorescent
# (4): warm-fluorescent
# (5): daylight
# (6): cloudy-daylight
# (7): twilight
# (8): shade
# (9): manual
# default: 1
wbmode = 1               

# Set exposure compensation
# range (float): -2 - 2, default: 0
exposurecompensation = 0.5

# Set auto exposure lock
# default: false
aelock = "false"

# Set auto white balance lock
# default: false
awblock = "false"

# Set temporal noise reduction
# tnr-mode:
# (0): NoiseReduction_Off
# (1): NoiseReduction_Fast
# (2): NoiseReduction_HighQuality
# default: 1
tnr_mode = 1
# tnr-strength:
# range (float): -1 - 1, default: -1
tnr_strength = -1

# Set color saturation
# range (float): 0 - 2, default: 1
saturation = 1.3

# Set edge enhnacement
# ee-mode:
# (0): EdgeEnhancement_Off
# (1): EdgeEnhancement_Fast
# (2): EdgeEnhancement_HighQuality
# default: 1
ee_mode = 2
# ee-strength:
# range (float): -1 - 1, default: -1
ee_strength = 0.5

# Anti flicker
# (0): AeAntibandingMode_Off
# (1): AeAntibandingMode_Auto
# (2): AeAntibandingMode_50HZ
# (3): AeAntibandingMode_60HZ
aeantibanding=2

# Set nvvidconv conversion parameters
# interpolation-method:
# (0): Nearest
# (1): Bilinear
# (2): 5-Tap
# (3): 10-Tap
# (4): Smart
# (5): Nicest
# default: 0
interpolation_method = 5

# flip-method:
# (0): none
# (1): counterclockwise 90 degrees
# (2): rotate-180
# (3): clockwise 90  degrees
# (4): horizontal-flip
# (5): upper-right/lower left-diagonal flip
# (6): vertical-flip
# (7): upper-left/lower right -diagonal flip
# default: 0
flip_method = 0




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Output video stream setup (to file or port)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

output_video_stream = 1              #(1-yes, 0-no)

send_to_file = 0            #(1-send to file, 0-do not write on disk)
send_to_port = 1            #(1-send to port, 0-do not stream over port)

# Output video stream encoding, format and location
vidout_encoder = "h265"     # h265, h264
vidout_format = "mp4"      # h265, h264, mp4

# Output video file location and name
vidout_location = os.path.join(os.getcwd(),"vidOut")
vidout_name =  "testVideoOut"

# Output video stream address
host_address = "127.0.0.1"      # default: "127.0.0.1"
host_port = 8001                # default: 8001

# Encoding stop criteria
max_filesize = 2000             # in MB (0-no limit)
max_duration = 3600            	# in sec. (0-no limit)

test_maxNframes = 0         # test switch (0-turn off)

# other
skip_Nframes = 30                    # skip N initial frames
camout_fps = target_fps              # output video stream fps

if target_fps == 1:
    skip_Nframes = 0


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (Additional) unprocessed output video stream setup
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Geenrate additional, unprocessed video output with the same settings
send_unprocessed = 1        #(1-yes, 0-no), in addition to processed vidout

# Exclusively make the unprocessed video (send_unprocessed needs to be true) 
send_only_unprocessed = 0   #(1-yes, 0-no)

# Unprocessed video stream address
host_address_unprocessed = host_address           # default: "127.0.0.1"
host_port_unprocessed = host_port + 10            # default: 8011

# Unprocessed video file location and name
# Same as vidout_location, name differs only in suffix "_unprocessed"


#RECEIVER:
# gst-launch-1.0 udpsrc address=127.0.0.1 port=8011 ! application/x-rtp, encoding-name=H265, payload=96 ! rtph265depay ! queue ! h265parse ! nvv4l2decoder ! nv3dsink sync=false -e -v



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Output image stream setup (to file or port)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

output_image_stream = 0              #(1-yes, 0-no)

sendImage_to_file = 1            #(1-send to file, 0-do not write on disk)
sendImage_to_port = 0            #(1-send to port, 0-do not stream over port)

# Output image stream encoding, format and folder location
imout_encoder = "nvjpegenc"
imout_format = "jpg"
imout_location = os.path.join(os.getcwd(),"imOut")

# Output image stream address
host_address_imout = "127.0.0.1"      # default: "127.0.0.1"
host_port_imout = 8002                # default: 8002

# Output image stopping criteria
imout_max_foldersize = 5000             # in MB (0-no limit)
imout_max_duration = 0                # in sec. (0-no limit)

test_maxNimagesOut = 0         # test switch (0-turn off)

# other
imout_fps = 30                  # image output frequency



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (Additional) unprocessed output image stream setup
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Geenrate additional, unprocessed image stream output with the same settings
send_unprocessed_image = 0        #(1-yes, 0-no), in addition to processed imout

# Exclusively output the unprocessed images (send_unprocessed_image needs to be true) 
send_only_unprocessed_image = 0   #(1-yes, 0-no)

# Unprocessed video stream address
host_address_imout_unprocessed = host_address_imout           # default: "127.0.0.1"
host_port_imout_unprocessed = host_port_imout + 10            # default: 8012

# Unprocessed image stream folder location
# Same as imout_location, differs only in suffix "_unprocessed"




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# H265/H264 hardware encoder setup
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Encoding Intra Frame occurance frequency
# Range: 0 - 4294967295 Default: 30
iframeinterval = 30

# IDR (Instantaneous Decoder Refresh) interval
# Special type of I-frame specifying that no frame after the IDR frame can reference any frame before it. This makes seeking video file easier and more responsive.
#idrinterval = 5
idrinterval = iframeinterval


# Option 1: Selectiong predefined encoder presets. Slower presets should result in higher quality at the same file size:

# (0): DisablePreset    - Disable HW-Preset
# (1): UltraFastPreset  - UltraFastPreset for high perf
# (2): FastPreset       - FastPreset
# (3): MediumPreset     - MediumPreset
# (4): SlowPreset       - SlowPreset
preset_level = 4

# Comment: 
# In standard H265 given presets also allow choice of CRF quality values, which usually range from 0 to 51 (28 as default for H265). However, in nvv4l2h265enc it is not possible to set CRF values. Instead, the presets provide some predefined encoder settings, and can also be used in combination with other options. This type of encoder setup would be called Constant Rate Factor (CRF) encoding mode, and would provide desired visual quality by controlling CRF values, but would not provide specific bitrate.
#
# On the other hand, in nvv4l2h265enc Constant Quantization Parameter (QP) encoder (CQP) setting can be achieved by setting "preset_level = 0" and fixing quantization levels for all I, P, and B frames in advance to a single value (by setting up options: ratecontrol-enable=0, preset-level=0, quant-i-frames, quant-p-frames, quant-b-frames). However, CQP should be avoided, since it would result in bitrate dependant on scene complexity.
#
# Given presets also define quality of motion vector estimation, which justifies their use in combination with other parameters 


# Option 2: Set CBR or VBR control mode
#
# (0): variable_bitrate (VBR mode)- GST_V4L2_VIDENC_VARIABLE_BITRATE
# (1): constant_bitrate (CBR mode)- GST_V4L2_VIDENC_CONSTANT_BITRATE

# 0-VBR 1-CBR (default)
control_rate = 0                    

# 1-yes (set custom CBR), 0-no (use default CBR target values), only in CBR mode
set_CBR_target = 1

# 1-yes (custom qp range), 0-no (do not set qp range)
set_VBR_qp_range = 1        

# 1-yes (set average VBR), 0-no (do not set desired bitrate)
set_VBR_target_AVBR = 1                  

# note: parameters set to empty string ("") will be omitted from the setup, regardles of control flag values


if control_rate == 1:
    # Option 2: Constant Bitrate (CBR) encoding. Guarantees file size or bandwith, but at the the cost of wasting bandwith if the source is easy to encode.

    if set_CBR_target == 1:
        # bitrate is in bps
        # 1080p (HD) with 30 fps: 8 Mbps
        # 480p with 30 fps: 2.5 Mbps
        #bitrate = "bitrate=2500000"          # 2.5 Mbps
        #bitrate = "bitrate=1000000"          #  1 Mbps
        #bitrate = "bitrate=250000"           #  0.25 Mbps
        #bitrate = "bitrate=10000000"         #  10 Mbps
        #bitrate = ""                         # default 4 Mbps
        bitrateValue = 1000000
        bitrate = "bitrate={bitrateValue}".format(bitrateValue=bitrateValue)

        #vbv-size in bits:  bitrate/fps ~ 2*(bitrate/fps). Example: if the source is 50fps and bitrate is 14Mbps. vbvin should be in the range 280000 ~ 560000.
        #vbv_size="vbv-size=125000"     # for 2.5 Mbps bitrate
        #vbv_size="vbv-size=33000"      # for 1 Mbps bitrate
        #vbv_size="vbv-size=330000"     # for 10 Mbps bitrate
        #vbv_size = ""                  # default 4000000
        vbvValue = round(1.5*bitrateValue/camout_fps)
        vbv_size="vbv-size={vbvValue}".format(vbvValue=vbvValue)     

        # do not set:        
        peak_bitrate = ""
        qp_range = ""
        ratecontrol_enable = ""
    else:
        # use default values
        bitrate = ""        # default setup, 4 Mbps
        vbv_size = ""       # default setup, 4000000

        # do not set:
        peak_bitrate = ""
        qp_range = ""
        ratecontrol_enable = ""


if control_rate == 0:
    # Option 3: Variable bitrate (VBR) encoding. 
    # Provides efficient use of resources: 
    #
    # a) by targeting specific minimum quality (setting quantization parameters range, QP values). Highly recommended for higher compression rations.
    
    if set_VBR_qp_range == 1:
        #qp_range="qp-range=\"50,51:30,31:10,35\""
        # qp_range=""   # will use the whole range
        qp_range = "qp-range=\"24,36:30,31:-1,-1\""
 
        # Setting qunatization range (QP values) for P, I and B frames. Range can be set only in VBR mode. String with values of Qunatization Ranges for different frame types should be formatted as:
        #  
        # MinQpP-MaxQpP:MinQpI-MaxQpI:MinQpB-MaxQpB
        #
        # allowed range values are from 1 to 51
        #
        # example: "qp-range = \"50,51:30,31:10,35\"" provides  high compression ratio (CR)
        #
        # qp range for B frames will be ignored by NVENC ... (B frames are not implemented in nvv4l2decoder Therefore, last two values can be set to -1.
    else:
        # do not set:
        qp_range = ""
        bitrate = ""
        vbv_size = ""
        peak_bitrate = ""
        ratecontrol_enable = ""

    # b) by “targeting" desired average bitrate (AVBR) (can often result in large quality variations over the video).
             
    if set_VBR_target_AVBR == 1:
        # bitrate is in bps
        # 1080p (HD) with 30 fps: 8 Mbps
        # 480p with 30 fps: 2.5 Mbps
        bitrateValue = 2500000         # 2.5 Mbps
        #bitrateValue = 1000000         #  1 Mbps
        #bitrateValue = 10000000        #  10 Mbps
        #bitrateValue = 250000          #  0.25 Mbps
        
        bitrate = "bitrate={bitrateValue}".format(bitrateValue=bitrateValue) 
               
        #vbv-size in bits:  bitrate/fps ~ 2*(bitrate/fps). Example: if the source is 50fps and bitrate is 14Mbps. vbv should be in the range 280000 ~ 560000.      
        vbvValue = round(1.5*bitrateValue/camout_fps)
        vbv_size="vbv-size={vbvValue}".format(vbvValue=vbvValue)     
        
        #vbv_size="vbv-size==125000"    # for 2.5 Mbps bitrate
        #vbv_size="vbv-size=33000"      # for 1 Mbps bitrate
        #vbv_size="vbv-size=330000"     # for 10 Mbps bitrate

        # peak-bitrate > bitrate, e.g. 1.2*bitrateValue
        pbitrateValue = round(1.2*bitrateValue)  
        peak_bitrate = "peak-bitrate={pbitrateValue}".format(pbitrateValue=pbitrateValue)     

    else:
        # do not set:
        bitrate = ""
        vbv_size = ""
        peak_bitrate = ""
        ratecontrol_enable = ""


    # c) combination of the previous two, VBR with QP and ABR:
    if set_VBR_qp_range == 1 and set_VBR_target_AVBR == 1:
        ratecontrol_enable = "ratecontrol-enable=1"
    else:
        ratecontrol_enable = ""



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# nvjpegenc hardware encoder setup
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

imout_quality = 100
idct_method = 2


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the input video stream with hardware acceleration
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# max-buffers: The maximum number of buffers to queue internally (0 = unlimited)
# drop: Drop old buffers when the buffer queue is filled

pipeline_caminput =  "nvarguscamerasrc sensor-id=0 wbmode={wbmode} aelock={aelock} awblock={awblock} {exposuretimerange} {gainrange} exposurecompensation={exposurecompensation} saturation={saturation} tnr-mode={tnr_mode} tnr-strength={tnr_strength} ee-mode={ee_mode} ee-strength={ee_strength} aeantibanding={aeantibanding} ! video/x-raw(memory:NVMM), width={camin_width}, height={camin_height}, format=(string)NV12, framerate=(fraction){camin_FR}/1 ! videorate ! nvvidconv interpolation-method={interpolation_method} flip-method={flip_method} ! video/x-raw(memory:NVMM), width={camout_width}, height={camout_height}, format=(string)NV12, framerate=(fraction){camin_endFR}/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=5 drop=true".format(camin_width=camin_width, camin_height=camin_height, camout_width=camout_width, camout_height=camout_height, wbmode=wbmode, aelock=aelock, awblock=awblock, exposuretimerange = exposuretimerange, gainrange=gainrange, exposurecompensation=exposurecompensation, saturation=saturation, tnr_mode=tnr_mode, tnr_strength=tnr_strength, ee_mode=ee_mode, ee_strength=ee_strength, aeantibanding=aeantibanding, camin_FR=camin_FR, camin_endFR=camout_fps, interpolation_method=interpolation_method, flip_method=flip_method)
pipeline_caminput = " ".join(pipeline_caminput.split())



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create an entry point for the defined processing pipeline
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# VideoCapture object frame grabber

if verboseStreamOutput:
    print("\n*** {datetimestr} *** Input video stream initialization ***\n".format(datetimestr=datetimeNow))

if not inVidRtp_address:
    inputVideoFromRTP = 0
if not inputVideoFilePath:
    inputVideoFromFile = 0
if not inVidImSeq_folderpath:
    inputVideoFromImageSequence = 0

if verboseStreamOutput and not inputVideoFromRTP and not inputVideoFromFile and not inputVideoFromImageSequence:
    print("\nInput camera processing pipeline:\n"+pipeline_caminput+"\n")

if inputVideoFromFile:
    # Capture video stream from file
    cap = get_video_input_from_file(inputVideoFilePath, testInVidRep, inVidTargetResolution, inVidInterpolationMethod, inVidFlipImage, inVidTargetFps, verboseStreamOutput)
    if testInVidRep and cap == 0:
        if verboseStreamOutput:
            print("\nget_video_input_from_file() succesfully finished!\n")
        exit()
    if testInVidRep == 0 and cap == -1:
        exit()
elif inputVideoFromRTP:
    # Capture incoming RTP video stream 
    cap = get_video_input_from_rtp(inVidRtp_address, inVidRtp_port, inVidRtp_encoding, testInVidRtpRep, inVidRtp_bufferSize, inVidRtp_latency, inVidRtp_dropOnLatency, inVidRtpTargetResolution, inVidRtpInterpolationMethod, inVidRtpFlipImage, inVidRtpTargetFps, verboseStreamOutput)
    if testInVidRtpRep and cap == 0:
        if verboseStreamOutput:
            print("\nget_video_input_from_rtp() succesfully finished!\n")
        exit()
    if testInVidRtpRep == 0 and cap == -1:
        exit()
elif inputVideoFromImageSequence:
    # Capture video stream from sequence of image files
    cap = get_video_input_from_images(inVidImSeq_folderpath, inVidImSeq_fileformat, inVidImSeq_encoding, inVidImSeq_fileindex, inVidImSeq_stopindex, inVidImSeq_fps, inVidImSeq_play_video, inVidImSeq_resolution, inVidImSeq_interpolation_method, inVidImSeq_flip_method,  inVidImSeq_vidout_location, inVidImSeq_vidout_encoder, inVidImSeq_vidout_format, inVidImSeq_preset_level, inVidImSeq_iframeinterval, inVidImSeq_control_rate, inVidImSeq_qp_range, inVidImSeq_bitrate, inVidImSeq_vbv_size,inVidImSeq_idrinterval, inVidImSeq_peak_bitrate, inVidImSeq_ratecontrol_enable, inVidImSeq_wait_for_videowriter, inVidImSeq_sync_video, inVidImSeq_demux_on_screen, verboseStreamOutput)
    if testInVidImSeqRep and cap == 0:
        if verboseStreamOutput:
            print("\nget_video_input_from_images() succesfully finished!\n")
        exit()
    if testInVidImSeqRep == 0 and cap == -1:
        exit()
else:
    # Capture video stream from camera
    cap = cv2.VideoCapture(pipeline_caminput, cv2.CAP_GSTREAMER)
    # Check was the VideoCapture successfully opened
    if not cap.isOpened():
        if verboseStreamOutput:
            print("\n\nFailed to open camera!\n")
        exit()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the output video stream with hardware acceleration
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if output_video_stream and send_to_file:
    vidout_location = os.path.join(vidout_location, datetimeNow)
    os.makedirs(vidout_location)
    vidout_location = os.path.join(vidout_location,vidout_name) 

# Format processing pipeline for the output video stream
fps = cap.get(cv2.CAP_PROP_FPS) # get input fps
fps = round(fps)

pipeline_gstout = "appsrc ! video/x-raw, format=BGR ! videoconvert ! video/x-raw,format=BGRx, framerate=(fraction){fps}/1 ! videorate ! nvvidconv ! video/x-raw(memory:NVMM), format=(string)NV12, framerate=(fraction){fps}/1 ! nvv4l2{encoder}enc profile=0 preset-level={preset_level} iframeinterval={iframeinterval} control-rate={control_rate} {ratecontrol_enable} {qp_range} {bitrate} {vbv_size} {peak_bitrate} insert-sps-pps=true insert-vui=true idrinterval={idrinterval} ! video/x-{encoder}, stream-format=(string)byte-stream, alignment=(string)au".format(fps=int(fps), encoder=vidout_encoder, preset_level=preset_level, iframeinterval=iframeinterval, control_rate=control_rate, qp_range=qp_range, bitrate=bitrate, vbv_size=vbv_size, idrinterval=idrinterval, peak_bitrate=peak_bitrate, ratecontrol_enable=ratecontrol_enable)
pipeline_gstout = " ".join(pipeline_gstout.split())

if send_to_file:
    if vidout_format=="h265" or vidout_format=="h264":
        # H265 stream without multiplexer and encoding in .mp4 container file format. h265 file can be decoded by using nvv4l2decoder hardware acceleration. Examples:
        # gst-launch-1.0 filesrc location=testVideoOut.h265  ! queue ! h265parse ! nvv4l2decoder ! nv3dsink -e -v 
        #
        # or: nvgstplayer-1.0 -i  testVideoOut.h265 --loop-forever --disable-fullscreen
        #
        # in order to display encoding metadata:
        # gst-discoverer-1.0 testVideoOut.h265 -c -v 
        #
        file_spec = " ! {encoder}parse ! queue ! filesink location={location}.{format}".format(encoder=vidout_encoder, location=vidout_location, format=vidout_format)

    if vidout_format=="mp4":
        # H265 stream multiplexed into .mp4 container file format. Can be decoded by using nvv4l2decoder hardware acceleration. Examples:
        # gst-launch-1.0 filesrc location=testVideoOut.mp4 ! qtdemux ! queue ! h265parse ! nvv4l2decoder ! nv3dsink -e -v 
        #
        # or: nvgstplayer-1.0 -i  testVideoOut.mp4 --loop-forever --disable-fullscreen
        #
        file_spec = " ! {encoder}parse ! queue ! qtmux ! filesink location={location}.{format}".format(encoder=vidout_encoder, location=vidout_location, format=vidout_format)
else:
    file_spec = ""

if send_to_port:
    # define output video stream by using UDP protocol and given address
    port_spec = " ! {encoder}parse ! queue ! rtp{encoder}pay pt=96 ! udpsink host={host_address} port={host_port} sync=false".format(encoder=vidout_encoder, host_address=host_address, host_port=host_port)

    # For test purposes, local transmitter can be launched by:
    #
    # gst-launch-1.0 nvarguscamerasrc sensor-id=0 wbmode=1 aelock=false awblock=false exposuretimerange="34000 358733000" gainrange="1 16" exposurecompensation=0.5 saturation=1.3 tnr-mode=1 tnr-strength=-1 ee-mode=2 ee-strength=0.5 aeantibanding=2 ! 'video/x-raw(memory:NVMM), width=2616, height=1946, format=(string)NV12, framerate=(fraction)30/1' ! videorate ! nvvidconv interpolation-method=5 flip-method=0 ! 'video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)30/1' ! nvv4l2h265enc  profile=0 preset-level=4 insert-sps-pps=true ! 'video/x-h265, stream-format=(string)byte-stream, alignment=(string)au'! h265parse ! queue ! rtph265pay pt=96 ! udpsink host=127.0.0.1 port=8001 sync=false -e -v
    #
    # and local receiver by:
    #
    # gst-launch-1.0 udpsrc address=127.0.0.1 port=8001 ! application/x-rtp, encoding-name=H265, payload=96 ! rtph265depay ! queue ! h265parse ! nvv4l2decoder ! nv3dsink sync=false -e -v
    # gst-launch-1.0 udpsrc address=127.0.0.1 port=8001 ! application/x-rtp, encoding-name=H264, payload=96 ! rtph264depay ! queue ! h264parse ! nvv4l2decoder ! nv3dsink sync=false -e -v
    #
    # local receiver with two screen outputs and additional jitter compensation:
    # gst-launch-1.0 udpsrc address=127.0.0.1 port=8001 ! application/x-rtp, encoding-name=H265, payload=96 ! rtpjitterbuffer latency=50 drop-on-latency=true ! rtph265depay ! queue ! h265parse ! tee name=vidout ! nvv4l2decoder ! nv3dsink sync=false -e -v vidout. ! nvv4l2decoder ! nv3dsink sync=false -e -v
    #
    # local receiver recording incoming video without re-encoding:
    #
    # gst-launch-1.0 udpsrc buffer-size=10485760 address=127.0.0.1 port=8001 ! application/x-rtp, media=video, encoding-name=H265, payload=96 ! rtpjitterbuffer latency=50 drop-on-latency=true ! rtph265depay ! queue ! h265parse config-interval=-1 ! queue ! qtmux ! filesink location=testRemoteVideoOut.mp4 -e -v 
    #
    # local receiver recording incoming video without re-encoding and showing decoded video:
    #
    # gst-launch-1.0 udpsrc buffer-size=10485760 address=127.0.0.1 port=8001 ! application/x-rtp, media=video, encoding-name=H265, payload=96 ! rtpjitterbuffer latency=50 drop-on-latency=true ! rtph265depay ! queue ! h265parse config-interval=-1 ! tee name=vidout ! queue ! filesink location=testRemoteVideoOut.h265 vidout. ! nvv4l2decoder ! nv3dsink -e -v 
    #
    # Note that in the case of recording received stream, video file needs proper EOS (end of stream) message at the end of recording (after stopping local receiver). This is forced by the -e flag at the end of gst-launch command.
else:
    port_spec = ""

if send_to_file and send_to_port:
    # split output video stream into two parallel 
    pipeline_gstout = pipeline_gstout + " ! tee name=vidout" + port_spec + "  vidout." + file_spec
else:
    # output exclusively to file or port
    pipeline_gstout = pipeline_gstout + file_spec + port_spec


# Initialize video stream writing, connect defined pipeline to the source
if output_video_stream and (send_to_file or send_to_port):
    if verboseStreamOutput:
        print("\nOutput video stream processing pipeline:\n"+pipeline_gstout)
    if camout_fps != fps:
        if verboseStreamOutput:
            print("Frame rate in the output stream setup is different from the input stream fps ...\n")
        exit()

    fourcc = 0 # set fourcc to 0 to push raw video

    # if specified, in addition to output containing processed video stream also create the unprocessed one, and output exlucsively latter (if specified)
    if send_unprocessed:
        pipeline_gstout_unprocessed = pipeline_gstout
        if send_to_port:
            pipeline_gstout_unprocessed = pipeline_gstout_unprocessed.replace("host={host_address}".format(host_address=host_address), "host={host_address}".format(host_address=host_address_unprocessed))
            pipeline_gstout_unprocessed = pipeline_gstout_unprocessed.replace("port={host_port}".format(host_port=host_port), "port={host_port}".format(host_port=host_port_unprocessed))
        if send_to_file:
            pipeline_gstout_unprocessed= pipeline_gstout_unprocessed.replace("location={location}".format(location=vidout_location), "location={location}".format(location=vidout_location+ "_unprocessed"))

        vidout_unprocessed = cv2.VideoWriter(pipeline_gstout_unprocessed, cv2.CAP_GSTREAMER, fourcc, float(fps), (int(camout_width), int(camout_height)))
    
        if not vidout_unprocessed.isOpened():
            if verboseStreamOutput:
                print("\n\nFailed to open unprocessed output video stream\n")
            exit()

        if not send_only_unprocessed:
            vidout = cv2.VideoWriter(pipeline_gstout, cv2.CAP_GSTREAMER, fourcc, float(fps), (int(camout_width), int(camout_height)))
        else:
            vidout = vidout_unprocessed
            vidout_location = vidout_location + "_unprocessed"
    else:
        vidout = cv2.VideoWriter(pipeline_gstout, cv2.CAP_GSTREAMER, fourcc, float(fps), (int(camout_width), int(camout_height)))

    if not vidout.isOpened():
        if verboseStreamOutput:
            print("\n\nFailed to open output video stream\n")
        exit()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the output image stream with hardware acceleration
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if output_image_stream and sendImage_to_file:
    imout_location = os.path.join(imout_location, datetimeNow)
    if send_unprocessed_image:
        os.makedirs(imout_location+"_unprocessed")
        if not send_only_unprocessed_image:
            os.makedirs(imout_location)
    else:
        os.makedirs(imout_location)

pipeline_gstout_image = "appsrc ! video/x-raw, format=BGR ! videoconvert ! video/x-raw,format=BGRx, framerate=(fraction){imout_fps}/1 ! videorate ! nvvidconv ! video/x-raw(memory:NVMM), format=(string)NV12, framerate=(fraction){imout_fps}/1 ! {imout_encoder} quality={imout_quality} idct-method={idct_method}".format(imout_encoder=imout_encoder, imout_fps=imout_fps, imout_quality=imout_quality, idct_method=idct_method)
pipeline_gstout_image = " ".join(pipeline_gstout_image.split())

if sendImage_to_file:
    if imout_format=="jpg":
        file_spec_imout = " ! queue ! multifilesink index=1 location={imout_location}.{format}".format(imout_location=os.path.join(imout_location, "image_%05d"),format=imout_format)
else:
    file_spec_imout = ""

if sendImage_to_port:
    port_spec_imout = " ! queue ! rtpjpegpay pt=26 ! udpsink buffer-size=10485760  host={host_address} port={host_port} sync=false".format(host_address=host_address_imout, host_port=host_port_imout)

    # for test purposes, local transmitter can be launched by:
    #
    # gst-launch-1.0 nvarguscamerasrc sensor-id=0 wbmode=1 aelock=false awblock=false exposuretimerange="34000 358733000" gainrange="1 16" exposurecompensation=0.5 saturation=1.3 tnr-mode=1 tnr-strength=-1 ee-mode=2 ee-strength=0.5 aeantibanding=2 ! 'video/x-raw(memory:NVMM), width=2616, height=1946, format=(string)NV12, framerate=(fraction)30/1' ! videorate ! nvvidconv interpolation-method=5 flip-method=0 ! 'video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)2/1' ! nvjpegenc quality=100 idct-method=2 ! queue ! rtpjpegpay pt=26 ! udpsink host=127.0.0.1 port=8002 sync=false -e -v
    #
    # and local receiver connected to screen by:
    #
    # gst-launch-1.0 udpsrc buffer-size=10485760 address=127.0.0.1 port=8002 ! application/x-rtp,encoding-name=JPEG, payload=26 ! rtpjitterbuffer latency=1000 drop-on-latency=true ! rtpjpegdepay ! queue ! nvjpegdec idct-method=2 ! nv3dsink -e -v
    #
    # or by recording received images (note that the specified location should have write permissions):
    #
    # gst-launch-1.0 udpsrc buffer-size=10485760 address=127.0.0.1 port=8002 ! application/x-rtp,encoding-name=JPEG, payload=26 ! rtpjitterbuffer latency=1000 drop-on-latency=true ! rtpjpegdepay ! queue ! multifilesink index=1 location=image_%05d.jpg -e -v
    #
    # or by doing both at the same time:
    # gst-launch-1.0 udpsrc buffer-size=10485760 address=127.0.0.1 port=8002 ! application/x-rtp,encoding-name=JPEG, payload=26 ! rtpjitterbuffer latency=1000 drop-on-latency=true ! rtpjpegdepay ! tee name=imout ! queue ! multifilesink index=1 location=image_%05d.jpg imout. ! queue ! nvjpegdec idct-method=2 ! nv3dsink -e -v
else:
    port_spec_imout = ""

if sendImage_to_file and sendImage_to_port:
    # split output image stream into two parallel 
    pipeline_gstout_image = pipeline_gstout_image + " ! tee name=imout" + port_spec_imout + "  imout." + file_spec_imout
else:
    # output exclusively to file or port
    pipeline_gstout_image = pipeline_gstout_image + file_spec_imout + port_spec_imout


# Initialize image stream writing, connect defined pipeline to the source
if output_image_stream and (sendImage_to_file or sendImage_to_port):
    if verboseStreamOutput:
        print("\nOutput image stream processing pipeline:\n"+pipeline_gstout_image)
    if imout_fps > camout_fps:
        if verboseStreamOutput:
            print("Output images cannot be generated more frequently than the camera output ...\n")
        exit()

    # if specified, in addition to output containing stream of processed images also create an additional, unprocessed one; in addition, if specified, output exlucsively the latter (unprocessed)
    if send_unprocessed_image:
        pipeline_gstout_image_unprocessed = pipeline_gstout_image
        if send_to_port:
            pipeline_gstout_image_unprocessed = pipeline_gstout_image_unprocessed.replace("host={host_address}".format(host_address=host_address_imout), "host={host_address}".format(host_address=host_address_imout_unprocessed))
            pipeline_gstout_image_unprocessed = pipeline_gstout_image_unprocessed.replace("port={host_port}".format(host_port=host_port_imout), "port={host_port}".format(host_port=host_port_imout_unprocessed))
        if send_to_file:
            pipeline_gstout_image_unprocessed= pipeline_gstout_image_unprocessed.replace("location={location}".format(location=imout_location), "location={location}".format(location=imout_location+ "_unprocessed"))

        imout_unprocessed = cv2.VideoWriter(pipeline_gstout_image_unprocessed, cv2.CAP_GSTREAMER, 0, float(imout_fps), (int(camout_width), int(camout_height)))
    
        if not imout_unprocessed.isOpened():
            if verboseStreamOutput:
                print("\n\nFailed to open unprocessed output image stream\n")
            exit()

        if not send_only_unprocessed_image:
            imout = cv2.VideoWriter(pipeline_gstout_image, cv2.CAP_GSTREAMER, 0, float(imout_fps), (int(camout_width), int(camout_height)))
        else:
            imout = imout_unprocessed
    else:
        imout = cv2.VideoWriter(pipeline_gstout_image, cv2.CAP_GSTREAMER, 0, float(imout_fps), (int(camout_width), int(camout_height)))
    
    if not imout.isOpened():
        if verboseStreamOutput:
            print("\n\nFailed to open output image stream\n")
        exit()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize camera capture preview window and counters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if show_camout_on_screen:
    cv2.namedWindow("Camera input", cv2.WINDOW_KEEPRATIO) 
    cv2.resizeWindow("Camera input", camout_width, camout_height)
    # create additional screen windows when necessary
    #if USE_TensorRT:
    #    cv2.namedWindow("Processing engine", cv2.WINDOW_KEEPRATIO)
    #    cv2.resizeWindow("Processing engine", camout_width, camout_height)
frameNumber = 0
skipping_finished = 0
vidout_sizeEstimate = 0
imoutNumber = 0;
imout_sizeEstimate = 0
imout_foldersizeEstimate = 0
imout_foldersizeEstimateIncrement = 0
if send_unprocessed or send_unprocessed_image:
    frame_unprocessed = np.zeros((camout_height, camout_width, 3), dtype=np.uint8)
cv2.waitKey(1)


if verboseStreamOutput:
    print("\n*** {datetimestr} *** Stream processing initialization ***\n".format(datetimestr=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    if output_video_stream:
        if send_to_file:
            print("Recording of {vidout_encoder} encoded output video stream to:\n".format(vidout_encoder=vidout_encoder)+"{location}.{format}\n".format (location=vidout_location, format=vidout_format))
            if max_filesize:
                print("Video file size limited to {max_filesize} MB\n".format(max_filesize=max_filesize))
            if max_duration:
                print("Video duration limited to {max_duration} seconds\n".format(max_duration=max_duration))
        if send_to_port:
            print("Sending of {vidout_encoder} encoded output video stream to:\n".format(vidout_encoder=vidout_encoder, vidout_format=vidout_format)+"host address: {host_address}, port: {host_port}\n".format (host_address=host_address, host_port=host_port))
        if not (send_to_file or send_to_port):
            print("Output video stream not specified ...\n")

    if output_image_stream:
        if sendImage_to_file:
            print("Recording of {imout_encoder} encoded output images to:\n".format(imout_encoder=imout_encoder)+"{imout_location}".format(imout_encoder=imout_encoder, imout_location=imout_location)+"\nevery {imout_interval} s".format(imout_interval=1/imout_fps))
            if imout_max_foldersize:
                print("Folder size limited to {imout_max_foldersize} MB\n".format(imout_max_foldersize=imout_max_foldersize))
            if imout_max_duration:
                print("Recording duration limited to {max_duration} seconds\n".format(max_duration=imout_max_duration))
        if sendImage_to_port:
            print("Sending of {imout_encoder} encoded output images to:".format(imout_encoder=imout_encoder)+"\nhost address: {host_address_imout}, port: {host_port},".format (host_address_imout=host_address_imout, host_port=host_port_imout) + " every {imout_interval} s\n".format(imout_interval=1/imout_fps))
        if not (sendImage_to_file or sendImage_to_port):
            print("Output image stream not specified ...\n")



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize inference engine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if USE_TensorRT:
    engine_initialization_finished = 0
    # Run computational graph using tensorRT engine
    if verboseStreamOutput:
        print("\n*** {datetimestr} *** Inference engine initialization ***\n".format(datetimestr=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    import numpy as np
    import pycuda.driver as cuda
    import pycuda.autoinit
    device = cuda.Device(0)     # Assuming device 0
    if verboseStreamOutput:
        print("pycuda.version: "+ str(cuda.get_version()))
        print("pycuda.driver_version: "+ str(cuda.get_driver_version()))
        print("cuda device count: "+ str(cuda.Device.count()))
        print("device: "+ device.name())
        print("compute capability: "+ str(device.compute_capability()))
        freeGpuMem, totalGpuMem = cuda.mem_get_info()
        print("free GPU memory (1GB=1024MB): "+ str(freeGpuMem) + " Bytes = " + str(round(freeGpuMem/(1024*1024*1024), 2))+" GB")
        print("total GPU memory (1GB=1024MB): "+ str(totalGpuMem) + " Bytes = " + str(round(totalGpuMem/(1024*1024*1024), 2))+" GB")

    import tensorrt as trt
    if verboseStreamOutput:
        print("tensorrt: "+ trt.__version__)
    assert trt.Builder(trt.Logger())

    TRT_LOGGER = trt.Logger()

    # TensorRT model (engine) base name, relative path, and parameters
    #model_base_name = "nyu_half_GuideDepth-S_FP16"
    model_base_name = "nyu_full_GuideDepth-S_FP16"				
    model_relative_path = "results/produced engines/"

    engine_input_width = 640           # trt input width
    engine_input_height= 480           # trt input height

    #engine_input_width = 320           # trt input width
    #engine_input_height= 240           # trt input height

    engine_input_depth = 3              # trt input number of channels
    # For given FCN semantic segmentation model it holds that:
    engine_output_width = engine_input_width
    engine_output_height = engine_input_height
    engine_output_depth = 1
    USE_FP16 = True                   # floating point precision
    if USE_FP16:
        fp_precision = 16
    else:
        fp_precision = 32
    #trt_model_name = os.path.join(os.getcwd(), model_relative_path, "{model_name}_pytorch_{tw}x{th}__fp{fp_precision}_engine.trt".format(model_name=model_base_name, tw=engine_input_width, th=engine_input_height, fp_precision = str(fp_precision)))
    trt_model_name = os.path.join(os.getcwd(), model_relative_path, "{model_name}.engine".format(model_name=model_base_name))

    engine_file = trt_model_name

    def load_engine(engine_file_path):
        assert os.path.exists(engine_file_path)
        if verboseStreamOutput:
            print("Loading TensorRT engine for" + model_base_name + "model")
            print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    engine =  load_engine(engine_file)

    if verboseStreamOutput:
        print("Creating execution context")
    context = engine.create_execution_context()

    # Set shape of the engine input based on image dimensions 
    #context.set_binding_shape(engine.get_binding_index("input"), (1, engine_input_depth, engine_input_height, engine_input_width))
    # Rest of the inference engine initialization is performed at the begining of processing loop

    def preprocess_engineInput(input, output, temp, interpolationMethod=cv2.INTER_LINEAR):
        # temp and output are preallocated to match resized input and required output shape, respectively
        # change input colorspace to RGB (required by PyTorch model)
        # check whether dimensions match
        height, width = output.shape[1:3]
        input_height, input_width = input.shape[0:2]
        if(input_height != height or input_width != width):
            # resize and change color space
            temp[:,:,:] = cv2.resize(cv2.cvtColor(input, cv2.COLOR_BGR2RGB), (width, height), interpolation=interpolationMethod)
        else:
            # just change color space
            temp[:,:,:] = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

        # since Pytorch model was trained on normalized dataset, additional whitening of input data is necesary (Mean and Std normalization)
        mean = np.array([0.485, 0.456, 0.406]).astype('float32')
        stddev = np.array([0.229, 0.224, 0.225]).astype('float32')
        temp[:,:,:] = (np.asarray(temp).astype('float32') / float(255.0) - mean) / stddev

        # Switch from Height Width Channel (HWC) dimensions order to CHW order, required by Pytorch model
        output[:,:,:] = np.moveaxis(temp, 2, 0)

    # depth map normalization
    def inverse_depth_norm(maxDepth, depth):
        depth = maxDepth / depth
        np.clip(depth, maxDepth / 100, maxDepth, depth)
        return depth

    def depth_norm(maxDepth, depth):
        np.clip(depth, maxDepth / 100, maxDepth, depth)
        depth = maxDepth / depth
        return depth


    def postprocess_engineOutput(input, output, temp1, temp2, interpolationMethod=cv2.INTER_LINEAR, colormap=cv2.COLORMAP_MAGMA):
        # based on input (predicted depth map) resize result to output resolution, normalize and create pseudocolor image
        # function does not return values, but modifies arguments: output contains depth map visualization
        # input, predicted values are not changed
        height, width, _ = output.shape
        in_height, in_width, _ = input.shape
        if(in_height != height or in_width != width):
            temp1[:,:] = cv2.resize(input, (width, height), interpolation=interpolationMethod)
        else:
            temp1[:,:] = input[:,:,0]
        # normalize depth map
        maxDepth_modeled = 10
        temp1[:,:] = inverse_depth_norm(maxDepth_modeled, temp1)     # 1 channel float
        # scale to uint8 range
        temp2[:,:] = ((temp1 - temp1.min()) / (temp1.max()- temp1.min())*255).astype(np.uint8)  # 1 channel uint8
        # apply colormap and overwrite original frame
        output[:,:] = cv2.applyColorMap(255-temp2, colormap)  # 3 channel uint8
        

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Processing loop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
while True:

    # frame grabbing
    ret, frame = cap.read()
    if not ret:
        if verboseStreamOutput:
            print("Failed to read frame!\n")
        break
    frameNumber += 1

    if USE_TensorRT:
        if not engine_initialization_finished and frameNumber == 1:
            # preallocate memory for input frame preprocessing in order to match engine input characteristics
            temp_pre = np.zeros((engine_input_height, engine_input_width, engine_input_depth), dtype='float32')    # HWC order of dimensions
            engine_input = np.zeros((engine_input_depth, engine_input_height, engine_input_width), dtype='float32')    # CHW order of dimensions
            # also for engine result postprocessing
            engine_output = np.zeros((engine_output_height, engine_output_width, engine_output_depth), dtype='float32')
            temp1_post = np.zeros((camout_height, camout_width), dtype='float32')
            temp2_post = np.zeros((camout_height, camout_width), dtype=frame.dtype)

        # Preprocess input frame into tensorRT engine format  
        preprocess_engineInput(frame, engine_input, temp_pre)

        # finish inference engine initialization
        if not engine_initialization_finished and frameNumber == 1:
            # Allocate host and device buffers for inference engine
            engine_bindings = []
            for binding in engine:
                binding_idx = engine.get_binding_index(binding)
                binding_size = trt.volume(context.get_binding_shape(binding_idx))
                trt_dtype = trt.nptype(engine.get_binding_dtype(binding))
                if engine.binding_is_input(binding):
                    # create row-major (C type) memory buffer to map engine_input from host to cuda device
                    engine_input_buffer = np.ascontiguousarray(engine_input, dtype=engine_input.dtype)
                    # allocate CUDA device memory for input data
                    engine_input_memory = cuda.mem_alloc(engine_input.nbytes)
                    # convert input memory address to integer value (pointer) and put it in the bindings list
                    engine_bindings.append(int(engine_input_memory))
                else:
                    # create pinned (page-locked) memory buffer to map engine_output from cuda device to host
                    engine_output_buffer = cuda.pagelocked_empty(binding_size, trt_dtype)
                    # allocate CUDA device memory for output data
                    engine_output_memory = cuda.mem_alloc(engine_output_buffer.nbytes)
                    # convert output memory address to integer value (pointer) and put it in the bindings list
                    engine_bindings.append(int(engine_output_memory))
                    # Create linear sequence of execution with given specifications
            engine_stream = cuda.Stream()
            engine_initialization_finished = 1

    if send_unprocessed or send_unprocessed_image:
        frame_unprocessed = frame.copy()

    if not skipping_finished:
        # skip output processing for N initial frames
        if frameNumber < skip_Nframes:
            if verboseStreamOutput and frameNumber==1:
                print("\n ... Camera calibration ... skipping first {} frames!\n".format(skip_Nframes))
            continue
        else:
            skipping_finished = 1
            frameNumber = 0
            continue

    # vidout test switch, turn off in deployment
    # print("
    # Frame: {}".format(frameNumber))
    if test_maxNframes > 0:
        if frameNumber > test_maxNframes:
            break 

    # imout test switch, turn off in deployment
    # print("
    # Frame: {}".format(frameNumber))
    if test_maxNimagesOut > 0:
        if imoutNumber > test_maxNimagesOut:
            break 
   
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # frame processing
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if USE_TensorRT:
        if frameNumber == 1:
            if skip_Nframes > 0:
                # since Nframes were skipped from the beginning of input stream, engine_input that was made based on the first frame needs to be re-created using new input (else just perform computation, frame is already preprocessed)
                preprocess_engineInput(frame, engine_input, temp_pre)
        else:
            # preprocess input frame into tensorRT engine format  
            preprocess_engineInput(frame, engine_input, temp_pre)
    
        # transfer input data to cuda device
        engine_input_buffer = np.ascontiguousarray(engine_input, dtype=engine_input.dtype)
        cuda.memcpy_htod_async(engine_input_memory, engine_input_buffer, engine_stream)

        # perform computation on cuda device
        #if not context.execute_async_v2(bindings=engine_bindings, stream_handle=engine_stream.handle):
        # note: implicit batching requires older version of the execute_async unction instead of "_v2"
        if not context.execute_async(bindings=engine_bindings, stream_handle=engine_stream.handle):
            print("Model execution did not succeed, please check context.execute_async() call ...")
            continue

        # collect results, cuda device to host copy
        cuda.memcpy_dtoh_async(engine_output_buffer, engine_output_memory, engine_stream)

        # synchronize operations
        engine_stream.synchronize()

        # collect results from the buffer
        engine_output = np.reshape(engine_output_buffer, (engine_output_height, engine_output_width, engine_output_depth))

        # postprocess the results and send downstream
        postprocess_engineOutput(engine_output, frame, temp1_post, temp2_post, interpolationMethod=cv2.INTER_LINEAR, colormap=cv2.COLORMAP_MAGMA)
    # continue with stream processing
    frame = cv2.putText(frame, "{}".format(frameNumber), (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 3, (0, 0, 255), 2, cv2.LINE_AA)

    if show_camout_on_screen:
        cv2.imshow("Camera input", frame)
 
    # write/send output video stream(s)
    if output_video_stream and (send_to_file or send_to_port):    
        if send_unprocessed:
            # write unprocessed output video stream
            vidout_unprocessed.write(frame_unprocessed)
            if not send_only_unprocessed:
                # write both output video streams
                vidout.write(frame)
        else:
            # write only processed video stream
            vidout.write(frame)      


    # write/send output image stream
    if output_image_stream and (sendImage_to_file or sendImage_to_port):
        # write periodically, depending on fps ratio
        if (frameNumber % math.ceil(camout_fps/imout_fps) == 1) or (camout_fps==imout_fps):            
            if send_unprocessed_image:
                # write unprocessed image stream
                imout_unprocessed.write(frame_unprocessed)
                if not send_only_unprocessed_image:
                # write both image streams
                    imout.write(frame)
            else:
                # write only processed image stream
                imout.write(frame)
    
            imoutNumber += 1
            imout_foldersizeEstimateIncrement = 1


    # stop sending/recording, predefined criteria
    if output_video_stream == 1:
        # limit output video file size
        if send_to_file and (max_filesize > 0):
            vidout_sizeEstimate = os.path.getsize("{location}.{format}".format (location=vidout_location, format=vidout_format))
            #print(vidout_sizeEstimate)
            if vidout_sizeEstimate > max_filesize *1024*1024:
                # stop recording/sending output video
                output_video_stream = 0
        
        # limit output video stream duration (in seconds)
        if max_duration > 0:
            if math.ceil(frameNumber/camout_fps) >= max_duration:
                # stop recording/sending output video
                output_video_stream = 0
        
        if output_video_stream == 0:
            if verboseStreamOutput:
                print("\n*** {datetimestr} ***".format(datetimestr=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + "\nRecording/sending of output video ended after ~ ", math.ceil(frameNumber/camout_fps), "seconds")
                if send_to_file:
                    print("Video file location: ", "{location}.{format}".format (location=vidout_location, format=vidout_format))
                    if vidout.isOpened():
                        vidout.release()
                    if vidout_sizeEstimate == 0:
                        vidout_sizeEstimate = os.path.getsize("{location}.{format}".format (location=vidout_location, format=vidout_format))
                    print("Expected video size: ", round(vidout_sizeEstimate/(1024*1024), 2), "MB\n")

    if output_image_stream == 1:
        # limit output folder size
        if sendImage_to_file and (imout_max_foldersize > 0):
            if imoutNumber == 5 and imout_sizeEstimate == 0:
                # get initial size estimate of the output image
                imout_sizeEstimate = os.path.getsize(os.path.join(imout_location,"image_{number:05d}.{format}".format(number=imoutNumber-2, format=imout_format)))
                imout_expectedNumber = math.ceil(imout_max_foldersize*1024*1024/imout_sizeEstimate)
                if verboseStreamOutput:
                    print("\n*** {datetimestr} ***".format(datetimestr=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + "\nSize estimate of single output image:", round(imout_sizeEstimate/(1024*1024), 2), "MB", "\nExpected number of output images:", imout_expectedNumber)

            elif imoutNumber > 5:
                if imoutNumber > round(0.95*imout_expectedNumber) and imout_foldersizeEstimate == 0:
                    # get initial output folder size estimate
                    imout_foldersizeEstimate = get_folder_size(*"{location},.{format}".format (location=imout_location, format=imout_format).split(","))
                    if verboseStreamOutput:
                        print("\n*** {datetimestr} ***".format(datetimestr=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + "\nOutput folder size after recording {NimagesStr} output images:".format(NimagesStr=imoutNumber), round(imout_foldersizeEstimate/(1024*1024), 2), "MB")
                    if imout_foldersizeEstimate > round(0.98*imout_max_foldersize*1024*1024):
                        # stop recording/sending output images if necessary
                        output_image_stream = 0                

                elif imout_foldersizeEstimate > round(0.98*imout_max_foldersize*1024*1024) and imout_foldersizeEstimateIncrement == 1:
                    # approaching size limit, start checking output folder size
                    imout_foldersizeEstimate = get_folder_size(*"{location},.{format}".format (location=imout_location, format=imout_format).split(","))
                    imout_foldersizeEstimateIncrement = 0
                    if imout_foldersizeEstimate > round(0.99*imout_max_foldersize*1024*1024):
                        # stop recording/sending output images
                        output_image_stream = 0
                elif imout_foldersizeEstimate > 0 and imout_foldersizeEstimateIncrement == 1:
                    imout_foldersizeEstimate += imout_sizeEstimate
                    imout_foldersizeEstimateIncrement = 0

        # limit output image stream duration (in seconds)
        if imout_max_duration > 0:
            if math.ceil(imoutNumber/imout_fps) >= imout_max_duration:
                # stop recording/sending output images
                output_image_stream = 0

        if output_image_stream == 0:
            if verboseStreamOutput:
                print("\n*** {datetimestr} ***".format(datetimestr=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + "\nRecording/sending of output images ended after ~ ", math.ceil(imoutNumber/imout_fps), "seconds", "\nExpected number of output images:", imoutNumber)
                if sendImage_to_file:
                    print("Folder location: ", imout_location)
                    if imout.isOpened():
                        imout.release()
                    if imout_foldersizeEstimate == 0:
                        imout_foldersizeEstimate = get_folder_size(*"{location},.{format}".format (location=imout_location, format=imout_format).split(","))
                    print("Expected folder size: ", round(imout_foldersizeEstimate/(1024*1024), 2), "MB\n")                  

    # stopping criteria
    if(not show_camout_on_screen and not output_video_stream and not output_image_stream):
        break
    # if camera capture is shown, continue capture until exit signal
    if show_camout_on_screen and (cv2.waitKey(1)&0xFF == ord("q")):
        break    
    # wait for ctrl+c



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Clean workspace
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Release the VideoCapture object, VideoWriter object, and close windows
cap.release()

if output_video_stream and (send_to_file or send_to_port):
    if verboseStreamOutput:
        print("\n*** {datetimestr} ***".format(datetimestr=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + "\nRecording/sending of output video ended after ~ ", math.ceil(frameNumber/camout_fps), "seconds")
        if send_to_file:
            print("Video file location: ", "{location}.{format}".format (location=vidout_location, format=vidout_format))
            if vidout.isOpened():
                vidout.release()
            if vidout_sizeEstimate == 0:
                vidout_sizeEstimate = os.path.getsize("{location}.{format}".format (location=vidout_location, format=vidout_format))
            print("Expected video size: ", round(vidout_sizeEstimate/(1024*1024), 2), "MB\n")
    else:
        if vidout.isOpened():
            vidout.release()
    if send_unprocessed and not send_only_unprocessed:
        if vidout_unprocessed.isOpened():
            vidout_unprocessed.release()
    # the case when unprocessed output is the only one is covered by: vidout = vidout_unprocessed


if output_image_stream and (sendImage_to_file or sendImage_to_port):
    if verboseStreamOutput:
        print("\n*** {datetimestr} ***".format(datetimestr=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + "\nRecording/sending of output images ended after ~ ", math.ceil(imoutNumber/imout_fps), "seconds", "\nExpected number of output images: {imoutNumber} \n".format(imoutNumber = imoutNumber))
        if sendImage_to_file:
            print("Folder location: ", imout_location)
            if imout.isOpened():
                imout.release()
            if imout_foldersizeEstimate == 0:
                imout_foldersizeEstimate = get_folder_size(*"{location},.{format}".format (location=imout_location, format=imout_format).split(","))
            print("Expected folder size: ", round(imout_foldersizeEstimate/(1024*1024), 2), "MB\n")
    else:
        if imout.isOpened():
            imout.release()
    if send_unprocessed and not send_only_unprocessed:
        if imout_unprocessed.isOpened():
            imout_unprocessed.release()
    # the case when unprocessed image output is the only one is covered by: imout = imout_unprocessed              

if show_camout_on_screen:
    cv2.destroyAllWindows()



