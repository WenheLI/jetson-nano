import jetson.inference
import jetson.utils
import sys

net = jetson.inference.detectNet("ssd-mobilenet-v2", .5)

camera = jetson.utils.gstCamera(1280, 720, sys.argv, "0")
display = jetson.utils.glDisplay()

while display.IsOpen():
    img, width, height = camera.CaptureRGBA()

    detections = net.Detect(img, width, height, "box,labels,conf")

    # print("detected {:d} objects in image".format(len(detections)))

    for detection in detections:
        print(detection)

    display.RenderOnce(img, width, height)

    display.SetTitle(" ssd-mobilenet-v2 | Network {:.0f} FPS".format(net.GetNetworkFPS()))

    net.PrintProfilerTimes()