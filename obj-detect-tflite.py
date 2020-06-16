import tflite_runtime.interpreter as tflite
import jetson.utils
import sys
import time
import numpy as np
from camera_helper import show_camera
from utils import *

HEIGHT = 1280
WIDTH = 720

def run_detection(img, interpreter, input_details, output_details):
  interpreter.set_tensor(input_details[0]['index'], img)
  interpreter.invoke()

  boxes = interpreter.get_tensor(output_details[0]['index'])
  classes = interpreter.get_tensor(output_details[1]['index'])
  scores = interpreter.get_tensor(output_details[2]['index'])
  num = interpreter.get_tensor(output_details[3]['index'])

  boxes, scores, classes = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes + 1).astype(np.int32)
  out_scores, out_boxes, out_classes = non_max_suppression(scores, boxes, classes) 
  return out_scores, out_boxes, out_classes

def detect(interpreter, colors, input_details, output_details):

  for frame in show_camera():

      start_time = time.time()
      image_data = preprocess_image_for_tflite(frame, model_image_size=300)
      out_scores, out_boxes, out_classes = run_detection(image_data, interpreter, input_details, output_details)
      result = draw_boxes(frame, out_scores, out_boxes, out_classes, class_names, colors)
      elapsed_ms = (time.time() - start_time) * 1000
      cv2.imshow("Object detection - ssdlite_mobilenet_v2", result)

if __name__ == "__main__":
  print("lite")
  interpreter = tflite.Interpreter(model_path="model_data/ssdlite_mobilenet_v2.tflite")
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  class_names = read_classes("model_data/coco_classes.txt")

  colors = generate_colors(class_names)

  detect(interpreter, colors, input_details, output_details)