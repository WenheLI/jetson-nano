import tflite_runtime.interpreter as tflite
import jetson.utils
import sys
import time
import numpy as np

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]


model_uri = "mobilenet_v1_1.0_224_quant"
interpreter = tf.lite.Interpreter(model_path=model_uri)
interpreter.allocate_tensors()
_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
camera = jetson.utils.gstCamera(1280, 720, sys.argv, "0")
display = jetson.utils.glDisplay()

while display.IsOpen():
    img, width, height = camera.CaptureRGBA()
    start_time = time.time()
    results = classify_image(interpreter, img)
    elapsed_ms = (time.time() - start_time) * 1000
    label_id, prob = results[0]
    display.RenderOnce(img, width, height)
    display.SetTitle(" ssd-mobilenet-v2 | Network {:.0f} FPS".format(60*1000/elapsed_ms))