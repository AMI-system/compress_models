import torchvision
from torchvision import transforms
import datetime
import numpy as np
import tensorflow as tf
from PIL import Image


image = Image.open('./3872946773_crop00_detections.jpg')

# Transform
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((300, 300)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
img = transform(image)

def tflite_inference(image, interpreter, print_time=False):
    a = datetime.datetime.now()
    interpreter.set_tensor(input_details[0]['index'], image.unsqueeze(0))
    interpreter.invoke()
    outputs_tf = interpreter.get_tensor(output_details[0]['index'])
    prediction_tf = np.squeeze(outputs_tf)
    prediction_tf = prediction_tf.argsort()[::-1][0]
    #print(prediction_tf)
    b = datetime.datetime.now()
    c = b - a
    if print_time: print(str(c.microseconds) + "\u03bcs")
    return(prediction_tf)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./mila_uk_denmark.tflite")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()

pred = tflite_inference(img, interpreter, print_time=True)
print(pred)