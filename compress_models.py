import onnx
from onnx_tf.backend import prepare

import torchvision
import torch
import os
import numpy as np
import tensorflow as tf
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch, torchvision
import sys

#sys.path.append('/bask/homes/f/fspo1218/amber/projects/species_classifier/')
sys.path.append('../species_classifier/models/')
sys.path.append('../species_classifier/data2/')
sys.path.append('../species_classifier/evaluation/')

from data2 import dataloader
import evaluation

# Load an example image
image_file='/bask/homes/f/fspo1218/amber/data/gbif_download_standalone/gbif_images/Noctuidae/Spodoptera/Spodoptera exigua/1211977745.jpg'


# Define the model and labels of interest 
region = 'costarica'
f = open(f"/bask/homes/f/fspo1218/amber/data/gbif_{region}/02_{region}_data_numeric_labels.json")
label_info = json.load(f)
label_info = label_info['species_list']
species_list_mila = list(label_info)
print(len(species_list_mila), " species in total")

num_classes = len(species_list_mila)

files = os.listdir("/bask/homes/f/fspo1218/amber/projects/species_classifier/outputs/")
PATH = os.path.join("/bask/homes/f/fspo1218/amber/projects/species_classifier/outputs/",
               [file for file in files if region in file and 'resnet50' in file and 'state' not in file][1])
print('model: ', PATH)

device = torch.device('cpu')

output_dir = f'/bask/homes/f/fspo1218/amber/data/compressed_models/gbif_{region}/'
os.makedirs(output_dir, exist_ok=True)

# Function to convert a pytorch model to tflite
def pytorch_to_tflite(model, output_dir, image, output_model_prefix="model"):

    # convert the model to onnx
    print("Converting to onnx")

    onnx_path = output_dir + "/" + output_model_prefix + ".onnx"
    torch.onnx.export(
            model=model.eval(),
            args=image.unsqueeze(0),
            f=onnx_path,
            verbose=False,
            export_params=True,
            do_constant_folding=False,
            input_names=['input'],
            opset_version=12,
            output_names=['output']
    )

    # Convert to tf
    print("Converting to tensorflow...")
    tf_path = output_dir + "/tf_" + output_model_prefix
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    tf_rep = prepare(onnx_model, device='CPU')
    tf_rep.export_graph(tf_path)

    # Convert to tfLite
    print("Converting to tensorflowlite")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops=True
    tflite_model = converter.convert()

    print("Saving converted model")
    with open(output_dir + "/" + output_model_prefix + ".tflite", 'wb') as f:
        f.write(tflite_model)

    return tflite_model


# Import image
image = Image.open(image_file)

# Transform
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((300, 300)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
img = transform(image)

if 'efficientnet' in PATH:
    model_py_mila = models.efficientnet_b0(pretrained=True)
    model_py_mila = model_py_mila.to(device)
    checkpoint = torch.load(PATH, map_location=device)
    model_py_mila.eval()

elif 'resnet' in PATH:
    model_py_mila = torchvision.models.resnet50(weights=None)
    num_ftrs = model_py_mila.fc.in_features
    model_py_mila.fc = torch.nn.Linear(num_ftrs, num_classes)
    model_py_mila = model_py_mila.to(device)
    model_py_mila = torch.load(PATH, map_location=device)
    model_py_mila.eval()

else:
    print('clarify model type')

print("loaded MILA model")

pref = 'resnet_' + region

tflite_model = pytorch_to_tflite(model_py_mila,
                  output_dir=output_dir,
                  image=img,
                  output_model_prefix=pref)