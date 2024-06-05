import onnx
from onnx_tf.backend import prepare

import torchvision
import torch
import os
import numpy as np
import tensorflow as tf
import json
from PIL import Image
import sys
import argparse

# #sys.path.append('~/amber/projects/species_classifier/')
# sys.path.append('../species_classifier/models/')
# sys.path.append('../species_classifier/data2/')
# sys.path.append('../species_classifier/evaluation/')

# from data2 import dataloader
# import evaluation


# Function to convert a pytorch model to tflite
def pytorch_to_tflite(model, output_dir, image, output_model_prefix="model"):

    # convert the model to onnx
    print("Converting to onnx")

    onnx_path = os.path.join(output_dir, output_model_prefix + ".onnx")
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
    tf_path = os.path.join(output_dir, "tf_" + output_model_prefix)
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
    tflite_path = os.path.join(output_dir, output_model_prefix + ".tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    return tflite_model


def load_model(model_path, device, num_classes):
    if 'efficientnet' in model_path:
        model = models.efficientnet_b0(pretrained=True)
        model = model.to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    elif 'resnet' in model_path:
        model = torchvision.models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
        model = model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        raise ValueError('Unsupported model type')
    return model

def main(model_path, labels_json_path, output_dir, image_path):
    # Load labels
    with open(labels_json_path) as f:
        label_info = json.load(f)['species_list']

    num_classes = len(label_info)

    device = torch.device('cpu')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load and transform image
    image = Image.open(image_path)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((300, 300)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    img = transform(image)

    # Load model
    model = load_model(model_path, device, num_classes)

    # Convert to TFLite
    pref = 'resnet_' + os.path.basename(labels_json_path).split('_')[1]  # Use region from filename
    tflite_model = pytorch_to_tflite(model, output_dir, img, output_model_prefix=pref)
    print("Conversion complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a PyTorch model to TFLite")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the PyTorch model")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory for the converted model")
    parser.add_argument('--image_path', type=str, required=True, help="Path to an example input image")
    parser.add_argument('--labels_json_path', type=str, required=True, help="Path to the numeric labels JSON file")
    args = parser.parse_args()
    
    main(args.model_path, args.output_dir, args.image_path, args.labels_json_path)