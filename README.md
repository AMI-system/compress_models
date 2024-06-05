# Compress models

Scripts to compress existing PyTorch models to TFLite models. 

Both models can subsequently evaluated with sandbox/Evaluation.ipynb. 


Example call: 

```sh
python convert_model.py \
    --model_path ~/amber/projects/species_classifier/outputs/turing-costarica_v03_resnet50_2024-06-04-16-17.pt \
    --output_dir ~/amber/data/compressed_models/gbif_costarica/ \
    --image_path ~/amber/data/gbif_download_standalone/gbif_images/Noctuidae/Spodoptera/Spodoptera exigua/1211977745.jpg \
    --labels_json_path ~/amber/data/gbif_costarica/03_costarica_data_numeric_labels.json
```


tflite_inference.py provides example for running the tflite model. 