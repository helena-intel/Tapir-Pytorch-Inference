# Tapir OpenVINO

This repository adds OpenVINO export and inference. Parts copied from https://github.com/jelyoussefi/Tapir-Pytorch-Inference

## Usage

Clone this repository, run `pip install requirements.txt` and export the model with the `openvino_export.py` script.

## Work In progress

- The model is exported with static shapes for better performance, but inference with the inference scripts/notebook currently only works with 25 and 100 points and inference may have other issues.

- The quantization script quantizes with random data. This is just for testing quantization

