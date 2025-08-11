"""
Quantize Tapir model with random data. For testing quantization only - the model outputs will not be good.
The quantized model will be saved as `{original_model}_int8.xml`

Requirements: `pip install openvino nncf`
Usage: `python quantize_random.py /path/to/predictor_model.xml`
"""

import sys
from pathlib import Path

import nncf
import numpy as np
import openvino as ov

model_path = sys.argv[1]

model = ov.Core().read_model(model_path)

# create random data
input_frame = np.random.rand(*tuple(model.inputs[0].shape))
query_feats = np.zeros(tuple(model.inputs[1].shape)).astype(np.float32)
hires_query_feats = np.zeros(tuple(model.inputs[2].shape)).astype(np.float32)
causal_state = np.zeros(tuple(model.inputs[3].shape)).astype(np.float32)
random_data = {"input_frame": input_frame, "query_feats": query_feats, "hires_query_feats": hires_query_feats, "causal_state": causal_state}
random_dataset = [random_data] * 5

# quantize model
calibration_dataset = nncf.Dataset(random_dataset)
quantized_model = nncf.quantize(model=model, calibration_dataset=calibration_dataset, fast_bias_correction=True)
quantized_model_file = Path(model_path).stem + "_int8.xml"
output_path = Path(model_path).with_name(quantized_model_file)
ov.save_model(quantized_model, output_path)
print(f"Quantized model saved to {output_path}")
