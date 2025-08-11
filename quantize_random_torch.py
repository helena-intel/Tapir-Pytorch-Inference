"""
Quantize PyTorch Tapir model with NNCF and convert to OpenVINO.
The model is quantized with random data. This is for testing purposes only!
Both a dynamic and static quantized model (based on input shapes provided) will be saved

Requirements: `pip install openvino nncf torch`
Usage: modify settings at top of the file and run script (in a Developer Command Prompt if you have Visual Studio installed)

"""
import copy
import time

import cv2
import nncf
import numpy as np
import openvino as ov
import torch

from tapnet.tapir_inference import TapirPointEncoder, TapirPredictor, build_model

model_path = "models/causal_bootstapir_checkpoint.pt"
resolution = 480
num_points = 100
num_iters = 4
device = "cpu"
dynamic_ir_path = "tapir_predictor_int8_torch_dynamic.xml"
static_ir_path = f"tapir_predictor_int8_torch_p{num_points}_n{num_iters}_{resolution}.xml"

model = build_model(model_path,(resolution, resolution), num_iters, True, device).eval()
predictor = TapirPredictor(model).to(device).eval()
encoder = TapirPointEncoder(model).to(device).eval()

causal_state_shape = (num_iters, model.num_mixer_blocks, num_points, 2, 512 + 2048)
causal_state = torch.zeros(causal_state_shape, dtype=torch.float32, device=device)

query_points = torch.zeros((num_points, 2), dtype=torch.float32, device=device)
feature_grid = torch.zeros((1, resolution//8, resolution//8, 256), dtype=torch.float32, device=device)
hires_feats_grid = torch.zeros((1, resolution//4, resolution//4, 128), dtype=torch.float32, device=device)

input_frame = torch.randn(1, 3, resolution, resolution, dtype=torch.float32, device=device)


with torch.no_grad():
    encoder_result = encoder(query_points[None], feature_grid, hires_feats_grid)
    predictor_input = (input_frame, encoder_result[0], encoder_result[1], causal_state)
    result = predictor(*predictor_input)

random_dataset = [predictor_input]
calibration_dataset = nncf.Dataset(random_dataset)
quantized_model = nncf.quantize(model=predictor, calibration_dataset=calibration_dataset, fast_bias_correction=True)

input_shapes = {"input_frame": tuple(input_frame.shape), "query_feats": tuple(encoder_result[0].shape), "hires_query_feats": tuple(encoder_result[1].shape), "causal_state": causal_state_shape}
ov_predictor_dynamic = ov.convert_model(quantized_model, example_input = predictor_input)
ov.save_model(ov_predictor_dynamic, static_ir_path, compress_to_fp16=False)

# for debugging, create static model both by reshaping dynamic model and by converting as static model
ov_predictor_static_reshaped = copy.deepcopy(ov_predictor_dynamic)
ov_predictor_static_reshaped.reshape(input_shapes)
ov.save_model(ov_predictor_static_reshaped, static_ir_path.replace(".xml", "_reshaped.xml"), compress_to_fp16=False)

ov_predictor_static = ov.convert_model(quantized_model, example_input = predictor_input, input=input_shapes) 
ov.save_model(ov_predictor_static, static_ir_path, compress_to_fp16=False)

print(f"Saved quantized models to {dynamic_ir_path} and {static_ir_path}")
