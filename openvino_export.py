import argparse
from pathlib import Path

import openvino as ov
import torch

from tapnet.tapir_inference import TapirPointEncoder, TapirPredictor, build_model

device = torch.device("cpu")


def get_parser():
    parser = argparse.ArgumentParser(description="Tapir OpenVINO Export")
    parser.add_argument("--model", default="models/causal_bootstapir_checkpoint.pt", type=str, help="path to Tapir checkpoint")
    parser.add_argument("--resolution", default=480, type=int, help="Input resolution")
    parser.add_argument("--num_points", default=100, type=int, help="Number of points")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic number of points")
    parser.add_argument("--num_iters", default=4, type=int, help="Number of iterations, 1 for faster inference, 4 for better results")
    parser.add_argument("--output_dir", default="./", type=str, help="Output directory for OpenVINO model")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    model_path = args.model
    resolution = args.resolution
    num_points = args.num_points
    dynamic = args.dynamic
    num_iters = args.num_iters
    output_dir = args.output_dir

    model = build_model(model_path, (resolution, resolution), num_iters, num_points,True, device).eval()
    predictor = TapirPredictor(model).to(device).eval()
    encoder = TapirPointEncoder(model).to(device).eval()

    causal_state_shape = (num_iters, model.num_mixer_blocks, num_points, 2, 512 + 2048)
    causal_state = torch.zeros(causal_state_shape, dtype=torch.float32, device=device)
    feature_grid = torch.zeros((1, resolution // 8, resolution // 8, 256), dtype=torch.float32, device=device)
    hires_feats_grid = torch.zeros((1, resolution // 4, resolution // 4, 128), dtype=torch.float32, device=device)
    query_points = torch.zeros((num_points, 2), dtype=torch.float32, device=device)
    input_frame = torch.zeros((1, 3, resolution, resolution), dtype=torch.float32, device=device)

    with torch.no_grad():
        query_feats, hires_query_feats = encoder(query_points[None], feature_grid, hires_feats_grid)
        tracks, visibles, causal_state, _, _ = predictor(input_frame, query_feats, hires_query_feats, causal_state)

    # Export model
    example_input = {"query_points": query_points[None], "feature_grid": feature_grid, "hires_feats_grid": hires_feats_grid}
    ov_encoder = ov.convert_model(encoder, example_input=example_input, input=[(key, value.shape) for key, value in example_input.items()])
    print(ov_encoder)

    example_input = {
        "input_frame": input_frame,
        "query_feats": query_feats,
        "hires_query_feats": hires_query_feats,
        "causal_state": causal_state,
    }
    ov_predictor = ov.convert_model(
        predictor, example_input=example_input, input=[(key, value.shape) for key, value in example_input.items()]
    )
    print(ov_predictor)

    target_directory = Path(args.output_dir)
    target_directory.mkdir(exist_ok=True)
    ov_encoder_filename = target_directory / f"tapir_encoder_fp16_p{num_points}.xml"
    ov_predictor_filename = target_directory / f"tapir_predictor_fp16_p{num_points}_n{num_iters}_{resolution}.xml"
    ov.save_model(ov_encoder, ov_encoder_filename, compress_to_fp16=True)
    ov.save_model(ov_predictor, ov_predictor_filename, compress_to_fp16=True)
    print(f"**** Models saved: {ov_encoder_filename} and {ov_predictor_filename} ****")
