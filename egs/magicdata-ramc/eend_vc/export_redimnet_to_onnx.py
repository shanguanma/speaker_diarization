#!/usr/bin/env python3
import torch
import argparse

from utils import  get_logger
logger = get_logger()
def get_args():
    parser = argparse.ArgumentParser(
        description="Export exist checkpoint to onnx file"
    )
    #parser.add_argument(
    #    "--experiment_path", required=True, type=str,
    #    help="Your experiment path, we could download or save something in this path, "
    #    "or you have trained your model using 3D-Speaker"
    #)
    parser.add_argument(
        "--redimnet_hubconfig_file_dir", default="ts_vad2/redimnet/", type=str, help="hubconfig.py path  in offical redimnet repo"
    )
    parser.add_argument(
        "--model_name", default="ReDimNetB2", type=str, help="Model name  in offical redimnet repo"
    ) 
    parser.add_argument(
        "--target_onnx_file", required=True, help="The target onnx file"
    )
    args = parser.parse_args()
    return args


def export_onnx_file(model, target_onnx_file):
    # build dummy input for export
    #       2. The model input shape is (batch_size, sample_point_num).
    dummy_input = torch.randn(1, 16000)
    torch.onnx.export(model,
                      dummy_input,
                      target_onnx_file,
                      export_params=True,
                      opset_version=17, #  Exporting the operator 'aten::stft' to ONNX opset version 11 is not supported. Support for this operator was added in version 17,
                      do_constant_folding=True,
                      input_names=['audio'],
                      output_names=['embedding'],
                      dynamic_axes={'feature': {0: 'batch_size', 1: 'sample_point_num'},
                                    'embedding': {0: 'batch_size'}})
    logger.info(f"Export model onnx to {target_onnx_file} finished")

def build_model(args):
    if args.model_name=="ReDimNetB2":
        model=torch.hub.load(args.redimnet_hubconfig_file_dir, 'ReDimNet',model_name="b2",train_type="ft_lm",dataset='vox2',source="local")
    model.eval()
    return model

if __name__ == "__main__":
   args = get_args()
   model = build_model(args)
   export_onnx_file(model,args.target_onnx_file)

