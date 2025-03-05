import onnx
try:
    #model_path="/data/maduo/model_hub/speaker_pretrain_model/en/wespeaker/resnet34-lm/pytorch_model.bin"
    #model_path="/data/maduo/model_hub/speaker_pretrain_model/en/wespeaker/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin"
    model_path="/data/maduo/model_hub/speaker_pretrain_model/en/wespeaker/voxceleb_resnet34_LM.onnx"
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print("ONNX model is valid!")
except onnx.onnx_cpp2py_export.checker.ValidationError as e:
    print(f"ONNX model validation failed: {e}")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
