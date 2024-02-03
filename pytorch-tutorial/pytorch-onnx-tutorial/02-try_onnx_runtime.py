import onnxruntime
import torch, onnx
from mymodel import MyModel

seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def setup():
    # Input amd ONNX model
    torch_input = torch.randn(1, 1, 32, 32, requires_grad=True)
    torch_model = MyModel()
    onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
    onnx_program.save("./data/my_image_classifier.onnx")
    return onnx_program, torch_model, torch_input

if __name__ == '__main__':
    onnx_program, torch_model, torch_input = setup()
    onnx_input = onnx_program.adapt_torch_inputs_to_onnx(torch_input)
    print(f"Input length: {len(onnx_input)}")
    print(f"Sample input: {onnx_input}")

    ort_session = onnxruntime.InferenceSession("./data/my_image_classifier.onnx", providers=['CPUExecutionProvider'])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    for k, v in zip(ort_session.get_inputs(), onnx_input):
        print(f"ONNX input name: {k.name}, shape: {v.shape}")
    
    onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}

    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
    print("Finish running ONNX model with ONNX Runtime")

    # 2つのmodelの出力を比較
    torch_outputs = torch_model(torch_input)
    torch_outputs = onnx_program.adapt_torch_outputs_to_onnx(torch_outputs)
    
    assert len(torch_outputs) == len(onnxruntime_outputs)
    for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
        torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

    print("PyTorch and ONNX Runtime output matched!")
    print(f"Output length: {len(onnxruntime_outputs)}")
    print(f"Sample output: {onnxruntime_outputs}")