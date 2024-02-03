import torch, onnx
from mymodel import MyModel

if __name__ == '__main__':
    model = MyModel()

    # Input to the model
    torch_input = torch.randn(1, 1, 32, 32, requires_grad=True)
    torch_out = model(torch_input)

    # Export the model
    onnx_program = torch.onnx.dynamo_export(model,               # model being run
                             torch_input,         # model input (or a tuple for multiple inputs)
                             )
    onnx_program.save("./data/my_image_classifier.onnx")

    # Verify the export
    onnx_model = onnx.load("./data/my_image_classifier.onnx")
    # モデルの構造をチェックし、定められたスキームに準拠しているかを確かめる
    # ONNXグラフの妥当性は、モデルのバージョン、グラフの構造、ノードとその入出力を確認して検証される
    onnx.checker.check_model(onnx_model)
    print('Exported model has been checked!')