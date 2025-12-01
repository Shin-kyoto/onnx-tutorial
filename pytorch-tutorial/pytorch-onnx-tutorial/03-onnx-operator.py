import torch
import onnxruntime
import onnxscript
from onnxscript import opset18  # opset 18 is the latest (and only) supported version for now

class Model(torch.nn.Module):
    def forward(self, input_x, input_y):
        return torch.ops.aten.add(input_x, input_y)  # generates a aten::add.Tensor node

input_add_x = torch.randn(3, 4)
input_add_y = torch.randn(3, 4)
aten_add_model = Model()


# Now we create a ONNX Script function that implements ``aten::add.Tensor``.
# The function name (e.g. ``custom_aten_add``) is displayed in the ONNX graph, so we recommend to use intuitive names.
custom_aten = onnxscript.values.Opset(domain="custom.aten", version=1)

# NOTE: The function signature must match the signature of the unsupported ATen operator.
# https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
# NOTE: All attributes must be annotated with type hints.
@onnxscript.script(custom_aten)
def custom_aten_add(input_x, input_y, alpha: float = 1.0):
    alpha = opset18.CastLike(alpha, input_y)
    input_y = opset18.Mul(input_y, alpha)
    return opset18.Add(input_x, input_y)


# Now we have everything we need to support unsupported ATen operators.
# Let's register the ``custom_aten_add`` function to ONNX registry, and export the model to ONNX again.
onnx_registry = torch.onnx.OnnxRegistry()
onnx_registry.register_op(
    namespace="aten", op_name="add", overload="Tensor", function=custom_aten_add
    )
print(f"aten::add.Tensor is supported by ONNX registry: \
      {onnx_registry.is_registered_op(namespace='aten', op_name='add', overload='Tensor')}"
      )
export_options = torch.onnx.ExportOptions(onnx_registry=onnx_registry)
onnx_program = torch.onnx.dynamo_export(
    aten_add_model, input_add_x, input_add_y, export_options=export_options
    )

# graph node domain is the custom domain we registered
assert onnx_program.model_proto.graph.node[0].domain == "custom.aten"
assert len(onnx_program.model_proto.graph.node) == 1
# graph node name is the function name
assert onnx_program.model_proto.graph.node[0].op_type == "custom_aten_add"
# function node domain is empty because we use standard ONNX operators
assert onnx_program.model_proto.functions[0].node[3].domain == ""
# function node name is the standard ONNX operator name
assert onnx_program.model_proto.functions[0].node[3].op_type == "Add"
print("Finish exporting the model to ONNX with custom ATen operator")

# Torchと比較してみよう
# Use ONNX Runtime to run the model, and compare the results with PyTorch
onnx_program.save("./data/custom_add_model.onnx")
ort_session = onnxruntime.InferenceSession(
    "./data/custom_add_model.onnx", providers=['CPUExecutionProvider']
    )

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

onnx_input = onnx_program.adapt_torch_inputs_to_onnx(input_add_x, input_add_y)
onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}
onnxruntime_outputs = ort_session.run(None, onnxruntime_input)

torch_outputs = aten_add_model(input_add_x, input_add_y)
torch_outputs = onnx_program.adapt_torch_outputs_to_onnx(torch_outputs)

assert len(torch_outputs) == len(onnxruntime_outputs)
for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
    torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))