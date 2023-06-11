import os
import argparse
import torch
# import onnxruntime
import hit_net_sf
# import numpy as np
import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = hit_net_sf.HITNet_SF()

    def forward(self, left, right):
        left = left * 2 - 1
        right = right * 2 - 1
        return self.model(left, right)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt")
    parser.add_argument("height", type=int)
    parser.add_argument("width", type=int)
    args = parser.parse_args()

    H = args.height
    W = args.width

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model = Model(**vars(args)).eval()
    model.load_state_dict(torch.load(args.ckpt)["state_dict"])

    X = torch.randn(1, 3, H, W)
    # torch_out = model(X, X)
    onnx_filename = os.path.join(output_dir, "tinyhitnet_{}x{}.onnx".format(H, W))

    torch.onnx.export(
        model,
        (X, X),
        onnx_filename,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["left", "right"],
        output_names=["disp"]
    )

    # ort_session = onnxruntime.InferenceSession(
    #     onnx_filename,  providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    # ort_inputs = {
    #     ort_session.get_inputs()[0].name: X.cpu().numpy(),
    #     ort_session.get_inputs()[1].name: X.cpu().numpy()
    # }
    # ort_outs = ort_session.run(None, ort_inputs)

    # np.testing.assert_allclose(torch_out.detach().cpu().numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)
    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")
