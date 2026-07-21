import torch
from SCVCNet import SCVCNet
from SCVCNet import SCVC, RandomConv1d

############################################
# 1 统计模型参数量
############################################

def count_scvc_params(scvc):

    visited = set()
    total = 0

    for module in scvc.modules():

        for name, val in vars(module).items():

            if isinstance(val, torch.Tensor):

                if id(val) not in visited:
                    total += val.numel()
                    visited.add(id(val))

    return total


def count_total_params(model):

    scvc_params = count_scvc_params(model.scvc)

    beta_params = model.out_channels * model.outputs

    return scvc_params + beta_params


############################################
# 2 计算 RandomConv1d FLOPs
############################################

def conv1d_flops(in_channels, out_channels, kernel_size, output_len):

    # MAC = Cin * K
    mac = in_channels * kernel_size

    # per output point
    flops = mac * 2

    total = out_channels * output_len * flops

    return total


############################################
# 3 计算 SCVC FLOPs
############################################

def compute_scvc_flops(scvc, x_shape, y_shape):

    B, Cx, Tx = x_shape
    _, Cy, Ty = y_shape

    flops = 0

    # cross product
    cross = Cx * Tx * Cy * Ty
    flops += cross * 2

    # kernel_C projection
    flops += scvc.out_channels * Cx * Cy * Tx * Ty * 2

    # conv part
    if hasattr(scvc, "conv_A"):

        for conv in scvc.conv_A:

            out_len = Tx

            flops += conv1d_flops(
                conv.in_channels,
                conv.out_channels,
                conv.kernel_size,
                out_len
            )

    if hasattr(scvc, "conv_B"):

        for conv in scvc.conv_B:

            out_len = Ty

            flops += conv1d_flops(
                conv.in_channels,
                conv.out_channels,
                conv.kernel_size,
                out_len
            )

    return flops


############################################
# 4 计算整个模型 FLOPs
############################################

def compute_model_flops(model, x1_shape, x2_shape):

    scvc_flops = compute_scvc_flops(
        model.scvc,
        x1_shape,
        x2_shape
    )

    # activation
    B = x1_shape[0]
    H = model.out_channels

    activation = B * H

    # final linear
    fc = model.out_channels * model.outputs * 2

    total = scvc_flops + activation + fc

    return total


############################################
# 5 测试
############################################

if __name__ == "__main__":

    model = SCVCNet(
        in1_channels=16,
        in2_channels=16,
        out_channels=800,
        outputs=2,
        kernel_size=3,
        function="sigmoid",
        reduce_dim="glbavg"
    )

    params = count_total_params(model)

    flops = compute_model_flops(
        model,
        (1,16,14),
        (1,16,14)
    )

    print("Params:", params)
    print("Params(M):", params/1e6)

    print("FLOPs:", flops)
    print("FLOPs(M):", flops/1e6)