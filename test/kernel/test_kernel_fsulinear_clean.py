import torch
from UnarySim.kernel.matmul import FSULinear
from UnarySim.stream import RNG, BinGen, BSGen
from UnarySim.metric.metric import ProgError
import matplotlib.pyplot as plt
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# default in-stream, batch = 1
def test_fsulinear():
    batch = 1
    in_feature = 1024
    out_feature = 512

    num_terms = 10

    width = 8
    mode = "bipolar"
    scale = False

    hwcfg_rng = {
        "width" : width,
        "rng" : "Sobol",
        "dimr" : 1
    }

    swcfg = {
        "btype" : torch.float,
        "rtype" : torch.float,
        "stype" : torch.float
    }

    hwcfg = {
        "width" : width,
        "mode" : mode,
        "scale" : scale,
        "depth" : width + 4,
        "rng" : "Sobol",
        "dimr" : 1
    }

    length = 2 ** width

    # ProgError scale: numeric divisor for reference
    pe_scale_linear = (in_feature + 1) if scale else 1
    pe_scale_matmul = in_feature if scale else 1

    hwcfg_pe_linear = {
        "width" : width,
        "mode" : mode,
        "scale" : pe_scale_linear
    }

    hwcfg_pe_matmul = {
        "width" : width,
        "mode" : mode,
        "scale" : pe_scale_matmul
    }

    # Weight generation
    if mode == "unipolar":
        weight_raw = torch.rand(out_feature, in_feature).mul(length).round().div(length).to(device)
        bias_raw = torch.rand(out_feature).mul(length).round().div(length).to(device)
    elif mode == "bipolar":
        weight_raw = torch.rand(out_feature, in_feature).mul(2).sub(1).mul(length).round().div(length).to(device)
        bias_raw = torch.rand(out_feature).mul(2).sub(1).mul(length).round().div(length).to(device)

    pe_cycle_sum_linear = torch.zeros(length)
    pe_cycle_sum_matmul = torch.zeros(length)

    for term in range(num_terms):

        # Generate new input for each term
        if mode == "unipolar":
            input_raw = torch.rand(batch, in_feature).mul(length).round().div(length).to(device)
        elif mode == "bipolar":
            input_raw = torch.rand(batch, in_feature).mul(2).sub(1).mul(length).round().div(length).to(device)

        matmul_exp_output = torch.matmul(input_raw, weight_raw.t())

        if not scale:
            if mode == "unipolar":
                matmul_exp_output = matmul_exp_output.clamp(0, 1)
            if mode == "bipolar":
                matmul_exp_output = matmul_exp_output.clamp(-1, 1)

        # Linear reference: matmul is already clamped by unary representation,
        # so bias adds on top of the clamped matmul, then clamp again
        linear_exp_output = matmul_exp_output.add(bias_raw)
        if not scale:
            if mode == "unipolar":
                linear_exp_output = linear_exp_output.clamp(0, 1)
            if mode == "bipolar":
                linear_exp_output = linear_exp_output.clamp(-1, 1)

        dut = FSULinear(
            in_features=in_feature,
            out_features=out_feature,
            bias=True,
            weight_ext=weight_raw,
            bias_ext=bias_raw,
            hwcfg=hwcfg,
            swcfg=swcfg
        ).to(device)

        linear_error_tracker = ProgError(linear_exp_output, hwcfg_pe_linear).to(device)
        matmul_error_tracker = ProgError(matmul_exp_output, hwcfg_pe_matmul).to(device)

        input_bin = BinGen(input_raw, hwcfg, swcfg)().to(device)
        input_rng = RNG(hwcfg_rng, swcfg)().to(device)
        input_bsg = BSGen(input_bin, input_rng, swcfg).to(device)

        with torch.no_grad():
            idx = torch.zeros(input_raw.size()).type(torch.long).to(device)

            for i in range(length):
                input_bs = input_bsg(idx + i)
                linear_output_bs = dut(input_bs)

                linear_error_tracker.Monitor(linear_output_bs)
                matmul_error_tracker.Monitor(dut.matmul_out)


                _, linear_pe_i = linear_error_tracker()
                linear_rmse_i = torch.sqrt(torch.mean(linear_pe_i ** 2)).item()
                pe_cycle_sum_linear[i] += linear_rmse_i

                _, matmul_pe_i = matmul_error_tracker()
                matmul_rmse_i = torch.sqrt(torch.mean(matmul_pe_i ** 2)).item()
                pe_cycle_sum_matmul[i] += matmul_rmse_i

    pe_cycle_avg_linear = (pe_cycle_sum_linear / num_terms).cpu().numpy()
    pe_cycle_avg_matmul = (pe_cycle_sum_matmul / num_terms).cpu().numpy()
    acc_cycle_avg_linear = 1 - pe_cycle_avg_linear
    acc_cycle_avg_matmul = 1 - pe_cycle_avg_matmul

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(range(length), pe_cycle_avg_matmul, label="MatMul")
    ax1.plot(range(length), pe_cycle_avg_linear, label="Linear (MatMul + Bias)")
    ax1.set_xlabel("Cycle")
    ax1.set_ylabel("RMSE")
    ax1.set_title("Progressive RMSE (%s, scaled=%s)" % (mode, scale))
    ax1.legend()

    ax2.plot(range(length), acc_cycle_avg_matmul, label="MatMul")
    ax2.plot(range(length), acc_cycle_avg_linear, label="Linear (MatMul + Bias)")
    ax2.set_xlabel("Cycle")
    ax2.set_ylabel("Accuracy (1 - RMSE)")
    ax2.set_title("Progressive Accuracy (%s, scaled=%s)" % (mode, scale))
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_fsulinear()
