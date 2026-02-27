import torch
import time
from UnarySim.kernel.matmul import FSUMatMul
from UnarySim.stream import RNG, BinGen, BSGen
from UnarySim.metric import ProgError

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_fsumatmul(): #TODO: clean up this code, clean_add.py code. check paper non-scaled add
    # the error of non-scaled are not accurate, because the actual output will not within [-1,1]

    # ====================== SETUP BEGIN ==================== #
    width = 8
    batch = 8
    in_feature = 16
    out_feature = 512

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
    # ====================== SETUP END ==================== #

    length = 2 ** width
    # Data Generate — snap to grid for fair comparison
    if mode == "unipolar":
        weight_raw = torch.rand(out_feature, in_feature).mul(length).round().div(length).to(device)
        input_raw = torch.rand(batch, in_feature).mul(length).round().div(length).to(device)
    elif mode == "bipolar":
        weight_raw = torch.rand(out_feature, in_feature).mul(2).sub(1).mul(length).round().div(length).to(device)
        input_raw = torch.rand(batch, in_feature).mul(2).sub(1).mul(length).round().div(length).to(device)

    # DUT
    dut = FSUMatMul(in_feature, out_feature, weight_raw, hwcfg, swcfg).to(device)

    # Expected output
    exp_output = torch.matmul(input_raw, weight_raw.t())

    # ProgError needs a numeric scale:
    # scaled=True  -> scale = in_feature
    # scaled=False -> scale = 1
    hwcfg_pe = {
        "width" : width,
        "mode" : mode,
        "scale" : in_feature if scale else 1
    }
    error_tracker = ProgError(exp_output, hwcfg_pe).to(device)

    # Input bit stream generator
    input_bin = BinGen(input_raw, hwcfg, swcfg)().to(device)
    input_rng = RNG(hwcfg_rng, swcfg)().to(device)
    input_bsg = BSGen(input_bin, input_rng, swcfg).to(device)

    with torch.no_grad():
        idx = torch.zeros(input_raw.size()).type(torch.long).to(device)

        for i in range(length):
            input_bs = input_bsg(idx + i)
            output_bs = dut(input_bs)
            error_tracker.Monitor(output_bs)

        _, pe = error_tracker()
        rmse = torch.sqrt(torch.mean(pe ** 2))
        print("----------------------------")
        print("mode: %s, scaled: %s, in_feature: %d, out_feature: %d" % (mode, scale, in_feature, out_feature))
        print("output error: min:", torch.min(pe).item(), "max:", torch.max(pe).item(), "RMSE:", rmse.item())
        print()
        error_tracker.plot()
            

if __name__ == '__main__':
    test_fsumatmul()
