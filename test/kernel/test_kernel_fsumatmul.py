import torch
import time
from UnarySim.kernel.matmul import FSUMatMul
from UnarySim.stream import RNG, BinGen, BSGen
from UnarySim.metric import ProgError, Stability

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_fsumatmul(): #TODO: clean up this code, clean_add.py code. check paper non-scaled add
    # the error of non-scaled are not accurate, because the actual output will not within [-1,1]

    # ====================== SETUP BEGIN ==================== #
    # matrix multiplication setting
    batch = 8
    in_feature = 32
    out_feature = 64

    # unary bit stream setting
    width = 8
    mode = "bipolar"
    scale = True

    # metric setting
    stability_threshold = 0.2

    # number of terms (different inputs, same weight)
    num_terms = 5

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
    pe_scale = in_feature if scale else 1

    hwcfg_pe = {
        "width" : width,
        "mode" : mode,
        "scale" : pe_scale
    }
    hwcfg_stab_in = {
        "width" : width,
        "mode" : mode,
        "scale" : 1,
        "threshold" : stability_threshold
    }
    hwcfg_stab_out = {
        "width" : width,
        "mode" : mode,
        "scale" : pe_scale,
        "threshold" : stability_threshold
    }

    # Generate one weight matrix
    if mode == "unipolar":
        weight_raw = torch.rand(out_feature, in_feature).mul(length).round().div(length).to(device)
    elif mode == "bipolar":
        weight_raw = torch.rand(out_feature, in_feature).mul(2).sub(1).mul(length).round().div(length).to(device)

    # Collect results across terms
    all_rmse = []
    all_in_cts_min = []
    all_in_cts_max = []
    all_out_cts_min = []
    all_out_cts_max = []
    all_in_stab_mean = []
    all_out_stab_mean = []

    for term in range(num_terms):
        # Generate new input for each term
        if mode == "unipolar":
            input_raw = torch.rand(batch, in_feature).mul(length).round().div(length).to(device)
        elif mode == "bipolar":
            input_raw = torch.rand(batch, in_feature).mul(2).sub(1).mul(length).round().div(length).to(device)

        # DUT — fresh instance per term (internal state resets)
        dut = FSUMatMul(in_feature, out_feature, weight_raw, hwcfg, swcfg).to(device)

        # Expected output
        exp_output = torch.matmul(input_raw, weight_raw.t())

        # Metric trackers
        error_tracker = ProgError(exp_output, hwcfg_pe).to(device)
        stability_in_tracker = Stability(input_raw, hwcfg_stab_in).to(device)
        stability_out_tracker = Stability(exp_output, hwcfg_stab_out).to(device)

        # Input bit stream generator
        input_bin = BinGen(input_raw, hwcfg, swcfg)().to(device)
        input_rng = RNG(hwcfg_rng, swcfg)().to(device)
        input_bsg = BSGen(input_bin, input_rng, swcfg).to(device)

        # ====================== CALCULATION BEGIN ==================== #
        with torch.no_grad():
            idx = torch.zeros(input_raw.size()).type(torch.long).to(device)

            for i in range(length):
                input_bs = input_bsg(idx + i)
                output_bs = dut(input_bs)
                stability_in_tracker.Monitor(input_bs)
                stability_out_tracker.Monitor(output_bs)
                error_tracker.Monitor(output_bs)

            _, pe = error_tracker()
            rmse = torch.sqrt(torch.mean(pe ** 2)).item()
            in_stab = stability_in_tracker()
            out_stab = stability_out_tracker()

            all_rmse.append(rmse)
            all_in_cts_min.append(torch.min(stability_in_tracker.cycle_to_stable).item())
            all_in_cts_max.append(torch.max(stability_in_tracker.cycle_to_stable).item())
            all_out_cts_min.append(torch.min(stability_out_tracker.cycle_to_stable).item())
            all_out_cts_max.append(torch.max(stability_out_tracker.cycle_to_stable).item())
            all_in_stab_mean.append(torch.mean(in_stab).item())
            all_out_stab_mean.append(torch.mean(out_stab).item())

            print("--- term %d ---" % (term + 1))
            print("output error: RMSE:", rmse)
            print("input  cycle to stable: min:", all_in_cts_min[-1], "max:", all_in_cts_max[-1], "stability mean:", all_in_stab_mean[-1])
            print("output cycle to stable: min:", all_out_cts_min[-1], "max:", all_out_cts_max[-1], "stability mean:", all_out_stab_mean[-1])
            print()
        # ====================== CALCULATION END ==================== #

    # ====================== AVERAGE RESULTS ==================== #
    print("===========================")
    print("mode: %s, scaled: %s, batch: %d, in_feature: %d, out_feature: %d, num_terms: %d" % (mode, scale, batch, in_feature, out_feature, num_terms))
    print("avg RMSE: ", sum(all_rmse) / num_terms)
    print(f"avg input cycle to stables [min:max]: [{sum(all_in_cts_min) / num_terms}:{sum(all_in_cts_max) / num_terms}]")
    print(f"avg output cycle to stable [min:max]: [{sum(all_in_cts_min) / num_terms}:{sum(all_in_cts_max) / num_terms}]")
    print("avg input stability: ", sum(all_in_stab_mean) / num_terms)
    print("avg output stability: ", sum(all_out_stab_mean) / num_terms)


if __name__ == '__main__':
    test_fsumatmul()
