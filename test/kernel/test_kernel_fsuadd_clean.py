import torch
import time
import matplotlib.pyplot as plt
from UnarySim.kernel.clean_add import FSUAdd
from UnarySim.stream import RNG, BinGen, BSGen
from UnarySim.metric import ProgError, Stability

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_fsuadd():

    # ====================== SETUP BEGIN ==================== #
    # addition setting
    entry = 1024
    out_size = [1, 512]
    acc_dim = 0

    # unary bit stream setting
    width = 8
    mode = "bipolar"
    scale = False

    # metric setting
    stability_threshold = 0.2

    # number of terms (different inputs)
    num_terms = 10

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

    hwcfg_add = {
        "mode" : mode,
        "scale" : scale,
        "depth" : width + 4,
        "dima" : acc_dim,
        "entry" : entry
    }
    # ====================== SETUP END ==================== #

    length = 2 ** width
    pe_scale = entry if scale else 1

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

    input_size = [entry] + out_size

    # Collect results across terms
    all_rmse = []
    all_in_cts_min = []
    all_in_cts_max = []
    all_out_cts_min = []
    all_out_cts_max = []
    all_in_stab_mean = []
    all_out_stab_mean = []

    pe_cycle_sum = torch.zeros(length)

    for term in range(num_terms):
        # Generate new input for each term
        if mode == "unipolar":
            input_raw = torch.rand(input_size).mul(length).round().div(length).to(device)
        elif mode == "bipolar":
            input_raw = torch.rand(input_size).mul(2).sub(1).mul(length).round().div(length).to(device)

        # DUT — fresh instance per term (internal state resets)
        dut = FSUAdd(hwcfg_add, swcfg).to(device)

        # Expected output
        exp_output = torch.sum(input_raw, acc_dim)

        if not scale:
            if mode == "unipolar":
                exp_output = exp_output.clamp(0, 1)
            if mode == "bipolar":
                exp_output = exp_output.clamp(-1, 1)

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

                _, pe_i = error_tracker()
                rmse_i = torch.sqrt(torch.mean(pe_i ** 2)).item()
                pe_cycle_sum[i] += rmse_i

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
    print("mode: %s, scaled: %s, entry: %d, num_terms: %d" % (mode, scale, entry, num_terms))
    print("avg RMSE: ", sum(all_rmse) / num_terms)
    print(f"avg input cycle to stable [min:max]: [{sum(all_in_cts_min) / num_terms}:{sum(all_in_cts_max) / num_terms}]")
    print(f"avg output cycle to stable [min:max]: [{sum(all_out_cts_min) / num_terms}:{sum(all_out_cts_max) / num_terms}]")
    print("avg input stability: ", sum(all_in_stab_mean) / num_terms)
    print("avg output stability: ", sum(all_out_stab_mean) / num_terms)

    # plot RMSE per cycle
    pe_cycle_avg = (pe_cycle_sum / num_terms).cpu().numpy()
    plt.figure()
    plt.plot(range(length), pe_cycle_avg)
    plt.xlabel("Cycle")
    plt.ylabel("RMSE")
    plt.title("Progressive RMSE (%s, scaled=%s, entry=%d)" % (mode, scale, entry))
    plt.show()


if __name__ == '__main__':
    test_fsuadd()

