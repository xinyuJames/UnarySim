import torch
from UnarySim.kernel import FSULinear
from UnarySim.stream import RNG, BinGen, BSGen
from UnarySim.metric import ProgError
import matplotlib.pyplot as plt
import time
import torch.autograd.profiler as profiler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# default in-stream, batch = 1
def test_fsulinear():
    plot_en=False
    cfg_width = 4
    hwcfg_input={
        "width" : cfg_width,
        "rng" : "Sobol",
        "dimr" : 1
    }
    hwcfg={
        "width" : cfg_width,
        "mode" : "bipolar",
        "scale" : None,
        "depth" : 20,
        "rng" : "Sobol",
        "dimr" : 1
    }
    swcfg={
        "btype" : torch.float, 
        "rtype" : torch.float, 
        "stype" : torch.float
    }

    rng = hwcfg["rng"]
    in_feature = 256
    out_feature = 1000
    bias = True
    modes = ["bipolar", "unipolar"]
    scaled = [True, False]
    result_pe = []
    
    for mode in modes:
        for scale in scaled:
            hwcfg["mode"] = mode
            hwcfg_input["mode"] = mode
            hwcfg["scale"] = (in_feature + bias) if scale else 1
            length = 2**hwcfg["width"] # length of unary bit stream, width is in binary
            length_input = 2**hwcfg_input["width"]

            result_pe_cycle = []

            # torch Linear instance
            fc = torch.nn.Linear(in_feature, out_feature, bias=bias).to(device)
            
            # initialize Torch Linear weight+data
            if mode == "unipolar":
                # quantization, snap to closest grid
                fc.weight.data = torch.rand(out_feature, in_feature).mul(length).round().div(length).to(device)
                if bias is True:
                    fc.bias.data = torch.rand(out_feature).mul(length).round().div(length).to(device)
            elif mode == "bipolar":
                fc.weight.data = torch.rand(out_feature, in_feature).mul(2).sub(1).mul(length).round().div(length).to(device)
                if bias is True:
                    fc.bias.data = torch.rand(out_feature).mul(2).sub(1).mul(length).round().div(length).to(device)

            # UGEMM Linear instance
            ufc = FSULinear(in_feature, out_feature, bias=bias, weight_ext=fc.weight, bias_ext=fc.bias, 
                              hwcfg=hwcfg, swcfg=swcfg).to(device)

            # quantization, snap to closest grid
            iVec = ((torch.rand(1, in_feature)*length_input).round()/length_input).to(device)
            oVec = fc(iVec)

            iVecSource = BinGen(iVec, hwcfg, swcfg)().to(device)
            iVecRNG = RNG(hwcfg_input, swcfg)().to(device)
            iVecBS = BSGen(iVecSource, iVecRNG, swcfg).to(device)

            hwcfg["scale"] = 1
            iVecPE = ProgError(iVec, hwcfg).to(device)

            hwcfg["scale"] = (in_feature + bias) if scale else 1
            oVecPE = ProgError(oVec, hwcfg).to(device)
            
            with torch.no_grad():
                idx = torch.zeros(iVecSource.size()).type(torch.long).to(device)
                start_time = time.time()
                for i in range(max(length, length_input)):
                    iBS = iVecBS(idx + i)
                    iVecPE.Monitor(iBS)

                    oVecU = ufc(iBS)
                    oVecPE.Monitor(oVecU)
                    rmse = torch.sqrt(torch.mean(torch.mul(oVecPE()[1], oVecPE()[1])))
                    if plot_en is True:
                        result_pe_cycle.append(1-rmse.item())
                print("--- %s seconds ---" % (time.time() - start_time))
                print("RNG: "+rng+", data: "+mode+", scaled: "+str(scale))
                print("input error:  ", "min: ", torch.min(iVecPE()[1]).item(), "max: ", torch.max(iVecPE()[1]).item())
                print("output error: ", "min: ", torch.min(oVecPE()[1]).item(), "max: ", torch.max(oVecPE()[1]).item(), "RMSE: ", rmse.item())
                print()
                oVecPE.plot()
                if plot_en is True:
                    result_pe = oVecPE()[1].cpu().numpy()
                    print("error distribution=========>")
                    plt.figure(figsize=(3,1.5))
                    fig = plt.hist(result_pe.flatten(), bins='auto')  # arguments are passed to np.histogram
                    plt.show()
                    print("progressive accuracy=========>")
                    plt.figure(figsize=(3,1.5))
                    fig = plt.plot(result_pe_cycle)  # arguments are passed to np.histogram
                    plt.show()


if __name__ == '__main__':
    test_fsulinear()

