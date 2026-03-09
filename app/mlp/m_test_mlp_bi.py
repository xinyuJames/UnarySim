# Unary MLP

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import os

from UnarySim.kernel.utils import *
from UnarySim.kernel.linear import FSULinear
from UnarySim.kernel.relu import FSUReLU
from UnarySim.stream.gen import RNG, BinGen, BSGen
from UnarySim.metric.metric import ProgError
from model import MLP3_clamp_eval

if __name__ == '__main__':
    cwd = os.getcwd()
    print(cwd)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # MNIST data loader
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    testset = torchvision.datasets.MNIST(root=cwd+'/data/mnist', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=4)

    # load binary model
    model_path = cwd + "/saved_model_state_dict" + "_8_clamp"
    model_clamp = MLP3_clamp_eval()
    model_clamp.to(device)
    model_clamp.load_state_dict(torch.load(model_path))
    model_clamp.eval()

    # binary accuracy check
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model_clamp(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size()[0]
            correct += (predicted == labels).sum().item()
    print('Binary accuracy on 10000 test images: %f %%' % (100 * correct / total))

    # ========== Configuration ==========
    bitwidth = 8
    mode = "bipolar"
    scaled = False
    bias = True
    rng = "Sobol"
    rng_dim = 1
    relu_buf_dep = 4
    sample_cnt = 1500
    length = 2 ** bitwidth

    # layer sizes
    fc1_in, fc1_out = 32 * 32, 512
    fc2_in, fc2_out = 512, 512
    fc3_in, fc3_out = 512, 10

    # per-layer scale: in_features + (1 if bias) for scaled, 1 for non-scaled
    if scaled:
        fc1_scale = fc1_in + (1 if bias else 0)
        fc2_scale = fc2_in + (1 if bias else 0)
        fc3_scale = fc3_in + (1 if bias else 0)
    else:
        fc1_scale = 1
        fc2_scale = 1
        fc3_scale = 1

    # hwcfg for FSULinear (scale=None lets FSULinear auto-compute for scaled)
    hwcfg = {
        "width" : bitwidth,
        "mode" : mode,
        "scale" : scaled,
        "depth" : 12,
        "rng" : rng,
        "dimr" : rng_dim
    }

    swcfg = {
        "btype" : torch.float,
        "rtype" : torch.float,
        "stype" : torch.float
    }

    hwcfg_rng = {"width" : bitwidth, "rng" : rng, "dimr" : rng_dim}

    hwcfg_relu = {"depth" : relu_buf_dep}
    swcfg_relu = {"btype" : torch.float, "stype" : torch.float}

    # per-layer ProgError configs
    hwcfg_pe_input = {"scale" : 1, "mode" : mode, "width" : bitwidth}
    hwcfg_pe_fc1   = {"scale" : fc1_scale, "mode" : mode, "width" : bitwidth}
    hwcfg_pe_relu1 = {"scale" : fc1_scale, "mode" : mode, "width" : bitwidth}
    hwcfg_pe_fc2   = {"scale" : fc2_scale, "mode" : mode, "width" : bitwidth}
    hwcfg_pe_relu2 = {"scale" : fc2_scale, "mode" : mode, "width" : bitwidth}
    hwcfg_pe_fc3   = {"scale" : fc3_scale, "mode" : mode, "width" : bitwidth}

    correct_binary = 0
    correct_unary = 0
    total = 0
    start_cnt = 0
    current_index = 0
    cycle_correct = torch.zeros(length).to(device)
    # per-layer RMSE (averaged over images)
    input_cycle_rmse = torch.zeros(length).to(device)
    fc1_cycle_rmse = torch.zeros(length).to(device)
    fc2_cycle_rmse = torch.zeros(length).to(device)
    fc3_cycle_rmse = torch.zeros(length).to(device)

    start_time = time.time()

    with torch.no_grad():
        for data in testloader:
            if current_index < start_cnt:
                current_index += 1
                continue
            current_index += 1

            images, labels = data[0].to(device), data[1].to(device)
            total += labels.size()[0]

            # reference binary mlp, golden model
            outputs_binary = model_clamp(images)
            _, predicted_binary = torch.max(outputs_binary.data, 1)
            correct_binary += (predicted_binary == labels).sum().item()

            # unary bit stream generator
            image = images.view(-1, 32 * 32)
            image_SRC = BinGen(image, hwcfg, swcfg)().to(device)
            image_RNG = RNG(hwcfg_rng, swcfg)().to(device)
            image_BSG = BSGen(image_SRC, image_RNG, swcfg).to(device)
            image_ERR = ProgError(image, hwcfg_pe_input).to(device)

            # unary mlp layers
            fc1_unary = FSULinear(fc1_in, fc1_out, bias=bias,
                                  weight_ext=model_clamp.fc1.weight.data,
                                  bias_ext=model_clamp.fc1.bias.data,
                                  hwcfg=hwcfg, swcfg=swcfg).to(device)
            fc1_ERR = ProgError(model_clamp.fc1_out, hwcfg_pe_fc1).to(device)

            relu1_unary = FSUReLU(hwcfg=hwcfg_relu, swcfg=swcfg_relu).to(device)
            relu1_ERR = ProgError(model_clamp.relu1_out, hwcfg_pe_relu1).to(device)

            fc2_unary = FSULinear(fc2_in, fc2_out, bias=bias,
                                  weight_ext=model_clamp.fc2.weight.data,
                                  bias_ext=model_clamp.fc2.bias.data,
                                  hwcfg=hwcfg, swcfg=swcfg).to(device)
            fc2_ERR = ProgError(model_clamp.fc2_out, hwcfg_pe_fc2).to(device)

            relu2_unary = FSUReLU(hwcfg=hwcfg_relu, swcfg=swcfg_relu).to(device)
            relu2_ERR = ProgError(model_clamp.relu2_out, hwcfg_pe_relu2).to(device)

            fc3_unary = FSULinear(fc3_in, fc3_out, bias=bias,
                                  weight_ext=model_clamp.fc3.weight.data,
                                  bias_ext=model_clamp.fc3.bias.data,
                                  hwcfg=hwcfg, swcfg=swcfg).to(device)
            fc3_ERR = ProgError(model_clamp.fc3_out, hwcfg_pe_fc3).to(device)

            if total % 100 == 0:
                print("--- %s seconds ---" % (time.time() - start_time))
                print(total, "images are done!!!")

            for i in range(length):
                idx = torch.zeros(image_SRC.size()).type(torch.long).to(device)
                image_bs = image_BSG(idx + i)
                image_ERR.Monitor(image_bs)

                # unary mlp calculation
                fc1_unary_out = fc1_unary(image_bs)
                fc1_ERR.Monitor(fc1_unary_out)
                relu1_unary_out = relu1_unary(fc1_unary_out)
                fc2_unary_out = fc2_unary(relu1_unary_out)
                fc2_ERR.Monitor(fc2_unary_out)
                relu2_unary_out = relu2_unary(fc2_unary_out)
                fc3_unary_out = fc3_unary(relu2_unary_out)
                fc3_ERR.Monitor(fc3_unary_out)

                # per-layer RMSE
                input_cycle_rmse[i] += torch.sqrt(torch.mean(image_ERR()[1] ** 2))
                fc1_cycle_rmse[i] += torch.sqrt(torch.mean(fc1_ERR()[1] ** 2))
                fc2_cycle_rmse[i] += torch.sqrt(torch.mean(fc2_ERR()[1] ** 2))
                fc3_cycle_rmse[i] += torch.sqrt(torch.mean(fc3_ERR()[1] ** 2))

                _, predicted_unary = torch.max(fc3_ERR()[0], 1)
                if predicted_unary == labels:
                    cycle_correct[i].add_(1)

            _, predicted_unary = torch.max(fc3_ERR()[0], 1)
            correct_unary += (predicted_unary == labels).sum().item()
            if total == sample_cnt:
                break

    # ========== Results ==========
    scale_str = "scaled" if scaled else "nonscaled"
    print('Binary accuracy on %d test images: %f %%' % (total, 100 * correct_binary / total))
    print('Unary (%s %s) accuracy on %d test images: %f %%' % (mode, scale_str, total, 100 * correct_unary / total))

    result = cycle_correct.cpu().numpy() / total
    fig = plt.plot([i for i in range(length)], result)
    plt.title("Cycle level accuracy (%s, %s)" % (mode, scale_str))
    plt.show()

    with open("cycle_accuracy_mlp_%s_%s.csv" % (mode, scale_str), "w+") as f:
        for i in result:
            f.write(str(i) + ", \n")

    # per-layer RMSE plot
    cycles = list(range(length))
    plt.figure()
    plt.plot(cycles, (input_cycle_rmse / total).cpu().numpy(), label="input")
    plt.plot(cycles, (fc1_cycle_rmse / total).cpu().numpy(), label="fc1")
    plt.plot(cycles, (fc2_cycle_rmse / total).cpu().numpy(), label="fc2")
    plt.plot(cycles, (fc3_cycle_rmse / total).cpu().numpy(), label="fc3")
    plt.xlabel("Cycle")
    plt.ylabel("RMSE")
    plt.title("Per-layer RMSE (%s, %s)" % (mode, scale_str))
    plt.legend()
    plt.show()

