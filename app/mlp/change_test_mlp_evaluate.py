# train model: MLP_clamp_eval
#

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
    transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST(root=cwd+'/data/mnist', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = torchvision.datasets.MNIST(root=cwd+'/data/mnist', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=4)

    """
    # test binary model clamp
    """

    model_path = cwd + "/saved_model_state_dict" + "_8_clamp"
    model_clamp = MLP3_clamp_eval()
    model_clamp.to(device)
    model_clamp.load_state_dict(torch.load(model_path))
    model_clamp.eval()
    model_clamp.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model_clamp(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size()[0]
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %f %%' % (
        100 * correct / total))


    """
    # test unary model nonscaled addition - clamp binary
    """


    correct_binary = 0
    correct_unary = 0

    bitwidth = 8
    total = 0

    # binary MLP3_clamp weight init
    rng = "Sobol"
    rng_dim = 1
    relu_buf_dep = 4
    mode = "bipolar"
    scaled = False
    bias = True
    sample_cnt = 50000

    start_cnt = 0
    current_index = 0

    cycle_correct = torch.zeros(2**(bitwidth)).to(device)

    # hwcfg/swcfg for stream generation and FSULinear
    hwcfg = {
        "width" : bitwidth,
        "mode" : mode,
        "scale" : None,     # None -> scaled (in_features+bias) in FSULinear; use 1 for non-scaled
        "depth" : 12,
        "rng" : rng,
        "dimr" : rng_dim
    }
    if not scaled:
        hwcfg["scale"] = 1

    swcfg = {
        "btype" : torch.float,
        "rtype" : torch.float,
        "stype" : torch.float
    }

    hwcfg_rng = {
        "width" : bitwidth,
        "rng" : rng,
        "dimr" : rng_dim
    }

    hwcfg_relu = {
        "depth" : relu_buf_dep
    }
    swcfg_relu = {
        "btype" : torch.float,
        "stype" : torch.float
    }

    hwcfg_pe = {
        "scale" : 1,
        "mode" : mode,
        "width" : bitwidth
    }

    start_time = time.time()

    with torch.no_grad():
        for data in testloader:
            if current_index < start_cnt:
                current_index = current_index + 1
                continue
            current_index = current_index + 1

            images, labels = data[0].to(device), data[1].to(device)

            total += labels.size()[0]

            # reference binary mlp
            outputs_binary = model_clamp(images)
            _, predicted_binary = torch.max(outputs_binary.data, 1)
            correct_binary += (predicted_binary == labels).sum().item()

            # unary part
            # input image check
            image = images.view(-1, 32*32)
            image_SRC = BinGen(image, hwcfg, swcfg)().to(device)
            image_RNG = RNG(hwcfg_rng, swcfg)().to(device)
            image_BSG = BSGen(image_SRC, image_RNG, swcfg).to(device)
            image_ERR = ProgError(image, hwcfg_pe).to(device)

            # unary mlp is decomposed into separate layers
            fc1_unary = FSULinear(32*32, 512, bias=bias,
                                    weight_ext=model_clamp.fc1.weight.data,
                                    bias_ext=model_clamp.fc1.bias.data,
                                    hwcfg=hwcfg, swcfg=swcfg).to(device)
            fc1_ERR = ProgError(model_clamp.fc1_out, hwcfg_pe).to(device)

            fc2_unary = FSULinear(512, 512, bias=bias,
                                    weight_ext=model_clamp.fc2.weight.data,
                                    bias_ext=model_clamp.fc2.bias.data,
                                    hwcfg=hwcfg, swcfg=swcfg).to(device)
            fc2_ERR = ProgError(model_clamp.fc2_out, hwcfg_pe).to(device)

            fc3_unary = FSULinear(512, 10, bias=bias,
                                    weight_ext=model_clamp.fc3.weight.data,
                                    bias_ext=model_clamp.fc3.bias.data,
                                    hwcfg=hwcfg, swcfg=swcfg).to(device)
            fc3_ERR = ProgError(model_clamp.fc3_out, hwcfg_pe).to(device)

            relu1_unary = FSUReLU(hwcfg=hwcfg_relu, swcfg=swcfg_relu).to(device)
            relu1_ERR = ProgError(model_clamp.relu1_out, hwcfg_pe).to(device)

            relu2_unary = FSUReLU(hwcfg=hwcfg_relu, swcfg=swcfg_relu).to(device)
            relu2_ERR = ProgError(model_clamp.relu2_out, hwcfg_pe).to(device)

            if total%100 == 0:
                print("--- %s seconds ---" % (time.time() - start_time))
                print(total, "images are done!!!")

            for i in range(2**(bitwidth)):
                idx = torch.zeros(image_SRC.size()).type(torch.long).to(device)
                image_bs = image_BSG(idx + i)
                image_ERR.Monitor(image_bs)
                # fc1
                fc1_unary_out   = fc1_unary(image_bs)
                # relu1
                relu1_unary_out = relu1_unary(fc1_unary_out)
                # fc2
                fc2_unary_out   = fc2_unary(relu1_unary_out)
                # relu2
                relu2_unary_out = relu2_unary(fc2_unary_out)
                # fc3
                fc3_unary_out   = fc3_unary(relu2_unary_out)
                fc3_ERR.Monitor(fc3_unary_out)

                _, predicted_unary = torch.max(fc3_ERR()[0], 1)
                if predicted_unary == labels:
                    cycle_correct[i].add_(1)

            _, predicted_unary = torch.max(fc3_ERR()[0], 1) # highest confidence score among dim=1
            correct_unary += (predicted_unary == labels).sum().item()
            if total == sample_cnt:
                break

    print('Accuracy of the network on %d test images: %f %%' % (total,
        100 * correct_binary / total))
    print('Accuracy of the network on %d test images: %f %%' % (total,
        100 * correct_unary / total))

    result = cycle_correct.cpu().numpy()/total
    fig = plt.plot([i for i in range(2**bitwidth)], result)
    plt.title("Cycle level accuracy")
    plt.show()

    with open("cycle_accuracy_mlp_nonscaled_clamp.csv", "w+") as f:
        for i in result:
            f.write(str(i)+", \n")

