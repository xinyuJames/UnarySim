import torch
from torch.amp import autocast

from UnarySim.kernel.clean_add import FSUAdd
from UnarySim.kernel import FSUMul
from UnarySim.stream import RNG, BinGen, BSGen

class FSUMatMul(torch.nn.Module):
    """
    This module uses uMul (static mode) and uAdd to achieve uGEMM.
    It computes (batch, in_features) x (out_features, in_features)^T -> (batch, out_features)
    using element-wise FSUMul for products and FSUAdd for accumulation.
    No bias is supported in this implementation.
    """

    def __init__(
            self,
            in_features,
            out_features,
            weight_ext=None,

            hwcfg={
                "width" : 8,
                "mode" : "bipolar",
                "scale" : True,
                "depth" : 12,
                "rng" : "Sobol",
                "dimr" : 1
            },
            swcfg={
                "btype" : torch.float,
                "rtype" : torch.float,
                "stype" : torch.float
            }
    ):

        super(FSUMatMul, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Hardware config
        self.hwcfg = {}
        self.hwcfg["width"] = hwcfg["width"]
        self.hwcfg["mode"] = hwcfg["mode"].lower()
        self.hwcfg["scale"] = hwcfg["scale"]
        self.hwcfg["depth"] = hwcfg["depth"]
        self.hwcfg["rng"] = hwcfg["rng"].lower()
        self.hwcfg["dimr"] = hwcfg["dimr"]

        # Software config
        self.swcfg = {}
        self.swcfg["btype"] = swcfg["btype"]
        self.swcfg["rtype"] = swcfg["rtype"]
        self.swcfg["stype"] = swcfg["stype"]

        if (hwcfg["mode"].lower() == "bipolar") and (not hwcfg["scale"]):
            assert self.hwcfg["rng"].lower() not in ["race", "tc", "race10", "tc10"], \
                "Error: the hw config 'rng' in " + str(self) + " class should avoid ['race', 'tc', 'race10', 'tc10'] for bipolar data with non-scaled addition."

        assert self.swcfg["btype"] == torch.float, "Error: the sw config 'btype' in " + str(self) + " class requires 'torch.float'."
        assert self.swcfg["rtype"] == torch.float, "Error: the sw config 'rtype' in " + str(self) + " class requires 'torch.float'."
        assert self.swcfg["stype"] == torch.float, "Error: the sw config 'stype' in " + str(self) + " class requires 'torch.float'."

        # Multiplier, W as [out_features, in_features]
        assert weight_ext is not None,  "Error: weight_ext is required in " + str(self) + " class."
        assert (weight_ext.size()[0], weight_ext.size()[1]) == (out_features, in_features), "Error: weight_ext shape must be (out_features, in_features) in " + str(self) + " class."

        hwcfg_mul = {
            "width" : self.hwcfg["width"],
            "mode" : self.hwcfg["mode"],
            "static" : True,
            "rng" : self.hwcfg["rng"],
            "dimr" : self.hwcfg["dimr"]
        }
        self.MUL = FSUMul(in_1_prob=weight_ext, hwcfg=hwcfg_mul, swcfg=swcfg)

        # Accumulator, in_feature parallel sum
        hwcfg_acc = {
            "mode" : self.hwcfg["mode"],
            "scale" : hwcfg["scale"],       # True/False passed directly to clean FSUAdd
            "depth" : self.hwcfg["depth"],
            "dima" : 2,
            "entry" : in_features
        }
        self.ACC = FSUAdd(hwcfg_acc, self.swcfg)

    @autocast('cuda')
    def forward(self, input):
        # [batch, in_feature] bit stream expected
        batch = input.size()[0]

        # Expand input to [batch, out_features, in_features]
        input_exp = input.unsqueeze(1).expand(batch, self.out_features, self.in_features)

        # static weight: [out_feature, in_feature] ; input: [batch, out_features, in_features] 
        self.products = self.MUL(input_exp)

        # Accumulate, dim=2, we have [batch, out_feature]
        output = self.ACC(self.products)

        return output


class FSULinear(torch.nn.Module):
    """
    Fully connected layer using FSUMatMul (uMul + uAdd) with optional bias.
    Step 1: matmul  — FSUMatMul computes input × weight^T
    Step 2: bias add — FSUMul generates bias bitstream, FSUAdd combines with matmul output
    """

    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            weight_ext=None,
            bias_ext=None,
            hwcfg={
                "width" : 8,
                "mode" : "bipolar",
                "scale" : True,
                "depth" : 12,
                "rng" : "Sobol",
                "dimr" : 1
            },
            swcfg={
                "btype" : torch.float,
                "rtype" : torch.float,
                "stype" : torch.float
            }
    ):
        super(FSULinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias and (bias_ext is not None)

        self.hwcfg = {}
        self.hwcfg["width"] = hwcfg["width"]
        self.hwcfg["mode"] = hwcfg["mode"].lower()
        self.hwcfg["scale"] = hwcfg["scale"]
        self.hwcfg["depth"] = hwcfg["depth"]
        self.hwcfg["rng"] = hwcfg["rng"].lower()
        self.hwcfg["dimr"] = hwcfg["dimr"]

        self.swcfg = {}
        self.swcfg["btype"] = swcfg["btype"]
        self.swcfg["rtype"] = swcfg["rtype"]
        self.swcfg["stype"] = swcfg["stype"]

        hwcfg_matmul = {
            "width" : self.hwcfg["width"],
            "mode" : self.hwcfg["mode"],
            "scale" : self.hwcfg["scale"],
            "depth" : self.hwcfg["depth"],
            "rng" : self.hwcfg["rng"],
            "dimr" : self.hwcfg["dimr"]
        }

        self.MatMul = FSUMatMul(
            self.in_features,
            self.out_features,
            weight_ext,
            hwcfg_matmul,
            swcfg
        )

        if self.has_bias:
            hwcfg_brng = {
                "width" : self.hwcfg["width"],
                "rng" : self.hwcfg["rng"],
                "dimr" : self.hwcfg["dimr"]
            }
            self.bias_bin = BinGen(bias_ext, self.hwcfg, swcfg)()
            self.bias_rng = RNG(hwcfg_brng, swcfg)()
            self.bias_bsg = BSGen(self.bias_bin, self.bias_rng, swcfg)
            self.bias_idx = torch.nn.Parameter(torch.zeros_like(bias_ext, dtype=torch.long), requires_grad=False)

            hwcfg_add = {
                "mode" : self.hwcfg["mode"],
                "scale" : hwcfg["scale"],
                "depth" : self.hwcfg["depth"],
                "dima" : 0,
                "entry" : 2
            }
            self.Add = FSUAdd(hwcfg_add, swcfg)

    @autocast('cuda')
    def forward(self, input):
        # matmul: [batch, in_features] -> [batch, out_features] bitstream
        self.matmul_out = self.MatMul(input)

        if self.has_bias:
            # generate bias bitstream from BSGen
            self.bias_bs = self.bias_bsg(self.bias_idx).type(self.swcfg["stype"])
            self.bias_idx.data.add_(1)
            # expand bias [out_features] -> [batch, out_features] to match matmul_out
            bias_exp = self.bias_bs.unsqueeze(0).expand_as(self.matmul_out)
            # stack along dim 0: [2, batch, out_features], then add
            combined = torch.stack([self.matmul_out, bias_exp], dim=0)
            output = self.Add(combined)
        else:
            output = self.matmul_out

        return output
