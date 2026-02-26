import torch
import math
import copy
from UnarySim.stream import RNG, BinGen, BSGen
from UnarySim.kernel import FSUMul, FSUAdd, rshift_offset
from torch.cuda.amp import autocast

class FSULinear(torch.nn.Module):
    """
    This module is the fully connected layer with unary input and unary output, and its API is similar to the Linear class (input/output feature count, bias flag), except:
    1) weight_ext: external binary weight
    2) bias_ext: external binary bias
    3) width: binary data width
    4) mode: unary data mode
    5) scale: accumulation scale
    6) depth: accumulator depth
    7) rng: weight rng type
    8) dimr: weight rng dimension

    The allowed coding for input, weight and bias with guaranteed accuracy can have the following three options.s
    (input, weight, bias):
    1) rate, rate, rate
    2) rate, temporal, rate
    3) temporal, rate, rate
    However, this module itself does not force the input coding. Thus, above coding constraints should be done by users.
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
            "scale" : None,
            "depth" : 12,
            "rng" : "Sobol",
            "dimr" : 1
        },
        swcfg={
            "btype" : torch.float, 
            "rtype" : torch.float, 
            "stype" : torch.float
        }):
        super(FSULinear, self).__init__()
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

        if (hwcfg["mode"].lower() == "bipolar") and (hwcfg["scale"] is not None) and (hwcfg["scale"] != (in_features + bias)):
            assert self.hwcfg["rng"].lower() not in ["race", "tc", "race10", "tc10"], \
                "Error: the hw config 'rng' in " + str(self) + " class should avoid ['race', 'tc', 'race10', 'tc10'] for bipolar data with non-scaled addition."

        assert self.swcfg["btype"] == torch.float, \
            "Error: the sw config 'btype' in " + str(self) + " class requires 'torch.float'."
        assert self.swcfg["rtype"] == torch.float, \
            "Error: the sw config 'rtype' in " + str(self) + " class requires 'torch.float'."
        assert self.swcfg["stype"] == torch.float, \
            "Error: the sw config 'stype' in " + str(self) + " class requires 'torch.float'."

        self.PC = FSULinearPC(
            in_features, 
            out_features, 
            bias=bias, 
            weight_ext=weight_ext, 
            bias_ext=bias_ext, 
            hwcfg=self.hwcfg,
            swcfg=self.swcfg)

        self.scale = hwcfg["scale"]
        if self.scale is None:
            scale_add = in_features + bias
        else:
            scale_add = self.scale
        hwcfg_acc = copy.deepcopy(self.hwcfg)
        hwcfg_acc["scale"] = scale_add
        hwcfg_acc["entry"] = in_features + bias # bias for addition
        # the pc result is unsqueezed before fed to the accumulator, so accumulation dim of FSUAdd is 0.
        hwcfg_acc["dima"] = 0
        self.ACC = FSUAdd(
            hwcfg_acc,
            self.swcfg)

    @autocast()
    def forward(self, input, scale=None, entry=None):
        pc = self.PC(input)
        output = self.ACC(pc.unsqueeze(0), scale, entry)
        return output


class FSULinearPC(torch.nn.Linear):
    """
    This module is the parallel counter result of FSULinear before generating the bitstreams.
    The allowed coding for input, weight and bias with guaranteed accuracy can have the following three options.
    (input, weight, bias):
    1) rate, rate, rate
    2) rate, temporal, rate
    3) temporal, rate, rate
    However, this module itself does not force the input coding. Thus, above coding constraints should be done by users.
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
            "rng" : "Sobol",
            "dimr" : 1
        },
        swcfg={
            "btype" : torch.float, 
            "rtype" : torch.float, 
            "stype" : torch.float
        }): # defined weight_in0 and weight_in1 generator for bipolar, and rng_idx for each of them.
        super(FSULinearPC, self).__init__(in_features, out_features, bias=bias)
        self.hwcfg = {}
        self.hwcfg["width"] = hwcfg["width"]
        self.hwcfg["mode"] = hwcfg["mode"].lower()
        self.hwcfg["rng"] = hwcfg["rng"].lower()
        self.hwcfg["dimr"] = hwcfg["dimr"]

        self.swcfg = {}
        self.swcfg["btype"] = swcfg["btype"]
        self.swcfg["rtype"] = swcfg["rtype"]
        self.swcfg["stype"] = swcfg["stype"]
        
        self.mode = hwcfg["mode"].lower()
        assert self.mode in ["unipolar", "bipolar"], \
            "Error: the hw config 'mode' in " + str(self) + " class requires one of ['unipolar', 'bipolar']."

        # bias indication for original linear layer
        self.has_bias = bias
        
        # RNG for weight
        hwcfg_wrng = {
            "width" : hwcfg["width"],
            "rng" : hwcfg["rng"],
            "dimr" : hwcfg["dimr"]
        }
        self.wrng = RNG(hwcfg_wrng, swcfg)()
        if hwcfg["rng"].lower() in ["race", "tc", "race10", "tc10"]:
            self.wtc = True
        else:
            self.wtc = False
        
        # define the linear weight and bias
        if weight_ext is not None:
            assert (weight_ext.size()[0], weight_ext.size()[1]) == (out_features, in_features), \
                "Error: the hw config 'out_features, in_features' in " + str(self) + " class unmatches the binary weight shape."
            self.weight.data = BinGen(weight_ext, self.hwcfg, self.swcfg)()
        
        if bias and (bias_ext is not None): # bias is 1D
            assert bias_ext.size()[0] == out_features, \
                "Error: the hw config 'out_features' in " + str(self) + " class unmatches the binary bias shape."
            self.bias.data = BinGen(bias_ext, self.hwcfg, self.swcfg)()
            # RNG for bias, same as RNG for weight
            hwcfg_brng = {
                "width" : hwcfg["width"],
                "rng" : hwcfg["rng"],
                "dimr" : hwcfg["dimr"]
            }
            self.brng = RNG(hwcfg_brng, swcfg)()

        # define the kernel linear for input bit 1
        self.wbsg_i1 = BSGen(self.weight, self.wrng, swcfg) 
        self.wrdx_i1 = torch.nn.Parameter(torch.zeros_like(self.weight, dtype=torch.long), requires_grad=False).unsqueeze(0)
        if self.has_bias is True:
            self.bbsg = BSGen(self.bias, self.brng, swcfg)
            self.brdx = torch.nn.Parameter(torch.zeros_like(self.bias, dtype=torch.long), requires_grad=False)
        
        # if bipolar, define a kernel for input bit 0, note that there is no bias required for this kernel
        if (self.mode == "bipolar") and (self.wtc is False):
            self.wbsg_i0 = BSGen(self.weight, self.wrng, swcfg)
            self.wrdx_i0 = torch.nn.Parameter(torch.zeros_like(self.weight, dtype=torch.long), requires_grad=False).unsqueeze(0)

    def FSULinear_PC_wrc(self, input):
        # this function is for weight with rate coding
        # first dim should always be batch
        batch = input.size()[0]

        # generate weight and bias bits for current cycle
        wbit_i1 = self.wbsg_i1(self.wrdx_i1).type(torch.float) # fancy indexing, the rng is [2**width], but every element index into it and compared with binary
        if wbit_i1.size()[0] != batch:
            wbit_i1 = torch.cat(batch*[wbit_i1], 0)
            self.wrdx_i1 = torch.cat(batch*[self.wrdx_i1], 0)
        torch.add(self.wrdx_i1, input.unsqueeze(1).type(torch.long), out=self.wrdx_i1) # this is a static multiplication, because weight is known
        
        ibit_i1 = input.unsqueeze(1).type(torch.float)
        obin_i1 = torch.empty(0, device=input.device)
        torch.matmul(ibit_i1, wbit_i1.transpose(1, 2), out=obin_i1)
        obin_i1.squeeze_(1)
        
        if self.has_bias is True:
            bbit = self.bbsg(self.brdx).type(torch.float)
            self.brdx.add_(1)
            obin_i1 += bbit.unsqueeze(0).expand_as(obin_i1)

        if self.mode == "unipolar":
            return obin_i1
        
        if self.mode == "bipolar":
            # generate weight and bias bits for current cycle
            wbit_i0 = 1 - self.wbsg_i0(self.wrdx_i0).type(torch.float)
            if wbit_i0.size()[0] != batch:
                wbit_i0 = torch.cat(batch*[wbit_i0], 0)
                self.wrdx_i0 = torch.cat(batch*[self.wrdx_i0], 0)
            torch.add(self.wrdx_i0, 1 - input.unsqueeze(1).type(torch.long), out=self.wrdx_i0)
            
            ibit_i0 = 1 - ibit_i1
            obin_i0 = torch.empty(0, device=input.device)
            torch.matmul(ibit_i0, wbit_i0.transpose(1, 2), out=obin_i0)
            obin_i0.squeeze_(1)

            return obin_i1 + obin_i0
    
    def FSULinear_PC_wtc(self, input):
        # this function is for weight with temporal coding
        # first dim should always be batch
        batch = input.size()[0]

        # generate weight and bias bits for current cycle
        wbit_i1 = self.wbsg_i1(self.wrdx_i1).type(torch.float)
        if wbit_i1.size()[0] != batch:
            wbit_i1 = torch.cat(batch*[wbit_i1], 0)
            self.wrdx_i1 = torch.cat(batch*[self.wrdx_i1], 0)
        torch.add(self.wrdx_i1, torch.ones_like(input).unsqueeze(1).type(torch.long), out=self.wrdx_i1)
        
        ibit_i1 = input.unsqueeze(1).type(torch.float)
        obin_i1 = torch.empty(0, device=input.device)
        torch.matmul(ibit_i1, wbit_i1.transpose(1, 2), out=obin_i1)
        obin_i1.squeeze_(1)
        
        if self.has_bias is True:
            bbit = self.bbsg(self.brdx).type(torch.float)
            self.brdx.add_(1)
            obin_i1 += bbit.unsqueeze(0).expand_as(obin_i1)

        if self.mode == "unipolar":
            return obin_i1
        
        if self.mode == "bipolar":
            # generate weight and bias bits for current cycle
            wbit_i0 = 1 - wbit_i1
            ibit_i0 = 1 - ibit_i1
            obin_i0 = torch.empty(0, device=input.device)
            torch.matmul(ibit_i0, wbit_i0.transpose(1, 2), out=obin_i0)
            obin_i0.squeeze_(1)

            return obin_i1 + obin_i0

    @autocast()
    def forward(self, input):
        assert len(input.size()) == 2, \
            "Error: the input of the " + str(self) + " class needs 2 dimensions."
        if self.wtc:
            return self.FSULinear_PC_wtc(input).type(self.swcfg["stype"])
        else:
            return self.FSULinear_PC_wrc(input).type(self.swcfg["stype"])
