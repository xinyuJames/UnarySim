import torch

class FSUAdd(torch.nn.Module):
    """
    This module is for unary addition for arbitrary scale, including scaled/non-scaled, unipolar/bipolar.
    """
    def __init__(
        self, 
        hwcfg={
            "mode" : "bipolar", 
            "scale" : None, # none->scaled ; 1->non-scaled
            # scale_carry: scaling factor
            "dima" : 0, # which axis to add together TODO: use case of non-zero?
            "depth" : 10, # for Accumulator bound calculation
            "entry" : None # operands in parallel
        }, 
        swcfg={
            "btype" : torch.float, # buffer type, used for accumulator
            "stype" : torch.float
        }):
        super(FSUAdd, self).__init__()
        self.hwcfg = {}
        self.hwcfg["depth"] = hwcfg["depth"]
        self.hwcfg["mode"] = hwcfg["mode"].lower()
        self.hwcfg["scale"] = hwcfg["scale"]
        self.hwcfg["dima"] = hwcfg["dima"]
        self.hwcfg["entry"] = hwcfg["entry"]

        self.swcfg = {}
        self.swcfg["btype"] = swcfg["btype"]
        self.swcfg["stype"] = swcfg["stype"]

        # data representation
        self.mode = hwcfg["mode"].lower()
        assert self.mode in ["unipolar", "bipolar"], \
            "Error: the hw config 'mode' in " + str(self) + " class requires one of ['unipolar', 'bipolar']."

        # scale is an arbitrary value that larger than 0
        self.scale = hwcfg["scale"]
        # dimension to do reduced sum
        self.dima = hwcfg["dima"]
        # depth of the accumulator
        self.depth = hwcfg["depth"]
        # number of entries in dima to do reduced sum
        self.entry = hwcfg["entry"]

        # max value in the accumulator
        self.acc_max = 2**(self.depth-2)
        # min value in the accumulator
        self.acc_min = -2**(self.depth-2)
        self.stype = swcfg["stype"]
        self.btype = swcfg["btype"]
        
        # the carry scale at the output
        self.scale_carry = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        # accumulation offset
        self.offset = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        # accumulator for (PC - offset)
        self.accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.first = True

    def forward(self, input, scale=None, entry=None):
        if self.first: # first to set up
            if entry is not None:
                # runtime entry will override the default value
                self.entry = entry
                self.hwcfg["entry"] = entry
            else:
                if self.entry is None:
                    self.entry = input.size()[self.dima]
                    self.hwcfg["entry"] = input.size()[self.dima]
                else:
                    self.entry = self.entry
                    self.hwcfg["entry"] = self.entry

            if scale is not None: # non-scaled
                # runtime scale will override the default value
                self.scale_carry.fill_(scale) # 1, since non-scaled
                self.hwcfg["scale"] = scale
            else: # scaled
                if self.scale is None:
                    self.scale_carry.fill_(self.entry) # N, since scaled
                    self.hwcfg["scale"] = self.entry
                else: 
                    self.scale_carry.fill_(self.scale)
                    self.hwcfg["scale"] = self.scale

            if self.mode == "bipolar":
                self.offset.data = (self.entry - self.scale_carry)/2
            self.hwcfg["offset"] = self.offset

            self.first = False
        else:
            pass
        
        # Parallel Counter
        acc_delta = torch.sum(input.type(self.btype), self.dima) - self.offset # in uLinear, this step is done in LinearPC
        # Accumulator
        self.accumulator.data = self.accumulator.add(acc_delta).clamp(self.acc_min, self.acc_max) # this clamp is because of hardware limitation
        # Compare with carry for output bit stream
        output = torch.ge(self.accumulator, self.scale_carry).type(self.btype)
        # substract carry from the Accumulator
        self.accumulator.sub_(output * self.scale_carry).clamp_(self.acc_min, self.acc_max)
        return output.type(self.stype)

