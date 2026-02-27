import torch

class FSUAdd(torch.nn.Module):
    """
    Unary addition supporting scaled/non-scaled, unipolar/bipolar.

    scale=True:  scaled addition, output represents sum/entry (average).
                 scale_carry = entry.
    scale=False: non-scaled addition, output represents raw sum.
                 scale_carry = 1. May saturate for large entry in unipolar mode.
    """
    def __init__(
        self,
        hwcfg={
            "mode" : "bipolar",
            "scale" : True,     # True -> scaled (average); False -> non-scaled (raw sum)
            "dima" : 0,         # dimension to reduce-sum along
            "depth" : 10,       # accumulator bit-width, acc range = [-2^(depth-2), 2^(depth-2)]
            "entry" : None      # number of operands along dima; None = infer from input
        },
        swcfg={
            "btype" : torch.float,
            "stype" : torch.float
        }):
        super(FSUAdd, self).__init__()

        # data representation
        self.mode = hwcfg["mode"].lower()
        assert self.mode in ["unipolar", "bipolar"], \
            "Error: 'mode' requires 'unipolar' or 'bipolar'."

        self.scaled = hwcfg["scale"]        # True or False
        self.dima = hwcfg["dima"]
        self.depth = hwcfg["depth"]
        self.entry = hwcfg["entry"]

        # accumulator bounds
        self.acc_max = 2**(self.depth - 2)
        self.acc_min = -2**(self.depth - 2)
        self.stype = swcfg["stype"]
        self.btype = swcfg["btype"]

        # parameters set on first forward call
        self.scale_carry = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.offset = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.initialized = False

    def forward(self, input):
        if not self.initialized:
            # infer entry from input shape if not provided
            if self.entry is None:
                self.entry = input.size()[self.dima]

            # set scale_carry based on scaled flag
            if self.scaled:
                self.scale_carry.fill_(self.entry)      # divide by N
            else:
                self.scale_carry.fill_(1)               # no division

            # bipolar offset centers the parallel counter output around 0
            if self.mode == "bipolar":
                self.offset.data = (self.entry - self.scale_carry) / 2

            self.initialized = True

        # Parallel Counter: sum along dima, subtract offset
        acc_delta = torch.sum(input.type(self.btype), self.dima) - self.offset
        # Accumulator: integrate
        self.accumulator.data = self.accumulator.add(acc_delta).clamp(self.acc_min, self.acc_max)
        # Output carry
        output = torch.ge(self.accumulator, self.scale_carry).type(self.btype)
        self.accumulator.sub_(output * self.scale_carry).clamp_(self.acc_min, self.acc_max)
        return output.type(self.stype)
