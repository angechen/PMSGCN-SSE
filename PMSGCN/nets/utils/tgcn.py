# The based unit of graph convolutional networks.

import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):

    """The basic module for applying a graph convolution.



    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels,1,  T*V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, T*V, T*V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, 1, T*V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, T*V, T*V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the kernel size`,
            :math:`T` is a length of sequence,
            :math:`V` is the number of graph nodes per frame.
    """

    def __init__(self,
                 in_channels, #2;128;128
                 out_channels,#128;128;256
                 kernel_size, #6
                 opt,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.inplace = True
        self.momentum = 0.1
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,                    #2;128;128
            out_channels * kernel_size,     #(128x6=768);128x6;256x6
            kernel_size=(t_kernel_size, 1), #(1,1)
            padding=(t_padding, 0),         #(0,0)
            stride=(t_stride, 1),           #(1,1)
            dilation=(t_dilation, 1),       #(1,1)
            bias=bias)
        if opt.tcnlayer == True:
            self.bn_relu_drop = nn.Sequential(
      #      nn.BatchNorm2d(out_channels, momentum=self.momentum),    #128,0.1;128;256
            nn.ELU(alpha=1.0, inplace=self.inplace),  #True
            nn.Dropout(0.05),)
        else:
            self.bn_relu_drop = nn.Sequential(
         #       nn.BatchNorm2d(out_channels, momentum=self.momentum),    #128,0.1;128;256
                nn.Dropout(0.05),)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        x = self.bn_relu_drop(x)

        return x.contiguous(), A

