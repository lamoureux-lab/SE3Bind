r"""
This is an implementation of voxel convolution
>>> test()
source code: https://github.com/e3nn/e3nn/blob/main/e3nn/nn/models/v2104/voxel_convolution.py
"""
import math

import e3nn.nn
import torch

from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Linear
from e3nn.math import soft_one_hot_linspace

from e3nn.nn import Gate, Activation

from ProcessCoords import ProcessCoords


class Convolution(torch.nn.Module):
    r"""convolution on voxels
    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        input irreps
    irreps_out : `e3nn.o3.Irreps`
        output irreps
    irreps_sh : `e3nn.o3.Irreps`
        set typically to ``o3.Irreps.spherical_harmonics(lmax)``
    diameter : float
        diameter of the filter in physical units
    num_radial_basis : int
        number of radial basis functions
    steps : tuple of float
        size of the pixel in physical units
    """

    def __init__(self, device, dtype, irreps_in, irreps_out, irreps_sh, diameter, num_radial_basis, steps=(1.0, 1.0, 1.0),
                 normalization='integral', debug=False, **kwargs):
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.lmax = self.irreps_sh.lmax
        self.num_radial_basis = num_radial_basis
        # print('voxel self.device', self.device)

        # self-connection
        self.sc = Linear(self.irreps_in, self.irreps_out).to(device=self.device, dtype=self.dtype)

        # connection with neighbors
        r = diameter / 2

        s = math.floor(r / steps[0])
        x = torch.arange(-s, s + 1.0) * steps[0]

        s = math.floor(r / steps[1])
        y = torch.arange(-s, s + 1.0) * steps[1]

        s = math.floor(r / steps[2])
        z = torch.arange(-s, s + 1.0) * steps[2]

        lattice = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1)  # [x, y, z, R^3]
        self.register_buffer("lattice", lattice)

        if "padding" not in kwargs:
            kwargs["padding"] = tuple(s // 2 for s in lattice.shape[:3])
        self.kwargs = kwargs

        # radial basis function
        emb = soft_one_hot_linspace(
            x=lattice.norm(dim=-1), # radial distance r = ||x||
            start=0.0,
            end=r,                  # cutoff radius = diameter/2
            number=self.num_radial_basis,
            basis="smooth_finite",
            cutoff=True,
        )
        self.register_buffer("emb", emb)

        sh = o3.spherical_harmonics(
            l=self.irreps_sh, x=lattice, normalize=True, normalization=normalization
        )  # [x, y, z, irreps_sh.dim]

        self.register_buffer("sh", sh)

        if debug:
            print('plotting embedding of radial basis=', self.num_radial_basis)
            print(emb.shape)
            for i in range(emb.shape[-1]):
                ProcessCoords(dim=diameter, device=self.device, dtype=self.dtype).plot_volume(emb[:, :, :, i], surface_count=10, isomin=None)

            print('plotting spherical harmonics of order lmax=', self.lmax)
            print(sh.shape)
            for i in range(sh.shape[-1]):
                ProcessCoords(dim=diameter, device=self.device, dtype=self.dtype).plot_volume(sh[:, :, :, i], surface_count=10, isomin=None)

        self.tp = FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            shared_weights=False,
            compile_left_right=False,
            compile_right=True,
        )

        self.weight = torch.nn.Parameter(torch.randn(self.num_radial_basis, self.tp.weight_numel, device=self.device, dtype=self.dtype))

    def kernel(self):
        weight = self.emb @ self.weight
        weight = weight / (self.sh.shape[0] * self.sh.shape[1] * self.sh.shape[2])
        kernel = self.tp.right(self.sh, weight)  # [x, y, z, irreps_in.dim, irreps_out.dim] #which mixes radial + angular + representation parts into the final kernel.
        kernel = torch.einsum("xyzio->oixyz", kernel)
        return kernel.to(device=self.device, dtype=self.dtype)

        # print('Kernel self.device', self.device)

    def forward(self, x):
        r"""
        Parameters
        ----------
        x : `torch.Tensor`
            tensor of shape ``(batch, irreps_in.dim, x, y, z)``
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, irreps_out.dim, x, y, z)``
        """
        sc = self.sc(x.transpose(1, 4)).transpose(1, 4)

        return sc + torch.nn.functional.conv3d(x, self.kernel(), **self.kwargs)
        # sc = self.sc(x.permute(1,2,3,0)).permute(3, 0, 1, 2)
        # return torch.nn.functional.conv3d(x, self.kernel(), **self.kwargs)


def test(kernel, num_radial_basis, lmax):
    # scalar_in = "0o"
    # vector_in = '4x0o'
    # gate = Gate(scalar_in, [torch.nn.ReLU], vector_in, [torch.nn.ReLU], "0o+1o")

    # gate =  Gate("16x0o", [torch.tanh], "32x0o", [torch.tanh], "16x1e+16x1o")

    # a = Activation("256x0o", [torch.abs])

    # a = Activation("2x0e", [torch.abs])

    # conv1 = Convolution(
    #     dtype=torch.float,
    #     device='cpu',
    #     irreps_in="1x0e",
    #     irreps_out="4x0e + 4x1o",
    #     irreps_sh=o3.Irreps.spherical_harmonics(lmax=lmax),
    #     diameter=kernel,
    #     num_radial_basis=num_radial_basis,
    #     steps=stride,
    #     debug=True,
    # )

    irreps_out = '1x3y'
    conv1 = Convolution(
        dtype=torch.float,
        device='cpu',
        irreps_in="1x0e",
        irreps_out=irreps_out,
        irreps_sh=o3.Irreps.spherical_harmonics(lmax=lmax),
        diameter=kernel,
        num_radial_basis=num_radial_basis,
        steps=stride,
        debug=True,
    )
    # conv2 = Convolution(
    #     irreps_in="0y + 4x1y",
    #     # irreps_out="0o + 1o",
    #     irreps_out="2x0y",
    #     irreps_sh=o3.Irreps.spherical_harmonics(lmax=2),
    #     diameter=5,
    #     num_radial_basis=3,
    #     steps=(1, 1, 1),
    # )

    x = torch.randn(1, 1, 50, 50, 50)
    x1 = conv1(x)
    # print(x1)
    # x2 = conv2(x1)
    # print(x.shape)
    # print(x1.shape)
    # print(x2.shape)

    # a(x2)

    # gate(x2)

    # m = torch.nn.GroupNorm(2, 4)
    # print(m(x2).shape)

    # n = e3nn.nn.NormActivation("0o + 4x1o", torch.nn.ReLU)
    # n(x2)


if __name__ == "__main__":
    # kernel = 9
    # num_radial_basis = 4
    # lmax = 2
    # stride = (1, 1, 1)
    # test(kernel=kernel, num_radial_basis=num_radial_basis, lmax=lmax)

    kernel = 9
    num_radial_basis = 7
    lmax = 3
    stride = (1, 1, 1)
    test(kernel=kernel, num_radial_basis=num_radial_basis, lmax=lmax)

    # #
    # kernel = 9
    # num_radial_basis = 2
    # lmax = 2
    # stride = (1, 1, 1)
    # test(kernel=kernel, num_radial_basis=num_radial_basis, lmax=lmax)


    #
    # kernel = 9
    # num_radial_basis = 1
    # lmax = 1
    # stride = (1, 1, 1)
    # test(kernel=kernel, num_radial_basis=num_radial_basis, lmax=lmax)
