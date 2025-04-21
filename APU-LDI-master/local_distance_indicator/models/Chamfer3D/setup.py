from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

import torch
print(torch.cuda.is_available())  # Should be True
print(torch.version.cuda)         # Should say 12.1

setup(
    name='chamfer_3D',
    ext_modules=[
        CUDAExtension('chamfer_3D', [
            "/".join(__file__.split('/')[:-1] + ['chamfer_cuda.cpp']),
            "/".join(__file__.split('/')[:-1] + ['chamfer3D.cu']),
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })