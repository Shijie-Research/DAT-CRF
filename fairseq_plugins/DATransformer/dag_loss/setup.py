from setuptools import setup
from torch.utils import cpp_extension

ext_modules = [
    cpp_extension.CUDAExtension(
        name="dag_loss_cuda",
        sources=[
            "dag_loss/dag_loss_kernel.cu",
            "dag_loss/dag_best_alignment.cu",
            "dag_loss/logsoftmax_gather.cu",
            "dag_loss/dag_loss.cpp",
        ],
    ),
]

setup(
    name="dag_loss",
    ext_modules=ext_modules,
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    include_package_data=True,
)
