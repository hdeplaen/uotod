from setuptools import setup
from torch.utils import cpp_extension
from sys import platform

if __name__ == "__main__":
    if platform == "win32":
        extra_compile_args = ['/O2', '/std:c++17']
    else:
        extra_compile_args = ['-O2', '-std=c++17']

    setup(
        ext_modules=[cpp_extension.CppExtension(name='uotod.compiled',
                                                sources=['src/cpp/sinkhorn.cpp'],
                                                include_dirs=['src/cpp'],
                                                extra_compile_args=extra_compile_args)]
    )