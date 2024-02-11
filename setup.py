from setuptools import setup
from sys import platform, version_info

if __name__ == "__main__":
    if platform == "win32" or platform: # windows (not supported yet)
        # extra_compile_args = ['/O2', '/std:c++17']
        setup()
    elif platform == 'darwin' and version_info.minor > 11 : # mac greater than python 3.11
        setup()
    else:
        from torch.utils import cpp_extension
        extra_compile_args = ['-O2', '-std=c++17']
        setup(
            ext_modules=[cpp_extension.CppExtension(name='uotod.compiled',
                                                    sources=['src/cpp/sinkhorn.cpp'],
                                                    include_dirs=['src/cpp'],
                                                    extra_compile_args=extra_compile_args)]
        )