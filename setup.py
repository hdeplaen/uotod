from setuptools import setup
from sys import platform

if __name__ == "__main__":
    if platform == "win32":  # windows (not supported yet)
        # extra_compile_args = ['/O2', '/std:c++17']
        # setup(
        #     ext_modules=[cpp_extension.CppExtension(name='uotod.compiled',
        #                                             sources=['src/cpp/sinkhorn.cpp'],
        #                                             include_dirs=['src/cpp'],
        #                                             extra_compile_args=extra_compile_args)]
        # )
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