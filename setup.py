from setuptools import setup
from torch.utils import cpp_extension

if __name__ == "__main__":
    setup(
        keywords="pytorch, machine learning",
        packages=['uotod'],
        ext_modules=[cpp_extension.CppExtension(name='uotod.compiled',
                                                sources=['src/cpp/sinkhorn.cpp'],
                                                extra_compile_args=['-O3','-w'])],
        cmdclass={'build_ext': cpp_extension.BuildExtension}
    )
