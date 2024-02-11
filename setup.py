from setuptools import setup
from torch.utils import cpp_extension

if __name__ == "__main__":
    setup(
        ext_modules=[cpp_extension.CppExtension(name='uotod.compiled',
                                                sources=['src/cpp/sinkhorn.cpp'],
                                                extra_compile_args=['-O3','-w', '-I src/cpp'])],
    )