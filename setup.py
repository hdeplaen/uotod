from setuptools import setup
import pathlib
import uotod
import sys
import platform
from torch.utils import cpp_extension

# DESSCRIPTION
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# REQUIREMENTS
with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

with open("LICENSE", "r", encoding="utf-8") as fh:
    license = fh.read()

# PYTHON VERSIONS
python_min_version = (3, 8, 0)
python_min_version_str = ".".join(map(str, python_min_version))
if sys.version_info < python_min_version:
    print(
        f"You are using Python {platform.python_version()}. Python >={python_min_version_str} is required."
    )
    sys.exit(-1)
version_range_max = max(sys.version_info[1], 10) + 1

# SETUP
setup(
    name='uotod',
    version=uotod.__version__,
    author=uotod.__author__,
    author_email='henri.deplaen@esat.kuleuven.be',
    description='Unbalanced Optimal Transport for Object Detection',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={
        "uotod": ["uotod/*"]
    },
    url='https://github.com/hdeplaen/uotod',
    download_url='https://github.com/hdeplaen/uotod/archive/{}.tar.gz'.format(uotod.__version__),
    project_urls={
        "Documentation": "https://hdeplaen.github.io/uotod/doc",
        "Bug Tracker": "https://github.com/hdeplaen/uotod/issues",
        "E-DUALITY": "https://www.esat.kuleuven.be/stadius/E/",
        "ESAT-STADIUS": "https://www.esat.kuleuven.be/stadius/",
        "ESAT-PSI": "https://www.esat.kuleuven.be/psi/",
    },
    platforms=['linux', 'macosx', 'windows'],
    license=license,
    install_requires=install_requires,
    classifiers=[
                    'Development Status :: 3 - Alpha',
                    'Environment :: GPU :: NVIDIA CUDA',
                    'Natural Language :: English',
                    'Programming Language :: Python :: 3',
                    'Programming Language :: C++',
                    'Topic :: Software Development :: Libraries :: Python Modules',
                    'Operating System :: OS Independent',
                    'Operating System :: POSIX :: Linux',
                    'Operating System :: MacOS',
                    'Operating System :: POSIX',
                    'Operating System :: Microsoft :: Windows',
                    'Topic :: Utilities',
                    'Topic :: Scientific/Engineering :: Artificial Intelligence',
                    'Topic :: Scientific/Engineering :: Mathematics',
                    'Topic :: Scientific/Engineering :: Information Analysis',
                    'Intended Audience :: Developers',
                    'Intended Audience :: Education',
                    'Intended Audience :: Science/Research',
                ]
                + [
                    f"Programming Language :: Python :: 3.{i}"
                    for i in range(python_min_version[1], version_range_max)
                ],
    keywords="pytorch, machine learning",
    packages=['uotod'],
    ext_modules=[cpp_extension.CppExtension(name='uotod.compiled',
                                            sources=['src/sinkhorn.cpp'],
                                            extra_compile_args=['-O3','-w'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
