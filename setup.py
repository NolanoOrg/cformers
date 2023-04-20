from setuptools import setup, find_packages
import codecs
import os
import subprocess

packages= ['cformers', 'cformers/cpp']
package_data = {'cformers': ['*'], 'cformers/cpp': ['*']}
build_main = subprocess.run(["make"], stdout=subprocess.PIPE, cwd="cformers/cpp")

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.4'
DESCRIPTION = 'SoTA Transformers with C-backend for fast inference on your CPU.'
LONG_DESCRIPTION = 'We identify three pillers to enable fast inference of SoTA AI models on your CPU:\n1. Fast C/C++ LLM inference kernels for CPU.\n2. Machine Learning Research & Exploration front - Compression through quantization, sparsification, training on more data, collecting data and training instruction & chat models.\n3. Easy to use API for fast AI inference in dynamically typed language like Python.\n\nThis project aims to address the third using LLaMa.cpp and GGML.'

# Setting up
setup(
    name="cformers",
    version=VERSION,
    author="Ayush Kaushal (Ayushk4)",
    author_email="ayush4@utexas.edu",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=packages,
    package_data=package_data,
    install_requires=['transformers', 'torch', 'wget'],
    keywords=['python', 'local inference', 'c++ inference', 'language models', 'cpu inference', 'quantization'],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)