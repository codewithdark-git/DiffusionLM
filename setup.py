from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="diffusion-llm",
    version="0.1.0",
    author="Dark Coder",
    author_email="codewithdark90@gmail.com",
    description="A diffusion-based language model implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/codewithdark-git/PIP-DifffusionLM",
    packages=find_packages(include=["diffusionLM", "diffusionLM.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.21.0",
        "datasets>=2.3.0",
        "numpy>=1.21.0",
        "tqdm>=4.62.0",
        "wandb>=0.12.0",
        "hydra-core>=1.1.0",
        "omegaconf>=2.1.0",
        "pytorch-lightning>=1.5.0",
        "accelerate>=0.10.0",
    ],
)
