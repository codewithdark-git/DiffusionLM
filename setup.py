from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="diffusionLM",
    version="0.1.5",
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
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "tqdm",
        "wandb",
        "hydra-core",
        "omegaconf",
        "pytorch-lightning",
        "accelerate",
    ],
)
