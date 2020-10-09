"""Setup script of hucrl."""
from setuptools import find_packages, setup

extras = {
    "test": [
        "pytest>=5.0,<5.1",
        "flake8>=3.7.8,<3.8",
        "pydocstyle==4.0.0",
        "black>=19.10b0",
        "isort>=5.0.0",
        "pytest_cov>=2.7,<3",
        "mypy==0.750",
    ],
    "mujoco": ["mujoco-py<2.1,>=2.0", "imageio-ffmpeg==0.4.1", "dm_control"],
    "logging": ["tensorboard>=2.0,<3"],
    "experiments": [
        "lsf_runner==0.0.5",
        "torchvision>=0.6.0",
        "Pillow==5.4.1",
        "pandas==0.25.0",
        "dotmap>=1.3.0,<1.4.0",
    ],
}
extras["all"] = [item for group in extras.values() for item in group]


setup(
    name="hucrl",
    version="0.0.1",
    author="Sebastian Curi",
    author_email="sebascuri@gmail.com",
    license="MIT",
    python_requires=">=3.7.0",
    packages=find_packages(exclude=["docs"]),
    install_requires=[
        "rllib @ git+ssh://git@github.com/sebascuri/rllib@dev#egg=rllib"
        "numpy>=1.14,<2",
        "scipy>=1.3.0,<1.4.0",
        "torch>=1.5.0,<1.7.0",
        "gym>=0.15.4",
        "atari_py>=0.2.6",
        "tqdm>=4.0.0,<5.0",
        "matplotlib>=3.1.0",
        "gpytorch>=1.1.1,<1.2.0",
        "tensorboardX>=2.0,<3",
    ],
    extras_require=extras,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
)
