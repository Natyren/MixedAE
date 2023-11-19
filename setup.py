from setuptools import setup

setup(
    name="mixedae",
    version="0.1.0",
    description="Package implemented the MixedAE model",
    url="https://github.com/Natyren/MixedAE",
    author="George Bredis",
    author_email="georgy.bredis@gmail.com",
    license="MIT",
    packages=["mixedae"],
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.26.1",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3 :: Only",
    ],
)
