from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="factqa-atropos",
    version="0.1.0",
    description="A factual question-answering environment for Atropos",
    author="sukrucildirr",
    author_email="sukrucildirr@gmail.com",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.10",
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
