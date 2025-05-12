from setuptools import setup, find_packages

setup(
    name="neural_component_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.1.0",
        "scipy>=1.6.0",
        "tqdm>=4.50.0",
    ],
    author="NCA Team",
    description="基于神经网络和Transformer的工业过程故障检测方法实现",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 