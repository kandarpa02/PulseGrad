import setuptools

setuptools.setup(
    name="PulseGrad",
    version="0.1.0",
    author="Kandarpa Sarkar",
    author_email="kandarpaexe@gmail.com",
    description="A automatic differentiation library with numpy backend for efficient matrix multiplication.",
    url="https://github.com/kandarpa02/PulseGrad.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy>=1.18.0"
    ],
)