from setuptools import setup, find_packages

setup(
    name="bise",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'transformers',
        'numpy',
        'pandas',
        'decord',
        'opencv-python',
        'mediapipe',
        'tqdm',
        'matplotlib',
        'seaborn'
    ],
)
