from setuptools import find_packages, setup

setup(
    name="bise",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch",
        "torchvision",
        "transformers",
        "numpy",
        "pandas",
        "decord",
        "opencv-python",
        "mediapipe",
        "tqdm",
        "matplotlib",
        "seaborn",
        "kornia",
        "flask",
        "faiss-cpu",
        "pytest",
    ],
)
