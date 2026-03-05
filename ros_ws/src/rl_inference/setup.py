from setuptools import setup

setup(
    name="rl_inference",
    version="0.1.0",
    packages=["rl_inference"],
    install_requires=["setuptools", "onnxruntime", "numpy", "opencv-python"],
    entry_points={
        "console_scripts": [
            "inference_node = rl_inference.inference_node:main",
        ],
    },
)
