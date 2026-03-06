from setuptools import setup

setup(
    name="safety_monitor",
    version="0.1.0",
    packages=["safety_monitor"],
    install_requires=["setuptools", "numpy", "opencv-python"],
    entry_points={
        "console_scripts": [
            "safety_node = safety_monitor.safety_node:main",
        ],
    },
)
