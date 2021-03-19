from distutils.core import setup
import setuptools


setup(
    name="flaxseed",
    version="0.1.0-alpha",
    author="Frank Odom",
    author_email="frank.odom.iii@gmail.com",
    url="https://github.com/fkodom/flaxseed",
    packages=setuptools.find_packages(),
    description="Library for training deep learning models with Jax/Flax.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    requires=open("requirements.txt").read().split("\n"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
