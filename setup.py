"""Create instructions to build the simSPI package."""

import setuptools

requirements = []

setuptools.setup(
    name="compSPI",
    maintainer="",
    version="0.0.1",
    maintainer_email="",
    description="computational Single Particle Imaging",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/compSPI/compSPI.git",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
