import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xr-random",
    version="0.0.1",
    author="Ondrej Grover",
    author_email="ondrej.grover@gmail.com",
    description="random number generation parametrized by xarray objects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smartass101/xr-random",
    packages=['xrrandom'],
    install_requires=['xarray', 'dask', 'scipy'],
)
