from setuptools import setup, Extension

setup(
    name='imagemorph',
    version='0.1',
    install_requires=['numpy', 'opencv-python'],
    ext_modules=[Extension('imagemorph', ["imagemorph.c"])]
)
