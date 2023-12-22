# Please don't change the order of following packages!
import sys
from distutils.core import setup

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, "python version >= 3.6 is required"

setup(
    name="marlray",
    install_requires=[
    "ray[all]==2.9.0",
    "gymnasium==0.28.1",
    "metadrive-simulator==0.4.1.2",
    "tensorboardX==2.6.2.2",
    "seaborn==0.13.0",
    "tqdm==4.66.1",
    "panda3d-simplepbr=0.11.2",
    "tensorboard"
    ],
    license="Apache 2.0",
)
