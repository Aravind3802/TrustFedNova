from __future__ import annotations
"""The try-import code block below is specific for my Macbook Pro M3 pro, because when I ran the code without increasing the soft limit on my MacOS, the code
tends to crash with Error saying "too many files open", so I had to increase the limit to 4096 files, as long as it does not exceed the hard limit we can keep increasing the soft limit."""
try:
    import resource
    SoftLimit, HardLimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(SoftLimit, 4096), HardLimit), HardLimit))
except Exception:
    pass

""" The try-except code block is kept seprately for importing trochvision package as the code is haveily dependent on the packages. It first checks for trochvision and if
it does not find it in the environment, it exits with runtime error asking us to install torchvision."""
try:
    import torchvision
    import torchvision.transforms as T
except Exception as ImportErrorException:
    raise RuntimeError("This script requires torchvision: pip install torchvision")


