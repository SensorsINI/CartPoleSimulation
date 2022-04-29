"""
This was the file to package SI_Toolkit and consequently to be able to install it with PIP.
The motivation was to easier imports between SI_Toolkit, SI_Toolkit application specific files and rest of projects where it is used
We abandoned the idea because it would not, or not necessarily be a sub-git-project
and this could once again cause that different versions of SI_Toolkit would start to diverge.
"""


import setuptools

import platform

if platform.machine() == 'arm64' and platform.system() == 'Darwin':  # For M1 Apple processor
    tensorflow_requirements = []
    print('Some functionalities of SI_Toolkit require Tensorflow.\n'
          'For M1 Apple processor you need to install it manually.\n'
          'Use commands:\n'
          'conda install -c apple tensorflow-deps\n'
          'pip install tensorflow-metal\n'
          'This is without GPU support which at the time of writing was working very poorly.\n'
          'You may want to check if anything better was released in the meantime.')
else:
    tensorflow_requirements = ['tensorflow']

version = platform.version()
if 'Ubuntu' in version and '18.04' in version:
    PyQt_requirements = ['PyQt5']
    print('SI_Toolkit is design to work with PyQt6 which does not support Ubuntu 18.04.\n'
          'You can find however on Github a separate branch with PyQt5 which will be installed now for you.')
else:
    PyQt_requirements = ['PyQt6']

requirements = []
requirements = requirements + tensorflow_requirements + PyQt_requirements

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SI_Toolkit_ASF_global",
    version="0.0.1",
    author="Sensors Group",
    author_email="marcin.p.paluch@gmail.com",
    description="Set of scripts for system identification with neural networks - globally available Application Specific Files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SensorsINI/SI_Toolkit",
    project_urls={
        "SI_Toolkit": "https://github.com/SensorsINI/SI_Toolkit",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where='src'),
    python_requires=">=3.8",
    install_requires=requirements
)