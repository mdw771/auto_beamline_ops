Welcome to the repository of *autobl*, a package that integrates tools and algoorthms for automating beamline operations at synchrotron facilities. 

# Installation

To install the package with basic functionality, download the repository and install it with
```
pip install -e .
```

To enable the autonomous experiment steering (including adaptive sampling) capability, do
```
pip install -e .[steering]
```

# Usage

## XANES adaptive sampling
Code examples for the adaptive XANES sampling tool are available in `workspace/spectroscopy/XANES`, *e.g.*, `XANES_sampling_Pt.py`. Data needed to run these scripts are available upon reasonable request from the repository admin.
