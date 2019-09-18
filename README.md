# Magnitude calibration
This repository provides scripts to calibrate, apply and analyze correction functions for magnitude estimation.
The correction functions consist of three parts:
- station bias
- 2D distance and depth correction
- 3D source correction (station specific)

It further provides functions to reduce uncertainty through multi-feature estimation using boosting tree regression.

## Requirements
The scripts use Python>=3.5. Please install python requirements with `pip install -r requirements.txt`.

Additionally the script relies on the Gurobi Optimizer for quadratic optimization.
Gurobi provides free Academic Licenses (http://www.gurobi.com/academia/for-universities).
Please also install the additional Gurobi python library.

## Usage
For details on usage please consult the Jupyter notebook and refer to the documentation in the code.

## Citation
```
@article{munchmeyer2019,
    author = {Münchmeyer, Jannes and Bindi, Dino and Sippl, Christian and Leser, Ulf and Tilmann, Frederik},
    title = "{Low uncertainty multi-feature magnitude estimation with 3D corrections and boosting tree regression: application to North Chile}",
    journal = {Geophysical Journal International},
    year = {2019},
    doi = {10.1093/gji/ggz416}
}
```
