# Tensor-Based Optimal Experimental Design (OED)

This repository contains code and examples for **tensor-based approaches to Optimal Experimental Design (OED)** for inverse problems.  
The focus is on scalable methods that exploit tensor structures for efficient computations in Bayesian inverse problems and related applications.

## 📂 Repository Structure

```text
tensor-oed/
│
├── data/                     # Data folder (initially empty)
│   └── README.md             # Instructions to download required files
│
├── examples/                         # Jupyter notebooks with experiments
│   ├── seismic_example.ipynb         # OED for seismic tomography
│   └── DEIMSSSP.ipynb                # OED for interpolation problem
│
├── sensor_placement/         # Source code (tensor-based OED implementations)
│   ├── __init__.py
│   ├── decomposition.py        # Core algorithms
│   ├── utils.py                # Helper functions
│   └── tests.py               # Other utilities
│
├── requirements.txt          # Python dependencies
├── LICENSE
└── README.md                 # Project documentation

