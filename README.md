# Cell-Cell-Communication

This repository contains the code for the Cell-Cell-Communication project.

```
Cell-Cell-Communication
├── projct
│   ├── earlystopping.py
    └── utils.py
├── main.py
├── preprocessing.py (now not available for project privacy reasons)
└── model.py
```

```
Before using those packages in gpu, be sure that all torch-related package is in correct cuda version.
- `nvidia-smi CUDA Version (12.5)`: This version indicates the maximum CUDA version supported by the installed NVIDIA driver. It reflects the driver’s capability to run CUDA applications that are built with any CUDA version up to 12.5.
- `nvcc --version CUDA Version (12.1)`: This version refers to the version of the CUDA toolkit installed on your system. The CUDA toolkit includes the CUDA compiler (nvcc), libraries, and other tools for developing CUDA applications.
- python == 3.10
- torch == 2.1.0+cu121 
- pyg_lib == 0.3.1+pt21cu121
- torch_cluster == 1.6.3+pt21cu121
- torch_scatter == 2.1.2+pt21cu121
- torch_sparse == 0.6.18+pt21cu121
- torch_spline_conv == 1.2.2+pt21cu121
```

## Tutorials

This repository contains the following codes and explainations:

- **earlystopping.py**: class for early stopping in training process.
- **utils.py**: models and functions for training and testing.
- **main.py**: code with cross-validation for hyperparameter tuning in regular GAT model.
- **preprocessing.py**: tutorial notebook for preprocessing the data.
- **model.py**: GAT model.


## rGAT Model:
[GitHub - rGAT Model Documentation](https://github.com/martina-yu/Cell-Cell-Communication/tutorial.html)  