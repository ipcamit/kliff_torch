# KIM-based Learning-Integrated Fitting Framework with Torch extensions (KLIFF-Torch)

> Note: This is temporary public fork of KLIFF for development purposes only. Once mature enough, all the capabilities developed 
> here will be incorporated in KLIFF.    

## Expected capabilities
1. Full compatibility with TorchScript ML models
2. Use new AD descriptor library
3. Use new TorchMLModel driver for universal ML backend support for exporting ML models to KIM API

### Original Documentation at: <https://kliff.readthedocs.io>

KLIFF is an interatomic potential fitting package that can be used to fit
physics-motivated (PM) potentials, as well as machine learning potentials such
as the neural network (NN) models.


## Install

### From source
```
git clone https://github.com/ipcamit/kliff_torch
cd kliff_torch; pip install -e .
```


## Why you want to use KLIFF (or not use it)

- Interacting seamlessly with[ KIM](https://openkim.org), the fitted model can
  be readily used in simulation codes such as LAMMPS and ASE via the `KIM API`
- Creating mixed PM and NN models
- High level API, fitting with a few lines of codes
- Low level API for creating complex NN models
- Parallel execution
- [PyTorch](https://pytorch.org) backend for NN (include GPU training)


## Cite

```
@Article{wen2021kliff,
  title   = {{KLIFF}: A framework to develop physics-based and machine learning interatomic potentials},
  author  = {Mingjian Wen and Yaser Afshar and Ryan S. Elliott and Ellad B. Tadmor},
  journal = {Computer Physics Communications},
  pages   = {108218},
  year    = {2021},
  doi     = {10.1016/j.cpc.2021.108218},
}
```
