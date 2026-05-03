# LENS-SLM-
__LENS-SLM__ (Learning Ensemble Confidence from Neural States for Multi-LLM Answer Integration) is an experimental attempt to replicate ensemble methods using small language models (__SLMs__) by combining outputs from multiple models. It learns a confidence scoring mechanism (__CMLP__) from neural states to evaluate and compare model responses. The goal is to improve answer quality by selecting or merging outputs based on learned confidence rather than relying on a single model.

## Requierments 
- Windows 10 or Windows 11  
- AMD GPU with ROCm support - [Compatibility list](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)
- Miniconda or Anaconda installed - [Getting Started anaconda](https://www.anaconda.com/docs/getting-started/main)
- Exact AMD GPU drivers or newer - [ROCm driver requierments](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)

## Environment Setup
```bash
conda env create -f environment.yml
```

___Enviorment creation is dependant on AMD repository download speed is throttled expect longer pip install time!___

```bash
conda activate rocm
```
### Verify Installation

#### Check python version

```bash
python --version
```

__Expected output__:

```bash
Python 3.12.13
```
#### Check torch and torch device

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

__Expected output__:

```bash
True
```

#### Get Device name

```bash
python -c "import torch; print(f'device name [0]:', torch.cuda.get_device_name(0))"
```

__Expected output__:

```bash
device name [0]: <Supported AMD GPU>
```

## Running the Project

Training CMLP (__Confidence multi layer perceptron__) for ensembling.
```bash
python Train.py
```

Benchmark ensembled models.
```bash
python Benchmark.py
```

Run interactive EnsembledChat.
```bash
python EnsembledChat.py
```

## Sources

https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html
https://arxiv.org/html/2507.23167v1