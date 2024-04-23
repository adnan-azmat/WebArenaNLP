### Step 1: Create venv with python 3.10
```
conda create -n webagent python=3.10
```

### Step 2: activate venv
```
conda activate webagent
```

### Step 3: Install PyTorch for CUDA 11.8
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Step 4: Install dependencies
```
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
pip install -q -U datasets scipy ipywidgets matplotlib
```
<!-- pip install -q wandb -U ... not needed-->