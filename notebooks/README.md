# This is a document to help you troubleshoot problems using this repository

### python virtual environment

```console
you@you chat-api % python3 -m venv venv
you@you chat-api % source venv/bin/activate
(venv) you@you chat-api % pip install --upgrade pip
(venv) you@you chat-api % pip install -r requirements.txt
```

### what directories are consuming the most memory

`df -h`

`du --max-depth=2 --human-readable /home/ | sort --human-numeric-sort`

### the torch version must match the CUDA version on your machine

https://pytorch.org/get-started/previous-versions/

for example if `nvcc --version` shows you have CUDA version 11.3 

`pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113` 


