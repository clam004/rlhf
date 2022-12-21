# This is a document to help you troubleshoot problems using this repository

### python virtual environment

```console
you@you chat-api % python3 -m venv venv
you@you chat-api % source venv/bin/activate
(venv) you@you chat-api % pip install --upgrade pip
(venv) you@you chat-api % pip install -r requirements.txt
```

### what directories or files are using the most disk memory

```
df -h
ls -alh *
du --max-depth=2 --human-readable /home/ | sort --human-numeric-sort
```

### the torch version must match the CUDA version on your machine

https://pytorch.org/get-started/previous-versions/

https://download.pytorch.org/whl/torch_stable.html

for example if `nvcc --version` shows you have CUDA version 11.3 

`pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113` 

you can remove this from requirements.txt where it is inserted like this

```
transformers
tqdm
--extra-index-url https://download.pytorch.org/whl/cu113
torch==1.11.0+cu113 
```

and pip install from the command line

CUDA version 11.1

`pip install --upgrade torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html`

### Huggingface Hub

You need this to use huggingface push_to_hub

`wget https://github.com/git-lfs/git-lfs/releases/download/v2.9.0/git-lfs-linux-amd64-v2.9.0.tar.gz`

and follow http://arfc.github.io/manual/guides/git-lfs

```
(env_ds) ubuntu@104-171-202-32:~/farafiles$ wget https://github.com/git-lfs/git-lfs/releases/download/v2.9.0/git-lfs-linux-amd64-v2.9.0.tar.gz
(env_ds) ubuntu@104-171-202-32:~/farafiles$ ls
Commonsense-Dialogues  git-lfs-linux-amd64-v2.9.0.tar.gz  ml_fara  modelstates
(env_ds) ubuntu@104-171-202-32:~/farafiles$ tar -xf git-lfs-linux-amd64-v2.9.0.tar.gz 
(env_ds) ubuntu@104-171-202-32:~/farafiles$ ls
CHANGELOG.md  Commonsense-Dialogues  README.md  git-lfs  git-lfs-linux-amd64-v2.9.0.tar.gz  install.sh  ml_fara  modelstates
(env_ds) ubuntu@104-171-202-32:~/farafiles$ chmod 755 install.sh
(env_ds) ubuntu@104-171-202-32:~/farafiles$ sudo ./install.sh
Git LFS initialized.
(env_ds) ubuntu@104-171-202-32:~/farafiles$ cd ml_fara
(env_ds) ubuntu@104-171-202-32:~/farafiles/ml_fara$ git lfs install
Updated git hooks.
Git LFS initialized.
(env_ds) ubuntu@104-171-202-32:~/farafiles/ml_fara$ 
```

### list processes using GPUs with 

`sudo fuser -v /dev/nvidia*`

### structure of a nohup command

`nohup python run.py > output.log &`

`nohup uvicorn main:app --host 0.0.0.0 --port 5000 > output.log &`
