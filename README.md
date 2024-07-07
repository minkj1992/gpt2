# Transformer

```bash
conda create -n gpt python=3.12 -y
conda activate gpt
# https://github.com/explosion/spaCy/issues/13528
conda install "numpy>=1.19.0,<2.0.0" spacy -y  
conda install pytorch::pytorch torchvision torchaudio torchtext -c pytorch -y
conda install matplotlib tensorboard seaborn -y
```
