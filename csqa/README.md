# ML Project 2: Commonsense Question Answering (Team: WGS)
This is the source code for the commonsense question answering part of our experiments.

## Development Environment
- OS: Ubuntu 16.04 LTS (64bit)
- GPU: Nvidia Titan Xp
- CUDA 10.1
- Python 3.6.13 (mandatory)

## Requirements
Manually install Knotty AllenNLP: 
```
pip install --no-dependencies git+git://github.com/allenai/allennlp.git@7142962d330ca5a95cade114c26a361c78f2042e
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz
```
Change code file of AllenNLP to fit PyTorch > 0.4.1:

In *{your_environment_root}/lib/python3.6/site-packages/allennlp/modules/encoder_base.py* line 93,

Change
```
num_valid = torch.sum(mask[:, 0]).int().data[0]
```
To
```
num_valid = torch.sum(mask[:, 0]).int().data
```

Install PyTorch 1.6.0:
```
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

Please also install the following library requirements specified in **requirements.txt**.

    transformers==4.6.0
    spacy==2.0.18
    scikit-learn==0.21.0
    overrides==3.1.0
    h5py==3.1.0
    nltk==3.6.0
    flask==0.12.1
    flask_cors==3.0.3
    gevent==1.2.1
    pytz==2017.3
    tensorboardX
    editdistance
    pyhocon
    pytest
    unidecode
    psycopg2

## Running Experiments
- Without Pretrained Language Model (None):
```
python -m allennlp.run train esim/none.json -s none --include-package esim
```
- With Random Initialized Language Model (Random):
```
python -m allennlp.run train esim/random.json -s random --include-package esim
```
- With BERT (base):
```
python -m allennlp.run train esim/bert.json -s bert --include-package esim
```
- With RoBERTa (base):
```
python -m allennlp.run train esim/roberta.json -s roberta --include-package esim
```
- With GPT-2 (small):
```
python -m allennlp.run train esim/gpt2.json -s gpt2 --include-package esim
```
- With BART (base):
```
python -m allennlp.run train esim/bart_encoder.json -s bart_encoder --include-package esim
python -m allennlp.run train esim/bart_decoder.json -s bart_decoder --include-package esim
python -m allennlp.run train esim/bart_both.json -s bart_both --include-package esim
```
