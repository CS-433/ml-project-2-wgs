# ML Project 2: Machine Reading Comprehension (Team: WGS)
This is the source code for the machine reading comprehension part of our experiments.

## Development Environment
- OS: Ubuntu 16.04 LTS (64bit)
- GPU: Nvidia Titan Xp
- CUDA 10.1
- Python 3.7

## Requirements
Install PyTorch 1.6.0:
```bash
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
Please also install the following library requirements specified in **requirements.txt**.

    transformers==4.6.0
    torchtext==0.3.1
    tensorboardX
    nltk
    matplotlib
    seaborn

## Running Experiments

### Download Dataset
SQuAD 1.1 dataset can be downloaded from [this link](https://drive.google.com/file/d/1bfrDR5N4BVS210cZQjjozIjRO57mm73r/view?usp=sharing), please unzip the file.

### Model Training and Evaluation
Note: Data loading will be slow for the first run, since data splits, pretrained GloVe and LM need to be prepared.

- Without Pretrained Language Model (None):
```bash
python run.py --add-lm-km none --dev-batch-size 64 --train-batch-size 64
```
- With Random Initialized Language Model (Random):
```bash
python run.py --add-lm-km random
```
- With BERT (base):
```bash
python run.py --lm-km-name bert-base-uncased
```
- With RoBERTa (base):
```bash
python run.py --lm-km-name roberta-base
```
- With GPT-2 (small):
```bash
python run.py --lm-km-name gpt2 --lm-km-type decoder
```
- With BART (base):
```bash
python run.py --lm-km-name facebook/bart-base
python run.py --lm-km-name facebook/bart-base --lm-km-type decoder
python run.py --lm-km-name facebook/bart-base --lm-km-type both
```
Note: Models will be saved under **saved_models**.

### Model Overview
![Model Overview](figures/model_mrc.png)


### Experimental Results

| Pretrained LM | Transformer Type | EM | F1 |
|--------------|:----------:|:----------:|:----------:|
| None | none | 43.85 | 56.43 |
| Random | none | 31.30 | 42.85 |
| BERT (base) | encoder | 47.46 | **59.97** |
| RoBERTa (base) | encoder | 47.16 | 59.39 |
| GPT-2 (small) | decoder | 40.69 | 53.45 |
| BART (base) | encoder | **47.52** | 59.83 |
| BART (base) | decoder | 47.24 | 59.72 |
| BART (base) | encoder + decoder | 44.86 | 57.34|



### Attention Score Analysis
Need to train Random and BERT (base),
and get *bert.pt*, *bert_ema.pkl*, *random.pt*, *random_ema.pkl* saved under **saved_models**.

- Analyze Random Initialized Language Model (Random):
```bash
python run_attention_analysis.py --add-lm-km random --model-path saved_models/random.pt --ema-path saved_models/random_ema.pkl --attn-tensor-save-path saved_models/random_attention_score_wrt_layer.pt
```

- Analyze BERT (base):
```bash
python run_attention_analysis.py --model-path saved_models/bert.pt --ema-path saved_models/bert_ema.pkl --attn-tensor-save-path saved_models/bert_attention_score_wrt_layer.pt
```
We have put analysis results *random_attention_score_wrt_layer.pt* and *bert_attention_score_wrt_layer.pt* under **saved_models**.

### Attention Score Visualization
```bash
python plot_heatmap.py
```
We will get: 
![Attetion Visualization](figures/heatmap.png)


## Reference
- https://github.com/galsang/BiDAF-pytorch


