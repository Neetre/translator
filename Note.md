# Note

## Model Goal

Translation (content from WMT19 dataset):
- cs-en (7.27M rows)
- de-en (34.8M rows)
- fi-en (6.59M rows)
- fr-de (9.83M rows)
- gu-en (13.7k rows)
- kk-en (129k rows)
- It-en (2.35M rows)
- ru-en (37.5M rows)
- zh-en (26M rows)

## Template models

- [annotated-transformer](https://github.com/harvardnlp/annotated-transformer/blob/master/AnnotatedTransformer.ipynb)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

## Datasets

- [WMT19](https://huggingface.co/datasets/wmt/wmt19)

## Evaluation

- [BLEU](https://www.digitalocean.com/community/tutorials/bleu-score-in-python)
- [TER](https://pypi.org/project/pyter/)
- [METEOR](https://spotintelligence.com/2024/08/26/meteor-metric-in-nlp-how-it-works-how-to-tutorial-in-python/)
- [BERTScore] (https://huggingface.co/spaces/evaluate-metric/bertscore ) 


## Step

- [X] Cercare i dataset delle varie lingue;
- [X] scegliere cosa fare on il tokenizer, o custom training, o tiktoken. Scelta: tiktoken cl100k_base;
- [X] Dataloaders;
- [X] Testing tokenizer;
- [] Creazione del modello;
- [] Ottimizzazione;
- [] Training del modello
- [] Finetune? (È necessaria?) 
- [] Evaluation
- [] Deploy (website, load the model on hf) 
 