# Note

## Model Goal

Machine translation is the task of automatically converting source text in one language to text in another language. The goal is to create a model that can translate text from one language to another. The model will be trained on a dataset of parallel text, which consists of examples of text in two languages and their translations. The model will learn to generate translations of text in one language into text in another language.

Languages extracted from WMT19 dataset:

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
- [nanoGPT](https://github.com/karpathy/nanoGPT) @karpathy
- [modded_nanoGPT](https://github.com/KellerJordan/modded-nanogpt.git) @KellerJordan

## Datasets

- [WMT19](https://huggingface.co/datasets/wmt/wmt19) : Multiple languages
- [Ko](https://github.com/ko-nlp/Korpora)
- [AI Hub Ko-En Parallel Corpus](https://ko-nlp.github.io/Korpora/en-docs/corpuslist/aihub_translation.html) : Only Korean-English
- [KrBert Corpus](https://www.kaggle.com/datasets/junbumlee/kcbert-pretraining-corpus-korean-news-comments/data) : Only Korean
- [laion-translated-to-en-korean-subset](Bingsu/laion-translated-to-en-korean-subset) : Also Korean-English

## Evaluation

- [Flores200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md)
- [BLEU](https://www.digitalocean.com/community/tutorials/bleu-score-in-python)
- [TER](https://pypi.org/project/pyter/)
- [METEOR](https://spotintelligence.com/2024/08/26/meteor-metric-in-nlp-how-it-works-how-to-tutorial-in-python/)
- [BERTScore](https://huggingface.co/spaces/evaluate-metric/bertscore) 

## Optimization

- [Muon](https://www.perplexity.ai/search/muon-optimizer-deep-learning-GcnuaC__Qu2FCqBFamaQwA#0)

Extract from Modded-NanoGPT:

- Modernized architecture: Rotary embeddings, QK-Norm, and ReLU^2.
- New optimizer: Muon - Momentum Orthogonalized by Newton-schulz.
- Untied head from embedding.
- Projection and classification layers initialized to zero (muP-like).
- Architectural shortcuts: value residual and embedding shortcut (partially following https://arxiv.org/abs/2410.17897).
- Momentum warmup.
- Tanh soft logit capping (following Gemma 2).
- FlexAttention with window size warmup.
- Extra embeddings which are fed into intermediate attention layers.


## Step

- [X] Cercare i dataset delle varie lingue;
- [X] scegliere cosa fare on il tokenizer, o custom training, o tiktoken. Scelta: tiktoken cl100k_base;
- [X] Dataloaders;
- [X] Testing tokenizer;
- [X] Creazione del modello;
- [X] Ottimizzazione;
- [] Training del modello;
- [] Finetune? (È necessaria?);
- [] Evaluation;
- [] Deploy (website, load the model on hf);

## Notes

- No Muon optimizer yet;

## Parameters

17/12/2024: mod_model.py: total parameters: 484.64M
            model_flex.py: total parameters: 484.50M
