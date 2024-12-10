from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from bert_score import BERTScorer


def evaluate_bleu(reference, candidate):
    return sentence_bleu([reference], candidate)


def evaluate_bert_score(reference, candidate):
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score([candidate], [reference])
    print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
    return F1.mean()


def evaluate_meteor(reference, candidate):
    return meteor_score([reference], candidate)