from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from bert_score import BERTScorer
import tiktoken
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Evaluator:
    def __init__(self, device, tokenizer):
        self.device = device
        self.tokenizer = tokenizer
        self.bert_scorer = BERTScorer(model_type='bert-base-uncased', device=device)
        
    def decode_predictions(self, logits):
        predictions = torch.argmax(logits, dim=-1)
        decoded_preds = [self.tokenizer.decode(pred) for pred in predictions]
        return decoded_preds
        
    def decode_targets(self, targets):
        decoded_targets = [self.tokenizer.decode(tgt) for tgt in targets]
        return decoded_targets
    
    def compute_bert_score(self, predictions, references):
        P, R, F1 = self.bert_scorer.score(predictions, references)
        return {
            'bert_precision': P.mean().item(),
            'bert_recall': R.mean().item(),
            'bert_f1': F1.mean().item()
        }


def evaluate_bleu(reference, candidate):
    return sentence_bleu([reference], candidate)


def evaluate_meteor(reference, candidate):
    return meteor_score([reference], candidate)


def main():
    enc = tiktoken.get_encoding("cl100k_base")
    evaluator = Evaluator(device, enc)
    
    logits = torch.randn(2, 5, 100352).to(device)
    targets = torch.randint(0, 100352, (2, 5)).to(device)
    
    predictions = evaluator.decode_predictions(logits)
    references = evaluator.decode_targets(targets)
    
    bert_scores = evaluator.compute_bert_score(predictions, references)
    print(bert_scores)
    
    bleu_score = evaluate_bleu(references, predictions)
    print(bleu_score)
    
    meteor_score = evaluate_meteor(references, predictions)
    print(meteor_score)


if __name__ == "__main__":
    main()