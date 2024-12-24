import os
import torch
from torch.optim import AdamW
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

import warnings
import transformers
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()


from tqdm import tqdm
from data_loader import get_dataloader, MTConfig
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', "opus")

class T5Trainer:
    def __init__(
        self,
        model_name="t5-base",
        source_lang="English",
        target_lang="Korean",
        pre_tokenized=True,
        learning_rate=2e-5,
        max_grad_norm=1.0,
        num_epochs=3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.prefix = f"translate {source_lang} to {target_lang}: "
        self.pre_tokenized = pre_tokenized
        
        if self.pre_tokenized:
            self.prefix_tokens = self.tokenizer(
                self.prefix,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids[0]

        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs

    def prepare_batch(self, batch, split="train"):
        """
        Prepare a batch for training/evaluation, handling both pre-tokenized
        and raw text inputs
        """
        if self.pre_tokenized:
            # For pre-tokenized data
            batch_size = batch['source'].size(0)
            prefix_expanded = self.prefix_tokens.repeat(batch_size, 1).to(self.device)

            input_ids = torch.cat([prefix_expanded, batch['source']], dim=1)

            print("prefix_expanded shape:", prefix_expanded.shape)
            print("src_attn_mask shape:", batch['src_attn_mask'].shape)

            attention_mask = torch.cat([
                torch.ones_like(prefix_expanded),
                batch['src_attn_mask']
            ], dim=1)

            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = batch['target_input'].to(self.device) if split == "train" else batch['target_output'].to(self.device)

            return {
                'source': input_ids,
                'src_attn_mask': attention_mask,
                'target_input': labels
            }
        else:
            # For raw text data
            input_texts = [self.prefix + text for text in batch['source_texts']]
            inputs = self.tokenizer(
                input_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)

            labels = self.tokenizer(
                batch['target_texts'],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).input_ids.to(self.device)

            return {
                'input_ids': inputs.input_ids,
                'src_attn_mask': inputs.attention_mask,
                'target_input': labels
            }

    def train(self, train_dataloader, val_dataloader):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_dataloader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        best_val_loss = float('inf')
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            self.model.train()
            total_train_loss = 0
            train_bar = tqdm(train_dataloader, desc="Training")
            
            for batch in train_bar:
                prepared_batch = self.prepare_batch(batch, "train")
                input_ids = prepared_batch['source'].to(self.device)
                attention_mask = prepared_batch['src_attn_mask'].to(self.device)
                labels = prepared_batch['target_input'].to(self.device)

                self.model.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                optimizer.step()
                scheduler.step()
                
                train_bar.set_postfix({
                    'loss': f"{loss.item():.4f}"
                })

            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f"\nAverage training loss: {avg_train_loss:.4f}")

            val_loss = self.evaluate(val_dataloader)
            print(f"Validation loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best_model")
                print("Saved new best model!")

    def evaluate(self, dataloader):
        self.model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            val_bar = tqdm(dataloader, desc="Validating")
            for batch in val_bar:
                prepared_batch = self.prepare_batch(batch, "val")
                input_ids = batch['source'].to(self.device)
                attention_mask = batch['src_attn_mask'].to(self.device)
                labels = batch['target_output'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_val_loss += loss.item()
                
                val_bar.set_postfix({
                    'loss': f"{loss.item():.4f}"
                })

        return total_val_loss / len(dataloader)

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def translate(self, text):
        self.model.eval()
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=512,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    trainer = T5Trainer()
    config = MTConfig()
    train_loader = get_dataloader(DATA_ROOT, 'train', config.batch_size, max_seq_len=config.max_seq_len)
    val_loader = get_dataloader(DATA_ROOT, 'val', config.batch_size, max_seq_len=config.max_seq_len)

    trainer.train(train_loader, val_loader)
    
    # Example translation
    source_text = "Hello, how are you?"
    translation = trainer.translate(source_text)
    print(f"Source: {source_text}")
    print(f"Translation: {translation}")

if __name__ == "__main__":
    main()