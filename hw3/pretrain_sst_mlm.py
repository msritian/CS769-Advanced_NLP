"""
Continued Pre-training on SST using MLM objective
This adapts BERT to SST domain before fine-tuning on labeled data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import argparse
from tqdm import tqdm
from tokenizer import BertTokenizer
from bert import BertModel
from optimizer import AdamW

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class MLMDataset(Dataset):
    """Dataset for Masked Language Modeling"""
    
    def __init__(self, texts, tokenizer, max_length=256, mask_prob=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.texts)

    def mask_tokens(self, input_ids):
        """Mask tokens for MLM"""
        labels = input_ids.clone()
        
        # Probability of masking each token
        probability_matrix = torch.full(labels.shape, self.mask_prob)
        
        # Don't mask special tokens
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels.tolist(), 
            already_has_special_tokens=True
        )
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        # Decide which tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only predict masked tokens
        
        # 80% replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids('[MASK]')
        
        # 10% replace with random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 1 / 9)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        # 10% keep original (already done, no action needed)
        
        return input_ids, labels

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0).clone()
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Mask tokens
        input_ids, labels = self.mask_tokens(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def load_texts_from_file(filename):
    """Load text sentences from file"""
    texts = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # SST format: label ||| sentence
            parts = line.strip().split(' ||| ')
            if len(parts) == 2:
                texts.append(parts[1])  # Get sentence
    return texts

def pretrain_mlm(args):
    """Run MLM pre-training"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    seed_everything(args.seed)
    
    # Load BERT
    print("Loading BERT model...")
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = model.to(device)
    model.train()
    
    # Load texts from SST/CFIMDB
    print(f"Loading texts from {args.input_file}...")
    texts = load_texts_from_file(args.input_file)
    print(f"Loaded {len(texts)} texts")
    
    # Create dataset
    dataset = MLMDataset(texts, tokenizer, max_length=args.max_length, mask_prob=0.15)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Training
    num_training_steps = len(dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1, 
        total_iters=num_training_steps // 10
    )
    
    print("\n" + "="*80)
    print("Starting MLM Pre-training")
    print("="*80 + "\n")
    
    for epoch in range(args.epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}", disable=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 🔥 FIX: Don't use output_hidden_states (not supported in custom BERT)
            # Just get the output directly
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get hidden states [batch_size, seq_len, hidden_size]
            # outputs is a dict with 'last_hidden_state' and 'pooler_output'
            if isinstance(outputs, dict):
                hidden_states = outputs['last_hidden_state']
            else:
                # If it's a tuple/list, first element is last_hidden_state
                hidden_states = outputs[0]
            
            # Simple MLM head: project to vocab
            # For proper BERT, you'd use the original MLM head
            # Here we use embeddings projection as approximation
            vocab_size = tokenizer.vocab_size
            
            # Project hidden states to vocab logits
            # This is a simplified version - real BERT has a dedicated MLM head
            mlm_logits = torch.matmul(hidden_states, model.word_embedding.weight.t())
            
            # Flatten for loss computation
            mlm_logits_flat = mlm_logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
            
            # MLM loss (only on masked positions)
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(mlm_logits_flat, labels_flat)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch}: MLM Loss = {avg_loss:.4f}\n")
    
    # Save pre-trained model
    print(f"\nSaving pre-trained model to {args.output_model}...")
    torch.save(model.state_dict(), args.output_model)
    print("✅ Pre-training complete!")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
                        help='SST/CFIMDB train file with format: label ||| sentence')
    parser.add_argument('--output_model', type=str, default='bert-mlm-pretrained.pt',
                        help='Path to save pre-trained model')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--seed', type=int, default=11711)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    pretrain_mlm(args)