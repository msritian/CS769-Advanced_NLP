import time, random, numpy as np, argparse, sys, re, os, math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score

from tokenizer import BertTokenizer
from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

TQDM_DISABLE = False

# ============= SEED & CONFIG =============
def seed_everything(seed=11711):
    """Fix random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ============= CLASSIFIER MODEL =============
class BertSentClassifier(torch.nn.Module):
    """
    Optimized BERT Sentence Classifier with:
    - Attention pooling (simple, effective)
    - Proper dropout scheduling
    - Minimal complexity for stability
    - Support for loading pre-trained MLM weights
    """
    
    def __init__(self, config):
        super(BertSentClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.config = config

        # Freeze/unfreeze BERT based on option
        if config.option == 'pretrain':
            for param in self.bert.parameters():
                param.requires_grad = False
        elif config.option == 'finetune':
            for param in self.bert.parameters():
                param.requires_grad = True

        hidden_size = 768
        
        # ========== Attention Pooling (Single Head) ==========
        # Simple but effective: learn to weight tokens
        self.attention = nn.Linear(hidden_size, 1)
        
        # ========== Feature Combination ==========
        # Combine: CLS token + attention-weighted pooling
        # Simple is better! Don't over-complicate
        combined_size = hidden_size * 2
        
        self.dense1 = nn.Linear(combined_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size // 2)
        
        # ========== Normalization ==========
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size // 2)
        
        # ========== Regularization ==========
        # Progressive dropout: higher early, lower late
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.1)
        
        # ========== Classification Head ==========
        self.classifier = nn.Linear(hidden_size // 2, self.num_labels)
        
        # ========== Activation ==========
        self.gelu = nn.GELU()
        
        # ========== Gradual Unfreezing ==========
        self.total_layers = 12
        self.current_unfrozen_layer = self.total_layers - 1

    def _unfreeze_bert_layer(self, layer_idx):
        """Unfreeze a specific BERT layer"""
        if 0 <= layer_idx < self.total_layers:
            layer = self.bert.bert_layers[layer_idx]
            for param in layer.parameters():
                param.requires_grad = True

    def unfreeze_next_layer(self):
        """Unfreeze the next layer (for gradual unfreezing)"""
        if self.current_unfrozen_layer > 0:
            self.current_unfrozen_layer -= 1
            self._unfreeze_bert_layer(self.current_unfrozen_layer)
            return True
        return False

    def forward(self, input_ids, attention_mask):
        """
        Clean forward pass with attention pooling
        """
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract hidden states
        if isinstance(outputs, dict):
            hidden_states = outputs['last_hidden_state']
        else:
            hidden_states = outputs[0]

        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # ========== 1. CLS Token Representation ==========
        cls_output = hidden_states[:, 0]  # [batch_size, hidden_size]
        
        # ========== 2. Attention Pooling ==========
        # Learn to weight which tokens are important
        attn_logits = self.attention(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        
        # Mask padding tokens (set to very negative so softmax → 0)
        attn_logits = attn_logits.masked_fill(attention_mask == 0, -1e9)
        attn_weights = F.softmax(attn_logits, dim=-1)  # [batch_size, seq_len]
        
        # Weighted average of hidden states
        weighted_output = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        # [batch_size, hidden_size]
        
        # ========== 3. Combine Representations ==========
        combined = torch.cat([cls_output, weighted_output], dim=1)  
        # [batch_size, hidden_size * 2]
        
        # ========== 4. Dense Layers with Normalization & Dropout ==========
        hidden = self.dense1(combined)  # [batch_size, hidden_size]
        hidden = self.layer_norm1(hidden)
        hidden = self.gelu(hidden)
        hidden = self.dropout1(hidden)
        
        # Second dense layer
        hidden = self.dense2(hidden)  # [batch_size, hidden_size // 2]
        hidden = self.layer_norm2(hidden)
        hidden = self.gelu(hidden)
        hidden = self.dropout2(hidden)
        
        # ========== 5. Classification ==========
        logits = self.classifier(hidden)
        
        return F.log_softmax(logits, dim=-1)

# ============= DATASET =============
class BertDataset(Dataset):
    """Custom Dataset for BERT classification"""
    
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        
        encoding = self.tokenizer(
            sents, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=self.p.max_length
        )
        
        token_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        token_type_ids = encoding['token_type_ids']
        labels = torch.LongTensor(labels)

        return token_ids, token_type_ids, attention_mask, labels, sents

    def collate_fn(self, all_data):
        all_data.sort(key=lambda x: -len(x[2]))

        batches = []
        num_batches = int(np.ceil(len(all_data) / self.p.batch_size))

        for i in range(num_batches):
            start_idx = i * self.p.batch_size
            data = all_data[start_idx: start_idx + self.p.batch_size]

            token_ids, token_type_ids, attention_mask, labels, sents = self.pad_data(data)
            batches.append({
                'token_ids': token_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
            })

        return batches

# ============= DATA LOADING =============
def create_data(filename, flag='train'):
    """Load data from file"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    num_labels = {}
    data = []

    with open(filename, 'r') as fp:
        for line in fp:
            label, org_sent = line.split(' ||| ')
            sent = org_sent.lower().strip()
            tokens = tokenizer.tokenize("[CLS] " + sent + " [SEP]")
            label = int(label.strip())
            if label not in num_labels:
                num_labels[label] = len(num_labels)
            data.append((sent, label, tokens))
    
    print(f"load {len(data)} data from {filename}")
    if flag == 'train':
        return data, len(num_labels)
    else:
        return data

# ============= EVALUATION =============
def model_eval(dataloader, model, device):
    """Evaluate model on dev/test set"""
    model.eval()
    y_true = []
    y_pred = []
    sents = []
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc='eval', disable=TQDM_DISABLE)):
            b_ids = batch[0]['token_ids'].to(device)
            b_mask = batch[0]['attention_mask'].to(device)
            b_labels = batch[0]['labels']
            b_sents = batch[0]['sents']

            logits = model(b_ids, b_mask)
            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1).flatten()

            b_labels = b_labels.flatten()
            y_true.extend(b_labels)
            y_pred.extend(preds)
            sents.extend(b_sents)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sents

# ============= CHECKPOINTING =============
def save_model(model, optimizer, args, config, filepath):
    """Save model checkpoint"""
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"💾 Saved model to {filepath}")

# ============= TRAINING =============
def train(args):
    """Main training function"""
    # ========== Device Setup ==========
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    elif args.use_gpu and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    seed_everything(args.seed)
    
    # ========== Load Data ==========
    train_data, num_labels = create_data(args.train, 'train')
    dev_data = create_data(args.dev, 'valid')

    train_dataset = BertDataset(train_data, args)
    dev_dataset = BertDataset(dev_data, args)

    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn
    )
    dev_dataloader = DataLoader(
        dev_dataset, 
        shuffle=False, 
        batch_size=args.batch_size,
        collate_fn=dev_dataset.collate_fn
    )

    # ========== Initialize Model ==========
    config = {
        'hidden_dropout_prob': args.hidden_dropout_prob,
        'num_labels': num_labels,
        'hidden_size': 768,
        'data_dir': '.',
        'option': args.option
    }

    config = SimpleNamespace(**config)
    model = BertSentClassifier(config)
    
    # 🔥 NEW: Load pre-trained MLM weights if provided
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"\n📦 Loading pre-trained MLM model from: {args.pretrained_model}")
        try:
            pretrained_state_dict = torch.load(args.pretrained_model, map_location='cpu')
            
            # Get current model state dict
            model_state_dict = model.state_dict()
            
            # Filter: only load BERT weights (not classifier head)
            pretrained_bert_dict = {}
            for k, v in pretrained_state_dict.items():
                if k in model_state_dict and k.startswith('bert.'):
                    pretrained_bert_dict[k] = v
            
            # Update model with pretrained weights
            model_state_dict.update(pretrained_bert_dict)
            model.load_state_dict(model_state_dict)
            
            print(f"✅ Successfully loaded {len(pretrained_bert_dict)} BERT parameters from pre-training")
        except Exception as e:
            print(f"⚠️  Warning: Could not load pre-trained model: {e}")
            print(f"   Continuing with randomly initialized BERT...")
    else:
        if args.pretrained_model:
            print(f"⚠️  Pre-trained model path not found: {args.pretrained_model}")
        print("ℹ️  Using randomly initialized BERT (no pre-training)")
    
    model = model.to(device)

    # ========== Setup Optimizer ==========
    lr = args.lr
    optimizer_grouped_parameters = []
    
    if args.discriminative_lr:
        # BERT layers: lower LR for earlier layers
        for i, layer in enumerate(model.bert.bert_layers):
            layer_lr = lr * (0.95 ** (model.total_layers - i - 1))
            optimizer_grouped_parameters.append({
                'params': layer.parameters(),
                'weight_decay': 0.01,
                'lr': layer_lr
            })
        
        # Embeddings: even lower LR
        optimizer_grouped_parameters.append({
            'params': model.bert.word_embedding.parameters(),
            'weight_decay': 0.01,
            'lr': lr * 0.5
        })
        
        # Classification head: highest LR
        head_params = [p for n, p in model.named_parameters() if not n.startswith('bert.')]
        if head_params:
            optimizer_grouped_parameters.append({
                'params': head_params,
                'weight_decay': 0.01,
                'lr': lr
            })
    else:
        # Standard weight decay
        decay_params = [p for n, p in model.named_parameters() 
                       if not any(nd in n for nd in ['bias', 'LayerNorm.weight'])]
        no_decay_params = [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in ['bias', 'LayerNorm.weight'])]
        
        if decay_params:
            optimizer_grouped_parameters.append({
                'params': decay_params,
                'weight_decay': 0.01
            })
        if no_decay_params:
            optimizer_grouped_parameters.append({
                'params': no_decay_params,
                'weight_decay': 0.0
            })

    if optimizer_grouped_parameters:
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    else:
        optimizer = AdamW(model.parameters(), lr=lr)

    # ========== Learning Rate Scheduler ==========
    num_training_steps = len(train_dataloader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=num_warmup_steps
    )

    # ========== Loss Function ==========
    try:
        criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)
    except TypeError:
        criterion = nn.CrossEntropyLoss(reduction='mean')

    # ========== Training State ==========
    best_dev_acc = 0
    patience_counter = 0
    best_model_states = []

    print(f"\n{'='*80}")
    print(f"Training Configuration:")
    print(f"  Option: {args.option}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {lr}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Discriminative LR: {args.discriminative_lr}")
    print(f"  Gradual Unfreeze: {args.gradual_unfreeze}")
    print(f"  Num Training Steps: {num_training_steps}")
    print(f"  Num Warmup Steps: {num_warmup_steps}")
    print(f"{'='*80}\n")

    # ========== Main Training Loop ==========
    for epoch in range(args.epochs):
        # Gradual unfreezing
        if args.gradual_unfreeze and epoch > 0 and epoch % 2 == 0:
            if model.unfreeze_next_layer():
                print(f"🔓 Unfroze BERT layer {model.current_unfrozen_layer}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.train()
        train_loss = 0
        num_batches = 0

        for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
            b_ids = batch[0]['token_ids'].to(device)
            b_mask = batch[0]['attention_mask'].to(device)
            b_labels = batch[0]['labels'].to(device)

            # Forward pass
            logits = model(b_ids, b_mask)
            loss = criterion(logits, b_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            num_batches += 1

            # Clear cache
            del logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_train_loss = train_loss / num_batches
        train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

        print(f"Epoch {epoch} | Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Dev Acc: {dev_acc:.4f} | Dev F1: {dev_f1:.4f}")

        # Save best model
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            patience_counter = 0
            save_model(model, optimizer, args, config, args.filepath)
            
            best_model_states.append(model.state_dict())
            if len(best_model_states) > 5:
                best_model_states.pop(0)
            
            print(f"✅ New best! Dev Acc: {dev_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏱️  Early stopping at epoch {epoch}")
                break

    # ========== Model Checkpoint Averaging ==========
    if len(best_model_states) > 1:
        print(f"\n📊 Averaging {len(best_model_states)} best checkpoints...")
        avg_state = {}
        for key in best_model_states[0].keys():
            try:
                avg_state[key] = torch.stack([state[key] for state in best_model_states]).mean(dim=0)
            except:
                avg_state[key] = best_model_states[-1][key]
        
        model.load_state_dict(avg_state)
        save_model(model, optimizer, args, config, args.filepath)
        print("✅ Averaged model saved!")

def test(args):
    """Test the model"""
    with torch.no_grad():
        if args.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
        elif args.use_gpu and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

        saved = torch.load(args.filepath)
        config = saved['model_config']
        model = BertSentClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model from {args.filepath}")

        dev_data = create_data(args.dev, 'valid')
        dev_dataset = BertDataset(dev_data, args)
        dev_dataloader = DataLoader(
            dev_dataset, 
            shuffle=False, 
            batch_size=args.batch_size,
            collate_fn=dev_dataset.collate_fn
        )

        test_data = create_data(args.test, 'test')
        test_dataset = BertDataset(test_data, args)
        test_dataloader = DataLoader(
            test_dataset, 
            shuffle=False, 
            batch_size=args.batch_size,
            collate_fn=test_dataset.collate_fn
        )

        dev_acc, dev_f1, dev_pred, dev_true, dev_sents = model_eval(dev_dataloader, model, device)
        test_acc, test_f1, test_pred, test_true, test_sents = model_eval(test_dataloader, model, device)

        print(f"\n{'='*80}")
        print(f"RESULTS")
        print(f"{'='*80}")
        print(f"Dev Accuracy:  {dev_acc:.4f}")
        print(f"Dev F1 Score:  {dev_f1:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"{'='*80}\n")

        with open(args.dev_out, "w+") as f:
            for s, t, p in zip(dev_sents, dev_true, dev_pred):
                f.write(f"{s} ||| {t} ||| {p}\n")

        with open(args.test_out, "w+") as f:
            for s, t, p in zip(test_sents, test_true, test_pred):
                f.write(f"{s} ||| {t} ||| {p}\n")

def get_args():
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument("--train", type=str, default="data/sst-train.txt")
    parser.add_argument("--dev", type=str, default="data/sst-dev.txt")
    parser.add_argument("--test", type=str, default="data/sst-test.txt")
    parser.add_argument("--dev_out", type=str, default="sst-dev-output.txt")
    parser.add_argument("--test_out", type=str, default="sst-test-output.txt")
    parser.add_argument("--filepath", type=str, default=None)
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="Path to pre-trained MLM model checkpoint")

    # Training arguments
    parser.add_argument("--option", type=str,
                        choices=('pretrain', 'finetune'), 
                        default="finetune")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--gradual_unfreeze", action='store_true')
    parser.add_argument("--discriminative_lr", action='store_true')
    parser.add_argument("--patience", type=int, default=4,
                        help='Early stopping patience')
    
    # Optimization arguments
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_length", type=int, default=256)

    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()
    if args.filepath is None:
        args.filepath = f'{args.option}-{args.epochs}-{args.lr}.pt'
    seed_everything(args.seed)
    train(args)
    test(args)