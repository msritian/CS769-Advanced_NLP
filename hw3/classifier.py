import time, random, numpy as np, argparse, sys, re, os, math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score

# change it with respect to the original model
from tokenizer import BertTokenizer
from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm


TQDM_DISABLE=True
# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# class BertSentClassifier(torch.nn.Module):
#     def __init__(self, config):
#         super(BertSentClassifier, self).__init__()
#         self.num_labels = config.num_labels
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         self.config = config

#         # Set up parameters based on mode
#         if config.option == 'pretrain':
#             for param in self.bert.parameters():
#                 param.requires_grad = False
#         elif config.option == 'finetune':
#             for param in self.bert.parameters():
#                 param.requires_grad = True

#         # Attention mechanism for sequence weighting
#         self.attention = torch.nn.Linear(config.hidden_size, 1)
        
#         # Two-stage classifier with higher capacity
#         self.classifier1 = torch.nn.Linear(config.hidden_size * 2, config.hidden_size * 2)
#         self.classifier2 = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)
#         self.classifier_out = torch.nn.Linear(config.hidden_size, config.num_labels)
        
#         # Normalization and regularization
#         self.layer_norm1 = torch.nn.LayerNorm(config.hidden_size * 2)
#         self.layer_norm2 = torch.nn.LayerNorm(config.hidden_size)
#         self.batch_norm1 = torch.nn.BatchNorm1d(config.hidden_size * 2)
#         self.batch_norm2 = torch.nn.BatchNorm1d(config.hidden_size)
        
#         # Dropout layers with different rates
#         self.dropout1 = torch.nn.Dropout(0.1)
#         self.dropout2 = torch.nn.Dropout(0.15)
        
#         # Activation functions
#         self.gelu = torch.nn.GELU()
#         self.tanh = torch.nn.Tanh()
        
#         # Classifier with larger intermediate
#         self.classifier_intermediate = torch.nn.Linear(config.hidden_size, config.hidden_size * 2)
#         self.classifier = torch.nn.Linear(config.hidden_size * 2, config.num_labels)
        
#         # Activation functions
#         self.gelu = torch.nn.GELU()
#         self.tanh = torch.nn.Tanh()

#     def forward(self, input_ids, attention_mask):
#         # Get BERT outputs
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
#         # Handle different possible output formats
#         if isinstance(outputs, dict):
#             hidden_states = outputs.get('last_hidden_state', outputs.get('hidden_states'))
#         elif isinstance(outputs, tuple):
#             hidden_states = outputs[0]
#         else:
#             hidden_states = outputs  # Assume it's the hidden states directly
            
#         # Get pooled output - either from pooler or use CLS token
#         pooled_output = outputs.get('pooler_output', hidden_states[:, 0])

#         # 1. CLS token representation
#         cls_output = hidden_states[:, 0]  # [batch_size, hidden_size]
        
#         # 2. Weighted attention pooling over sequence
#         attention_weights = torch.tanh(self.attention(hidden_states))
#         attention_weights = attention_weights.squeeze(-1) * attention_mask
#         attention_weights = F.softmax(attention_weights, dim=1)
#         weighted_output = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)
        
#         # 3. Mean pooling
#         mask = attention_mask.unsqueeze(-1).float()
#         mean_output = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        
#         # Concatenate all representations
#         combined = torch.cat([cls_output, weighted_output, mean_output], dim=-1)
        
#         # 1. Get BERT outputs with proper handling
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs['last_hidden_state']
        
#         # 2. CLS token representation
#         cls_output = hidden_states[:, 0]  # [batch_size, hidden_size]
        
#         # 3. Attention-weighted pooling
#         attention_weights = torch.tanh(self.attention(hidden_states))  # [batch_size, seq_len, 1]
#         attention_weights = attention_weights.squeeze(-1) * attention_mask  # Apply mask
#         attention_weights = F.softmax(attention_weights, dim=1)  # [batch_size, seq_len]
#         weighted_output = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)  # [batch_size, hidden_size]
        
#         # 4. Average pooling
#         mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
#         sum_embeddings = (hidden_states * mask_expanded).sum(1)  # [batch_size, hidden_size]
#         avg_embeddings = sum_embeddings / mask_expanded.sum(1).clamp(min=1e-9)  # [batch_size, hidden_size]
        
#         # Global attention over the local contexts
#         global_weights = self.global_attention(context)  # [batch_size, seq_len, 1]
#         global_weights = global_weights.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
#         global_weights = F.softmax(global_weights, dim=1)
#         global_context = torch.sum(global_weights * context, dim=1)  # [batch_size, hidden_size]
        
#         # Get CLS token and mean pooling
#         cls_output = hidden_states[:, 0]
#         mean_output = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9)
        
#         # Combine all representations
#         combined = torch.cat([cls_output, local_context, global_context, mean_output], dim=1)  # [batch_size, hidden_size*4]
        
#         # First dense transformation
#         hidden = self.dropout1(combined)
#         hidden = self.dense1(hidden)
#         hidden = self.layer_norm2(hidden)
#         hidden = self.batch_norm1(hidden)
#         hidden = self.gelu(hidden)
        
#         # Second dense transformation
#         hidden = self.dropout2(hidden)
#         hidden = self.dense2(hidden)
#         hidden = self.layer_norm1(hidden)
#         hidden = self.batch_norm2(hidden)
#         hidden = self.gelu(hidden)
        
#         # Classification with intermediate layer
#         hidden = self.classifier_intermediate(hidden)
#         hidden = self.gelu(hidden)
#         hidden = self.dropout2(hidden)
#         logits = self.classifier(hidden)
        
#         return F.log_softmax(logits, dim=-1)
        
#         # Apply log softmax for numerical stability
#         logits = F.log_softmax(logits, dim=-1)
        
#         return logits

class BertSentClassifier(torch.nn.Module):
    def __init__(self, config):
        super(BertSentClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.config = config

        # Set up parameters based on mode
        if config.option == 'pretrain':
            for param in self.bert.parameters():
                param.requires_grad = False
        elif config.option == 'finetune':
            for param in self.bert.parameters():
                param.requires_grad = True

        # Attention mechanism for sequence weighting
        self.attention = torch.nn.Linear(config.hidden_size, 1)
        
        # Two-stage classifier with higher capacity
        self.classifier1 = torch.nn.Linear(config.hidden_size * 2, config.hidden_size * 2)
        self.classifier2 = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.classifier_out = torch.nn.Linear(config.hidden_size, config.num_labels)
        
        # Normalization and regularization
        self.layer_norm1 = torch.nn.LayerNorm(config.hidden_size * 2)
        self.layer_norm2 = torch.nn.LayerNorm(config.hidden_size)
        self.batch_norm1 = torch.nn.BatchNorm1d(config.hidden_size * 2)
        self.batch_norm2 = torch.nn.BatchNorm1d(config.hidden_size)
        
        # Dropout layers with different rates
        self.dropout1 = torch.nn.Dropout(0.1)
        self.dropout2 = torch.nn.Dropout(0.15)
        
        # Activation functions
        self.gelu = torch.nn.GELU()
        self.tanh = torch.nn.Tanh()

    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs['last_hidden_state']
        
        # 1. CLS token representation
        cls_output = hidden_states[:, 0]  # [batch_size, hidden_size]
        
        # 2. Attention weighted representation
        attention_weights = torch.tanh(self.attention(hidden_states))  # [batch_size, seq_len, 1]
        attention_weights = attention_weights.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
        attention_weights = F.softmax(attention_weights, dim=1)
        weighted_output = torch.bmm(attention_weights.transpose(1, 2), hidden_states).squeeze(1)  # [batch_size, hidden_size]
        
        # 3. Combine representations
        combined = torch.cat([cls_output, weighted_output], dim=1)  # [batch_size, hidden_size*2]
        
        # 4. First classification stage
        hidden = self.dropout1(combined)
        hidden = self.classifier1(hidden)
        hidden = self.layer_norm1(hidden)
        hidden = self.batch_norm1(hidden)
        hidden = self.gelu(hidden)
        
        # 5. Second classification stage
        hidden = self.dropout2(hidden)
        hidden = self.classifier2(hidden)
        hidden = self.layer_norm2(hidden)
        hidden = self.batch_norm2(hidden)
        hidden = self.gelu(hidden)
        
        # 6. Output layer
        logits = self.classifier_out(hidden)
        
        return F.log_softmax(logits, dim=-1)

# create a custom Dataset Class to be used for the dataloader
class BertDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ele = self.dataset[idx]
        return ele

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        # Use max_length from args
        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True, max_length=self.p.max_length)
        token_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        token_type_ids = encoding['token_type_ids']
        labels = torch.LongTensor(labels)

        return token_ids, token_type_ids, attention_mask, labels, sents

    def collate_fn(self, all_data):
        all_data.sort(key=lambda x: -len(x[2]))  # sort by number of tokens

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


# create the data which is a list of (sentence, label, token for the labels)
def create_data(filename, flag='train'):
    # specify the tokenizer
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

# perform model evaluation in terms of the accuracy and f1 score.
def model_eval(dataloader, model, device):
    model.eval() # switch to eval model, will turn off randomness like dropout
    y_true = []
    y_pred = []
    sents = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_type_ids, b_mask, b_labels, b_sents = batch[0]['token_ids'], batch[0]['token_type_ids'], \
                                                       batch[0]['attention_mask'], batch[0]['labels'], batch[0]['sents']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

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

def save_model(model, optimizer, args, config, filepath):
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
    print(f"save the model to {filepath}")

def train(args):
    # Prefer CUDA (NVIDIA) if available, otherwise use MPS (Apple), else CPU
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()  # Clear GPU memory
    elif args.use_gpu and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # Set seeds for reproducibility
    seed_everything(args.seed)
    #### Load data
    # create the data and its corresponding datasets and dataloader
    train_data, num_labels = create_data(args.train, 'train')
    dev_data = create_data(args.dev, 'valid')

    train_dataset = BertDataset(train_data, args)
    dev_dataset = BertDataset(dev_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    #### Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    # initialize the Senetence Classification Model
    model = BertSentClassifier(config)
    model = model.to(device)

    lr = args.lr
    ## specify the optimizer with weight decay differentiation
    # Parameters that will use weight decay
    decay_parameters = [p for n, p in model.named_parameters() 
                       if not any(nd in n for nd in ['bias', 'LayerNorm.weight'])]
    # Parameters that won't use weight decay
    no_decay_parameters = [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in ['bias', 'LayerNorm.weight'])]
    
    # Verify we have parameters before creating groups
    optimizer_grouped_parameters = []
    if decay_parameters:
        optimizer_grouped_parameters.append({
            'params': decay_parameters,
            'weight_decay': 0.01
        })
    if no_decay_parameters:
        optimizer_grouped_parameters.append({
            'params': no_decay_parameters,
            'weight_decay': 0.0
        })
    
    # Create optimizer groups with different learning rates
    optimizer_grouped_parameters = []

    
    if not optimizer_grouped_parameters:
        # Fallback if no parameters were found
        optimizer = AdamW(model.parameters(), lr=lr)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    best_dev_acc = 0
    best_test_acc = 0
    patience = 2  # Number of epochs to wait for improvement
    patience_counter = 0
    best_model_state = None

    # Create learning rate scheduler with warmup
    num_training_steps = len(train_dataloader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, 
                                                  total_steps=num_training_steps,
                                                  pct_start=args.warmup_ratio,
                                                  anneal_strategy='linear')
    
    # For model averaging
    model_states = []
    model_weights = []
    
    # Create loss function with label smoothing
    smoothing = 0.1
    try:
        criterion = torch.nn.CrossEntropyLoss(reduction='sum', label_smoothing=smoothing)
    except TypeError:
        # Fallback for older PyTorch versions
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    ## run for the specified number of epochs
    for epoch in range(args.epochs):
        # Clear GPU memory at the start of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model.train()
        train_loss = 0
        num_batches = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
            b_ids, b_type_ids, b_mask, b_labels, b_sents = batch[0]['token_ids'], batch[0]['token_type_ids'], batch[0][
                'attention_mask'], batch[0]['labels'], batch[0]['sents']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            # Clear gradients
            if step % args.grad_accumulation_steps == 0:
                optimizer.zero_grad()

            # Forward pass
            logits = model(b_ids, b_mask)
            loss = criterion(logits, b_labels.view(-1)) / args.grad_accumulation_steps
            
            # Backward pass
            loss.backward()

            # Update weights after accumulating gradients
            if (step + 1) % args.grad_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            # Free up memory
            del logits
            torch.cuda.empty_cache()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)
            # Store model state for averaging
            model_states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
            # Keep only the best 3 model states
            if len(model_states) > 3:
                model_states.pop(0)

        print(f"epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
    
        # Average the best model states
    if len(model_states) > 1:
        avg_state = {}
        for key in model_states[0].keys():
            try:
                # Only average floating point parameters
                if model_states[0][key].dtype in [torch.float32, torch.float64]:
                    avg_state[key] = torch.stack([state[key] for state in model_states]).mean(dim=0).to(device)
                else:
                    # For non-floating point parameters (like ints), just keep the last one
                    avg_state[key] = model_states[-1][key].to(device)
            except:
                # If any error occurs, just keep the last state for this parameter
                avg_state[key] = model_states[-1][key].to(device)
        
        model.load_state_dict(avg_state)
        save_model(model, optimizer, args, config, args.filepath)
def test(args):
    with torch.no_grad():
        # Prefer CUDA (NVIDIA) if available, otherwise use MPS (Apple), else CPU
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
        print(f"load model from {args.filepath}")
        dev_data = create_data(args.dev, 'valid')
        dev_dataset = BertDataset(dev_data, args)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

        test_data = create_data(args.test, 'test')
        test_dataset = BertDataset(test_data, args)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

        dev_acc, dev_f1, dev_pred, dev_true, dev_sents = model_eval(dev_dataloader, model, device)
        test_acc, test_f1, test_pred, test_true, test_sents = model_eval(test_dataloader, model, device)

        with open(args.dev_out, "w+") as f:
            print(f"dev acc :: {dev_acc :.3f}")
            for s, t, p in zip(dev_sents, dev_true, dev_pred):
                f.write(f"{s} ||| {t} ||| {p}\n")

        with open(args.test_out, "w+") as f:
            print(f"test acc :: {test_acc :.3f}")
            for s, t, p in zip(test_sents, test_true, test_pred):
                f.write(f"{s} ||| {t} ||| {p}\n")


def get_args():
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument("--train", type=str, default="data/cfimdb-train.txt")
    parser.add_argument("--dev", type=str, default="data/cfimdb-dev.txt")
    parser.add_argument("--test", type=str, default="data/cfimdb-test.txt")
    parser.add_argument("--dev_out", type=str, default="cfimdb-dev-output.txt")
    parser.add_argument("--test_out", type=str, default="cfimdb-test-output.txt")
    parser.add_argument("--filepath", type=str, default=None)

    # Training arguments
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action='store_true')
    
    # Optimization arguments
    parser.add_argument("--lr", type=float, default=2e-5,
                        help='Initial learning rate')
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU',
                        type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1,
                        help='Dropout probability for hidden layers')
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help='Ratio of warmup steps to total steps')
    parser.add_argument("--grad_accumulation_steps", type=int, default=1,
                        help='Number of steps to accumulate gradients')
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help='Weight decay coefficient')
    parser.add_argument("--max_length", type=int, default=256,
                        help='Maximum sequence length')

    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()
    if args.filepath is None:
        args.filepath = f'{args.option}-{args.epochs}-{args.lr}.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train(args)
    test(args)
