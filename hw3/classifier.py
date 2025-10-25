# import time, random, numpy as np, argparse, sys, re, os, math
# from types import SimpleNamespace

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score

# # change it with respect to the original model
# from tokenizer import BertTokenizer
# from bert import BertModel
# from optimizer import AdamW
# from tqdm import tqdm


# TQDM_DISABLE=True
# # fix the random seed
# def seed_everything(seed=11711):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True

# # class BertSentClassifier(torch.nn.Module):
# #     def __init__(self, config):
# #         super(BertSentClassifier, self).__init__()
# #         self.num_labels = config.num_labels
# #         self.bert = BertModel.from_pretrained('bert-base-uncased')
# #         self.config = config

# #         # Set up parameters based on mode
# #         if config.option == 'pretrain':
# #             for param in self.bert.parameters():
# #                 param.requires_grad = False
# #         elif config.option == 'finetune':
# #             for param in self.bert.parameters():
# #                 param.requires_grad = True

# #         # Attention mechanism for sequence weighting
# #         self.attention = torch.nn.Linear(config.hidden_size, 1)
        
# #         # Two-stage classifier with higher capacity
# #         self.classifier1 = torch.nn.Linear(config.hidden_size * 2, config.hidden_size * 2)
# #         self.classifier2 = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)
# #         self.classifier_out = torch.nn.Linear(config.hidden_size, config.num_labels)
        
# #         # Normalization and regularization
# #         self.layer_norm1 = torch.nn.LayerNorm(config.hidden_size * 2)
# #         self.layer_norm2 = torch.nn.LayerNorm(config.hidden_size)
# #         self.batch_norm1 = torch.nn.BatchNorm1d(config.hidden_size * 2)
# #         self.batch_norm2 = torch.nn.BatchNorm1d(config.hidden_size)
        
# #         # Dropout layers with different rates
# #         self.dropout1 = torch.nn.Dropout(0.1)
# #         self.dropout2 = torch.nn.Dropout(0.15)
        
# #         # Activation functions
# #         self.gelu = torch.nn.GELU()
# #         self.tanh = torch.nn.Tanh()
        
# #         # Classifier with larger intermediate
# #         self.classifier_intermediate = torch.nn.Linear(config.hidden_size, config.hidden_size * 2)
# #         self.classifier = torch.nn.Linear(config.hidden_size * 2, config.num_labels)
        
# #         # Activation functions
# #         self.gelu = torch.nn.GELU()
# #         self.tanh = torch.nn.Tanh()

# #     def forward(self, input_ids, attention_mask):
# #         # Get BERT outputs
# #         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
# #         # Handle different possible output formats
# #         if isinstance(outputs, dict):
# #             hidden_states = outputs.get('last_hidden_state', outputs.get('hidden_states'))
# #         elif isinstance(outputs, tuple):
# #             hidden_states = outputs[0]
# #         else:
# #             hidden_states = outputs  # Assume it's the hidden states directly
            
# #         # Get pooled output - either from pooler or use CLS token
# #         pooled_output = outputs.get('pooler_output', hidden_states[:, 0])

# #         # 1. CLS token representation
# #         cls_output = hidden_states[:, 0]  # [batch_size, hidden_size]
        
# #         # 2. Weighted attention pooling over sequence
# #         attention_weights = torch.tanh(self.attention(hidden_states))
# #         attention_weights = attention_weights.squeeze(-1) * attention_mask
# #         attention_weights = F.softmax(attention_weights, dim=1)
# #         weighted_output = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)
        
# #         # 3. Mean pooling
# #         mask = attention_mask.unsqueeze(-1).float()
# #         mean_output = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        
# #         # Concatenate all representations
# #         combined = torch.cat([cls_output, weighted_output, mean_output], dim=-1)
        
# #         # 1. Get BERT outputs with proper handling
# #         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
# #         hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs['last_hidden_state']
        
# #         # 2. CLS token representation
# #         cls_output = hidden_states[:, 0]  # [batch_size, hidden_size]
        
# #         # 3. Attention-weighted pooling
# #         attention_weights = torch.tanh(self.attention(hidden_states))  # [batch_size, seq_len, 1]
# #         attention_weights = attention_weights.squeeze(-1) * attention_mask  # Apply mask
# #         attention_weights = F.softmax(attention_weights, dim=1)  # [batch_size, seq_len]
# #         weighted_output = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)  # [batch_size, hidden_size]
        
# #         # 4. Average pooling
# #         mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
# #         sum_embeddings = (hidden_states * mask_expanded).sum(1)  # [batch_size, hidden_size]
# #         avg_embeddings = sum_embeddings / mask_expanded.sum(1).clamp(min=1e-9)  # [batch_size, hidden_size]
        
# #         # Global attention over the local contexts
# #         global_weights = self.global_attention(context)  # [batch_size, seq_len, 1]
# #         global_weights = global_weights.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
# #         global_weights = F.softmax(global_weights, dim=1)
# #         global_context = torch.sum(global_weights * context, dim=1)  # [batch_size, hidden_size]
        
# #         # Get CLS token and mean pooling
# #         cls_output = hidden_states[:, 0]
# #         mean_output = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9)
        
# #         # Combine all representations
# #         combined = torch.cat([cls_output, local_context, global_context, mean_output], dim=1)  # [batch_size, hidden_size*4]
        
# #         # First dense transformation
# #         hidden = self.dropout1(combined)
# #         hidden = self.dense1(hidden)
# #         hidden = self.layer_norm2(hidden)
# #         hidden = self.batch_norm1(hidden)
# #         hidden = self.gelu(hidden)
        
# #         # Second dense transformation
# #         hidden = self.dropout2(hidden)
# #         hidden = self.dense2(hidden)
# #         hidden = self.layer_norm1(hidden)
# #         hidden = self.batch_norm2(hidden)
# #         hidden = self.gelu(hidden)
        
# #         # Classification with intermediate layer
# #         hidden = self.classifier_intermediate(hidden)
# #         hidden = self.gelu(hidden)
# #         hidden = self.dropout2(hidden)
# #         logits = self.classifier(hidden)
        
# #         return F.log_softmax(logits, dim=-1)
        
# #         # Apply log softmax for numerical stability
# #         logits = F.log_softmax(logits, dim=-1)
        
# #         return logits

# class BertSentClassifier(torch.nn.Module):
#     def __init__(self, config):
#         super(BertSentClassifier, self).__init__()
#         self.num_labels = config.num_labels
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         self.config = config

#         # Advanced attention with multi-scale features
#         self.attention = torch.nn.Linear(config.hidden_size, 1)
#         self.scale_attention = torch.nn.Linear(config.hidden_size, 1)
        
#         # Multi-scale feature combination
#         self.dense = torch.nn.Linear(config.hidden_size * 3, config.hidden_size)
#         self.intermediate = torch.nn.Linear(config.hidden_size, config.hidden_size)
        
#         # Classification head with enhanced regularization
#         self.dropout = torch.nn.Dropout(0.1)
#         self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        
#         # Layer normalization and batch normalization for stability
#         self.layer_norm = torch.nn.LayerNorm(config.hidden_size)
#         self.batch_norm = torch.nn.BatchNorm1d(config.hidden_size)
        
#         # Advanced activation functions
#         self.gelu = torch.nn.GELU()
#         self.mish = lambda x: x * torch.tanh(F.softplus(x))

#         # Set up parameters based on mode
#         if config.option == 'pretrain':
#             for param in self.bert.parameters():
#                 param.requires_grad = False
#         elif config.option == 'finetune':
#             for param in self.bert.parameters():
#                 param.requires_grad = True        # Advanced attention with multi-scale features
#         self.attention = torch.nn.Linear(config.hidden_size, 1)
#         self.scale_attention = torch.nn.Linear(config.hidden_size, 1)
        
#         # Multi-scale feature combination
#         self.dense = torch.nn.Linear(config.hidden_size * 3, config.hidden_size)
#         self.intermediate = torch.nn.Linear(config.hidden_size, config.hidden_size)
        
#         # Classification head with enhanced regularization
#         self.dropout = torch.nn.Dropout(0.1)
#         self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        
#         # Layer normalization and batch normalization for stability
#         self.layer_norm = torch.nn.LayerNorm(config.hidden_size)
#         self.batch_norm = torch.nn.BatchNorm1d(config.hidden_size)
        
#         # Advanced activation functions
#         self.gelu = torch.nn.GELU()
#         self.mish = lambda x: x * torch.tanh(F.softplus(x))
        
#         # Virtual adversarial training parameters
#         self.vat_eps = 1e-6
#         self.vat_xi = 1e-6
        
#         # Initialize unfreezing parameters
#         self.total_layers = 12  # Total number of BERT layers
#         self.current_unfrozen_layer = self.total_layers - 1  # Start with the last layer unfrozen

#     def _unfreeze_bert_layer(self, layer_idx):
#         """Unfreeze a specific BERT layer with discriminative learning rates (custom model)"""
#         if 0 <= layer_idx < self.total_layers:
#             layer = self.bert.bert_layers[layer_idx]
#             for param in layer.parameters():
#                 param.requires_grad = True
#             # Apply discriminative learning rates (lower for earlier layers)
#             base_lr = self.config.lr if hasattr(self.config, 'lr') else 2e-5
#             layer_lr = base_lr * (0.95 ** (self.total_layers - layer_idx - 1))
#             # Store layer-specific learning rate for optimizer (optional, for reference)
#             for param in layer.parameters():
#                 param.layer_lr = layer_lr
    
#     def unfreeze_next_layer(self):
#         """Unfreeze the next layer in the gradual unfreezing process"""
#         if self.current_unfrozen_layer > 0:
#             self.current_unfrozen_layer -= 1
#             self._unfreeze_bert_layer(self.current_unfrozen_layer)
#             return True
#         return False
        
#     def compute_vat_perturbation(self, hidden_states, attention_mask):
#         """Compute virtual adversarial perturbation"""
#         d = torch.randn_like(hidden_states)
#         d = F.normalize(d, dim=-1) * self.vat_xi
#         d.requires_grad_()
        
#         with torch.enable_grad():
#             logits_ori = self.forward_from_hidden(hidden_states, attention_mask)
#             logits_pert = self.forward_from_hidden(hidden_states + d, attention_mask)
#             kl_loss = F.kl_div(F.log_softmax(logits_pert, dim=1),
#                               F.softmax(logits_ori, dim=1),
#                               reduction='batchmean')
#             grad = torch.autograd.grad(kl_loss, d)[0]
            
#         r_vadv = F.normalize(grad, dim=-1) * self.vat_eps
#         return r_vadv

#     def forward_from_hidden(self, hidden_states, attention_mask):
#         """Forward pass starting from hidden states (for VAT)"""
#         # Multi-scale attention mechanisms
#         attention_weights = self.attention(hidden_states)
#         attention_weights = attention_weights.squeeze(-1)
#         attention_weights = attention_weights.masked_fill(attention_mask == 0, -1e9)
#         attention_weights = F.softmax(attention_weights, dim=1)
#         context_vector = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)
        
#         # Scale-aware attention
#         scale_weights = self.scale_attention(hidden_states)
#         scale_weights = scale_weights.squeeze(-1)
#         scale_weights = scale_weights.masked_fill(attention_mask == 0, -1e9)
#         scale_weights = F.softmax(scale_weights, dim=1)
#         scale_context = torch.bmm(scale_weights.unsqueeze(1), hidden_states).squeeze(1)
        
#         # Get CLS and combine all features
#         cls_output = hidden_states[:, 0]
#         combined = torch.cat([cls_output, context_vector, scale_context], dim=1)
        
#         # Multi-layer feature processing
#         hidden = self.dense(combined)
#         hidden = self.mish(hidden)
#         hidden = self.layer_norm(hidden)
#         hidden = self.dropout(hidden)
        
#         # Intermediate processing
#         hidden = self.intermediate(hidden)
#         hidden = self.mish(hidden)
#         hidden = self.batch_norm(hidden)
#         hidden = self.dropout(hidden)
        
#         # Classification
#         logits = self.classifier(hidden)
#         return logits
    
#     def forward(self, input_ids, attention_mask):
#         # Get BERT outputs
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
#         # Handle different output formats
#         if isinstance(outputs, dict):
#             hidden_states = outputs['last_hidden_state']
#         else:
#             hidden_states = outputs[0]

#         # Get CLS token representation
#         cls_output = hidden_states[:, 0]  # [batch_size, hidden_size]
        
#         # First attention mechanism
#         attention_weights = self.attention(hidden_states)  # [batch_size, seq_len, 1]
#         attention_weights = attention_weights.squeeze(-1)  # [batch_size, seq_len]
#         attention_weights = attention_weights.masked_fill(attention_mask == 0, -1e9)
#         attention_weights = F.softmax(attention_weights, dim=1)
#         context_vector = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)
        
#         # Scale attention mechanism
#         scale_weights = self.scale_attention(hidden_states)  # [batch_size, seq_len, 1]
#         scale_weights = scale_weights.squeeze(-1)  # [batch_size, seq_len]
#         scale_weights = scale_weights.masked_fill(attention_mask == 0, -1e9)
#         scale_weights = F.softmax(scale_weights, dim=1)
#         scale_context = torch.bmm(scale_weights.unsqueeze(1), hidden_states).squeeze(1)
        
#         # Combine all features
#         combined = torch.cat([cls_output, context_vector, scale_context], dim=1)  # [batch_size, hidden_size * 3]
        
#         # Feature processing
#         hidden = self.dense(combined)  # [batch_size, hidden_size]
#         hidden = self.mish(hidden)
#         hidden = self.layer_norm(hidden)
        
#         # Intermediate processing
#         hidden = self.intermediate(hidden)
#         hidden = self.mish(hidden)
#         hidden = self.batch_norm(hidden)
#         hidden = self.dropout(hidden)
        
#         # Classification
#         logits = self.classifier(hidden)
        
#         return F.log_softmax(logits, dim=-1)
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs['last_hidden_state']
        
#         # 1. CLS token representation
#         cls_output = hidden_states[:, 0]  # [batch_size, hidden_size]
        
#         # 2. Attention weighted representation
#         attention_weights = torch.tanh(self.attention(hidden_states))  # [batch_size, seq_len, 1]
#         attention_weights = attention_weights.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
#         attention_weights = F.softmax(attention_weights, dim=1)
#         weighted_output = torch.bmm(attention_weights.transpose(1, 2), hidden_states).squeeze(1)  # [batch_size, hidden_size]
        
#         # 3. Combine representations
#         combined = torch.cat([cls_output, weighted_output], dim=1)  # [batch_size, hidden_size*2]
        
#         # 4. First classification stage
#         hidden = self.dropout1(combined)
#         hidden = self.classifier1(hidden)
#         hidden = self.layer_norm1(hidden)
#         hidden = self.batch_norm1(hidden)
#         hidden = self.gelu(hidden)
        
#         # 5. Second classification stage
#         hidden = self.dropout2(hidden)
#         hidden = self.classifier2(hidden)
#         hidden = self.layer_norm2(hidden)
#         hidden = self.batch_norm2(hidden)
#         hidden = self.gelu(hidden)
        
#         # 6. Output layer
#         logits = self.classifier_out(hidden)
        
#         return F.log_softmax(logits, dim=-1)

# # Add VAT loss computation
# class VirtualAdversarialTraining:
#     def __init__(self, model, epsilon=1e-6, xi=10.0):
#         self.model = model
#         self.epsilon = epsilon
#         self.xi = xi

#     def generate_adversarial_perturbation(self, input_ids, attention_mask):
#         # Generate random noise
#         noise = torch.randn_like(input_ids, dtype=torch.float, requires_grad=True)
#         noise = noise / torch.norm(noise, dim=-1, keepdim=True)

#         # Forward pass with noise
#         perturbed_input = input_ids + self.xi * noise
#         logits = self.model(perturbed_input, attention_mask)
#         loss = F.cross_entropy(logits, logits.detach())

#         # Backpropagate to compute gradients
#         loss.backward()
#         noise_grad = noise.grad

#         # Normalize the gradient to create adversarial perturbation
#         perturbation = self.epsilon * noise_grad / torch.norm(noise_grad, dim=-1, keepdim=True)
#         return perturbation

#     def compute_vat_loss(self, input_ids, attention_mask):
#         perturbation = self.generate_adversarial_perturbation(input_ids, attention_mask)
#         perturbed_input = input_ids + perturbation
#         logits = self.model(perturbed_input, attention_mask)
#         return F.cross_entropy(logits, logits.detach())

# # create a custom Dataset Class to be used for the dataloader
# class BertDataset(Dataset):
#     def __init__(self, dataset, args):
#         self.dataset = dataset
#         self.p = args
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         ele = self.dataset[idx]
#         return ele

#     def pad_data(self, data):
#         sents = [x[0] for x in data]
#         labels = [x[1] for x in data]
#         # Use max_length from args
#         encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True, max_length=self.p.max_length)
#         token_ids = encoding['input_ids']
#         attention_mask = encoding['attention_mask']
#         token_type_ids = encoding['token_type_ids']
#         labels = torch.LongTensor(labels)

#         return token_ids, token_type_ids, attention_mask, labels, sents

#     def collate_fn(self, all_data):
#         all_data.sort(key=lambda x: -len(x[2]))  # sort by number of tokens

#         batches = []
#         num_batches = int(np.ceil(len(all_data) / self.p.batch_size))

#         for i in range(num_batches):
#             start_idx = i * self.p.batch_size
#             data = all_data[start_idx: start_idx + self.p.batch_size]

#             token_ids, token_type_ids, attention_mask, labels, sents = self.pad_data(data)
#             batches.append({
#                 'token_ids': token_ids,
#                 'token_type_ids': token_type_ids,
#                 'attention_mask': attention_mask,
#                 'labels': labels,
#                 'sents': sents,
#             })

#         return batches


# # create the data which is a list of (sentence, label, token for the labels)
# def create_data(filename, flag='train'):
#     # specify the tokenizer
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     num_labels = {}
#     data = []

#     with open(filename, 'r') as fp:
#         for line in fp:
#             label, org_sent = line.split(' ||| ')
#             sent = org_sent.lower().strip()
#             tokens = tokenizer.tokenize("[CLS] " + sent + " [SEP]")
#             label = int(label.strip())
#             if label not in num_labels:
#                 num_labels[label] = len(num_labels)
#             data.append((sent, label, tokens))
#     print(f"load {len(data)} data from {filename}")
#     if flag == 'train':
#         return data, len(num_labels)
#     else:
#         return data

# # perform model evaluation in terms of the accuracy and f1 score.
# def model_eval(dataloader, model, device):
#     model.eval() # switch to eval model, will turn off randomness like dropout
#     y_true = []
#     y_pred = []
#     sents = []
#     for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
#         b_ids, b_type_ids, b_mask, b_labels, b_sents = batch[0]['token_ids'], batch[0]['token_type_ids'], \
#                                                        batch[0]['attention_mask'], batch[0]['labels'], batch[0]['sents']

#         b_ids = b_ids.to(device)
#         b_mask = b_mask.to(device)

#         logits = model(b_ids, b_mask)
#         logits = logits.detach().cpu().numpy()
#         preds = np.argmax(logits, axis=1).flatten()

#         b_labels = b_labels.flatten()
#         y_true.extend(b_labels)
#         y_pred.extend(preds)
#         sents.extend(b_sents)

#     f1 = f1_score(y_true, y_pred, average='macro')
#     acc = accuracy_score(y_true, y_pred)

#     return acc, f1, y_pred, y_true, sents

# def save_model(model, optimizer, args, config, filepath):
#     save_info = {
#         'model': model.state_dict(),
#         'optim': optimizer.state_dict(),
#         'args': args,
#         'model_config': config,
#         'system_rng': random.getstate(),
#         'numpy_rng': np.random.get_state(),
#         'torch_rng': torch.random.get_rng_state(),
#     }

#     torch.save(save_info, filepath)
#     print(f"save the model to {filepath}")

# def train(args):
#     # Prefer CUDA (NVIDIA) if available, otherwise use MPS (Apple), else CPU
#     if args.use_gpu and torch.cuda.is_available():
#         device = torch.device('cuda')
#         torch.cuda.empty_cache()  # Clear GPU memory
#     elif args.use_gpu and torch.backends.mps.is_available():
#         device = torch.device('mps')
#     else:
#         device = torch.device('cpu')
    
#     # Set seeds for reproducibility
#     seed_everything(args.seed)
#     #### Load data
#     # create the data and its corresponding datasets and dataloader
#     train_data, num_labels = create_data(args.train, 'train')
#     dev_data = create_data(args.dev, 'valid')

#     train_dataset = BertDataset(train_data, args)
#     dev_dataset = BertDataset(dev_data, args)

#     train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
#                                   collate_fn=train_dataset.collate_fn)
#     dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
#                                 collate_fn=dev_dataset.collate_fn)

#     #### Init model
#     config = {'hidden_dropout_prob': args.hidden_dropout_prob,
#               'num_labels': num_labels,
#               'hidden_size': 768,
#               'data_dir': '.',
#               'option': args.option}

#     config = SimpleNamespace(**config)

#     # initialize the Senetence Classification Model
#     model = BertSentClassifier(config)
#     model = model.to(device)

#     lr = args.lr
#     # Gradual unfreezing and discriminative learning rates
#     if args.discriminative_lr:
#         # Assign different learning rates to each BERT layer (custom model)
#         optimizer_grouped_parameters = []
#         # BERT encoder layers (custom: bert_layers)
#         for i, layer in enumerate(model.bert.bert_layers):
#             layer_lr = lr * (0.95 ** (model.total_layers - i - 1))
#             optimizer_grouped_parameters.append({
#                 'params': layer.parameters(),
#                 'weight_decay': 0.01,
#                 'lr': layer_lr
#             })
#         # Embeddings
#         optimizer_grouped_parameters.append({
#             'params': model.bert.word_embedding.parameters(),
#             'weight_decay': 0.01,
#             'lr': lr * 0.5
#         })
#         # Classifier and other head params
#         head_params = [p for n, p in model.named_parameters() if not n.startswith('bert.')]
#         if head_params:
#             optimizer_grouped_parameters.append({
#                 'params': head_params,
#                 'weight_decay': 0.01,
#                 'lr': lr
#             })
#         optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
#     else:
#         # Standard optimizer
#         decay_parameters = [p for n, p in model.named_parameters() 
#                            if not any(nd in n for nd in ['bias', 'LayerNorm.weight'])]
#         no_decay_parameters = [p for n, p in model.named_parameters() 
#                               if any(nd in n for nd in ['bias', 'LayerNorm.weight'])]
#         optimizer_grouped_parameters = []
#         if decay_parameters:
#             optimizer_grouped_parameters.append({
#                 'params': decay_parameters,
#                 'weight_decay': 0.01
#             })
#         if no_decay_parameters:
#             optimizer_grouped_parameters.append({
#                 'params': no_decay_parameters,
#                 'weight_decay': 0.0
#             })
#         if not optimizer_grouped_parameters:
#             optimizer = AdamW(model.parameters(), lr=lr)
#         else:
#             optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
#     best_dev_acc = 0
#     best_test_acc = 0
#     patience = 2  # Number of epochs to wait for improvement
#     patience_counter = 0
#     best_model_state = None

#     # Create learning rate scheduler with warmup
#     num_training_steps = len(train_dataloader) * args.epochs
#     num_warmup_steps = int(num_training_steps * args.warmup_ratio)
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, 
#                                                   total_steps=num_training_steps,
#                                                   pct_start=args.warmup_ratio,
#                                                   anneal_strategy='linear')
    
#     # For model averaging
#     model_states = []
#     model_weights = []
    
#     # Create loss function with label smoothing
#     smoothing = 0.1
#     try:
#         criterion = torch.nn.CrossEntropyLoss(reduction='sum', label_smoothing=smoothing)
#     except TypeError:
#         # Fallback for older PyTorch versions
#         criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
#     ## run for the specified number of epochs
#     for epoch in range(args.epochs):
#         # Gradual unfreezing: unfreeze one more BERT layer each epoch if enabled
#         if hasattr(args, 'gradual_unfreeze') and args.gradual_unfreeze:
#             model.unfreeze_next_layer()
#         # Clear GPU memory at the start of each epoch
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         model.train()
#         train_loss = 0
#         num_batches = 0
#         for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
#             b_ids, b_type_ids, b_mask, b_labels, b_sents = batch[0]['token_ids'], batch[0]['token_type_ids'], batch[0][
#                 'attention_mask'], batch[0]['labels'], batch[0]['sents']

#             b_ids = b_ids.to(device)
#             b_mask = b_mask.to(device)
#             b_labels = b_labels.to(device)

#             # Clear gradients
#             if step % args.grad_accumulation_steps == 0:
#                 optimizer.zero_grad()

#             # Forward pass
#             logits = model(b_ids, b_mask)
#             loss = criterion(logits, b_labels.view(-1)) / args.grad_accumulation_steps

#             # Backward pass
#             loss.backward()

#             # Update weights after accumulating gradients
#             if (step + 1) % args.grad_accumulation_steps == 0:
#                 # Gradient clipping
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 optimizer.step()
#                 scheduler.step()

#             # Free up memory
#             del logits
#             torch.cuda.empty_cache()

#             train_loss += loss.item()
#             num_batches += 1

#         train_loss = train_loss / (num_batches)

#         train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
#         dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

#         if dev_acc > best_dev_acc:
#             best_dev_acc = dev_acc
#             save_model(model, optimizer, args, config, args.filepath)
#             # Store model state for averaging
#             model_states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
#             # Keep only the best 3 model states
#             if len(model_states) > 3:
#                 model_states.pop(0)

#         print(f"epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
    
#     # Average the best model states
#     if len(model_states) > 1:
#         avg_state = {}
#         for key in model_states[0].keys():
#             try:
#                 # Only average floating point parameters
#                 if model_states[0][key].dtype in [torch.float32, torch.float64]:
#                     avg_state[key] = torch.stack([state[key] for state in model_states]).mean(dim=0).to(device)
#                 else:
#                     # For non-floating point parameters (like ints), just keep the last one
#                     avg_state[key] = model_states[-1][key].to(device)
#             except:
#                 # If any error occurs, just keep the last state for this parameter
#                 avg_state[key] = model_states[-1][key].to(device)
        
#         model.load_state_dict(avg_state)
#         save_model(model, optimizer, args, config, args.filepath)
# def test(args):
#     with torch.no_grad():
#         # Prefer CUDA (NVIDIA) if available, otherwise use MPS (Apple), else CPU
#         if args.use_gpu and torch.cuda.is_available():
#             device = torch.device('cuda')
#         elif args.use_gpu and torch.backends.mps.is_available():
#             device = torch.device('mps')
#         else:
#             device = torch.device('cpu')
#         saved = torch.load(args.filepath)
#         config = saved['model_config']
#         model = BertSentClassifier(config)
#         model.load_state_dict(saved['model'])
#         model = model.to(device)
#         print(f"load model from {args.filepath}")
#         dev_data = create_data(args.dev, 'valid')
#         dev_dataset = BertDataset(dev_data, args)
#         dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

#         test_data = create_data(args.test, 'test')
#         test_dataset = BertDataset(test_data, args)
#         test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

#         dev_acc, dev_f1, dev_pred, dev_true, dev_sents = model_eval(dev_dataloader, model, device)
#         test_acc, test_f1, test_pred, test_true, test_sents = model_eval(test_dataloader, model, device)

#         with open(args.dev_out, "w+") as f:
#             print(f"dev acc :: {dev_acc :.3f}")
#             for s, t, p in zip(dev_sents, dev_true, dev_pred):
#                 f.write(f"{s} ||| {t} ||| {p}\n")

#         with open(args.test_out, "w+") as f:
#             print(f"test acc :: {test_acc :.3f}")
#             for s, t, p in zip(test_sents, test_true, test_pred):
#                 f.write(f"{s} ||| {t} ||| {p}\n")


# def get_args():
#     parser = argparse.ArgumentParser()
#     # Data arguments
#     parser.add_argument("--train", type=str, default="data/cfimdb-train.txt")
#     parser.add_argument("--dev", type=str, default="data/cfimdb-dev.txt")
#     parser.add_argument("--test", type=str, default="data/cfimdb-test.txt")
#     parser.add_argument("--dev_out", type=str, default="cfimdb-dev-output.txt")
#     parser.add_argument("--test_out", type=str, default="cfimdb-test-output.txt")
#     parser.add_argument("--filepath", type=str, default=None)

#     # Training arguments
#     parser.add_argument("--option", type=str,
#                         help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
#                         choices=('pretrain', 'finetune'), default="pretrain")
#     parser.add_argument("--seed", type=int, default=11711)
#     parser.add_argument("--epochs", type=int, default=10)
#     parser.add_argument("--use_gpu", action='store_true')
#     parser.add_argument("--gradual_unfreeze", action='store_true',
#                         help='Enable gradual unfreezing of BERT layers during fine-tuning')
#     parser.add_argument("--discriminative_lr", action='store_true',
#                         help='Enable discriminative learning rates for different BERT layers')
    
#     # Optimization arguments
#     parser.add_argument("--lr", type=float, default=2e-5,
#                         help='Initial learning rate')
#     parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU',
#                         type=int, default=8)
#     parser.add_argument("--hidden_dropout_prob", type=float, default=0.1,
#                         help='Dropout probability for hidden layers')
#     parser.add_argument("--warmup_ratio", type=float, default=0.1,
#                         help='Ratio of warmup steps to total steps')
#     parser.add_argument("--grad_accumulation_steps", type=int, default=1,
#                         help='Number of steps to accumulate gradients')
#     parser.add_argument("--weight_decay", type=float, default=0.01,
#                         help='Weight decay coefficient')
#     parser.add_argument("--max_length", type=int, default=256,
#                         help='Maximum sequence length')

#     args = parser.parse_args()
#     print(f"args: {vars(args)}")
#     return args

# if __name__ == "__main__":
#     args = get_args()
#     if args.filepath is None:
#         args.filepath = f'{args.option}-{args.epochs}-{args.lr}.pt' # save path
#     seed_everything(args.seed)  # fix the seed for reproducibility
#     train(args)
#     test(args)

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
    Enhanced BERT Sentence Classifier with:
    - Multi-head attention pooling
    - Advanced regularization (label smoothing, dropout)
    - Gradual unfreezing support
    - Discriminative learning rates
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
        
        # ========== Multi-Head Attention Pooling ==========
        # This provides richer representations than just CLS token
        self.attention_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(4)
        ])
        
        # ========== Feature Combination ==========
        # Combine: CLS + 4 attention heads + mean pooling + max pooling
        # Total: hidden_size * 7
        combined_size = hidden_size * 7
        
        self.dense1 = nn.Linear(combined_size, hidden_size * 2)
        self.dense2 = nn.Linear(hidden_size * 2, hidden_size)
        
        # ========== Normalization ==========
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # ========== Regularization ==========
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.1)
        
        # ========== Classification Head ==========
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        
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
        Clean forward pass with multiple pooling strategies
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
        
        # ========== 2. Multi-Head Attention Pooling ==========
        attention_outputs = []
        for attn_layer in self.attention_heads:
            # Calculate attention weights
            attn_logits = attn_layer(hidden_states).squeeze(-1)  # [batch_size, seq_len]
            
            # Mask padding tokens
            attn_logits = attn_logits.masked_fill(attention_mask == 0, -1e9)
            attn_weights = F.softmax(attn_logits, dim=-1)  # [batch_size, seq_len]
            
            # Weighted average of hidden states
            context = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
            attention_outputs.append(context)
        
        # ========== 3. Mean Pooling ==========
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        mean_output = (hidden_states * mask_expanded).sum(1) / mask_expanded.sum(1)  # [batch_size, hidden_size]
        
        # ========== 4. Max Pooling ==========
        hidden_states_masked = hidden_states.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
        max_output = torch.max(hidden_states_masked, dim=1)[0]  # [batch_size, hidden_size]
        
        # ========== 5. Combine All Representations ==========
        combined = torch.cat([cls_output] + attention_outputs + [mean_output, max_output], dim=1)
        # Shape: [batch_size, hidden_size * 7]
        
        # ========== 6. Dense Layers with Normalization & Dropout ==========
        hidden = self.dense1(combined)  # [batch_size, hidden_size * 2]
        hidden = self.layer_norm1(hidden)
        hidden = self.gelu(hidden)
        hidden = self.dropout1(hidden)
        
        # Second dense layer
        hidden = self.dense2(hidden)  # [batch_size, hidden_size]
        hidden = self.layer_norm2(hidden)
        hidden = self.batch_norm(hidden)
        hidden = self.gelu(hidden)
        hidden = self.dropout2(hidden)
        
        # ========== 7. Classification ==========
        logits = self.classifier(hidden)
        hidden = self.dropout3(hidden)
        
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
    print(f"save the model to {filepath}")

# ============= TRAINING =============
def train(args):
    """Main training function with advanced techniques"""
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
    model = model.to(device)

    # ========== Setup Optimizer with Discriminative Learning Rates ==========
    lr = args.lr
    optimizer_grouped_parameters = []
    
    if args.discriminative_lr:
        # BERT layers: lower LR for earlier layers (more general knowledge)
        for i, layer in enumerate(model.bert.bert_layers):
            # Discriminative LR: multiply by 0.95^(12-i-1)
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
        
        # Classification head: highest LR (task-specific)
        head_params = [p for n, p in model.named_parameters() if not n.startswith('bert.')]
        if head_params:
            optimizer_grouped_parameters.append({
                'params': head_params,
                'weight_decay': 0.01,
                'lr': lr
            })
    else:
        # Standard weight decay: decay for all except bias and LayerNorm
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
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        total_steps=num_training_steps,
        pct_start=args.warmup_ratio,
        anneal_strategy='linear',
        final_div_factor=100
    )

    # ========== Loss Function with Label Smoothing ==========
    try:
        criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)
    except TypeError:
        # Fallback for older PyTorch versions
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
    print(f"{'='*80}\n")

    # ========== Main Training Loop ==========
    for epoch in range(args.epochs):
        # Gradual unfreezing: unfreeze one layer every 2 epochs
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

            # Gradient accumulation
            if step % args.grad_accumulation_steps == 0:
                optimizer.zero_grad()

            # Forward pass
            logits = model(b_ids, b_mask)
            loss = criterion(logits, b_labels)

            # Backward pass
            loss = loss / args.grad_accumulation_steps
            loss.backward()

            # Update weights after accumulation
            if (step + 1) % args.grad_accumulation_steps == 0:
                # Gradient clipping for stability
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

        # Save best model and keep track for averaging
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            patience_counter = 0
            save_model(model, optimizer, args, config, args.filepath)
            
            # Keep best model states for averaging
            best_model_states.append(model.state_dict())
            if len(best_model_states) > 5:
                best_model_states.pop(0)
            
            print(f"✅ New best! Saved model.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏱️  Early stopping at epoch {epoch} (patience: {args.patience})")
                break

    # ========== Model Checkpoint Averaging ==========
    if len(best_model_states) > 1:
        print(f"\n📊 Averaging {len(best_model_states)} best model checkpoints...")
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

    # Training arguments
    parser.add_argument("--option", type=str,
                        choices=('pretrain', 'finetune'), 
                        default="finetune",
                        help='pretrain: freeze BERT; finetune: update BERT')
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--gradual_unfreeze", action='store_true',
                        help='Enable gradual unfreezing of BERT layers')
    parser.add_argument("--discriminative_lr", action='store_true',
                        help='Enable discriminative learning rates')
    parser.add_argument("--patience", type=int, default=3,
                        help='Early stopping patience')
    
    # Optimization arguments
    parser.add_argument("--lr", type=float, default=2e-5,
                        help='Initial learning rate')
    parser.add_argument("--batch_size", type=int, default=8,
                        help='Batch size')
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help='Warmup ratio for scheduler')
    parser.add_argument("--grad_accumulation_steps", type=int, default=2,
                        help='Gradient accumulation steps')
    parser.add_argument("--max_length", type=int, default=256,
                        help='Maximum sequence length')

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