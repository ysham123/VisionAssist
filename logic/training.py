import torch 
import torch.nn as nn 


criterion = nn.CrossEntropyLoss(ignore_index=0)

def compute_loss(outputs, targets):
  outputs = outputs.reshape(-1, outputs.size(-1))
  targets = targets.reshape(-1)
  return criterion(outputs, targets)

import random 

def teacher_forcing_ratio(epoch, max_epochs):
  return max(0.5, 1.0 - (epoch/max_epochs))

def forward_pass(encoder, decoder, images, captions, forcing_ratio = 1.0):
  features = encoder(images)
  outputs = []
  hidden = decoder.init_hidden(features)

  input_token = captions[:,0]

  for t in range(1, captions.size(1)):
    output, hidden = decoder.step(input_token, features, hidden)
    outputs.append(output)

    if random.random() < forcing_ratio:
      input_token = captions[:, t]
    else:
      input_token = output.argmax(dim=-1)
  return torch.stack(outputs, dim = 1)

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR

# Warmup scheduler (linear increase over first 5 epochs)
def warmup_lambda(epoch):
    if epoch < 5:
        return (epoch + 1) / 5
    return 1.0

def create_optimizer(model, lr=1e-4):
    """Create optimizer for the model"""
    return Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)

def create_scheduler(optimizer, scheduler_type='warmup'):
    """Create learning rate scheduler"""
    if scheduler_type == 'warmup':
        return LambdaLR(optimizer, lr_lambda=warmup_lambda)
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    else:
        return None

import torch.nn.utils as utils

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, max_epochs):
    model.train()
    total_loss = 0
    forcing_ratio = teacher_forcing_ratio(epoch, max_epochs)
    
    for images, captions in dataloader:
        images, captions = images.to(device), captions.to(device)
        
        optimizer.zero_grad()
        outputs = forward_pass(model.encoder, model.decoder, images, captions, forcing_ratio)
        loss = compute_loss(outputs, captions[:, 1:])  # Shift targets by 1 (ignore <START>)
        
        loss.backward()
        utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Validation (no gradients)
def validate(model, val_dataloader, criterion, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for images, captions in val_dataloader:
            # Similar to train, but use inference mode (no teacher forcing)
            outputs = model.generate(images)  # Assume you have a generate method for inference
            loss = compute_loss(outputs, captions[:, 1:])
            total_loss += loss.item()
    return total_loss / len(val_dataloader)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

class CaptionDataset(Dataset):
    def __init__(self, images, captions, vocab, transform=None):
        self.images = images  # List of image tensors/paths
        self.captions = captions  # List of token ID lists
        self.vocab = vocab
        self.transform = transform or transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)  # Apply augmentation
        caption = torch.tensor([self.vocab['<START>']] + self.captions[idx] + [self.vocab['<END>']])
        return image, caption

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)
    captions_padded = torch.nn.utils.rnn.pad_sequence(list(captions), batch_first=True, padding_value=0)  # <PAD>=0
    return images, captions_padded

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

from collections import Counter

def build_vocab(captions, threshold=5):
    counter = Counter(word for cap in captions for word in cap)
    vocab = {word: idx + 4 for idx, (word, count) in enumerate(counter.items()) if count >= threshold}
    vocab.update({'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3})
    return vocab, {idx: word for word, idx in vocab.items()}

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

def evaluate_captions(references, predictions):
    """
    Compute BLEU-1 to BLEU-4 and METEOR for a list of reference and predicted captions.
    Args:
        references: list of list of tokens (ground truth)
        predictions: list of list of tokens (model output)
    Returns:
        dict with average BLEU-1, BLEU-2, BLEU-3, BLEU-4, and METEOR scores
    """
    smoothie = SmoothingFunction().method4
    bleu_scores = {'bleu1': [], 'bleu2': [], 'bleu3': [], 'bleu4': []}
    meteor_scores = []

    for ref, pred in zip(references, predictions):
        # Convert integer tokens to strings and filter out special tokens
        ref_str = [str(token) for token in ref if token not in [0, 1, 2]]  # Remove PAD, START, END
        pred_str = [str(token) for token in pred if token not in [0, 1, 2]]  # Remove PAD, START, END
        
        if len(ref_str) == 0 or len(pred_str) == 0:
            continue
            
        bleu_scores['bleu1'].append(sentence_bleu([ref_str], pred_str, weights=(1, 0, 0, 0), smoothing_function=smoothie))
        bleu_scores['bleu2'].append(sentence_bleu([ref_str], pred_str, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
        bleu_scores['bleu3'].append(sentence_bleu([ref_str], pred_str, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie))
        bleu_scores['bleu4'].append(sentence_bleu([ref_str], pred_str, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))
        meteor_scores.append(meteor_score([ref_str], pred_str))

    avg_bleu = {k: sum(v)/len(v) for k, v in bleu_scores.items()}
    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    return {**avg_bleu, 'meteor': avg_meteor}

# Optional: CIDEr (requires pycocoevalcap)
def compute_cider(gts, res):
    """
    gts: dict {image_id: [list of reference captions]}
    res: dict {image_id: [list of generated captions]}
    Returns: average CIDEr score or None if pycocoevalcap is not installed
    """
    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError:
        print("pycocoevalcap not installed. Skipping CIDEr.")
        return None
    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    return score

from nltk.translate.meteor_score import meteor_score

def compute_meteor(reference, predicted):
    return meteor_score([reference], predicted)  # Strings or tokenized lists

# For CIDEr, install pycocoevalcap and use:
# from pycocoevalcap.cider.cider import Cider
# scorer = Cider()
# score, _ = scorer.compute_score(gts, res)  # gts/res as dicts


def train_model(model, train_loader, val_loader, device, num_epochs=40):
    """
    Complete training function with three phases
    """
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = create_optimizer(model, lr=1e-4)
    scheduler = create_scheduler(optimizer, 'warmup')
    
    # Phase 1: Freeze encoder, train decoder only
    print("Phase 1: Training decoder only...")
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    for epoch in range(10):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, 10)
        if scheduler:
            scheduler.step()
        print(f"Epoch {epoch+1}/10, Loss: {train_loss:.4f}")
    
    # Phase 2: Unfreeze and joint train
    print("Phase 2: Joint training...")
    for param in model.encoder.parameters():
        param.requires_grad = True
    optimizer = create_optimizer(model, lr=1e-5)
    scheduler = create_scheduler(optimizer, 'cosine')
    
    for epoch in range(20):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, 20)
        if scheduler:
            scheduler.step()
        print(f"Epoch {epoch+1}/20, Loss: {train_loss:.4f}")
    
    # Phase 3: Refinement
    print("Phase 3: Refinement...")
    optimizer = create_optimizer(model, lr=1e-6)
    
    for epoch in range(10):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, 10)
        print(f"Epoch {epoch+1}/10, Loss: {train_loss:.4f}")
    
    return model

def beam_search(decoder, features, vocab, beam_size=3, max_len=20):
    start = torch.tensor([vocab['<START>']])
    beams = [(start, 0.0, decoder.init_hidden(features))]  # (seq, score, hidden)
    
    for _ in range(max_len):
        new_beams = []
        for seq, score, hidden in beams:
            output, hidden = decoder.step(seq[-1], features, hidden)
            probs = torch.log_softmax(output, dim=-1)
            topk_probs, topk_idx = probs.topk(beam_size)
            
            for p, idx in zip(topk_probs, topk_idx):
                new_seq = torch.cat([seq, idx.unsqueeze(0)])
                new_beams.append((new_seq, score + p.item(), hidden))
        
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
    
    best_seq = beams[0][0]
    # Note: vocab_inv should be passed as parameter or created from vocab
    return [idx.item() for idx in best_seq[1:]]  # Skip <START>

# Example usage of gradient accumulation and mixed precision
def train_with_accumulation(model, dataloader, optimizer, criterion, device, accum_steps=4):
    """Training with gradient accumulation"""
    model.train()
    total_loss = 0
    
    for batch_idx, (images, captions) in enumerate(dataloader):
        images, captions = images.to(device), captions.to(device)
        
        outputs = forward_pass(model.encoder, model.decoder, images, captions, forcing_ratio=1.0)
        loss = compute_loss(outputs, captions[:, 1:]) / accum_steps
        loss.backward()
        
        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps
    
    return total_loss / len(dataloader)

# Mixed precision training (requires CUDA)
def train_with_mixed_precision(model, dataloader, optimizer, criterion, device):
    """Training with mixed precision"""
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    
    model.train()
    total_loss = 0
    
    for images, captions in dataloader:
        images, captions = images.to(device), captions.to(device)
        
        with autocast():
            outputs = forward_pass(model.encoder, model.decoder, images, captions, forcing_ratio=1.0)
            loss = compute_loss(outputs, captions[:, 1:])
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


import torch
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True