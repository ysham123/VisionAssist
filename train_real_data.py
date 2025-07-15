#!/usr/bin/env python3
"""
Train the image captioning model end-to-end on real VizWiz data using raw images and tokenized captions.
"""
import os
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from logic.architecture import ImageCaptioningModel
from logic.training import CaptionDataset, collate_fn, train_model

from torchvision import transforms

# 1. Load processed data
processed_dir = os.path.join('logic', 'processed_data')

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    print("Loading processed data...")
    train_filenames = np.load(os.path.join(processed_dir, 'train_filenames.npy'))
    val_filenames = np.load(os.path.join(processed_dir, 'val_filenames.npy'))
    train_captions = load_pickle(os.path.join(processed_dir, 'train_captions.pkl'))
    val_captions = load_pickle(os.path.join(processed_dir, 'val_captions.pkl'))
    vocab = np.load(os.path.join(processed_dir, 'vocab.npy'), allow_pickle=True).tolist()
    word_to_idx = load_pickle(os.path.join(processed_dir, 'word_to_idx.pkl'))

    print(f"Train samples: {len(train_filenames)} | Val samples: {len(val_filenames)}")
    print(f"Vocab size: {len(vocab)}")

    # 2. Set up image transforms
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Create datasets
    train_image_dir = os.path.join('Data', 'train')
    val_image_dir = os.path.join('Data', 'val')
    train_image_paths = [os.path.join(train_image_dir, fname) for fname in train_filenames]
    val_image_paths = [os.path.join(val_image_dir, fname) for fname in val_filenames]

    train_dataset = CaptionDataset(train_image_paths, train_captions, word_to_idx, transform=image_transform)
    val_dataset = CaptionDataset(val_image_paths, val_captions, word_to_idx, transform=image_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 4. Create model
    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        embed_dim=256,
        hidden_dim=256,
        encoder_type='mobilenet',
        decoder_type='lstm',
        num_layers=1,
        dropout=0.1,
        freeze_encoder=True
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 5. Train
    print("Starting end-to-end training...")
    train_model(model, train_loader, val_loader, device, num_epochs=40)
    print("Training complete!")

if __name__ == "__main__":
    main() 