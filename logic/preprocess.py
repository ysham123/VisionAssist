import json
from collections import defaultdict, Counter
import nltk
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image
import torch
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import pickle
import re

# Set paths
image_dir = '../Data/train/'
annotation_file = '../annotations/train.json'

# Configuration
MAX_CAPTION_LENGTH = 30  # Maximum caption length for padding
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
START_TOKEN = '<START>'
END_TOKEN = '<END>'

# 1. Parse VizWiz annotation JSON
def parse_vizwiz_json(annotation_file):
    """Parse VizWiz annotation JSON and return mapping from filename to all captions"""
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # The data has keys: 'info', 'images', 'annotations'
    annotations = data['annotations']
    filename_to_captions = defaultdict(list)
    
    for ann in annotations:
        # Skip rejected or precanned captions
        if ann.get('is_rejected') or ann.get('is_precanned'):
            continue
        
        image_id = ann['image_id']
        caption = ann['caption']
        
        # Convert image_id to filename format
        filename = f"VizWiz_train_{image_id:08d}.jpg"
        filename_to_captions[filename].append(caption)
    
    return filename_to_captions

# 2. Analyze dataset diversity
def analyze_diversity(captions_dict):
    """Analyze caption diversity and context"""
    print("Analyzing dataset diversity...")
    
    all_captions = []
    for captions in captions_dict.values():
        all_captions.extend(captions)
    
    # Basic statistics
    total_captions = len(all_captions)
    unique_images = len(captions_dict)
    avg_captions_per_image = total_captions / unique_images
    
    print(f"Total captions: {total_captions}")
    print(f"Unique images: {unique_images}")
    print(f"Average captions per image: {avg_captions_per_image:.2f}")
    
    # Context analysis (simple keyword-based)
    indoor_keywords = ['room', 'kitchen', 'bedroom', 'bathroom', 'office', 'indoor', 'inside']
    outdoor_keywords = ['outdoor', 'outside', 'street', 'park', 'garden', 'building', 'car']
    
    indoor_count = sum(1 for caption in all_captions 
                      if any(keyword in caption.lower() for keyword in indoor_keywords))
    outdoor_count = sum(1 for caption in all_captions 
                       if any(keyword in caption.lower() for keyword in outdoor_keywords))
    
    print(f"Indoor context captions: {indoor_count} ({indoor_count/total_captions*100:.1f}%)")
    print(f"Outdoor context captions: {outdoor_count} ({outdoor_count/total_captions*100:.1f}%)")
    
    return {
        'total_captions': total_captions,
        'unique_images': unique_images,
        'avg_captions_per_image': avg_captions_per_image,
        'indoor_ratio': indoor_count / total_captions,
        'outdoor_ratio': outdoor_count / total_captions
    }

# 3. Build vocabulary from all captions
def build_vocabulary(captions_dict, min_freq=2):
    """Build vocabulary from all captions"""
    print("Building vocabulary...")
    
    # Collect all captions
    all_captions = []
    for captions in captions_dict.values():
        all_captions.extend(captions)
    
    # Tokenize all captions
    word_freq = Counter()
    for caption in all_captions:
        tokens = nltk.word_tokenize(caption.lower())
        word_freq.update(tokens)
    
    # Build vocabulary
    vocab = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]
    word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1, START_TOKEN: 2, END_TOKEN: 3}
    
    for word, freq in word_freq.most_common():
        if freq >= min_freq:
            vocab.append(word)
            word_to_idx[word] = len(word_to_idx)
    
    print(f"Vocabulary size: {len(vocab)}")
    return vocab, word_to_idx

# 4. Process captions with padding and all captions
def process_captions(captions_dict, word_to_idx, max_length=MAX_CAPTION_LENGTH):
    """Process all captions with padding and special tokens"""
    print("Processing captions with padding...")
    
    processed_captions = []
    image_to_caption_indices = defaultdict(list)
    
    for filename, captions in tqdm(captions_dict.items(), desc="Processing captions"):
        for caption in captions:  # Process ALL captions per image
            # Tokenize
            tokens = nltk.word_tokenize(caption.lower())
            
            # Convert to indices
            indices = [word_to_idx.get(token, word_to_idx[UNK_TOKEN]) for token in tokens]
            
            # Add start and end tokens
            indices = [word_to_idx[START_TOKEN]] + indices + [word_to_idx[END_TOKEN]]
            
            # Truncate if too long
            if len(indices) > max_length:
                indices = indices[:max_length-1] + [word_to_idx[END_TOKEN]]
            
            # Pad if too short
            while len(indices) < max_length:
                indices.append(word_to_idx[PAD_TOKEN])
            
            processed_captions.append(indices)
            image_to_caption_indices[filename].append(len(processed_captions) - 1)
    
    return processed_captions, image_to_caption_indices

# 5. Stratified split based on context diversity
def stratified_split(captions_dict, test_size=0.1, val_size=0.1):
    """Split data with stratification based on context diversity"""
    print("Performing stratified split...")
    
    # Calculate diversity score for each image
    image_diversity = {}
    for filename, captions in captions_dict.items():
        # Simple diversity metric: number of unique words across all captions
        all_words = set()
        for caption in captions:
            tokens = nltk.word_tokenize(caption.lower())
            all_words.update(tokens)
        image_diversity[filename] = len(all_words)
    
    # Create diversity bins for stratification
    diversity_values = list(image_diversity.values())
    diversity_bins = np.percentile(diversity_values, [25, 50, 75])
    
    def get_diversity_bin(diversity_score):
        if diversity_score <= diversity_bins[0]:
            return 0  # Low diversity
        elif diversity_score <= diversity_bins[1]:
            return 1  # Medium-low diversity
        elif diversity_score <= diversity_bins[2]:
            return 2  # Medium-high diversity
        else:
            return 3  # High diversity
    
    # Create stratification labels
    filenames = list(captions_dict.keys())
    diversity_labels = [get_diversity_bin(image_diversity[f]) for f in filenames]
    
    # Split with stratification
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        filenames, diversity_labels, test_size=test_size, 
        stratify=diversity_labels, random_state=42
    )
    
    # Split train/val
    val_size_adjusted = val_size / (1 - test_size)
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels, test_size=val_size_adjusted,
        stratify=train_val_labels, random_state=42
    )
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    return train_files, val_files, test_files

# 6. Setup MobileNetV2 for feature extraction
def setup_model():
    """Setup MobileNetV2 for feature extraction"""
    print("Setting up MobileNetV2...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    # Remove the classifier layer to get features
    model.classifier = nn.Identity()  # type: ignore
    model.eval()
    model.to(device)
    
    return model, device

# 7. Extract features
def extract_features(image_files, model, device, image_dir):
    """Extract features from images"""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    features = []
    filenames = []
    
    for filename in tqdm(image_files, desc=f"Extracting features"):
        image_path = os.path.join(image_dir, filename)
        try:
            image = Image.open(image_path).convert('RGB')
            tensor_img = preprocess(image)
            tensor_img = tensor_img.unsqueeze(0).to(device)  # type: ignore
            
            with torch.no_grad():
                feature = model(tensor_img)
                features.append(feature.cpu().numpy().flatten())
                filenames.append(filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    return np.array(features), np.array(filenames)

# Main preprocessing pipeline
def main():
    print("Starting enhanced preprocessing pipeline...")
    
    # 1. Parse annotations
    print("Loading annotations...")
    filename_to_captions = parse_vizwiz_json(annotation_file)
    print(f"Processing {sum(len(captions) for captions in filename_to_captions.values())} annotations...")
    
    # 2. Filter images that exist
    print("Filtering images...")
    valid_images = []
    for filename in tqdm(list(filename_to_captions.keys()), desc="Checking image files"):
        image_path = os.path.join(image_dir, filename)
        if os.path.exists(image_path) and filename_to_captions[filename]:
            valid_images.append(filename)
    
    # Filter captions dict to only valid images
    filename_to_captions = {f: filename_to_captions[f] for f in valid_images}
    print(f"Found {len(valid_images)} valid images")
    
    # 3. Analyze diversity
    diversity_stats = analyze_diversity(filename_to_captions)
    
    # 4. Build vocabulary
    vocab, word_to_idx = build_vocabulary(filename_to_captions)
    
    # 5. Process captions with padding
    all_captions, image_to_caption_indices = process_captions(filename_to_captions, word_to_idx)
    
    # 6. Stratified split
    train_files, val_files, test_files = stratified_split(filename_to_captions)
    
    # 7. Setup model
    model, device = setup_model()
    
    # 8. Extract features for each split
    print("Extracting features for train split...")
    train_features, train_filenames = extract_features(train_files, model, device, image_dir)
    
    print("Extracting features for validation split...")
    val_features, val_filenames = extract_features(val_files, model, device, image_dir)
    
    print("Extracting features for test split...")
    test_features, test_filenames = extract_features(test_files, model, device, image_dir)
    
    # 9. Prepare caption data for each split
    def get_split_captions(split_files, all_captions, image_to_caption_indices):
        split_captions = []
        for filename in split_files:
            if filename in image_to_caption_indices:
                for idx in image_to_caption_indices[filename]:
                    split_captions.append(all_captions[idx])
        return split_captions
    
    train_captions = get_split_captions(train_files, all_captions, image_to_caption_indices)
    val_captions = get_split_captions(val_files, all_captions, image_to_caption_indices)
    test_captions = get_split_captions(test_files, all_captions, image_to_caption_indices)
    
    # 10. Save processed data
    print("Saving processed data...")
    output_dir = 'processed_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save features
    np.save(os.path.join(output_dir, 'train_features.npy'), train_features)
    np.save(os.path.join(output_dir, 'val_features.npy'), val_features)
    np.save(os.path.join(output_dir, 'test_features.npy'), test_features)
    
    # Save filenames
    np.save(os.path.join(output_dir, 'train_filenames.npy'), train_filenames)
    np.save(os.path.join(output_dir, 'val_filenames.npy'), val_filenames)
    np.save(os.path.join(output_dir, 'test_filenames.npy'), test_filenames)
    
    # Save captions (now padded)
    with open(os.path.join(output_dir, 'train_captions.pkl'), 'wb') as f:
        pickle.dump(train_captions, f)
    with open(os.path.join(output_dir, 'val_captions.pkl'), 'wb') as f:
        pickle.dump(val_captions, f)
    with open(os.path.join(output_dir, 'test_captions.pkl'), 'wb') as f:
        pickle.dump(test_captions, f)
    
    # Save vocabulary
    np.save(os.path.join(output_dir, 'vocab.npy'), vocab)
    with open(os.path.join(output_dir, 'word_to_idx.pkl'), 'wb') as f:
        pickle.dump(word_to_idx, f)
    
    # Save diversity stats
    with open(os.path.join(output_dir, 'diversity_stats.pkl'), 'wb') as f:
        pickle.dump(diversity_stats, f)
    
    print("Enhanced preprocessing completed!")
    print(f"Saved processed data to {output_dir}/")
    print(f"Train samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}")
    print(f"Total captions processed: {len(all_captions)}")
    print(f"Average captions per image: {diversity_stats['avg_captions_per_image']:.2f}")

if __name__ == "__main__":
    main()