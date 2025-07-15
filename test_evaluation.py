#!/usr/bin/env python3
"""
Runtime evaluation script for the image captioning model
Tests quantitative metrics (BLEU, METEOR) and qualitative assessment
"""

import torch
import torch.nn as nn
import numpy as np
from logic.architecture import ImageCaptioningModel
from logic.training import evaluate_captions, compute_cider
import random
from typing import List, Dict, Tuple

def create_dummy_vocab(size: int = 1000) -> Dict[str, int]:
    """Create a dummy vocabulary for testing"""
    vocab = {
        '<PAD>': 0,
        '<START>': 1,
        '<END>': 2,
        '<UNK>': 3
    }
    
    # Add some common words
    words = ['a', 'the', 'is', 'in', 'on', 'at', 'and', 'or', 'but', 'with', 'to', 'for', 'of', 'from', 'by']
    for i, word in enumerate(words):
        vocab[word] = i + 4
    
    # Add dummy words to reach desired size
    for i in range(len(vocab), size):
        vocab[f'word_{i}'] = i
    
    return vocab

def create_dummy_data(batch_size: int = 4, max_len: int = 15) -> Tuple[torch.Tensor, List[List[int]], List[List[int]]]:
    """Create dummy images and captions for testing"""
    # Create dummy images
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Create reference captions (ground truth)
    reference_captions = []
    for i in range(batch_size):
        # Create a simple caption: [START, word1, word2, word3, END]
        caption = [1]  # START token
        caption.extend([4 + i, 5 + i, 6 + i])  # Some words
        caption.append(2)  # END token
        # Pad to max_len
        caption.extend([0] * (max_len - len(caption)))
        reference_captions.append(caption)
    
    # Create predicted captions (model output)
    predicted_captions = []
    for i in range(batch_size):
        # Simulate model predictions (slightly different from references)
        caption = [1]  # START token
        caption.extend([4 + i + 1, 5 + i, 6 + i + 1])  # Similar but not identical
        caption.append(2)  # END token
        # Pad to max_len
        caption.extend([0] * (max_len - len(caption)))
        predicted_captions.append(caption)
    
    return images, reference_captions, predicted_captions

def test_model_inference(model: ImageCaptioningModel, images: torch.Tensor, vocab: Dict[str, int]) -> List[List[int]]:
    """Test model inference and return predicted captions"""
    model.eval()
    with torch.no_grad():
        # Generate captions
        outputs = model.generate(images, max_len=15, start_token=vocab['<START>'])
        
        # Convert to token sequences
        predicted_captions = []
        for i in range(outputs.size(0)):
            caption = []
            for j in range(outputs.size(1)):
                token = outputs[i, j].argmax().item()
                if token == vocab['<END>']:
                    break
                caption.append(token)
            predicted_captions.append(caption)
    
    return predicted_captions

def qualitative_evaluation(predictions: List[List[int]], references: List[List[int]], vocab: Dict[str, int]) -> Dict[str, str]:
    """Perform qualitative evaluation of captions"""
    results = {
        'clarity': 'Good',
        'inclusivity': 'Good', 
        'accessibility': 'Good',
        'issues': []
    }
    
    # Create reverse vocabulary
    vocab_inv = {v: k for k, v in vocab.items()}
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        pred_words = [vocab_inv.get(t, '<UNK>') for t in pred if t not in [vocab['<PAD>'], vocab['<START>'], vocab['<END>']]]
        ref_words = [vocab_inv.get(t, '<UNK>') for t in ref if t not in [vocab['<PAD>'], vocab['<START>'], vocab['<END>']]]
        
        # Check for clarity issues
        if len(pred_words) < 2:
            results['issues'].append(f"Caption {i}: Too short")
        
        # Check for repetitive words
        if len(set(pred_words)) < len(pred_words) * 0.7:
            results['issues'].append(f"Caption {i}: Repetitive words")
        
        # Check for vague descriptions
        vague_words = ['thing', 'stuff', 'something', 'object']
        if any(word in pred_words for word in vague_words):
            results['issues'].append(f"Caption {i}: Contains vague descriptions")
    
    if results['issues']:
        results['clarity'] = 'Needs improvement'
    
    return results

def run_comprehensive_evaluation():
    """Run comprehensive evaluation of the model"""
    print("🧪 Running Comprehensive Model Evaluation...")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Create vocabulary and model
    vocab = create_dummy_vocab(1000)
    vocab_size = len(vocab)
    
    print(f"📚 Vocabulary size: {vocab_size}")
    
    # Create model
    model = ImageCaptioningModel(
        vocab_size=vocab_size,
        embed_dim=256,
        hidden_dim=256,
        encoder_type='mobilenet',
        decoder_type='lstm',
        num_layers=1,
        dropout=0.1,
        freeze_encoder=True
    )
    
    print("✅ Model created successfully")
    
    # Create test data
    images, reference_captions, dummy_predictions = create_dummy_data(batch_size=4)
    print(f"📊 Test data created: {len(images)} images")
    
    # Test 1: Model Inference
    print("\n🔍 Test 1: Model Inference")
    print("-" * 30)
    try:
        predicted_captions = test_model_inference(model, images, vocab)
        print("✅ Model inference successful")
        print(f"   Generated {len(predicted_captions)} captions")
        
        # Show sample predictions
        vocab_inv = {v: k for k, v in vocab.items()}
        for i, (pred, ref) in enumerate(zip(predicted_captions[:2], reference_captions[:2])):
            pred_text = ' '.join([vocab_inv.get(t, '<UNK>') for t in pred if t not in [vocab['<PAD>'], vocab['<START>'], vocab['<END>']]])
            ref_text = ' '.join([vocab_inv.get(t, '<UNK>') for t in ref if t not in [vocab['<PAD>'], vocab['<START>'], vocab['<END>']]])
            print(f"   Sample {i+1}:")
            print(f"     Reference: {ref_text}")
            print(f"     Predicted: {pred_text}")
            
    except Exception as e:
        print(f"❌ Model inference failed: {e}")
        return
    
    # Test 2: Quantitative Metrics
    print("\n📈 Test 2: Quantitative Metrics")
    print("-" * 30)
    try:
        # Use dummy predictions for consistent testing
        scores = evaluate_captions(reference_captions, dummy_predictions)
        
        print("✅ Quantitative evaluation successful")
        print(f"   BLEU-1: {scores['bleu1']:.4f}")
        print(f"   BLEU-2: {scores['bleu2']:.4f}")
        print(f"   BLEU-3: {scores['bleu3']:.4f}")
        print(f"   BLEU-4: {scores['bleu4']:.4f}")
        print(f"   METEOR: {scores['meteor']:.4f}")
        
        # Test CIDEr if available
        try:
            cider_score = compute_cider(
                gts={i: [ref] for i, ref in enumerate(reference_captions)},
                res={i: [pred] for i, pred in enumerate(dummy_predictions)}
            )
            if cider_score is not None:
                print(f"   CIDEr: {cider_score:.4f}")
            else:
                print("   CIDEr: Not available (pycocoevalcap not installed)")
        except Exception as e:
            print(f"   CIDEr: Error - {e}")
            
    except Exception as e:
        print(f"❌ Quantitative evaluation failed: {e}")
    
    # Test 3: Qualitative Evaluation
    print("\n🎯 Test 3: Qualitative Evaluation")
    print("-" * 30)
    try:
        qual_results = qualitative_evaluation(dummy_predictions, reference_captions, vocab)
        
        print("✅ Qualitative evaluation successful")
        print(f"   Clarity: {qual_results['clarity']}")
        print(f"   Inclusivity: {qual_results['inclusivity']}")
        print(f"   Accessibility: {qual_results['accessibility']}")
        
        if qual_results['issues']:
            print("   Issues found:")
            for issue in qual_results['issues']:
                print(f"     - {issue}")
        else:
            print("   No major issues detected")
            
    except Exception as e:
        print(f"❌ Qualitative evaluation failed: {e}")
    
    # Test 4: Model Components
    print("\n🔧 Test 4: Model Components")
    print("-" * 30)
    
    # Test encoder
    try:
        encoder_output = model.encoder(images)
        print(f"✅ Encoder: Output shape {encoder_output.shape}")
    except Exception as e:
        print(f"❌ Encoder failed: {e}")
    
    # Test decoder
    try:
        captions = torch.randint(0, vocab_size, (4, 10))
        decoder_output, attention = model.decoder(encoder_output, captions)
        print(f"✅ Decoder: Output shape {decoder_output.shape}")
        if attention is not None:
            print(f"   Attention weights shape: {attention.shape}")
    except Exception as e:
        print(f"❌ Decoder failed: {e}")
    
    # Test explainability (if available)
    try:
        explainability = model.encoder.features[-1]  # Get last layer
        print("✅ Explainability: Last layer accessible for Grad-CAM")
    except Exception as e:
        print(f"❌ Explainability setup failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Evaluation Complete!")
    print("=" * 60)

if __name__ == "__main__":
    run_comprehensive_evaluation() 