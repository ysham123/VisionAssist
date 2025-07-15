#!/usr/bin/env python3
"""
Test script to verify training components work correctly
"""

import torch
import torch.nn as nn
from logic.architecture import ImageCaptioningModel
from logic.training import forward_pass, compute_loss, teacher_forcing_ratio
import random

def test_training_components():
    """Test the training components with a small model"""
    print("🧪 Testing Training Components...")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # Create a small model for testing
    vocab_size = 1000
    model = ImageCaptioningModel(
        vocab_size=vocab_size,
        embed_dim=256,
        hidden_dim=256,
        encoder_type='mobilenet',
        decoder_type='lstm',
        num_layers=1,
        dropout=0.1,
        freeze_encoder=False  # Don't freeze for testing
    )
    
    # Test data
    batch_size = 2
    max_len = 10
    device = torch.device('cpu')
    model.to(device)
    
    # Create dummy data
    images = torch.randn(batch_size, 3, 224, 224)
    captions = torch.randint(1, vocab_size, (batch_size, max_len))
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test 1: Forward pass through the full model
    print("\n🧪 Test 1: Full model forward pass...")
    try:
        outputs, attention_weights = model(images, captions)
        print(f"✅ Full model output shape: {outputs.shape}")
        print(f"✅ Attention weights shape: {attention_weights.shape if attention_weights is not None else 'None'}")
    except Exception as e:
        print(f"❌ Full model forward pass failed: {e}")
        return False
    
    # Test 2: Step-wise forward pass (for training)
    print("\n🧪 Test 2: Step-wise forward pass...")
    try:
        outputs = forward_pass(model.encoder, model.decoder, images, captions, forcing_ratio=1.0)
        print(f"✅ Step-wise output shape: {outputs.shape}")
    except Exception as e:
        print(f"❌ Step-wise forward pass failed: {e}")
        return False
    
    # Test 3: Loss computation
    print("\n🧪 Test 3: Loss computation...")
    try:
        loss = compute_loss(outputs, captions[:, 1:])
        print(f"✅ Loss value: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        return False
    
    # Test 4: Teacher forcing ratio
    print("\n🧪 Test 4: Teacher forcing ratio...")
    try:
        ratio = teacher_forcing_ratio(epoch=5, max_epochs=20)
        print(f"✅ Teacher forcing ratio: {ratio:.3f}")
    except Exception as e:
        print(f"❌ Teacher forcing ratio failed: {e}")
        return False
    
    # Test 5: Model generation
    print("\n🧪 Test 5: Model generation...")
    try:
        generated_outputs = model.generate(images, max_len=5, start_token=1)
        print(f"✅ Generated output shape: {generated_outputs.shape}")
    except Exception as e:
        print(f"❌ Model generation failed: {e}")
        return False
    
    # Test 6: Backward pass
    print("\n🧪 Test 6: Backward pass...")
    try:
        loss.backward()
        print("✅ Backward pass successful")
        
        # Check gradients
        total_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        print(f"✅ Total gradient norm: {total_grad_norm:.4f}")
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return False
    
    print("\n🎉 All training component tests passed!")
    return True

if __name__ == "__main__":
    success = test_training_components()
    if success:
        print("\n✅ Training components are ready for use!")
    else:
        print("\n❌ Some training components need fixes.") 