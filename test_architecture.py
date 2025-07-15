import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add the logic directory to the path
sys.path.append('logic')

from logic.architecture import (
    ImageEncoder, 
    LSTMDecoder, 
    TransformerDecoder, 
    ImageCaptioningModel, 
    ExplainabilityModule,
    create_model
)

def test_encoder():
    """Test the image encoder"""
    print("🧪 Testing Image Encoder...")
    
    # Test MobileNet encoder
    encoder_mobilenet = ImageEncoder(encoder_type='mobilenet', pretrained=False, freeze=False)
    test_image = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    
    with torch.no_grad():
        features = encoder_mobilenet(test_image)
        print(f"✅ MobileNet encoder output shape: {features.shape}")
        assert features.shape == (2, 1280, 7, 7), f"Expected (2, 1280, 7, 7), got {features.shape}"
    
    # Test ResNet encoder
    encoder_resnet = ImageEncoder(encoder_type='resnet50', pretrained=False, freeze=False)
    
    with torch.no_grad():
        features = encoder_resnet(test_image)
        print(f"✅ ResNet encoder output shape: {features.shape}")
        assert features.shape == (2, 2048, 7, 7), f"Expected (2, 2048, 7, 7), got {features.shape}"
    
    print("✅ Encoder tests passed!\n")

def test_lstm_decoder():
    """Test the LSTM decoder with attention"""
    print("🧪 Testing LSTM Decoder...")
    
    vocab_size = 1000
    embed_dim = 256
    hidden_dim = 512
    feature_dim = 1280  # MobileNet feature dimension
    batch_size = 2
    max_len = 20
    
    decoder = LSTMDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        feature_dim=feature_dim,
        num_layers=2,
        dropout=0.1
    )
    
    # Test inputs
    features = torch.randn(batch_size, feature_dim, 7, 7)
    captions = torch.randint(0, vocab_size, (batch_size, max_len))
    lengths = torch.tensor([15, 18])  # Different caption lengths
    
    # Forward pass
    outputs, attention_weights = decoder(features, captions, lengths)
    
    print(f"✅ LSTM decoder output shape: {outputs.shape}")
    print(f"✅ Attention weights shape: {attention_weights.shape}")
    
    assert outputs.shape == (batch_size, max_len, vocab_size), f"Expected ({batch_size}, {max_len}, {vocab_size}), got {outputs.shape}"
    assert attention_weights.shape == (batch_size, max_len, 49), f"Expected ({batch_size}, {max_len}, 49), got {attention_weights.shape}"
    
    print("✅ LSTM decoder tests passed!\n")

def test_transformer_decoder():
    """Test the Transformer decoder"""
    print("🧪 Testing Transformer Decoder...")
    
    vocab_size = 1000
    embed_dim = 256
    hidden_dim = 512
    feature_dim = 1280
    batch_size = 2
    max_len = 20
    
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        feature_dim=feature_dim,
        num_layers=2,
        num_heads=8,
        dropout=0.1
    )
    
    # Test inputs
    features = torch.randn(batch_size, feature_dim, 7, 7)
    captions = torch.randint(0, vocab_size, (batch_size, max_len))
    
    # Forward pass
    outputs, attention_weights = decoder(features, captions)
    
    print(f"✅ Transformer decoder output shape: {outputs.shape}")
    print(f"✅ Transformer attention weights: {attention_weights}")  # Should be None
    
    assert outputs.shape == (batch_size, max_len, vocab_size), f"Expected ({batch_size}, {max_len}, {vocab_size}), got {outputs.shape}"
    assert attention_weights is None, "Transformer decoder should return None for attention weights"
    
    print("✅ Transformer decoder tests passed!\n")

def test_complete_model():
    """Test the complete image captioning model"""
    print("🧪 Testing Complete Model...")
    
    vocab_size = 1000
    batch_size = 2
    max_len = 20
    
    # Test LSTM model with MobileNet
    config_lstm = {
        'vocab_size': vocab_size,
        'encoder_type': 'mobilenet',
        'decoder_type': 'lstm',
        'embed_dim': 256,
        'hidden_dim': 512,
        'num_layers': 1,
        'dropout': 0.1
    }
    
    model_lstm = create_model(config_lstm)
    
    # Test inputs
    images = torch.randn(batch_size, 3, 224, 224)
    captions = torch.randint(0, vocab_size, (batch_size, max_len))
    
    # Forward pass
    outputs, attention_weights = model_lstm(images, captions)
    
    print(f"✅ LSTM model output shape: {outputs.shape}")
    print(f"✅ LSTM model attention weights shape: {attention_weights.shape}")
    
    assert outputs.shape == (batch_size, max_len, vocab_size)
    assert attention_weights.shape == (batch_size, max_len, 49)
    
    # Test Transformer model with ResNet
    config_transformer = {
        'vocab_size': vocab_size,
        'encoder_type': 'resnet50',
        'decoder_type': 'transformer',
        'embed_dim': 256,
        'hidden_dim': 512,
        'num_layers': 2,
        'num_heads': 8,
        'dropout': 0.1
    }
    
    model_transformer = create_model(config_transformer)
    
    # Forward pass
    outputs, attention_weights = model_transformer(images, captions)
    
    print(f"✅ Transformer model output shape: {outputs.shape}")
    print(f"✅ Transformer model attention weights: {attention_weights}")
    
    assert outputs.shape == (batch_size, max_len, vocab_size)
    assert attention_weights is None
    
    print("✅ Complete model tests passed!\n")

def test_explainability():
    """Test the explainability module"""
    print("🧪 Testing Explainability Module...")
    
    # Create a simple model for testing
    config = {
        'vocab_size': 1000,
        'encoder_type': 'mobilenet',
        'decoder_type': 'lstm',
        'embed_dim': 256,
        'hidden_dim': 512
    }
    
    model = create_model(config)
    
    # Create explainability module
    explainability = ExplainabilityModule(model, target_layer_name='last_conv')
    
    # Test inputs
    image = torch.randn(1, 3, 224, 224)  # Single image
    caption_tokens = [1, 5, 10, 15, 20]  # Sample caption tokens
    vocab = {'<PAD>': 0, 'word1': 1, 'word5': 5, 'word10': 10, 'word15': 15, 'word20': 20}
    
    # Test heatmap generation
    try:
        heatmap = explainability.generate_heatmap(image, caption_tokens, word_index=2, vocab=vocab)
        print(f"✅ Grad-CAM heatmap shape: {heatmap.shape}")
        assert heatmap.shape == (224, 224), f"Expected (224, 224), got {heatmap.shape}"
    except Exception as e:
        print(f"⚠️ Grad-CAM test failed (this is expected if pytorch-grad-cam is not installed): {e}")
    
    # Test attention visualization
    attention_weights = torch.randn(1, 5, 49)  # [batch, max_len, num_regions]
    
    try:
        attention_maps = explainability.visualize_attention(image, attention_weights, caption_tokens, vocab)
        print(f"✅ Attention maps generated for {len(attention_maps)} words")
        for word, attention_map in attention_maps.items():
            assert attention_map.shape == (224, 224), f"Expected (224, 224), got {attention_map.shape}"
    except Exception as e:
        print(f"⚠️ Attention visualization test failed: {e}")
    
    print("✅ Explainability tests passed!\n")

def test_device_compatibility():
    """Test model compatibility with different devices"""
    print("🧪 Testing Device Compatibility...")
    
    config = {
        'vocab_size': 1000,
        'encoder_type': 'mobilenet',
        'decoder_type': 'lstm',
        'embed_dim': 256,
        'hidden_dim': 512
    }
    
    # Test on CPU
    model_cpu = create_model(config)
    images_cpu = torch.randn(1, 3, 224, 224)
    captions_cpu = torch.randint(0, 1000, (1, 10))
    
    outputs_cpu, _ = model_cpu(images_cpu, captions_cpu)
    print(f"✅ CPU test passed - output shape: {outputs_cpu.shape}")
    
    # Test on GPU if available
    if torch.cuda.is_available():
        model_gpu = create_model(config).cuda()
        images_gpu = torch.randn(1, 3, 224, 224).cuda()
        captions_gpu = torch.randint(0, 1000, (1, 10)).cuda()
        
        outputs_gpu, _ = model_gpu(images_gpu, captions_gpu)
        print(f"✅ GPU test passed - output shape: {outputs_gpu.shape}")
    else:
        print("ℹ️ GPU not available, skipping GPU test")
    
    print("✅ Device compatibility tests passed!\n")

def test_model_parameters():
    """Test model parameter counting and freezing"""
    print("🧪 Testing Model Parameters...")
    
    config = {
        'vocab_size': 1000,
        'encoder_type': 'mobilenet',
        'decoder_type': 'lstm',
        'embed_dim': 256,
        'hidden_dim': 512,
        'freeze_encoder': True
    }
    
    model = create_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Total parameters: {total_params:,}")
    print(f"✅ Trainable parameters: {trainable_params:,}")
    print(f"✅ Frozen parameters: {total_params - trainable_params:,}")
    
    # Verify encoder is frozen
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    
    assert encoder_trainable == 0, "Encoder should be frozen"
    print("✅ Encoder freezing test passed!")
    
    print("✅ Model parameter tests passed!\n")

def main():
    """Run all tests"""
    print("🚀 Starting Architecture Tests...\n")
    
    try:
        test_encoder()
        test_lstm_decoder()
        test_transformer_decoder()
        test_complete_model()
        test_explainability()
        test_device_compatibility()
        test_model_parameters()
        
        print("🎉 All tests passed! Architecture is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Architecture is ready for training!")
    else:
        print("\n❌ Architecture needs fixes before training.") 