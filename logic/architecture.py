import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertTokenizer, BertModel
import numpy as np
try:
    from pytorch_grad_cam import GradCAM # type: ignore
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget # type: ignore
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False
    print("Warning: pytorch-grad-cam not available. Grad-CAM functionality will be disabled.")
from typing import Optional, Tuple, Dict, Any
import os

class ImageEncoder(nn.Module):
    """Modular image encoder supporting MobileNet and ResNet"""
    
    def __init__(self, encoder_type: str = 'mobilenet', pretrained: bool = True, freeze: bool = True):
        super().__init__()
        self.encoder_type = encoder_type
        
        if encoder_type == 'mobilenet':
            self.model = models.mobilenet_v2(pretrained=pretrained)
            # Remove classifier, keep feature extractor
            self.features = self.model.features
            self.feature_dim = 1280  # MobileNetV2 feature dimension
            self.spatial_size = 7    # 7x7 spatial features
            
        elif encoder_type == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            # Remove classifier, keep layers up to avgpool
            self.features = nn.Sequential(*list(self.model.children())[:-2])
            self.feature_dim = 2048  # ResNet50 feature dimension
            self.spatial_size = 7    # 7x7 spatial features
            
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        # Freeze encoder weights if requested
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
            self.features.eval()
            # Also freeze the model parameters
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: Input images [batch_size, 3, 224, 224]
        Returns:
            features: [batch_size, feature_dim, spatial_size, spatial_size]
        """
        return self.features(x)

class AttentionModule(nn.Module):
    """Attention mechanism for focusing on relevant image regions"""
    
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Attention scoring function
        self.attention = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, features, hidden):
        """
        Args:
            features: [batch_size, num_regions, feature_dim]
            hidden: [batch_size, hidden_dim]
        Returns:
            context: [batch_size, feature_dim]
            attention_weights: [batch_size, num_regions]
        """
        batch_size, num_regions, _ = features.size()
        
        # Expand hidden state to match regions
        hidden_expanded = hidden.unsqueeze(1).expand(-1, num_regions, -1)
        
        # Concatenate features and hidden state
        attention_input = torch.cat([features, hidden_expanded], dim=2)
        
        # Calculate attention scores
        attention_scores = self.attention(attention_input).squeeze(2)  # [batch_size, num_regions]
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted context vector
        context = torch.sum(features * attention_weights.unsqueeze(2), dim=1)
        
        return context, attention_weights

class LSTMDecoder(nn.Module):
    """LSTM decoder with attention mechanism"""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 feature_dim: int, num_layers: int = 1, dropout: float = 0.5,
                 use_pretrained_embeddings: bool = False, embedding_path: Optional[str] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Load pre-trained embeddings if requested
        if use_pretrained_embeddings and embedding_path:
            self._load_pretrained_embeddings(embedding_path)
        
        # Attention mechanism
        self.attention = AttentionModule(feature_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embed_dim + feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout_layer = nn.Dropout(dropout)
    
    def _load_pretrained_embeddings(self, embedding_path: str):
        """Load pre-trained embeddings (e.g., GloVe)"""
        if os.path.exists(embedding_path):
            # This is a placeholder - you'd need to implement loading logic
            # based on your embedding format (GloVe, Word2Vec, etc.)
            print(f"Loading pre-trained embeddings from {embedding_path}")
            # embeddings = load_embeddings(embedding_path)
            # self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
        else:
            print(f"Warning: Embedding file {embedding_path} not found")
    
    def forward(self, features, captions, lengths=None):
        """
        Args:
            features: [batch_size, feature_dim, spatial_size, spatial_size]
            captions: [batch_size, max_len]
            lengths: [batch_size] - actual lengths of captions
        Returns:
            outputs: [batch_size, max_len, vocab_size]
            attention_weights: [batch_size, max_len, num_regions]
        """
        batch_size, max_len = captions.size()
        num_regions = features.size(2) * features.size(3)
        
        # Reshape features for attention: [batch_size, num_regions, feature_dim]
        features_flat = features.view(batch_size, self.feature_dim, -1).transpose(1, 2)
        
        # Embed captions
        embeddings = self.embedding(captions)  # [batch_size, max_len, embed_dim]
        
        # Initialize LSTM hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(features.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(features.device)
        
        outputs = []
        attention_weights_list = []
        
        # Process each timestep
        for t in range(max_len):
            # Get current word embedding
            word_embed = embeddings[:, t, :]  # [batch_size, embed_dim]
            
            # Get LSTM hidden state (use last layer)
            lstm_hidden = h0[-1]  # [batch_size, hidden_dim]
            
            # Apply attention
            context, attn_weights = self.attention(features_flat, lstm_hidden)
            
            # Concatenate word embedding and context
            lstm_input = torch.cat([word_embed, context], dim=1)  # [batch_size, embed_dim + feature_dim]
            lstm_input = lstm_input.unsqueeze(1)  # [batch_size, 1, embed_dim + feature_dim]
            
            # Pass through LSTM
            lstm_out, (h0, c0) = self.lstm(lstm_input, (h0, c0))
            
            # Generate output
            output = self.fc(self.dropout_layer(lstm_out.squeeze(1)))
            outputs.append(output)
            attention_weights_list.append(attn_weights)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # [batch_size, max_len, vocab_size]
        attention_weights = torch.stack(attention_weights_list, dim=1)  # [batch_size, max_len, num_regions]
        
        return outputs, attention_weights

class TransformerDecoder(nn.Module):
    """Transformer-based decoder (alternative to LSTM)"""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 feature_dim: int, num_layers: int = 6, num_heads: int = 8,
                 dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.max_len = max_len
        
        # Embedding layers
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        
        # Feature projection
        self.feature_projection = nn.Linear(feature_dim, embed_dim)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features, captions, lengths=None):
        """
        Args:
            features: [batch_size, feature_dim, spatial_size, spatial_size]
            captions: [batch_size, max_len]
            lengths: [batch_size] - actual lengths of captions
        Returns:
            outputs: [batch_size, max_len, vocab_size]
        """
        batch_size, max_len = captions.size()
        num_regions = features.size(2) * features.size(3)
        
        # Reshape and project features
        features_flat = features.view(batch_size, self.feature_dim, -1).transpose(1, 2)
        features_projected = self.feature_projection(features_flat)  # [batch_size, num_regions, embed_dim]
        
        # Create position embeddings
        positions = torch.arange(max_len, device=captions.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding(positions)
        
        # Word embeddings
        word_embeddings = self.word_embedding(captions)
        
        # Combine word and position embeddings
        embeddings = word_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        
        # Create causal mask for decoder
        mask = torch.triu(torch.ones(max_len, max_len, device=captions.device), diagonal=1).bool()
        
        # Pass through transformer decoder
        decoder_output = self.transformer_decoder(
            tgt=embeddings,
            memory=features_projected,
            tgt_mask=mask
        )
        
        # Generate outputs
        outputs = self.fc(decoder_output)
        
        return outputs, None  # No attention weights for transformer

class ImageCaptioningModel(nn.Module):
    """Complete image captioning model with modular encoder and decoder"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 512, hidden_dim: int = 512,
                 encoder_type: str = 'mobilenet', decoder_type: str = 'lstm',
                 num_layers: int = 1, dropout: float = 0.5,
                 use_pretrained_embeddings: bool = False, embedding_path: Optional[str] = None,
                 freeze_encoder: bool = True):
        super().__init__()
        
        # Initialize encoder
        self.encoder = ImageEncoder(
            encoder_type=encoder_type,
            pretrained=True,
            freeze=freeze_encoder
        )
        
        # Get feature dimensions from encoder
        feature_dim = self.encoder.feature_dim
        
        # Initialize decoder
        if decoder_type == 'lstm':
            self.decoder = LSTMDecoder(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                feature_dim=feature_dim,
                num_layers=num_layers,
                dropout=dropout,
                use_pretrained_embeddings=use_pretrained_embeddings,
                embedding_path=embedding_path
            )
        elif decoder_type == 'transformer':
            self.decoder = TransformerDecoder(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                feature_dim=feature_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_type}")
        
        self.decoder_type = decoder_type
    
    def forward(self, images, captions, lengths=None):
        """
        Args:
            images: [batch_size, 3, 224, 224]
            captions: [batch_size, max_len]
            lengths: [batch_size] - actual lengths of captions
        Returns:
            outputs: [batch_size, max_len, vocab_size]
            attention_weights: [batch_size, max_len, num_regions] or None
        """
        # Encode images
        features = self.encoder(images)
        
        # Decode captions
        outputs, attention_weights = self.decoder(features, captions, lengths)
        
        return outputs, attention_weights

class ExplainabilityModule:
    """Module for generating Grad-CAM visualizations"""
    
    def __init__(self, model: ImageCaptioningModel, target_layer_name: str = 'last_conv'):
        self.model = model
        self.target_layer_name = target_layer_name
        
        if not GRAD_CAM_AVAILABLE:
            print("Warning: Grad-CAM not available. ExplainabilityModule will be limited.")
            return
        
        # Get the target layer for Grad-CAM
        if target_layer_name == 'last_conv':
            if hasattr(model.encoder.features, 'conv'):
                # For MobileNet
                target_layer = model.encoder.features[-1]
            else:
                # For ResNet
                target_layer = model.encoder.features[-1]
        else:
            target_layer = self._get_layer_by_name(target_layer_name)
        
        # Initialize Grad-CAM
        self.cam = GradCAM(
            model=model.encoder,
            target_layers=[target_layer],
            use_cuda=next(model.parameters()).is_cuda
        )
    
    def _get_layer_by_name(self, layer_name: str):
        """Get a specific layer by name"""
        for name, layer in self.model.encoder.features.named_modules():
            if name == layer_name:
                return layer
        raise ValueError(f"Layer {layer_name} not found")
    
    def generate_heatmap(self, image: torch.Tensor, caption_tokens: list, 
                        word_index: int, vocab: Dict[str, int]) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a specific word in the caption
        
        Args:
            image: [1, 3, 224, 224] - single image
            caption_tokens: list of token indices
            word_index: index of the word to explain
            vocab: vocabulary dictionary
        Returns:
            heatmap: [H, W] numpy array
        """
        if not GRAD_CAM_AVAILABLE:
            print("Grad-CAM not available. Returning dummy heatmap.")
            return np.zeros((224, 224))
        
        # Create a target function that focuses on the specific word
        def target_function(model_output):
            # For simplicity, we'll use the output at the word_index
            # In practice, you might want to create a more sophisticated target
            return model_output[:, word_index, :].max()
        
        # Generate heatmap
        heatmap = self.cam(
            input_tensor=image,
            targets=[target_function]
        )
        
        return heatmap[0]  # Remove batch dimension
    
    def visualize_attention(self, image: torch.Tensor, attention_weights: torch.Tensor, 
                           caption_tokens: list, vocab: Dict[str, int]) -> Dict[str, np.ndarray]:
        """
        Visualize attention weights for each word in the caption
        
        Args:
            image: [1, 3, 224, 224] - single image
            attention_weights: [1, max_len, num_regions] - attention weights
            caption_tokens: list of token indices
            vocab: vocabulary dictionary
        Returns:
            attention_maps: dict mapping word to attention heatmap
        """
        attention_maps = {}
        
        # Convert attention weights to spatial maps
        batch_size, max_len, num_regions = attention_weights.size()
        spatial_size = int(np.sqrt(num_regions))
        
        for t in range(max_len):
            if t < len(caption_tokens) and caption_tokens[t] != vocab.get('<PAD>', 0):
                # Get attention weights for this timestep
                weights = attention_weights[0, t, :].detach().cpu().numpy()
                
                # Reshape to spatial map
                attention_map = weights.reshape(spatial_size, spatial_size)
                
                # Upsample to image size
                attention_map = F.interpolate(
                    torch.from_numpy(attention_map).unsqueeze(0).unsqueeze(0).float(),
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()
                
                # Get word from vocabulary
                word = [k for k, v in vocab.items() if v == caption_tokens[t]][0]
                attention_maps[word] = attention_map
        
        return attention_maps

# Utility functions
def load_glove_embeddings(embedding_path: str, vocab: Dict[str, int], embed_dim: int) -> torch.Tensor:
    """Load GloVe embeddings and align with vocabulary"""
    embeddings = torch.randn(len(vocab), embed_dim) * 0.1  # Initialize randomly
    
    # Load GloVe embeddings
    glove_embeddings = {}
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(x) for x in values[1:]], dtype=torch.float32)
            glove_embeddings[word] = vector
    
    # Align with vocabulary
    for word, idx in vocab.items():
        if word in glove_embeddings:
            embeddings[idx] = glove_embeddings[word]
    
    return embeddings

def create_model(config: Dict[str, Any]) -> ImageCaptioningModel:
    """Factory function to create model from configuration"""
    return ImageCaptioningModel(
        vocab_size=config['vocab_size'],
        embed_dim=config.get('embed_dim', 512),
        hidden_dim=config.get('hidden_dim', 512),
        encoder_type=config.get('encoder_type', 'mobilenet'),
        decoder_type=config.get('decoder_type', 'lstm'),
        num_layers=config.get('num_layers', 1),
        dropout=config.get('dropout', 0.5),
        use_pretrained_embeddings=config.get('use_pretrained_embeddings', False),
        embedding_path=config.get('embedding_path', None),
        freeze_encoder=config.get('freeze_encoder', True)
    )
