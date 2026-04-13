"""
Image Captioning Model Architecture
CNN Encoder (ResNet50) + RNN Decoder (LSTM)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    """CNN Encoder using pretrained ResNet50"""
    
    def __init__(self, embed_size):
        """
        Args:
            embed_size: Dimension of the embedding space
        """
        super(CNNEncoder, self).__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=True)
        
        # Remove the last fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Linear layer to project ResNet output to embedding space
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        
        # Freeze ResNet parameters (optional - can fine-tune later)
        for param in self.resnet.parameters():
            param.requires_grad = False
    
    def forward(self, images):
        """
        Extract features from images
        
        Args:
            images: Batch of images (batch_size, 3, 224, 224)
        
        Returns:
            features: Image features (batch_size, embed_size)
        """
        with torch.no_grad():
            features = self.resnet(images)
        
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        
        return features
    
    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent fine-tuning of the CNN encoder
        
        Args:
            fine_tune: If True, allows fine-tuning
        """
        for param in self.resnet.parameters():
            param.requires_grad = fine_tune


class RNNDecoder(nn.Module):
    """RNN Decoder using LSTM"""
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Args:
            embed_size: Dimension of word embeddings
            hidden_size: Dimension of LSTM hidden states
            vocab_size: Size of vocabulary
            num_layers: Number of LSTM layers
        """
        super(RNNDecoder, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Word embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # Linear layer to produce vocabulary distribution
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, features, captions):
        """
        Decode image features and generate captions
        
        Args:
            features: Image features from encoder (batch_size, embed_size)
            captions: Ground truth captions (batch_size, caption_length)
        
        Returns:
            outputs: Predicted word distributions (batch_size, caption_length, vocab_size)
        """
        # Embed captions
        embeddings = self.embed(captions)  # (batch_size, caption_length, embed_size)
        
        # Concatenate image features with caption embeddings
        features = features.unsqueeze(1)  # (batch_size, 1, embed_size)
        inputs = torch.cat((features, embeddings), dim=1)  # (batch_size, caption_length+1, embed_size)
        
        # Pass through LSTM
        hiddens, _ = self.lstm(inputs)  # (batch_size, caption_length+1, hidden_size)
        
        # Apply dropout
        hiddens = self.dropout(hiddens)
        
        # Generate outputs
        outputs = self.linear(hiddens)  # (batch_size, caption_length+1, vocab_size)
        
        return outputs
    
    def generate_caption(self, features, max_length=20, start_token=1, end_token=2):
        """
        Generate caption using greedy search
        
        Args:
            features: Image features (batch_size, embed_size)
            max_length: Maximum caption length
            start_token: Index of <START> token
            end_token: Index of <END> token
        
        Returns:
            captions: Generated caption indices (batch_size, max_length)
        """
        batch_size = features.size(0)
        captions = []
        
        # Initialize LSTM state
        states = None
        
        # Start with image features
        inputs = features.unsqueeze(1)  # (batch_size, 1, embed_size)
        
        for _ in range(max_length):
            # Forward pass through LSTM
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))  # (batch_size, vocab_size)
            
            # Get predicted word
            _, predicted = outputs.max(1)  # (batch_size)
            captions.append(predicted)
            
            # Prepare next input
            inputs = self.embed(predicted).unsqueeze(1)  # (batch_size, 1, embed_size)
            
            # Stop if all sequences have generated <END> token
            if (predicted == end_token).all():
                break
        
        # Stack captions
        captions = torch.stack(captions, 1)  # (batch_size, length)
        
        return captions
    
    def generate_caption_beam_search(self, features, beam_width=3, max_length=20, start_token=1, end_token=2):
        """
        Generate caption using beam search
        
        Args:
            features: Image features (1, embed_size) - single image
            beam_width: Beam width for search
            max_length: Maximum caption length
            start_token: Index of <START> token
            end_token: Index of <END> token
        
        Returns:
            best_caption: Best generated caption
        """
        # Initialize beam
        k = beam_width
        
        # Start with image features
        inputs = features.unsqueeze(1)  # (1, 1, embed_size)
        
        # Get first prediction
        hiddens, states = self.lstm(inputs)
        outputs = self.linear(hiddens.squeeze(1))  # (1, vocab_size)
        
        # Get top k predictions
        log_probs, indices = torch.topk(torch.log_softmax(outputs, dim=1), k)
        
        # Initialize beams
        beams = []
        for i in range(k):
            beam = {
                'tokens': [indices[0, i].item()],
                'log_prob': log_probs[0, i].item(),
                'states': (states[0][:, :, :], states[1][:, :, :]),
                'finished': indices[0, i].item() == end_token
            }
            beams.append(beam)
        
        # Generate captions
        for _ in range(max_length - 1):
            all_candidates = []
            
            for beam in beams:
                if beam['finished']:
                    all_candidates.append(beam)
                    continue
                
                # Get last token
                token = torch.tensor([[beam['tokens'][-1]]], device=features.device)
                inputs = self.embed(token)
                
                # Forward pass
                hiddens, states = self.lstm(inputs, beam['states'])
                outputs = self.linear(hiddens.squeeze(1))
                
                # Get top k predictions
                log_probs, indices = torch.topk(torch.log_softmax(outputs, dim=1), k)
                
                # Create new candidates
                for i in range(k):
                    candidate = {
                        'tokens': beam['tokens'] + [indices[0, i].item()],
                        'log_prob': beam['log_prob'] + log_probs[0, i].item(),
                        'states': (states[0].clone(), states[1].clone()),
                        'finished': indices[0, i].item() == end_token
                    }
                    all_candidates.append(candidate)
            
            # Select top k beams
            all_candidates.sort(key=lambda x: x['log_prob'] / len(x['tokens']), reverse=True)
            beams = all_candidates[:k]
            
            # Check if all beams are finished
            if all(beam['finished'] for beam in beams):
                break
        
        # Return best beam
        best_beam = beams[0]
        return torch.tensor(best_beam['tokens'], device=features.device)


class ImageCaptioningModel(nn.Module):
    """Complete Image Captioning Model"""
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Args:
            embed_size: Dimension of embeddings
            hidden_size: Dimension of LSTM hidden states
            vocab_size: Size of vocabulary
            num_layers: Number of LSTM layers
        """
        super(ImageCaptioningModel, self).__init__()
        
        self.encoder = CNNEncoder(embed_size)
        self.decoder = RNNDecoder(embed_size, hidden_size, vocab_size, num_layers)
    
    def forward(self, images, captions):
        """
        Forward pass
        
        Args:
            images: Batch of images (batch_size, 3, 224, 224)
            captions: Ground truth captions (batch_size, caption_length)
        
        Returns:
            outputs: Predicted word distributions
        """
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    
    def generate_caption(self, images, max_length=20, start_token=1, end_token=2, beam_search=False, beam_width=3):
        """
        Generate captions for images
        
        Args:
            images: Batch of images
            max_length: Maximum caption length
            start_token: Index of <START> token
            end_token: Index of <END> token
            beam_search: Whether to use beam search
            beam_width: Beam width for beam search
        
        Returns:
            captions: Generated captions
        """
        features = self.encoder(images)
        
        if beam_search and images.size(0) == 1:
            captions = self.decoder.generate_caption_beam_search(
                features, beam_width, max_length, start_token, end_token
            )
        else:
            captions = self.decoder.generate_caption(
                features, max_length, start_token, end_token
            )
        
        return captions


if __name__ == '__main__':
    # Test the model
    print("Testing Image Captioning Model...")
    
    # Hyperparameters
    embed_size = 256
    hidden_size = 512
    vocab_size = 1000
    num_layers = 2
    batch_size = 4
    
    # Create model
    model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_layers)
    
    # Test input
    images = torch.randn(batch_size, 3, 224, 224)
    captions = torch.randint(0, vocab_size, (batch_size, 15))
    
    # Forward pass
    outputs = model(images, captions)
    
    print(f"\nModel Architecture Summary:")
    print(f"Input image shape: {images.shape}")
    print(f"Input caption shape: {captions.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Expected output shape: (batch_size, caption_length+1, vocab_size)")
    
    # Test caption generation
    print("\nTesting caption generation...")
    generated_captions = model.generate_caption(images, max_length=20)
    print(f"Generated captions shape: {generated_captions.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\nModel test successful!")
