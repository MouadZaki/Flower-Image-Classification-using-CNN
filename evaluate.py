"""
Model Evaluation and Caption Generation
"""

import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from torchvision import transforms

from model import ImageCaptioningModel
from data_loader import Vocabulary


def load_model(checkpoint_path, device='cpu'):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        model, vocab, hyperparameters
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get hyperparameters
    hyperparams = checkpoint['hyperparameters']
    
    # Load vocabulary
    vocab = checkpoint['vocab']
    
    # Create model
    model = ImageCaptioningModel(
        embed_size=hyperparams['embed_size'],
        hidden_size=hyperparams['hidden_size'],
        vocab_size=hyperparams['vocab_size'],
        num_layers=hyperparams['num_layers']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
    
    return model, vocab, hyperparams


def load_image(image_path, transform=None):
    """
    Load and preprocess image
    
    Args:
        image_path: Path to image
        transform: Image transformations
    
    Returns:
        image_tensor: Preprocessed image tensor
    """
    image = Image.open(image_path).convert('RGB')
    
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image


def generate_caption(model, image_tensor, vocab, idx_to_word, device, max_length=20, beam_search=False, beam_width=3):
    """
    Generate caption for an image
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        vocab: Vocabulary dict
        idx_to_word: Index to word mapping
        device: Device
        max_length: Maximum caption length
        beam_search: Whether to use beam search
        beam_width: Beam width
    
    Returns:
        caption: Generated caption string
    """
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # Generate caption
        start_token = vocab['<START>']
        end_token = vocab['<END>']
        
        caption_indices = model.generate_caption(
            image_tensor,
            max_length=max_length,
            start_token=start_token,
            end_token=end_token,
            beam_search=beam_search,
            beam_width=beam_width
        )
        
        # Convert indices to words
        if beam_search:
            words = [idx_to_word[idx.item()] for idx in caption_indices 
                    if idx.item() not in [vocab['<START>'], vocab['<END>'], vocab['<PAD>']]]
        else:
            words = [idx_to_word[idx.item()] for idx in caption_indices[0] 
                    if idx.item() not in [vocab['<START>'], vocab['<END>'], vocab['<PAD>']]]
        
        caption = ' '.join(words)
    
    return caption


def evaluate_model(model, dataloader, criterion, device, vocab_size):
    """
    Evaluate model on a dataset
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        criterion: Loss function
        device: Device
        vocab_size: Vocabulary size
    
    Returns:
        avg_loss: Average loss
        perplexity: Perplexity score
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, captions, lengths in dataloader:
            images = images.to(device)
            captions = captions.to(device)
            
            # Forward pass
            outputs = model(images, captions[:, :-1])
            
            # Calculate loss
            outputs = outputs.reshape(-1, vocab_size)
            targets = captions[:, 1:].reshape(-1)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    perplexity = np.exp(avg_loss)
    
    return avg_loss, perplexity


def visualize_predictions(model, image_paths, vocab, idx_to_word, device, save_path='predictions.png'):
    """
    Visualize model predictions
    
    Args:
        model: Trained model
        image_paths: List of image paths
        vocab: Vocabulary dict
        idx_to_word: Index to word mapping
        device: Device
        save_path: Path to save visualization
    """
    num_images = len(image_paths)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    
    if num_images == 1:
        axes = [axes]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    for idx, image_path in enumerate(image_paths):
        # Load image
        image_tensor, original_image = load_image(image_path, transform)
        
        # Generate caption
        caption = generate_caption(model, image_tensor, vocab, idx_to_word, device)
        
        # Display
        axes[idx].imshow(original_image)
        axes[idx].set_title(f"Caption:\n{caption}", fontsize=10, wrap=True)
        axes[idx].axis('off')
    
    plt.tight_layout()
    # Ensure directory exists
    outdir = os.path.dirname(save_path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Predictions saved to {save_path}")


def calculate_bleu_score(reference, candidate):
    """
    Calculate BLEU score (simplified version)
    
    Args:
        reference: Reference caption (string)
        candidate: Generated caption (string)
    
    Returns:
        bleu: BLEU score
    """
    from collections import Counter
    
    reference_tokens = reference.lower().split()
    candidate_tokens = candidate.lower().split()
    
    # Calculate precision for unigrams
    reference_counts = Counter(reference_tokens)
    candidate_counts = Counter(candidate_tokens)
    
    overlap = sum((candidate_counts & reference_counts).values())
    precision = overlap / len(candidate_tokens) if len(candidate_tokens) > 0 else 0
    
    # Brevity penalty
    bp = 1.0 if len(candidate_tokens) >= len(reference_tokens) else np.exp(1 - len(reference_tokens) / len(candidate_tokens))
    
    bleu = bp * precision
    
    return bleu


def evaluate_with_metrics(model, dataloader, vocab, idx_to_word, device):
    """
    Evaluate model with various metrics
    
    Args:
        model: Trained model
        dataloader: Data loader
        vocab: Vocabulary dict
        idx_to_word: Index to word mapping
        device: Device
    
    Returns:
        metrics: Dictionary of metrics
    """
    model.eval()
    
    bleu_scores = []
    
    with torch.no_grad():
        for images, captions, lengths in dataloader:
            images = images.to(device)
            batch_size = images.size(0)
            
            for i in range(batch_size):
                # Generate caption
                image_tensor = images[i].unsqueeze(0)
                generated_caption = generate_caption(model, image_tensor, vocab, idx_to_word, device)
                
                # Get ground truth caption
                caption_indices = captions[i]
                true_caption = ' '.join([idx_to_word[idx.item()] for idx in caption_indices 
                                       if idx.item() not in [vocab['<START>'], vocab['<END>'], vocab['<PAD>']]])
                
                # Calculate BLEU score
                bleu = calculate_bleu_score(true_caption, generated_caption)
                bleu_scores.append(bleu)
    
    metrics = {
        'average_bleu': np.mean(bleu_scores),
        'median_bleu': np.median(bleu_scores),
        'min_bleu': np.min(bleu_scores),
        'max_bleu': np.max(bleu_scores)
    }
    
    return metrics


def main():
    """Main evaluation function"""
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate and visualize predictions')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help='Model checkpoint path')
    parser.add_argument('--dataset-root', type=str, default='data/flower_photos', help='Dataset root for sampling images')
    parser.add_argument('--outdir', type=str, default='outputs', help='Directory to save visualization')
    args = parser.parse_args()
    
    # Configuration
    checkpoint_path = args.checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading model...")
    model, vocab, hyperparams = load_model(checkpoint_path, device)
    
    # Create index to word mapping
    idx_to_word = {v: k for k, v in vocab.items()}
    
    # Test on some images
    print("\nTesting on sample images...")
    
    # Collect a few random images from data/flower_photos
    root_dir = args.dataset_root
    if not os.path.exists(root_dir):
        print("No dataset found at data/flower_photos. Please ensure the dataset is placed there.")
        return

    class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    image_paths = []
    for cls in class_dirs:
        cls_dir = os.path.join(root_dir, cls)
        imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        image_paths.extend(imgs)

    if len(image_paths) == 0:
        print("No images found in data/flower_photos.")
        return

    # Pick up to 3 random images
    import random
    sample_images = random.sample(image_paths, k=min(3, len(image_paths)))
    save_path = os.path.join(args.outdir, 'predictions.png')
    visualize_predictions(model, sample_images, vocab, idx_to_word, device, save_path=save_path)
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
