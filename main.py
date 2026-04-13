"""
Image Captioning with Flower Dataset - Main Training Script
Using CNN (ResNet50) for feature extraction and RNN (LSTM) for caption generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

from model import ImageCaptioningModel
from data_loader import FlowerDataset, get_data_loaders
from evaluate import evaluate_model, generate_caption


def train_epoch(model, dataloader, criterion, optimizer, device, vocab_size):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    
    for batch_idx, (images, captions, lengths) in enumerate(dataloader):
        images = images.to(device)
        captions = captions.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass: teacher forcing with inputs excluding last token
        outputs = model(images, captions[:, :-1])

        # Align time steps: remove the first output step (image features)
        # so outputs length matches targets length (captions[:, 1:])
        outputs = outputs[:, :-1, :]

        # Reshape for loss calculation
        outputs = outputs.reshape(-1, vocab_size)
        targets = captions[:, 1:].reshape(-1)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        # Update weights
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    return epoch_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device, vocab_size):
    """Validate for one epoch"""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for images, captions, lengths in dataloader:
            images = images.to(device)
            captions = captions.to(device)
            
            # Forward pass
            outputs = model(images, captions[:, :-1])
            
            # Align time steps and reshape
            outputs = outputs[:, :-1, :]
            outputs = outputs.reshape(-1, vocab_size)
            targets = captions[:, 1:].reshape(-1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def plot_losses(train_losses, val_losses, save_path='training_curves.png'):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to {save_path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train image captioning model')
    parser.add_argument('--outdir', type=str, default='outputs', help='Directory to save training curves and history')
    args = parser.parse_args()
    # Hyperparameters
    BATCH_SIZE = 32
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    SAVE_DIR = 'checkpoints'
    OUT_DIR = args.outdir
    
    # Create checkpoint directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading data...')
    train_loader, val_loader, vocab = get_data_loaders(
        root_dir='data',
        batch_size=BATCH_SIZE
    )
    
    vocab_size = len(vocab)
    print(f'Vocabulary size: {vocab_size}')
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')
    
    # Initialize model
    print('Initializing model...')
    model = ImageCaptioningModel(
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        vocab_size=vocab_size,
        num_layers=NUM_LAYERS
    ).to(device)
    
    # Print model architecture
    print(f'\nModel Architecture:')
    print(f'CNN Encoder: ResNet50 (pretrained)')
    print(f'Embedding size: {EMBED_SIZE}')
    print(f'LSTM Hidden size: {HIDDEN_SIZE}')
    print(f'LSTM Layers: {NUM_LAYERS}')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f'\nStarting training for {NUM_EPOCHS} epochs...')
    print('=' * 60)
    
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 60)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, vocab_size)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device, vocab_size)
        val_losses.append(val_loss)
        
        # Adjust learning rate
        scheduler.step(val_loss)
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'vocab': vocab,
                'hyperparameters': {
                    'embed_size': EMBED_SIZE,
                    'hidden_size': HIDDEN_SIZE,
                    'vocab_size': vocab_size,
                    'num_layers': NUM_LAYERS
                }
            }
            save_path = os.path.join(SAVE_DIR, 'best_model.pth')
            torch.save(checkpoint, save_path)
            print(f'✓ Best model saved to {save_path}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f'✓ Checkpoint saved to {checkpoint_path}')
        
        # Plot training curves
        plot_path = os.path.join(OUT_DIR, 'training_curves.png')
        plot_losses(train_losses, val_losses, save_path=plot_path)
    
    print('\n' + '=' * 60)
    print('Training completed!')
    print(f'Best validation loss: {best_val_loss:.4f}')
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'embed_size': EMBED_SIZE,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS
        },
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    history_path = os.path.join(OUT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f'\nTraining history saved to {history_path}')
    
    # Test the model with some examples
    print('\n' + '=' * 60)
    print('Testing model on validation samples...')
    model.eval()
    
    # Load best model
    checkpoint = torch.load(os.path.join(SAVE_DIR, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Generate some sample captions
    from PIL import Image
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Get a few validation samples
    val_dataset = val_loader.dataset
    idx_to_word = {v: k for k, v in vocab.items()}
    
    for i in range(min(5, len(val_dataset))):
        image, caption, _ = val_dataset[i]
        image_tensor = image.unsqueeze(0).to(device)
        
        # Generate caption
        predicted_caption = generate_caption(model, image_tensor, vocab, idx_to_word, device, max_length=20)
        
        # Get ground truth caption
        true_caption = ' '.join([idx_to_word[idx.item()] for idx in caption if idx.item() not in [vocab['<PAD>'], vocab['<START>'], vocab['<END>']]])
        
        print(f'\nSample {i+1}:')
        print(f'True caption: {true_caption}')
        print(f'Predicted caption: {predicted_caption}')
    
    print('\n' + '=' * 60)
    print('All done! Check the checkpoints folder for saved models.')


if __name__ == '__main__':
    main()
