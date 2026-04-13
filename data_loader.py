"""
Data loading and preprocessing for Flower Dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import numpy as np
from collections import Counter
import pickle


class Vocabulary:
    """Vocabulary for captions"""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
        # Add special tokens
        self.add_word('<PAD>')
        self.add_word('<START>')
        self.add_word('<END>')
        self.add_word('<UNK>')
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<UNK>']
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.word2idx.get(key, self.word2idx['<UNK>'])
        elif isinstance(key, int):
            return self.idx2word.get(key, '<UNK>')
        else:
            raise TypeError("Key must be str or int")


class FlowerDataset(Dataset):
    """Dataset for flower images with captions"""
    
    def __init__(self, root_dir, vocab=None, transform=None, build_vocab=False):
        """
        Args:
            root_dir: Root directory containing flower_photos folder
            vocab: Vocabulary object (if None, will create new one)
            transform: Image transformations
            build_vocab: Whether to build vocabulary from captions
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all image paths and create captions
        self.images = []
        self.captions = []
        
        # Check if flower_photos directory exists
        photos_dir = os.path.join(root_dir, 'flower_photos')
        if not os.path.exists(photos_dir):
            # Alternative: look for subdirectories with flower names
            photos_dir = root_dir
        
        # Flower categories (typical in flower datasets)
        flower_types = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips',
                       'rose', 'sunflower', 'tulip', 'daisies', 'dandelions']
        
        # Collect images
        if os.path.exists(photos_dir):
            for item in os.listdir(photos_dir):
                item_path = os.path.join(photos_dir, item)
                if os.path.isdir(item_path):
                    # This is a category folder
                    category = item.lower()
                    for img_file in os.listdir(item_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            img_path = os.path.join(item_path, img_file)
                            self.images.append(img_path)
                            # Generate caption based on folder name
                            caption = self.generate_caption(category)
                            self.captions.append(caption)
                elif item.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    # Image directly in root
                    self.images.append(item_path)
                    caption = self.generate_caption('flower')
                    self.captions.append(caption)
        
        # If no images found, create dummy data for demonstration
        if len(self.images) == 0:
            print("Warning: No images found. Creating dummy dataset for demonstration.")
            self.create_dummy_data()
        
        # Build or use vocabulary
        if build_vocab or vocab is None:
            self.vocab = Vocabulary()
            self.build_vocabulary()
        else:
            self.vocab = vocab
        
        print(f"Loaded {len(self.images)} images with captions")
    
    def generate_caption(self, category):
        """Generate descriptive captions for flower images"""
        category = category.lower().replace('_', ' ')
        
        # Caption templates
        templates = [
            f"a beautiful {category} flower",
            f"a colorful {category} in bloom",
            f"a {category} flower with vibrant petals",
            f"a lovely {category} flower",
            f"a {category} flower in the garden",
            f"a close up of a {category} flower",
            f"a bright {category} flower",
            f"a {category} flower with green leaves",
        ]
        
        # Random selection for variety
        import random
        return random.choice(templates)
    
    def create_dummy_data(self):
        """Create dummy data for demonstration"""
        categories = ['daisy', 'rose', 'tulip', 'sunflower', 'dandelion']
        
        for i in range(100):  # Create 100 dummy samples
            category = categories[i % len(categories)]
            self.images.append(f"dummy_image_{i}.jpg")
            caption = self.generate_caption(category)
            self.captions.append(caption)
    
    def build_vocabulary(self):
        """Build vocabulary from all captions"""
        word_freq = Counter()
        
        for caption in self.captions:
            tokens = caption.lower().split()
            word_freq.update(tokens)
        
        # Add words to vocabulary (filter low frequency words)
        for word, freq in word_freq.items():
            if freq >= 1:  # You can increase threshold if needed
                self.vocab.add_word(word)
        
        print(f"Vocabulary size: {len(self.vocab)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        
        # Handle dummy images
        if img_path.startswith('dummy_'):
            # Create a dummy image
            image = Image.new('RGB', (224, 224), color=(np.random.randint(0, 255), 
                                                         np.random.randint(0, 255), 
                                                         np.random.randint(0, 255)))
        else:
            try:
                image = Image.open(img_path).convert('RGB')
            except:
                # If image fails to load, create dummy
                image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        # Process caption
        caption = self.captions[idx]
        tokens = ['<START>'] + caption.lower().split() + ['<END>']
        caption_ids = [self.vocab(token) for token in tokens]
        caption_tensor = torch.tensor(caption_ids, dtype=torch.long)
        
        return image, caption_tensor, len(caption_ids)


def collate_fn(batch):
    """Custom collate function for batching variable-length captions"""
    # Sort batch by caption length (descending)
    batch.sort(key=lambda x: x[2], reverse=True)
    
    images, captions, lengths = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Pad captions
    max_length = lengths[0]
    padded_captions = torch.zeros(len(captions), max_length, dtype=torch.long)
    
    for i, caption in enumerate(captions):
        end = lengths[i]
        padded_captions[i, :end] = caption[:end]
    
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return images, padded_captions, lengths


def get_data_loaders(root_dir='flower_dataset', batch_size=32, num_workers=0, train_split=0.8):
    """
    Create train and validation data loaders
    
    Args:
        root_dir: Root directory containing images
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        train_split: Fraction of data to use for training
    
    Returns:
        train_loader, val_loader, vocab
    """
    
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Build vocabulary from full dataset
    full_dataset = FlowerDataset(root_dir, transform=transform, build_vocab=True)
    vocab = full_dataset.vocab
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Save vocabulary
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("Vocabulary saved to vocab.pkl")
    
    return train_loader, val_loader, vocab.word2idx


if __name__ == '__main__':
    # Test data loader
    print("Testing data loader...")
    train_loader, val_loader, vocab = get_data_loaders(
        root_dir='flower_dataset',
        batch_size=4
    )
    
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test one batch
    for images, captions, lengths in train_loader:
        print(f"\nBatch shapes:")
        print(f"Images: {images.shape}")
        print(f"Captions: {captions.shape}")
        print(f"Lengths: {lengths}")
        break
    
    print("\nData loader test successful!")
