"""
Interactive Inference: Generate captions for flower photos using PyTorch.
Loads the trained Image Captioning model checkpoint and runs GPU inference.
"""

import os
import random
import argparse
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import csv

from evaluate import load_model, load_image, generate_caption


def pick_random_image_from_dataset(root_dir='data/flower_photos'):
    """Pick a random image path from the dataset."""
    if not os.path.exists(root_dir):
        return None

    # Collect class folders
    class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    if not class_dirs:
        return None

    cls = random.choice(class_dirs)
    cls_dir = os.path.join(root_dir, cls)
    imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not imgs:
        return None

    img = random.choice(imgs)
    return os.path.join(cls_dir, img)


def show_image_with_caption(image, caption, save_path='prediction_result.png'):
    """Display image with generated caption and save the figure."""
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(f"Caption: {caption}", fontsize=12, wrap=True)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def save_caption_text(out_path, caption, image_path, checkpoint_path, device):
    """Save caption and metadata to a text file."""
    lines = [
        f"Image: {image_path}",
        f"Checkpoint: {checkpoint_path}",
        f"Device: {device}",
        f"Caption: {caption}",
    ]
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def append_caption_csv(outdir, timestamp, image_path, caption, device, checkpoint_path):
    """Append a record to captions_log.csv in the provided outdir."""
    log_path = os.path.join(outdir, 'captions_log.csv')
    file_exists = os.path.exists(log_path)
    with open(log_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'image_path', 'caption', 'device', 'checkpoint'])
        writer.writerow([timestamp, image_path, caption, str(device), checkpoint_path])


def main():
    print("=" * 60)
    print("🌸 FLOWER IMAGE CAPTIONING - PYTORCH GPU INFERENCE 🌸")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Flower image captioning inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                        help="Path to the trained model checkpoint (.pth)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an image file. If not provided, a random dataset image will be used.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference even if CUDA is available")
    parser.add_argument("--outdir", type=str, default="outputs",
                        help="Directory to save results (caption image and text)")
    parser.add_argument("--dataset-root", type=str, default="data/flower_photos",
                        help="Root folder containing class subfolders with images for random selection")
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f"Using device: {device}")

    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("➡️  Train the model first by running: python main.py")
        return

    # Load model and vocabulary
    model, vocab, hyperparams = load_model(checkpoint_path, device)
    idx_to_word = {v: k for k, v in vocab.items()}

    # Determine image path
    image_path = args.image
    if image_path is None:
        print("\n" + "=" * 60)
        print("CHOOSE IMAGE SOURCE:")
        print("=" * 60)
        print("1. Random image from dataset")
        print("2. Provide a custom image path")
        print("=" * 60)
        choice = input("\nEnter choice (1 or 2): ").strip()

        if choice == '1':
            image_path = pick_random_image_from_dataset(args.dataset_root)
            if image_path is None:
                print(f"❌ No images found in {args.dataset_root}. Provide --image <path>.")
                return
            print(f"\n🎲 Testing random image: {image_path}")
        elif choice == '2':
            image_path = input("\nEnter image path: ").strip().strip('"').strip("'")
            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path}")
                return
        else:
            print("❌ Invalid choice!")
            return
    else:
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return

    # Ensure output directory exists
    os.makedirs(args.outdir, exist_ok=True)

    # Load and preprocess the image
    image_tensor, original_image = load_image(image_path)

    # Generate caption on GPU if available
    caption = generate_caption(model, image_tensor, vocab, idx_to_word, device, max_length=20)

    print("\n" + "=" * 60)
    print("CAPTION RESULT")
    print("=" * 60)
    print(f"📝 Generated Caption: {caption}")
    print("=" * 60 + "\n")

    # Compose output file paths
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = os.path.splitext(os.path.basename(image_path))[0]
    img_out = os.path.join(args.outdir, f"caption_{base}_{ts}.png")
    txt_out = os.path.join(args.outdir, f"caption_{base}_{ts}.txt")

    # Save outputs
    show_image_with_caption(original_image, caption, save_path=img_out)
    save_caption_text(txt_out, caption, image_path, checkpoint_path, device)
    append_caption_csv(args.outdir, ts, image_path, caption, device, checkpoint_path)
    print(f"📁 Saved: {img_out}")
    print(f"📁 Saved: {txt_out}")
    print(f"📁 Logged: {os.path.join(args.outdir, 'captions_log.csv')}")


if __name__ == "__main__":
    main()
