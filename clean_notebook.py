#!/usr/bin/env python3
"""
Script to remove emojis and AI-like elements from Jupyter notebook
"""

import json
import re

def clean_notebook(input_file, output_file):
    """Remove emojis and verbose AI-like elements from notebook"""
    
    # Load the notebook
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Define patterns to remove
    emoji_patterns = [
        r'[\u2600-\u27BF]',  # Misc symbols
        r'[\u2700-\u27BF]',  # Dingbats
        r'[\u23E9-\u23EC]',  # Fast forward/rewind buttons
        r'[\u23F0-\u23F3]',  # Clock symbols
        r'[\u25C0-\u25FF]',  # Geometric shapes
        r'[\u2601-\u2610]',  # Weather symbols
        r'[\u2614-\u2615]',  # Weather/beverage
        r'[\u2680-\u2689]',  # Dice
        r'[\u26A0-\u26A1]',  # Warning/electric
        r'[\u26BD-\u26BE]',  # Sports
        r'[\u2713\u2714]',  # Checkmarks
        r'[\u2717\u2718]',  # X marks
        r'[\u274C]',        # Red X
        r'[\u2757]',        # Exclamation
    ]
    
    # Combined emoji regex
    emoji_regex = '|'.join(emoji_patterns)
    
    # AI-like patterns to remove
    ai_patterns = [
        r'All libraries imported successfully!',
        r'Model compiled successfully!',
        r'Training completed!',
        r'Best model saved!',
        r'CNN Model created successfully',
        r'PyTorch transforms created',
        r'Training configuration set',
        r'Dataset Split Summary',
        r'Model Configuration',
        r'Dataset Information',
        r'Directory structure',
        r'Images per class',
        r'Basic Statistics',
        r'Dataset Balance Analysis',
        r'Average image dimensions by class',
        r'Sample Images from Each Flower Class',
        r'Sample Augmented Training Images',
        r'Detailed Model Summary',
        r'Training Configuration',
        r'Starting model training on GPU',
        r'Early stopping triggered',
    ]
    
    # Process each cell
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            # Clean source code
            source = ''.join(cell['source'])
            
            # Remove emojis
            source = re.sub(emoji_regex, '', source)
            
            # Remove AI-like verbose statements
            for pattern in ai_patterns:
                source = re.sub(f'.*{pattern}.*\n?', '', source, flags=re.MULTILINE)
            
            # Remove excessive print statements with checkmarks
            source = re.sub(r'print\(f"[^"]*\\u2713[^"]*"\)\n?', '', source)
            source = re.sub(r'print\(f"[^"]*\\u2714[^"]*"\)\n?', '', source)
            source = re.sub(r'print\(f"[^"]*\\u2717[^"]*"\)\n?', '', source)
            source = re.sub(r'print\(f"[^"]*\\u2718[^"]*"\)\n?', '', source)
            source = re.sub(r'print\(f"[^"]*\\u274C[^"]*"\)\n?', '', source)
            source = re.sub(r'print\(f"[^"]*\\u26A0[^"]*"\)\n?', '', source)
            source = re.sub(r'print\(f"[^"]*\\u26A1[^"]*"\)\n?', '', source)
            
            # Remove verbose print statements
            source = re.sub(r'print\(".*All libraries.*"\)\n?', '', source)
            source = re.sub(r'print\(".*Model compiled.*"\)\n?', '', source)
            source = re.sub(r'print\(".*Training completed.*"\)\n?', '', source)
            source = re.sub(r'print\(".*Best model saved.*"\)\n?', '', source)
            source = re.sub(r'print\(".*CNN Model created.*"\)\n?', '', source)
            source = re.sub(r'print\(".*PyTorch transforms.*"\)\n?', '', source)
            source = re.sub(r'print\(".*Training configuration.*"\)\n?', '', source)
            
            # Clean up multiple empty lines
            source = re.sub(r'\n\s*\n\s*\n', '\n\n', source)
            
            cell['source'] = source.splitlines(keepends=True)
            
        elif cell['cell_type'] == 'markdown':
            # Clean markdown cells
            source = ''.join(cell['source'])
            
            # Remove emojis
            source = re.sub(emoji_regex, '', source)
            
            # Remove overly enthusiastic language
            source = re.sub(r'Importing all necessary libraries.*', '', source)
            source = re.sub(r'Explore the dataset directory.*', '', source)
            source = re.sub(r'Perform detailed analysis.*', '', source)
            source = re.sub(r'Create comprehensive visualizations.*', '', source)
            source = re.sub(r'Prepare images for CNN model.*', '', source)
            source = re.sub(r'Split data into training.*', '', source)
            source = re.sub(r'Design a deep convolutional.*', '', source)
            source = re.sub(r'Configure the model with.*', '', source)
            source = re.sub(r'Configure callbacks for.*', '', source)
            source = re.sub(r'Train the CNN model.*', '', source)
            
            cell['source'] = source.splitlines(keepends=True)
    
    # Save cleaned notebook
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"Cleaned notebook saved to: {output_file}")

if __name__ == "__main__":
    input_notebook = "Flower_Classification_CNN_Project.ipynb"
    output_notebook = "Flower_Classification_CNN_Project_Clean.ipynb"
    
    clean_notebook(input_notebook, output_notebook)
