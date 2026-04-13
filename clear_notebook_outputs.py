#!/usr/bin/env python3
"""
Script to clear all outputs from Jupyter notebook to reduce file size
"""

import json

def clear_notebook_outputs(input_file, output_file):
    """Clear all outputs from notebook cells to reduce file size"""
    
    # Load the notebook
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Clear outputs from all code cells
    cleared_cells = 0
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'outputs' in cell:
            if cell['outputs']:  # Only count if there were outputs
                cleared_cells += 1
            cell['outputs'] = []
            cell['execution_count'] = None
    
    # Save the cleaned notebook
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"Notebook cleared and saved to: {output_file}")
    print(f"Cleared outputs from {cleared_cells} cells")
    
    # Show file size comparison
    import os
    original_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
    new_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"Original size: {original_size:.2f} MB")
    print(f"New size: {new_size:.2f} MB")
    print(f"Size reduction: {original_size - new_size:.2f} MB ({((original_size - new_size)/original_size*100):.1f}%)")

if __name__ == "__main__":
    input_notebook = "Flower_Classification_CNN_Project.ipynb"
    output_notebook = "Flower_Classification_CNN_Project_Cleared.ipynb"
    
    clear_notebook_outputs(input_notebook, output_notebook)
