import numpy as np
import torch
from pathlib import Path

def convert_npz_to_th(npz_path):
    """
    Convert a NumPy .npz file to a PyTorch .th file
    
    Parameters
    ----------
    npz_path : str or Path
        Path to the .npz file
    
    Returns
    -------
    dict
        Dictionary of PyTorch tensors
    """
    # Convert to Path object if it's not already
    npz_path = Path(npz_path)
    
    # Create the output path with .th extension
    th_path = npz_path.with_suffix('.th')
    
    # Load the NumPy arrays from the .npz file
    data = np.load(npz_path)
    
    # Convert to PyTorch tensors
    torch_dict = {}
    for key in data.files:
        torch_dict[key] = torch.from_numpy(data[key])
    
    # Save as a PyTorch file
    torch.save(torch_dict, th_path)
    
    print(f"Converted {npz_path} to {th_path}")
    
    return torch_dict

import sys

if __name__ == "__main__":
    """
    Usage:
        python convert_npz_to_th.py path/to/file.npz
    
    This will convert the NumPy .npz file to a PyTorch .th file with the same name
    but different extension, saving it in the same directory.
    
    Example:
        python convert_npz_to_th.py /data/melody_salience.npz
        # Creates /data/melody_salience.th
    """
    if len(sys.argv) < 2:
        print("Error: Please provide the path to the .npz file")
        print("Usage: python convert_npz_to_th.py path/to/file.npz")
        sys.exit(1)
    
    npz_path = sys.argv[1]
    convert_npz_to_th(npz_path)
