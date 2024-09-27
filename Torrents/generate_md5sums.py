#!/usr/bin/env python3

import hashlib
import sys
import subprocess
import os
from pathlib import Path

def get_commit_hash(directory):
    """
    Get the current commit hash from the git repository at the specified
    directory.
    """
    try:
        os.chdir(directory)
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            stderr=subprocess.STDOUT
        ).strip().decode('utf-8')
        return commit_hash
    except subprocess.CalledProcessError:
        return None
    except FileNotFoundError:
        print(f"Error: The directory '{directory}' does not exist.")
        return None

def get_md5sum(file_path):
    """Generate the MD5 checksum for the given file."""
    hash_md5 = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    if len(sys.argv) != 2:
        print("Usage: generate_md5sums.py <directory_path>")
        print("Provide a relative or absolute path to the directory.")
        sys.exit(1)

    # Get the path from the command line argument
    dir_path = Path(sys.argv[1])

    # Ensure parent exists
    parent_dir = dir_path.parent
    if not parent_dir:
        print("Error: The given directory has no parent.")
        sys.exit(1)

    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Error: The path '{dir_path}' does not exist or is not a directory.")
        sys.exit(1)

    # Get the commit hash
    commit_hash = get_commit_hash(dir_path)
    if commit_hash:
        output_file = parent_dir / f"md5sums_{commit_hash}.txt"
    else:
        output_file = parent_dir / "md5sums.txt"

    # Use the absolute or relative path as required
    if dir_path.is_absolute():
        paths = [p for p in dir_path.rglob("*") if p.is_file()]
    else:
        paths = [p.relative_to(Path.cwd()) for p in dir_path.rglob("*") if p.is_file()]

    with output_file.open("w") as f:
        for file in paths:
            try:
                file_md5 = get_md5sum(file)
                # Write the file path relative to dir_path
                relative_path = file.relative_to(dir_path)
                f.write(f"{relative_path}: {file_md5}\n")
            except Exception as e:
                print(f"Warning: Could not compute MD5 for {file}: {e}")

    print(f"MD5 sums written to {output_file}")

if __name__ == "__main__":
    main()
