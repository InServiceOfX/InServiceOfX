#!/usr/bin/env python3

import sys
import subprocess
import os
from pathlib import Path

# List of default public trackers if none is provided
DEFAULT_TRACKERS = [
    'udp://tracker.openbittorrent.com:80/announce',
    'udp://tracker.opentrackr.org:1337/announce',
    'udp://tracker.leechers-paradise.org:6969/announce'
]

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

def create_torrent(dir_path, trackers):
    """
    Create a torrent file for the directory using transmission-create.
    https://help.ubuntu.com/community/TransmissionHowTo
    """
    if not dir_path.exists() or not dir_path.is_dir():
        print(
            f"Error: The path '{dir_path}' does not exist or is not a directory.")
        sys.exit(1)

    # Get the commit hash
    commit_hash = get_commit_hash(dir_path)
    if commit_hash:
        torrent_file = dir_path.parent / f"{dir_path.name}_{commit_hash}.torrent"
    else:
        torrent_file = dir_path.parent / f"{dir_path.name}.torrent"

    # Command to create the torrent
    cmd = ["transmission-create", "-o", str(torrent_file), str(dir_path)]

    # Add the trackers to the transmission-create command
    for tracker in trackers:
        cmd.extend(["-t", tracker])

    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print(f"Torrent file created: {torrent_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating torrent: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: transmission-create is not installed or not found in PATH.")
        sys.exit(1)

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: create_torrent.py <directory_path> [tracker_url]")
        print(
            "Provide a relative or absolute path to the repository directory, and optionally a tracker URL.")
        sys.exit(1)

    # Get the path from the command line argument
    dir_path = Path(sys.argv[1]).resolve()

    # Use the provided tracker or default trackers
    if len(sys.argv) == 3:
        trackers = [sys.argv[2]]
    else:
        trackers = DEFAULT_TRACKERS

    create_torrent(dir_path, trackers)

if __name__ == "__main__":
    main()
