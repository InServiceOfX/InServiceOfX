import subprocess
import os
from pathlib import Path
import warnings

try:
    from git import Repo
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    warnings.warn(
        "git is not installed. Please install it with `pip install git` if you need it.")

def _parse_repo_url_into_target_path(
        repo_url: str,
        local_base_path: str | Path | None):

    # Parse the URL to extract author and repo name
    # Expected format: https://github.com/author/repo.git
    parts = repo_url.rstrip('/').split('/')
    if len(parts) < 5 or parts[0] != 'https:' or parts[2] != 'github.com':
        raise ValueError(
            "Invalid GitHub URL format. Expected: https://github.com/author/repo.git")
    
    author = parts[3]
    repo_name = parts[4].replace('.git', '')

    if local_base_path is None:
        local_base_path = Path.cwd()

    local_base_path = Path(local_base_path)
    author_dir = local_base_path / author
    target_dir = author_dir / repo_name

    return target_dir, author_dir

def git_clone_repo(repo_url: str, local_base_path: str | Path | None):
    """
    Args:
        repo_url: The URL of the repository to clone. It's expected to follow
        this format: https://github.com/jujumilk3/leaked-system-prompts.git
    """
    try:
        target_dir, author_dir = _parse_repo_url_into_target_path(
            repo_url,
            local_base_path)
        
        # Create directories if they don't exist
        author_dir.mkdir(parents=True, exist_ok=True)

        if GIT_AVAILABLE:
            result = Repo.clone_from(repo_url, target_dir)
        else:
            cmd = ['git', 'clone', repo_url, str(target_dir)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
        
        print(f"Successfully cloned {repo_url}")
        print(f"Repository location: {target_dir}")
        print(f"Result: {result}")

        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Git is not installed or not in PATH")
        return False
    except ValueError as e:
        print(f"Invalid URL format: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
