from pathlib import Path
import argparse
import sys

corecode_directory = Path(__file__).resolve().parents[2]

if not str(corecode_directory) in sys.path:
    sys.path.append(str(corecode_directory))

from corecode.SetupProjectData.SetupPrompts import \
    git_clone_repo_into_prompts_path

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Clone a GitHub repository into the prompts path",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_git_clone_repo_into_prompts_path.py https://github.com/jujumilk3/leaked-system-prompts.git
        """
    )

    parser.add_argument(
        "repo_url",
        type=str,
        help="The URL of the repository to clone. It's expected to follow this format: https://github.com/jujumilk3/leaked-system-prompts.git"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    repo_url = args.repo_url
    git_clone_repo_into_prompts_path(repo_url)

    print(f"Repository cloned successfully: {repo_url}")