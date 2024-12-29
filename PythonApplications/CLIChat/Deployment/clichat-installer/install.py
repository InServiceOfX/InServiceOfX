from pathlib import Path
import subprocess
import sys
import shutil
from prompt_toolkit.shortcuts import yes_no_dialog, message_dialog, ProgressBar
import os

def run_with_spinner(cmd, label):
    print(f"\n{label}...")
    print(f"Running command: {' '.join(cmd)}")  # Print the command being run
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Real-time output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                
        _, stderr = process.communicate()
        
        if stderr:
            print(f"Error: {stderr}")
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, cmd, None, stderr)
    except Exception as e:
        print(f"Failed to execute command: {e}")
        raise
    return process

def create_virtual_environment():
    venv_path = Path.cwd() / "venv"
    
    create = yes_no_dialog(
        title="Virtual Environment",
        text=(
            "It's strongly recommended to create a virtual environment.\n"
            "Create one now?")
    ).run()
    
    if create:
        try:
            print(f"\nCreating virtual environment at: {venv_path}")
            
            # Try to find python3 executable
            python_exe = shutil.which('python3') or sys.executable
            
            steps = [
                (
                    "Creating Python environment",
                    [python_exe, "-m", "venv", str(venv_path)]
                ),
                (
                    "Installing pip",
                    [str(venv_path / "bin" / "python3"), "-m", "ensurepip", "--upgrade"]
                ),
                (
                    "Upgrading pip",
                    [str(venv_path / "bin" / "pip3"), "install", "--upgrade", "pip"]
                )
            ]
            
            for label, cmd in steps:
                run_with_spinner(cmd, label)
                
            print("\nVirtual environment created successfully! ✓")
            return venv_path
            
        except subprocess.CalledProcessError as e:
            print(f"\nError output:\n{e.stderr}")
            message_dialog(
                title="Virtual Environment Error",
                text=f"Failed to create virtual environment:\n{e.stderr}"
            ).run()
            sys.exit(1)
    return None

def run_installation(venv_path):
    script_dir = Path(__file__).parent
    moregroq_path = script_dir / "moregroq-0.1.0.tar.gz"
    clichat_path = script_dir / "clichat-0.1.0.tar.gz"
    
    if not moregroq_path.exists() or not clichat_path.exists():
        message_dialog(
            title="Installation Error",
            text="Required package files not found!\n"
                 f"moregroq: {moregroq_path}\n"
                 f"clichat: {clichat_path}"
        ).run()
        sys.exit(1)
        
    pip_cmd = str(venv_path / "bin" / "pip") if venv_path else "pip"
    python_cmd = str(venv_path / "bin" / "python") if venv_path else "python"
    
    try:
        print("\nInstalling required packages...")
        subprocess.run([pip_cmd, "install", str(moregroq_path)], check=True)
        subprocess.run([pip_cmd, "install", str(clichat_path)], check=True)
        
        print("\nRunning initial setup...")
        subprocess.run([python_cmd, "-m", "Executables.main_setup"], check=True)
        
        message_dialog(
            title="Installation Complete",
            text=(
                "CLIChat has been installed successfully!\n\n"
                "To start using CLIChat:\n"
                "1. Activate the virtual environment:\n"
                f"   source {venv_path}/bin/activate\n\n"
                "2. Run CLIChat from anywhere:\n"
                "   clichat\n\n"
            )
        ).run()
        
    except Exception as e:
        message_dialog(
            title="Installation Error",
            text=f"An error occurred during installation:\n{str(e)}"
        ).run()
        sys.exit(1)

def main():
    print("Welcome to CLIChat Installer")
    venv_path = create_virtual_environment()
    
    # No need for activation, just use the full paths
    run_installation(venv_path)

if __name__ == "__main__":
    main()
