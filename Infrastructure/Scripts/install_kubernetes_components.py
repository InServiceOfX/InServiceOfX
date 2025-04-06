#!/usr/bin/env python3
# Scripts/install_kubernetes_components.py
# See also
# https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/

import subprocess
import getpass
import os
import sys
import argparse

def run_command(command):
    """Run a command and return its output and success status"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout, result.returncode == 0
    except Exception as e:
        print(f"Error executing command: {e}")
        return "", False

def run_sudo_command(command, password=None):
    """
    Run a command with sudo, optionally providing the password.
    
    Args:
        command (str): The command to run
        password (str, optional): The sudo password
        
    Returns:
        bool: True if the command succeeded, False otherwise
    """
    if password:
        # Use a secure way to pass the password to sudo
        sudo_command = f"sudo -S {command}"
        try:
            process = subprocess.Popen(
                sudo_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                text=True
            )
            stdout, stderr = process.communicate(input=password + '\n')
            
            if process.returncode != 0:
                print(f"Command failed: {command}")
                print(f"Error: {stderr}")
                return False
            
            print(stdout)
            return True
        except Exception as e:
            print(f"Error executing command: {e}")
            return False
    else:
        # If no password provided, just run with sudo and let it prompt
        try:
            result = subprocess.run(
                f"sudo {command}",
                shell=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {command}")
            print(f"Error: {e}")
            return False

def is_package_installed(package_name):
    """Check if a package is already installed"""
    output, success = run_command(
        f"dpkg-query -W -f='${{Status}}' {package_name} 2>/dev/null")
    return success and "installed" in output

def is_package_on_hold(package_name):
    """Check if a package is on hold"""
    output, success = run_command(f"apt-mark showhold | grep '^{package_name}$'")
    return success and package_name in output

def add_kubernetes_repository(password, k8s_version):
    """Add the Kubernetes repository to apt sources.
    
    See
    https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/"""
    print("\nAdding Kubernetes repository...")
    
    # Install prerequisites
    if not run_sudo_command("apt-get update", password):
        return False
    
    if not run_sudo_command("apt-get install -y apt-transport-https ca-certificates curl gnupg", password):
        return False
    
    # Download and add the Kubernetes GPG key
    key_cmd = f"curl -fsSL https://pkgs.k8s.io/core:/stable:/v{k8s_version}/deb/Release.key | " \
              f"sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg"
    if not run_command(key_cmd)[1]:
        return False
    
    # Set proper permissions for the keyring
    if not run_sudo_command(
        "chmod 644 /etc/apt/keyrings/kubernetes-apt-keyring.gpg",
        password):
        return False
    
    # Add the Kubernetes repository
    repo_cmd = f"echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] " \
               f"https://pkgs.k8s.io/core:/stable:/v{k8s_version}/deb/ /' | " \
               f"sudo tee /etc/apt/sources.list.d/kubernetes.list"
    if not run_command(repo_cmd)[1]:
        return False
    
    # Set proper permissions for the sources list
    if not run_sudo_command(
        "chmod 644 /etc/apt/sources.list.d/kubernetes.list",
        password):
        return False
    
    # Update package lists again
    if not run_sudo_command("apt-get update", password):
        return False
    
    return True

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Install Kubernetes components")
    parser.add_argument("--k8s-version", default="1.32", help="Kubernetes version to install (default: 1.32)")
    return parser.parse_args()

def main():
    args = parse_arguments()
    k8s_version = args.k8s_version
    
    print("Kubernetes Components Installer")
    print("===============================")
    print("This script will install and configure the following Kubernetes components:")
    print()
    print("1. kubelet - The primary node agent that runs on each node in the cluster")
    print("   It ensures containers are running in a Pod and healthy.")
    print()
    print("2. kubeadm - A tool to bootstrap a Kubernetes cluster, simplifying the")
    print("   process of setting up a minimum viable Kubernetes cluster.")
    print()
    print("3. kubectl - The command-line tool for interacting with the Kubernetes API")
    print("   Used to deploy applications, inspect and manage cluster resources.")
    print()
    print(f"Kubernetes version: v{k8s_version}")
    print()
    
    # Ask for sudo password once
    password = getpass.getpass("Enter sudo password (or press Enter to be prompted for each command): ")
    
    # Check which components are already installed
    components = ["kubelet", "kubeadm", "kubectl"]
    to_install = []
    already_installed = []
    on_hold = []
    
    for component in components:
        if is_package_installed(component):
            already_installed.append(component)
            if is_package_on_hold(component):
                on_hold.append(component)
        else:
            to_install.append(component)
    
    # Report status
    if already_installed:
        print(f"\nAlready installed: {', '.join(already_installed)}")
    if on_hold:
        print(f"On hold: {', '.join(on_hold)}")
    if to_install:
        print(f"Will install: {', '.join(to_install)}")
    
    # Ask for confirmation if some components are already on hold
    if on_hold and to_install:
        confirm = input(
            "\nSome components are already on hold. Continue with installation? (y/n): ")
        if confirm.lower() != 'y':
            print("Installation aborted.")
            sys.exit(0)
    
    # Update package lists
    print("\nUpdating package lists...")
    if not run_sudo_command("apt-get update", password):
        print("Failed to update package lists. Aborting.")
        sys.exit(1)
    
    # Install components if needed
    if to_install:
        install_cmd = f"apt-get install -y {' '.join(to_install)}"
        print(f"\nInstalling: {' '.join(to_install)}...")
        
        # Try standard installation first
        standard_install_success = run_sudo_command(install_cmd, password)
        
        # If standard installation fails, try adding the Kubernetes repository
        if not standard_install_success:
            print("\nStandard installation failed. Adding Kubernetes repository...")
            if not add_kubernetes_repository(password, k8s_version):
                print("Failed to add Kubernetes repository. Aborting.")
                sys.exit(1)
            
            # Try installation again
            print(f"\nRetrying installation: {' '.join(to_install)}...")
            if not run_sudo_command(install_cmd, password):
                print("Installation failed even after adding the repository. Please check your network connection and try again.")
                sys.exit(1)
    else:
        print("\nAll components are already installed.")
    
    # Mark components on hold
    not_on_hold = [c for c in components if c not in on_hold]
    if not_on_hold:
        hold_cmd = f"apt-mark hold {' '.join(not_on_hold)}"
        print(f"\nMarking as held: {' '.join(not_on_hold)}...")
        if not run_sudo_command(hold_cmd, password):
            print("Failed to mark packages as held. Please do this manually.")
            sys.exit(1)
    else:
        print("\nAll components are already on hold.")
    
    print("\nKubernetes components setup completed successfully.")
    print("You can now proceed with Kubernetes cluster setup.")

if __name__ == "__main__":
    # Check if running as root
    if os.geteuid() == 0:
        print("This script should not be run directly as root.")
        print("Please run as a normal user with sudo privileges.")
        sys.exit(1)
    
    main()
