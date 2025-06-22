# InServiceOfX
Monorepo (single or "mono" repository) for deep learning.

## Creating and starting a uv environment for Python 3 to get started

```
uv venv
source .venv/bin/activate
```

Then proceed to build one of the Docker images in `Scripts/DockerBuilds/Builds/` (Make a local copy and configure the necessary `*.txt` files, then run `python BuildDocker.py`)

## (Deprecating) Creating and starting a virtual environment for Python 3

Create a directory for a virtual environment:

```
/InServiceOfX$ python3 -m venv ./venv/
```

Activate it:
```
/InServiceOfX$ source ./venv/bin/activate
```
You should see the prompt have a prefix `(venv)`.

Deactivate it:
```
deactivate
```

## Running Python tests

From the "base directory" of this repository, you may run the Python unit tests and integration tests as follows:

```
$ pytest ./ThirdParty/NeuralOperators
```
This is a specific example (you may change it) illustrating running the integration tests for `neuraloperators`, and it seems pytest will recursively run all the tests it detects in all the subdirectories.

## Setting up a Virtual Machine

Requiring manual effort:

- edit `/etc/docker/daemon.json` for place to put Docker containers, in field "data-root"
  * References:
  	- https://medium.com/@calvineotieno010/change-docker-default-root-data-directory-a1d9271056f4
  	- https://diegocarrasco.com/change-docker-data-directory-vps-optimization/
  * 
```
docker info | grep 'Docker Root Dir'
```

Example `/etc/docker/daemon.json`
```
{
    "data-root": "/mnt/ey/docker",
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
```
However, read about the difference between storage driver "overlay2" for Docker vs. storage driver "vfs":

https://docs.docker.com/storage/storagedriver/overlayfs-driver/
https://docs.docker.com/storage/storagedriver/vfs-driver/

vfs is very slow compared to overlayfs because vfs does a deep copy (of each layer). If the filesystem mounted is virtiofs type, then if you try to force "storage-driver": "overlay2", it will return error, for instance 

```
Aug 01 08:03:32 ernest-yeung dockerd[18151]: time="2024-08-01T08:03:32.397154488Z" level=error msg="failed to mount overlay: invalid argument" storage-driver=overlay2
Aug 01 08:03:32 ernest-yeung dockerd[18151]: failed to start daemon: error initializing graphdriver: driver not supported: overlay2
Aug 01 08:03:32 ernest-yeung systemd[1]: docker.service: Main process exited, code=exited, status=1/FAILURE
```
When starting and stopping Docker services,
don't do sudo systemctl stop docker.socket, you'll need root access to restart it again. When I inadvertently did so, what I ended up doing was deleting the virtual machine and starting over again.
https://www.ibm.com/docs/en/z-logdata-analytics/5.1.0?topic=software-relocating-docker-root-directory

- Register ssh keys to github:
  * `ssh-keygen -t ed25519 -C "your_email@example.com"``
  	- https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent



In `SetupVirtualMachine.sh` script
- git lfs install
- manage Docker as a non-root user, so that running docker doesn't require sudo.
- Install nvidia-container-toolkit as 1 of 3 prerequisites for using NVIDIA Pytorch Docker.
- Let's follow https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#keyconcepts and do a `docker pull` on the Docker container we'll build on top of, from NVIDIA Container Pytorch.
  * 