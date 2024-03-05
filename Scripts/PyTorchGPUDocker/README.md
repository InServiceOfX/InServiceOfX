# Installing NVIDIA Container Toolkit

From https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html,

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg   && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

Then do the following:
```
# Configure the container runtime (for docker)
sudo nvidia-ctk runtime configure --runtime=docker

# Restart the Docker daemon
sudo systemctl restart docker
```

# NVIDIA PyTorch container image from [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

See the release notes:
https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html



# Dockerfile

## Making changes

After you make changes to Dockerfile, you'll need to build the new Docker image.

1. Check name of current image. I typically reuse the current image I am on and have named, for instance `from-nvidia-python-23.08`. You can check for the name with

```
docker images
```

2. After updating your Dockerfile, build Docker image:

```
docker build -t from-nvidia-python-24.02 .
```
where `-t` is for tag, and if it's a previously used name or tag, it'll simply refer to that previously used (image) name or tag.

# Running jupyter notebook from Pytorch GPU Docker

In the command line when you run

```
jupyter notebook
```

You may see

```
(venv) root@7b83500a510c:/InServiceOfX/InServiceOfX# jupyter notebook
[I 16:12:52.667 NotebookApp] Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/notebook_cookie_secret
[I 16:12:53.565 NotebookApp] jupyter_tensorboard extension loaded.
[I 16:12:53.832 NotebookApp] JupyterLab extension loaded from /usr/local/lib/python3.10/dist-packages/jupyterlab
[I 16:12:53.832 NotebookApp] JupyterLab application directory is /usr/local/share/jupyter/lab
[I 16:12:53.834 NotebookApp] [Jupytext Server Extension] NotebookApp.contents_manager_class is (a subclass of) jupytext.TextFileContentsManager already - OK
[I 16:12:53.839 NotebookApp] Serving notebooks from local directory: /InServiceOfX/InServiceOfX
[I 16:12:53.839 NotebookApp] Jupyter Notebook 6.4.10 is running at:
[I 16:12:53.839 NotebookApp] http://hostname:8888/?token=6fa6967ab7a28a0eb853f8d506bdf1fb7358cdc8263fd555
[I 16:12:53.839 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 16:12:53.842 NotebookApp] 
    
    To access the notebook, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/nbserver-381-open.html
    Or copy and paste this URL:
        http://hostname:8888/?token=6fa6967ab7a28a0eb853f8d506bdf1fb7358cdc8263fd555
[I 16:13:14.390 NotebookApp] 302 GET /?token=6fa6967ab7a28a0eb853f8d506bdf1fb7358cdc8263fd555 (172.17.0.1) 0.500000ms

```

What worked for me was to replace the `<hostname>` with `localhost` and keep the token; so in the browser I entered this:

```
http://localhost:8888/?token=6fa6967ab7a28a0eb853f8d506bdf1fb7358cdc8263fd555
```

## LangChain and Docker file

The [langchain repository](https://github.com/ernestyalumni/langchain) consists of a number of "top level" subdictories, including 'libs', 'docker', 'cookbook', 'templates'. If you do `pip install -e .` at the "root" directory of the repository, such as `langchain/` then you obtain this error:

```
Obtaining file:///home/propdev/Prop/langchain
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... error
  error: subprocess-exited-with-error
  
  × Getting requirements to build editable did not run successfully.
  │ exit code: 1
  ╰─> [14 lines of output]
      error: Multiple top-level packages discovered in a flat-layout: ['libs', 'docker', 'cookbook', 'templates'].
      
      To avoid accidental inclusion of unwanted files or directories,
      setuptools will not proceed with this build.
      
      If you are trying to create a single distribution with multiple packages
      on purpose, you should not rely on automatic discovery.
      Instead, consider the following options:
      
      1. set up custom discovery (`find` directive with `include` or `exclude`)
      2. use a `src-layout`
      3. explicitly set `py_modules` or `packages` with a list of names
      
      To find more information, look for "package discovery" on setuptools docs.
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error

× Getting requirements to build editable did not run successfully.
│ exit code: 1
╰─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.

``` 

What you want to do is to change directory (`cd`) into a desired subdirectory and then run `pip install -e .`. I wanted `langchain` and so I changed directories to `langchain/libs/langchain`.