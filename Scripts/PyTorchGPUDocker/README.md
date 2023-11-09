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