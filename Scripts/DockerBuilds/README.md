# Docker (itself) Installation

To install Docker onto a new computer/system, follow these instructions:

https://docs.docker.com/engine/install/debian/

including the Linux post-installation steps:

https://docs.docker.com/engine/install/linux-postinstall/

## Move where Docker stores the containers

If you're like me, you'll want to move where the Docker stores the containers to another mounted file system/hard drive/SSD that has more storage, because containers can get large in size.

### Change `/etc/docker/daemon.json`

Following https://www.simplified.guide/docker/change-data-directory

where are the dockers stored? See Docker Rootdir in
docker info 

sudo systemctl stop docker
sudo mv /var/lib/docker <new-path-todockers>
e.g.
sudo mv /var/lib/docker /media/tx2i/Samsung256/docker

sudo nano /etc/docker/daemon.json

e.g.
$ cat /etc/docker/daemon.json
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
   "data-root": "/media/tx2i/Samsung256/docker"
}


sudo systemctl restart docker

Confirm change:
docker info

