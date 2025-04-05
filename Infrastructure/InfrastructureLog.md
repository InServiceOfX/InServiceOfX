Install `kubeadm`, `kubectl`, `kubelet` using `Scripts/install_kubernetes_components.py`.

```
# Get your IP address
ip addr show

sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --apiserver-advertise-address=192.168.86.91

# --v=5 for verbose output for backtrace
sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --apiserver-advertise-address=192.168.86.91 --v=5
```

It was advised not to use for CIDR 192.168.X.X because it may conflict with local network; TODO: investigate how to assign 10.244.X.X to other nodes.

# `kind`

## Installation

https://kind.sigs.k8s.io/docs/user/quick-start/

```
[ $(uname -m) = x86_64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.27.0/kind-linux-amd64
```

Diagnose if a cluster (and nodes)exists already or still doesn't.
```
kind get clusters
kubectl cluster-info
kubectl get nodes
```

```
kind create cluster --name kind-dev
```
