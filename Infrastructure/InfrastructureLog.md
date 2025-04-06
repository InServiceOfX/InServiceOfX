Install `kubeadm`, `kubectl`, `kubelet` using `Scripts/install_kubernetes_components.py`.

```
# Get your IP address
ip addr show

sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --apiserver-advertise-address=192.168.86.91

# --v=5 for verbose output for backtrace
sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --apiserver-advertise-address=192.168.8X.9X --v=5
```

It was advised not to use for CIDR 192.168.X.X because it may conflict with local network; TODO: investigate how to assign 10.244.X.X to other nodes.

I needed to troubleshoot this when I ran:

```
sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --apiserver-advertise-address=192.168.86.46

```
(I used 192.168.8X.4X, the IP address of my wifi since for some reason with Apple's smaller router, the ethernet connection was slower)


```
[init] Using Kubernetes version: v1.32.3
[preflight] Running pre-flight checks
W0405 18:41:23.539873   33610 checks.go:1077] [preflight] WARNING: Couldn't create the interface used for talking to the container runtime: failed to create new CRI runtime service: validate service connection: validate CRI v1 runtime API for endpoint "unix:///var/run/containerd/containerd.sock": rpc error: code = Unimplemented desc = unknown service runtime.v1.RuntimeService
	[WARNING Swap]: swap is supported for cgroup v2 only. The kubelet must be properly configured to use swap. Please refer to https://kubernetes.io/docs/concepts/architecture/nodes/#swap-memory, or disable swap on the node
[preflight] Pulling images required for setting up a Kubernetes cluster
[preflight] This might take a minute or two, depending on the speed of your internet connection
[preflight] You can also perform this action beforehand using 'kubeadm config images pull'
error execution phase preflight: [preflight] Some fatal errors occurred:
failed to create new CRI runtime service: validate service connection: validate CRI v1 runtime API for endpoint "unix:///var/run/containerd/containerd.sock": rpc error: code = Unimplemented desc = unknown service runtime.v1.RuntimeService[preflight] If you know what you are doing, you can make a check non-fatal with `--ignore-preflight-errors=...`
To see the stack trace of this error execute with --v=5 or higher

```
I commented out the following in `/etc/containerd/config.toml`,

```
#disabled_plugins = ["cri"]
```
and then run

```
sudo systemctl restart containerd
sudo systemctl status containerd
```

and then re-ran the `sudo kubeadm init` command.

If you see a "failed" error involving swap, then disable swap:

```
# Check if swap is enabled or not
free -h
```

```
sudo swapoff -a
free -h
```

Rerunning `sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --apiserver-advertise-address=192.168.8X.9X`, I saw this for success:

```
I0405 19:07:49.207500   36405 request.go:661] Waited for 178.214669ms due to client-side throttling, not priority and fairness, request: POST:https://192.168.86.46:6443/apis/rbac.authorization.k8s.io/v1/namespaces/kube-system/roles?timeout=10s
[addons] Applied essential addon: kube-proxy

Your Kubernetes control-plane has initialized successfully!

To start using your cluster, you need to run the following as a regular user:

  mkdir -p $HOME/.kube
  sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
  sudo chown $(id -u):$(id -g) $HOME/.kube/config

Alternatively, if you are the root user, you can run:

  export KUBECONFIG=/etc/kubernetes/admin.conf

You should now deploy a pod network to the cluster.
Run "kubectl apply -f [podnetwork].yaml" with one of the options listed at:
  https://kubernetes.io/docs/concepts/cluster-administration/addons/

Then you can join any number of worker nodes by running the following on each as root:

kubeadm join 192.168.8X.4X:6443 --token 6vg0q4.5fw37oulgrydtx9f \
	--discovery-token-ca-cert-hash sha256:9afcffd507f415bfa34f468135a2a1062d8a2d78903b195593518e241368XXXX
```

Then

```
kubectl apply -f https://raw.githubusercontent.com/flannel-io/flannel/master/Documentation/kube-flannel.yml
```
If this fails, for instance,

```
error: error validating "https://raw.githubusercontent.com/flannel-io/flannel/master/Documentation/kube-flannel.yml": error validating data: failed to download openapi: Get "https://192.168.86.46:6443/openapi/v2?timeout=32s": dial tcp 192.168.86.46:6443: connect: connection refused; if you choose to ignore these errors, turn validation off with --validate=false
```

then try

```
sudo kubeadm reset -f
```

and then redo these steps:

```
sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --apiserver-advertise-address=192.168.86.46
mkdir -p $HOME/.kube
sudo cp /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
kubectl apply -f https://raw.githubusercontent.com/flannel-io/flannel/master/Documentation/kube-flannel.yml
```
After getting this (success)

```
$ kubectl apply -f https://raw.githubusercontent.com/flannel-io/flannel/master/Documentation/kube-flannel.yml
namespace/kube-flannel created
clusterrole.rbac.authorization.k8s.io/flannel created
clusterrolebinding.rbac.authorization.k8s.io/flannel created
serviceaccount/flannel created
configmap/kube-flannel-cfg created
daemonset.apps/kube-flannel-ds created
```

# `minikube`

## Get started with `minikube`

See
https://minikube.sigs.k8s.io/docs/start/?arch=%2Flinux%2Fx86-64%2Fstable%2Fbinary+download

```
curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64
```

```
minikube start

# po is for pods
kubectl get po -A
# or
kubectl get pods -A

minikube dashboard
```

### Start and Stop minikube

```
minikube start
minikube stop
```


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

# `helm`

https://helm.sh/docs/intro/quickstart/

```
helm repo list

helm search repo prometheus
```

# `prometheus`

## Installing prometheus using helm

https://artifacthub.io/packages/helm/prometheus-community/prometheus

```
helm install --generate-name prometheus-community/prometheus
```

## Running prometheus on minikube

```
# Find the name of the service
kubectl get services -A

minikube service prometheus-1743949465-server
```