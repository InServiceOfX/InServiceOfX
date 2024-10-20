https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuquantum-appliance

From this page:

NVIDIA cuQuantum Appliance

The NVIDIA cuQuantum Appliance is a highly performant multi-GPU multi-node
solution for quantum circuit simulation. It contains NVIDIA’s cuStateVec and
cuTensorNet libraries which optimize state vector and tensor network simulation,
respectively. The cuTensorNet library functionality is accessible through Python
for Tensor Network operations. With the cuStateVec libraries, NVIDIA provides
the following simulators:

    IBM’s Qiskit Aer frontend via cusvaer, NVIDIA’s distributed state vector
    backend solver.
    An optimized multi-GPU Google Cirq frontend via qsim, Google’s state vector
    simulator.

Prerequisites

Using NVIDIA’s cuQuantum Appliance NGC Container requires the host system to
have the following installed:

    Docker Engine
    NVIDIA GPU Drivers
    NVIDIA Container Toolkit
    For supported versions, see the container release notes. No other installation, compilation, or dependency management is required.

Running the NVIDIA cuQuantum Appliance with Cirq or Qiskit

Note: ${march} is one of [x86_64, arm64].


# pull the image
...$ docker pull nvcr.io/nvidia/cuquantum-appliance:24.08-${march}
# launch the container interactively
...$ docker run --gpus all \
       -it --rm nvcr.io/nvidia/cuquantum-appliance:24.08-${march}
# interactive launch, but enumerate only GPUs 0,3
...$ docker run --gpus '"device=0,3"' \
       -it --rm nvcr.io/nvidia/cuquantum-appliance:24.08-${march}

The examples are located under /home/cuquantum/examples. Confirm this with the
following command:


...$ docker run --gpus all --rm \
...$ nvcr.io/nvidia/cuquantum-appliance:24.08-${march} ls \
       -la /home/cuquantum/examples
...

==========================================================================
===                 NVIDIA CUQUANTUM APPLIANCE v24.08                  ===
==========================================================================
=== COPYRIGHT © NVIDIA CORPORATION & AFFILIATES.  All rights reserved. ===
==========================================================================

INFO: nvidia devices detected
INFO: gpu functionality will be available

total 36
drwxr-xr-x 2 cuquantum cuquantum 4096 Nov 10 01:52 .
drwxr-x--- 1 cuquantum cuquantum 4096 Nov 10 01:54 ..
-rw-r--r-- 1 cuquantum cuquantum 2150 Nov 10 01:52 ghz.py
-rw-r--r-- 1 cuquantum cuquantum 7436 Nov 10 01:52 hidden_shift.py
-rw-r--r-- 1 cuquantum cuquantum 1396 Nov 10 01:52 qiskit_ghz.py
-rw-r--r-- 1 cuquantum cuquantum 8364 Nov 10 01:52 simon.py

Running the examples is straightforward:


#### without an interactive session:
...$ docker run --gpus all --rm \
       nvcr.io/nvidia/cuquantum-appliance:24.08-${march} \
         python /home/cuquantum/examples/{example_name}.py
#### with an interactive session:
...$ docker run --gpus all --rm -it \
       nvcr.io/nvidia/cuquantum-appliance:24.08-${march}
...
(cuquantum-24.08) cuquantum@...:~$ cd examples && python {example_name}.py

The examples all accept runtime arguments. To see what they are, pass --help
to the python + script command. Looking at two examples, ghz.py and
qiskit_ghz.py, the help messages are as follows:


(cuquantum-24.08) cuquantum@...:~/examples$ python ghz.py --help
usage: ghz.py [-h] [--nqubits NQUBITS] [--nsamples NSAMPLES] [--ngpus NGPUS]

GHZ circuit

options:
  -h, --help           show this help message and exit
  --nqubits NQUBITS    the number of qubits in the circuit
  --nsamples NSAMPLES  the number of samples to take
  --ngpus NGPUS        the number of GPUs to use


(cuquantum-24.08) cuquantum@...:~/examples$ python qiskit_ghz.py --help
usage: qiskit_ghz.py [-h] [--nbits NBITS] [--precision {single,double}] [--disable-cusvaer]

Qiskit ghz.

options:
  -h, --help            show this help message and exit
  --nbits NBITS         the number of qubits
  --precision {single,double}
                        numerical precision
  --disable-cusvaer     disable cusvaer

Importantly, ghz.py implements the GHZ circuit using Cirq as a frontend, and
qiskit_ghz.py implements the GHZ circuit using Qiskit as a frontend. The
cuQuantum Appliance modifies the backends of these frameworks, optimizing them
for use with Nvidia's platforms. Information regarding any alterations are
available in the Appliance section of the Nvidia cuQuantum documentation.

Running cd examples && python ghz.py --nqubits 30 will create and simulate a
GHZ circuit running on a single GPU. To run on 4 available GPUs, use
... python ghz.py --nqubits 30 --ngpus 4. The output will look something like this:


(cuquantum-24.08) cuquantum@...:~/examples$ python ghz.py --nqubits 30
q(0),...,q(29)=111,...,111

Likewise, cd examples && python qiskit_ghz.py --nbits 30 will create and simulate a GHZ circuit. This script will assign one GPU per process. To run on 4 GPUs, you need to explicitly enumerate
the GPUs you want to use and execute with MPI:


#### interactively:
...$ docker run --gpus '"device=0,1,2,3"' \
       -it --rm nvcr.io/nvidia/cuquantum-appliance:24.08-${march}
(cuquantum-24.08) cuquantum@...:~$ cd examples
(cuquantum-24.08) cuquantum@...:~$ mpirun -np 4 python qiskit_ghz.py --nbits 30
#### noninteractively:
...$ docker run --gpus '"device=0,1,2,3"' \
       --rm nvcr.io/nvidia/cuquantum-appliance:24.08-${march} \
       mpirun -np 4 python /home/cuquantum/examples/qiskit_ghz.py --nbits 30

The output from qiskit_ghz.py looks like this:


(cuquantum-24.08) cuquantum@...:~$ cd examples
(cuquantum-24.08) cuquantum@...:~$ python qiskit_ghz.py --nbits 30
...
precision: single
{'0...0': 520, '1...1': 504}

More information, examples, and utilities are available in the NVIDIA cuQuantum
repository on GitHub. Notably, you can
find useful guides for getting started with multi-node multi-GPU simulation
using the benchmarks tools.
Known issues
For tags: *23.10-*-arm64

Note: this issue is fixed in the 24.03 release.

When using ssh in the container, the following error is emitted:


(cuquantum-23.10) cuquantum@...:~$ ssh ...
OpenSSL version mismatch. ...

As a workaround, please specify LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libcrypto.so.3:


LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libcrypto.so.3 ssh ...

Software in the container
Default user environment

The default user in the container is cuquantum with user ID 1000. The
cuquantum user is a member of the sudo group. By default, executing commands
with sudo using the cuquantum user requires a password which can be obtained
by reading the file located at /home/cuquantum/.README formatted as
{user}:{password}.

To acquire new packages, we recommend using conda install -c conda-forge ...
in the default environment (cuquantum-24.03). You may clone this environment
and change the name using conda create --name {new_name} --clone cuquantum-24.03.
This may be useful in isolating your changes from the default environment.

CUDA is available under /usr/local/cuda. /usr/local/cuda is a symbolic
directory managed by update-alternatives. To query configuration information,
use update-alternatives --config cuda.
MPI

We provide Open MPI v4.1 in the container located at /usr/local/openmpi. The
default mpirun runtime configuration can be queried with ompi_info --all --parseable.
When using the multi-GPU features in the cuQuantum Appliance, a valid and compatible
mpirun runtime configuration must be exposed to the
container. It must also be accessible to the container runtime.

Important change notices
version == 24.08

Introducing Python 3.11.9

The version of Python in the container was updated from 3.10.13 to 3.11.9 to accommodate required security remediation and compliance.

The following image tags are available:


nvcr.io/nvidia/cuquantum-appliance:24.08-cuda11.8.0-devel-ubuntu20.04-x86_64
nvcr.io/nvidia/cuquantum-appliance:24.08-cuda12.2.2-devel-ubuntu20.04-x86_64
nvcr.io/nvidia/cuquantum-appliance:24.08-cuda11.8.0-devel-ubuntu22.04-x86_64
nvcr.io/nvidia/cuquantum-appliance:24.08-cuda12.2.2-devel-ubuntu22.04-x86_64

nvcr.io/nvidia/cuquantum-appliance:24.08-cuda11.8.0-devel-ubuntu20.04-arm64
nvcr.io/nvidia/cuquantum-appliance:24.08-cuda12.2.2-devel-ubuntu20.04-arm64
nvcr.io/nvidia/cuquantum-appliance:24.08-cuda11.8.0-devel-ubuntu22.04-arm64
nvcr.io/nvidia/cuquantum-appliance:24.08-cuda12.2.2-devel-ubuntu22.04-arm64

nvcr.io/nvidia/cuquantum-appliance:24.08-${march} is equivalent to
nvcr.io/nvidia/cuquantum-appliance:24.08-cuda12.2.2-devel-ubuntu22.04-${march}.
The following two docker pull commands will download the same image.


docker nvcr.io/nvidia/cuquantum-appliance:24.08-${march}


docker pull nvcr.io/nvidia/cuquantum-appliance:24.08-cuda12.2.2-devel-ubuntu22.04-${march}

version == 24.03

The following image tags are available:


nvcr.io/nvidia/cuquantum-appliance:24.03-cuda11.8.0-devel-ubuntu20.04-x86_64
nvcr.io/nvidia/cuquantum-appliance:24.03-cuda12.2.2-devel-ubuntu20.04-x86_64
nvcr.io/nvidia/cuquantum-appliance:24.03-cuda11.8.0-devel-ubuntu22.04-x86_64
nvcr.io/nvidia/cuquantum-appliance:24.03-cuda12.2.2-devel-ubuntu22.04-x86_64

nvcr.io/nvidia/cuquantum-appliance:24.03-cuda11.8.0-devel-ubuntu20.04-arm64
nvcr.io/nvidia/cuquantum-appliance:24.03-cuda12.2.2-devel-ubuntu20.04-arm64
nvcr.io/nvidia/cuquantum-appliance:24.03-cuda11.8.0-devel-ubuntu22.04-arm64
nvcr.io/nvidia/cuquantum-appliance:24.03-cuda12.2.2-devel-ubuntu22.04-arm64

nvcr.io/nvidia/cuquantum-appliance:24.03-${march} is equivalent to
nvcr.io/nvidia/cuquantum-appliance:24.03-cuda12.2.2-devel-ubuntu22.04-${march}.
The following two docker pull commands will download the same image.


docker nvcr.io/nvidia/cuquantum-appliance:24.03-${march}


docker pull nvcr.io/nvidia/cuquantum-appliance:24.03-cuda12.2.2-devel-ubuntu22.04-${march}
