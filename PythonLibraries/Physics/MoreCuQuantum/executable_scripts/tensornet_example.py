import numpy as np
import cupy as cp

import cuquantum
from cuquantum import custatevec as cusv
from cuquantum import cutensornet as cutn

print("========================")
print("========================")
print("cuTensorNet-vers:", cutn.get_version())
dev = cp.cuda.Device()  # get current device
props = cp.cuda.runtime.getDeviceProperties(dev.id)
print("===== device info ======")
print("GPU-local-id:", dev.id)
print("GPU-name:", props["name"].decode())
print("GPU-clock:", props["clockRate"])
print("GPU-memoryClock:", props["memoryClockRate"])
print("GPU-nSM:", props["multiProcessorCount"])
print("GPU-major:", props["major"])
print("GPU-minor:", props["minor"])
print("========================")
print("========================")

# |00>, |01>, |10>, |11> space.
#NIndexBits = 2

# |000>, |001>, |010>, |011>, ... space.
NIndexBits = 3
svSize = (1 << NIndexBits)

n_targets = 1
n_controls = 2
adjoint = 0

# 0.2|001> + 0.4|011> - 0.4|101> - 0.8|111>
sv = cp.asarray([0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j,
                         0.2+0.2j, 0.3+0.3j, 0.3+0.4j, 0.4+0.5j],
                dtype=cp.complex128)

nTargets   = 1
nControls  = 2
adjoint    = 0

targets    = np.asarray([2], dtype=np.int32)
controls   = np.asarray([0, 1], dtype=np.int32)

# the gate matrix can live on either host (np) or device (cp)
# CNOT gate on qubits 0 and 1.
matrix     = cp.asarray([0.0+0.0j, 1.0+0.0j, 1.0+0.0j, 0.0+0.0j], dtype=np.complex64)

if isinstance(matrix, cp.ndarray):
    matrix_ptr = matrix.data.ptr
elif isinstance(matrix, np.ndarray):
    matrix_ptr = matrix.ctypes.data
else:
    raise ValueError

#d_sv = cp.asarray(h_sv)

####################################################################################

# cuStateVec handle initialization
handle = cusv.create()

workspaceSize = cusv.apply_matrix_get_workspace_size(
    handle,
    cuquantum.cudaDataType.CUDA_C_32F,
    NIndexBits,
    matrix_ptr,
    cuquantum.cudaDataType.CUDA_C_32F,
    cusv.MatrixLayout.ROW,
    adjoint,
    nTargets,
    nControls,
    cuquantum.ComputeType.COMPUTE_32F)

# check the size of external workspace
if workspaceSize > 0:
    workspace = cp.cuda.memory.alloc(workspaceSize)
    workspace_ptr = workspace.ptr
else:
     workspace_ptr = 0

# Print the initial state vector
print("\nInitial state vector:")
sv_initial = sv.get()
for i, value in enumerate(sv_initial):
    print(f"|{i:03b}>: {value}")

# Print the initial probabilities
probabilities_initial = np.abs(sv_initial)**2
print("\nInitial probabilities:")
for i, prob in enumerate(probabilities_initial):
    print(f"|{i:03b}>: {prob:.4f}")

print("\nApplying matrix...")

# apply gate
cusv.apply_matrix(
    handle,
    sv.data.ptr,
    cuquantum.cudaDataType.CUDA_C_32F,
    NIndexBits,
    matrix_ptr,
    cuquantum.cudaDataType.CUDA_C_32F,
    cusv.MatrixLayout.ROW,
    adjoint,
    targets.ctypes.data,
    nTargets,
    controls.ctypes.data,
    0,
    nControls,
    cuquantum.ComputeType.COMPUTE_32F, workspace_ptr, workspaceSize)


# Convert the device array to host and print the results
sv_host = sv.get()
print("\nState vector after applying the matrix:")
for i, value in enumerate(sv_host):
    print(f"|{i:03b}>: {value}")

# Print the probabilities
probabilities = np.abs(sv_host)**2
print("\nProbabilities:")
for i, prob in enumerate(probabilities):
    print(f"|{i:03b}>: {prob:.4f}")

# destroy handle
cusv.destroy(handle)
