#!/bin/bash

# USAGE:
# bash BuildOpenCVWithCUDA.sh <ARCH> <PTX> [OPENCV_VERSION] [PYTHON_VERSION] [BUILD_NAME]

# We will compare CMake flags from multiple sources, including
# from doing docker history --no-trunc 2ef8676cffb9 | grep -A 3 -B 3 opencv
# and seeing the layer,
# https://github.com/Qengineering/Install-OpenCV-Jetson-Nano/blob/main/OpenCV-4-9-0.sh
generate_cmake_flags()
{
  local ARCH="$1"
  local PTX="$2"
  # Example value, OPENCV_VERSION=4.10.0
  # :- checks if $3 in this case is unset or null (empty), called parameter
  # expansion.
  # https://www.gnu.org/software/bash/manual/html_node/Shell-Parameter-Expansion.html
  # ${parameter:-word}
  # if parameter unset or null, word is substituted.
  local OPENCV_VERSION="${3:-4.11.0}"
  # Example value, PYTHON_VERSION=3.10, which is found in the NVIDIA PyTorch
  # Container.
  local PYTHON_VERSION="${4:-3.10}"

  # From opencv/CMakelists.txt
  # CMAKE_INSTALL_PREFIX default is "/usr/local"
  # EIGEN_INCLUDE_PATH, in opencv/cmake/OpenCVFindLibPerf.cmake, doesn't appear
  # in NVIDIA Docker container. TODO: Some Eigen functions could be used in CUDA
  # kernel, so consider adding it.

  local cmake_flags="-D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local "

  # OpenCV cmake options
  # 3rd party libs
  # TBB, Intel Threading Building Block, C++ template library for multicore,
  # parallel programming.
  cmake_flags+="-D OPENCV_ENABLE_NONFREE=ON \
    -D BUILD_TIFF=OFF \
    -D BUILD_TBB=OFF "

  # Optional 3rd party components
  # 1394, include IEEE1394 support, default ON. This was turned OFF in the
  # NVIDIA PyTorch Docker; I will leave it ON.
  # NVCUVID, NVIDIA Video decoding library support, default WITH_CUDA value.
  # NVCUVENC, NVIDIA Video encoding library support, default WITH_CUDA value.
  # EIGEN, include Eigen2/Eigen3 support
  # TBB, include intel TBB support, default, OFF.
  # V4L, Video 4 Linux support, turned off by NVIDIA PyTorch Docker.
  # IPP, Intel IPP support.
  # PROTOBUF, enables libprotobuf, default ON, but was turned OFF in NVIDIA
  # PyTorch Docker; I will leave it on.
  # WITH_1393 was not in QEngineering's cmake configuration.
  #cmake_flags+="-D WITH_1394=ON \
  # WITH_NVCUVID, WITH_NVCUVENC not specified in QEngineering's cmake
  # configuration.
  #     -D WITH_NVCUVID=ON \
  #  -D WITH_NVCUVENC=ON \
  # WITH_TBB=OFF on QEngineering's cmake configuration.
  #     -D WITH_TBB=ON \
  cmake_flags+="-D WITH_CUDA=ON \
    -D WITH_CUFFT=ON \
    -D WITH_CUBLAS=ON \
    -D WITH_CUDNN=ON \
    -D WITH_EIGEN=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_IPP=OFF \
    -D WITH_QT=OFF \
    -D WITH_TBB=OFF \
    -D WITH_OPENMP=ON \
    -D WITH_V4L=ON \
    -D WITH_OPENCL=OFF \
    -D WITH_PROTOBUF=ON "

  # OpenCV build components
  # -D BUILD_JAVA=OFF from QEngineering's cmake configuration.
  cmake_flags+="-D BUILD_DOCS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_JAVA=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_TESTS=OFF "

  # OpenCV installation options
  cmake_flags+="-D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_TESTS=OFF "

  # OpenCV build options
  # NEON, NEON instructions. Originally, in QEngineering, NEON was enabled,
  #     -D ENABLE_NEON=ON \
  # but disabled because it fails hard on x86_64
  # ENABLE_FAST_MATH default OFF, turn it on following QEngineering.
  # OPENCV_GENERATE_PKGCONFIG, generates .pc file for pkg-config, but is
  # deprecated.
  cmake_flags+="-D ENABLE_FAST_MATH=ON \
    -D OPENCV_GENERATE_PKGCONFIG=OFF "

  # Path for additional modules
  cmake_flags+="-D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib-${OPENCV_VERSION}/modules "

  # Other third-party libraries
  cmake_flags+="-D CUDA_FAST_MATH=ON "

  # opencv/cmake/OpenCVDetectPython.cmake
  cmake_flags+="-D PYTHON3_PACKAGES_PATH=/usr/local/lib/python${PYTHON_VERSION}/dist-packages "

  # opencv/cmake/templates/cvconfig.h.in
  cmake_flags+="-D CUDA_ARCH_BIN=${ARCH} \
    -D CUDA_ARCH_PTX=${PTX} "

  # opencv/modules/dnn/CMakeLists.txt
  cmake_flags+="-D OPENCV_DNN_CUDA=ON \
    -D OPENCV_DNN_OPENCL=OFF "
  # opencv/modules/stitching/CMakeLists.txt
  # opencv_cudalegacy, set to OFF in NVIDIA Pytorch container.
  # Commented out because not in QEngineering's cmake configuration.
  cmake_flags+="-D opencv_cudalegacy=OFF "

  # opencv/modules/python/CMakeLists.txt
  cmake_flags+="-D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=ON "

  # opencv/platforms/js/build_js.py
  # BUILD_opencv_stitching set to OFF in build_js.py and NVIDIA Pytorch
  # container.
  # Commented out because not in QEngineering's cmake configuration.
  cmake_flags+="-D BUILD_opencv_stitching=OFF "

  echo "$cmake_flags"
}

generate_one_line_cmake_flags()
{
  local OPENCV_VERSION="$1"
  local PYTHON_VERSION="$2"
  local ARCH="$3"
  local PTX="$4"

  local cmake_flags="-D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_ENABLE_NONFREE=ON -D BUILD_TIFF=OFF -D BUILD_TBB=OFF -D WITH_1394=ON -D WITH_CUDA=ON -D WITH_CUFFT=ON -D WITH_CUBLAS=ON -D WITH_CUDNN=ON -D WITH_NVCUVID=ON -D WITH_NVCUVENC=ON -D WITH_EIGEN=ON -D WITH_FFMPEG=ON -D WITH_GSTREAMER=ON -D WITH_IPP=OFF -D WITH_QT=OFF -D WITH_TBB=ON -D WITH_OPENMP=ON -D WITH_V4L=ON -D WITH_OPENCL=OFF -D WITH_PROTOBUF=ON -D BUILD_DOCS=OFF -D BUILD_EXAMPLES=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D ENABLE_FAST_MATH=ON -D ENABLE_NEON=ON -D OPENCV_GENERATE_PKGCONFIG=OFF -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib-${OPENCV_VERSION}/modules -D CUDA_FAST_MATH=ON -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python${PYTHON_VERSION}/dist-packages -D CUDA_ARCH_BIN=${ARCH} -D CUDA_ARCH_PTX=${PTX} -D OPENCV_DNN_CUDA=ON -D OPENCV_DNN_OPENCL=OFF -D opencv_cudalegacy=OFF -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=ON -D BUILD_opencv_stitching=OFF "
  echo "$cmake_flags"
}

main()
{
  local ARCH="$1"
  local PTX="$2"
  # Example value, OPENCV_VERSION=4.9.0
  # :- checks if $3 in this case is unset or null (empty), called parameter
  # expansion.
  # https://www.gnu.org/software/bash/manual/html_node/Shell-Parameter-Expansion.html
  # ${parameter:-word}
  # if parameter unset or null, word is substituted.
  local OPENCV_VERSION="${3:-4.11.0}"
  # Example value, PYTHON_VERSION=3.10, which is found in the NVIDIA PyTorch
  # Container.
  local PYTHON_VERSION="${4:-3.10}"

  local BUILD_NAME="${5:-Build}"

  # Generate CMake flags
  local cmake_flags=$(generate_cmake_flags "$ARCH" "$PTX" "$OPENCV_VERSION" "$PYTHON_VERSION")

  # Essentially, we'll do
  # mkdir -p Build
  # cd Build
  # cmake $cmake_flags ..
  # make

  mkdir -p "$BUILD_NAME"
  cd "$BUILD_NAME" || exit 1

  cmake $cmake_flags ..

  make -j4
  make install
}

main "$@"