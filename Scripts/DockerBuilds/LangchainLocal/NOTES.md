By running

```

```
we see that NVIDIA doesn't even build OpenCV with CUDA on!

```
<missing>                                                                 3 weeks ago   RUN |6 NVIDIA_PYTORCH_VERSION=24.03 PYTORCH_BUILD_VERSION=2.3.0a0+40ec155e58 NVFUSER_BUILD_VERSION=f73ff1bc6a TARGETARCH=amd64 PYVER=3.10 L4T=0 /bin/sh -c OPENCV_VERSION=4.7.0 &&     cd / &&     wget -q -O - https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz | tar -xzf - &&     cd /opencv-${OPENCV_VERSION} &&     cmake -GNinja -Bbuild -H.           -DWITH_CUDA=OFF -DWITH_1394=OFF           -DPYTHON3_PACKAGES_PATH="/usr/local/lib/python${PYVER}/dist-packages"           -DBUILD_opencv_cudalegacy=OFF -DBUILD_opencv_stitching=OFF -DWITH_IPP=OFF -DWITH_PROTOBUF=OFF &&     cmake --build build --target install &&     cd modules/python/package &&     pip install --no-cache-dir --disable-pip-version-check -v . &&     rm -rf /opencv-${OPENCV_VERSION} # buildkit  
```