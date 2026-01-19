# RTAB-Map "Triple Threat" Installation Guide
### (ORB-SLAM3 + SuperPoint + SuperGlue)

This guide details the process for building a high-performance SLAM stack tailored for autonomous vehicles. It integrates **ORB-SLAM3** for Visual-Inertial Odometry and **SuperPoint/SuperGlue** for deep-learning-based loop closure detection.

---

## 1. Directory Structure & Prerequisites

Ensure your workspace is organized as follows:
- **Workspace Root**: `/media/thippe/SDV/Ubuntu/rtab_ws`
- **Dependencies**: `/media/thippe/SDV/Ubuntu/rtab_ws/dependencies`
    - `libtorch/`: [LibTorch C++ Binaries](https://pytorch.org/get-started/locally/) (v2.4.0+ recommended)
    - `rtabmap/`: Standalone RTAB-Map Source
    - `SuperGluePretrainedNetwork/`: [SuperGlue Source](https://github.com/magicleap/SuperGluePretrainedNetwork)
- **ORB-SLAM3**: `/home/thippe/ws_slam/ORB_SLAM3` (Pre-compiled as a library)

### System Requirements
```bash
sudo apt update
sudo apt install python3-dev python3-numpy libsqlite3-dev libpcl-dev libopencv-dev


# 1. Export required build paths
export INSTALL_DIR="/media/thippe/SDV/Ubuntu/rtab_ws/install_standalone"
export LIBTORCH_DIR="/media/thippe/SDV/Ubuntu/rtab_ws/dependencies/libtorch"
export ORB_SLAM_ROOT_DIR="/home/thippe/ws_slam/ORB_SLAM3"

# 2. Configure Build
cd /media/thippe/SDV/Ubuntu/rtab_ws/src/rtabmap
mkdir -p build && cd build
rm -rf *

cmake .. \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DWITH_TORCH=ON \
    -DWITH_PYTHON=ON \
    -DWITH_ORB_SLAM=ON \
    -DWITH_G2O=OFF \
    -DORB_SLAM_ROOT_DIR="$ORB_SLAM_ROOT_DIR" \
    -DTorch_DIR="$LIBTORCH_DIR/share/cmake/Torch" \
    -DCMAKE_BUILD_TYPE=Release

# 3. Compile and Install
make -j$(nproc)
make install