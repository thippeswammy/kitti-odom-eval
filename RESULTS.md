# Evaluation Results

## Gitam Sequence 00 Comparison

| Method | ATE (m) |
| :--- | :--- |
| **Features** | 39.314 |
| **Optical Flow** | 45.594 |
| **ORB SLAM** | 0.674 |
| **DPVO** | 16.541 |

### Trajectory Plots

| Features | Optical Flow | ORB SLAM | DPVO |
| :---: | :---: | :---: | :---: |
| ![Features](./result/features/plot_path/sequence_00.png) | ![Optical Flow](./result/optical_flow/plot_path/sequence_00.png) | ![ORB SLAM](./result/orb_slam/plot_path/sequence_00.png) | ![DPVO](./result/DPVO/plot_path/sequence_00.png) |

---

## Gitam Sequence 01 Comparison

| Method | ATE (m) |
| :--- | :--- |
| **Features** | 68.535 |
| **Optical Flow** | 72.086 |
| **ORB SLAM** | 15.283 |
| **DPVO** | 38.602 |

### Trajectory Plots

| Features | Optical Flow | ORB SLAM | DPVO |
| :---: | :---: | :---: | :---: |
| ![Features](./result/features/plot_path/sequence_01.png) | ![Optical Flow](./result/optical_flow/plot_path/sequence_01.png) | ![ORB SLAM](./result/orb_slam/plot_path/sequence_01.png) | ![DPVO](./result/DPVO/plot_path/sequence_01.png) |

---

## KITTI Sequences

### Sequence 09

| Method | ATE (m) |
| :--- | :--- |
| **ORB KITTI** | 9.705 |

**Trajectory Plot:**
![ORB KITTI](./result/orb_kitti/plot_path/sequence_09.png)

### Sequence 03

| Method | Status |
| :--- | :--- |
| **ORB KITTI** | Failure |

> **Note:** Evaluation failed for this sequence.

### Other Methods

| Method | Status |
| :--- | :--- |
| **DFVO** | Failure |

> **Note:** DFVO failed on our dataset.
