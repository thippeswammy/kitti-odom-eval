import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from Kitti_odometry2 import KittiEvalOdom

def plot_gt(gt_dir, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(gt_dir, "*.txt"))
    tool = KittiEvalOdom()
    
    for f in files:
        seq = os.path.basename(f).split('.')[0]
        try:
            poses, _ = tool.load_poses_from_txt(f, step=1)
            
            # Extract positions
            pos_xz = np.array([[poses[k][0, 3], poses[k][2, 3]] for k in sorted(poses.keys())])
            
            # Plot
            plt.figure(figsize=(10, 6), dpi=300)
            plt.plot(pos_xz[:, 0], pos_xz[:, 1], label=f"GT {name} {seq}", linewidth=2.5, color='green', linestyle='--')
            plt.xlabel('x (m)', fontsize=18)
            plt.ylabel('z (m)', fontsize=18)
            plt.legend(fontsize=16, loc='best', framealpha=0.8)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title(f"GT Trajectory {name} Seq {seq}", fontsize=20)
            plt.axis('equal')
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, f"{name}_{seq}_2D.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            print(f"Saved plot to {save_path}")
            
        except Exception as e:
            print(f"Error processing {f}: {e}")

if __name__ == "__main__":
    # buggy_odom
    plot_gt("dataset/buggy_odom/gt_poses/", "result/gt_plt/buggy_odom", "Buggy")
    
    # kitti_odom
    plot_gt("dataset/kitti_odom/gt_poses/", "result/gt_plt/kitti_odom", "KITTI")
