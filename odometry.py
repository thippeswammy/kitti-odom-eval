# Copyright (C) Huangying Zhan 2019. All rights reserved.

import argparse
import copy
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from tabulate import tabulate


def scale_lse_solver(X, Y):
    """Least-square-error solver for scaling factor."""
    scale = np.sum(X * Y) / np.sum(X ** 2)
    return scale


def umeyama_alignment(x, y, with_scale=False):
    """Computes Sim(m) transformation parameters (Umeyama, 1991)."""
    if x.shape != y.shape:
        raise ValueError("x.shape must equal y.shape")
    m, n = x.shape
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis]) ** 2)
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer(y[:, i] - mean_y, x[:, i] - mean_x)
    cov_xy = outer_sum / n
    u, d, v = np.linalg.svd(cov_xy)
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        s[m - 1, m - 1] = -1
    r = u.dot(s).dot(v)
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - c * r.dot(mean_x)
    return r, t, c


import ast


def parse_computational_metrics(file_path):
    """Parse computational metrics from txt file with robust error handling."""
    metrics = {
        'total_time': 0.0, 'frames_processed': 0, 'fps': 0.0,
        'avg_cpu': 0.0, 'avg_cpu_overall': 0.0, 'avg_gpu': 0.0,
        'avg_gpu_mem': 0.0, 'avg_gpu_power': 0.0, 'avg_ram': 0.0,
        'monitor_duration': 0.0
    }
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                stripped = line.strip()
                if 'Total Time' in stripped:
                    metrics['total_time'] = float(stripped.split('=')[1].split()[0])
                elif 'Frames processed' in stripped:
                    metrics['frames_processed'] = int(stripped.split('=')[1].split()[0])
                elif 'FPS' in stripped:
                    metrics['fps'] = float(stripped.split('=')[1].split()[0])
                elif '[AVERAGE USAGE]' in stripped:
                    # Expect the dictionary to be in the next line(s)
                    dict_lines = []
                    for l in lines[i + 1:]:
                        if '[RESULT]' in l:
                            break  # stop at result block
                        dict_lines.append(l.strip())
                    try:
                        usage_dict = ast.literal_eval(''.join(dict_lines))
                        for key in usage_dict:
                            mapped_key = key.strip().lower()
                            if mapped_key in metrics:
                                metrics[mapped_key] = float(usage_dict[key])
                            elif mapped_key == 'monitor_duration_seconds':
                                metrics['monitor_duration'] = float(usage_dict[key])
                    except Exception as e:
                        print(f"Error parsing [AVERAGE USAGE] block in {file_path}: {e}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return metrics


class KittiEvalOdom:
    """Evaluate odometry results for KITTI dataset."""

    def __init__(self):
        self.lengths = [100, 200, 300, 400, 500, 600, 700, 800]
        self.step_size = 10

    @staticmethod
    def plot_3d_trajectories_interactive(estimated, groundtruth, title="3D Trajectory Comparison",
                                         file_name="trajectory_3d_interactive.html"):
        """Plot interactive 3D trajectories using Plotly."""
        estimated = np.asarray(estimated, dtype=np.float32)
        groundtruth = np.asarray(groundtruth, dtype=np.float32)

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=estimated[:, 0], y=estimated[:, 1], z=estimated[:, 2],
            mode='lines+markers',
            marker=dict(size=3, color='blue'),
            line=dict(color='blue'),
            name='Estimated'
        ))
        fig.add_trace(go.Scatter3d(
            x=groundtruth[:, 0], y=groundtruth[:, 1], z=groundtruth[:, 2],
            mode='lines+markers',
            marker=dict(size=3, color='green'),
            line=dict(color='green', dash='dash'),
            name='Groundtruth'
        ))
        fig.update_layout(
            title=title,
            scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)'),
            legend=dict(x=0.02, y=0.98)
        )
        fig.write_html(file_name)

    @staticmethod
    def load_poses_from_txt(file_name, step=1):
        """
        Load poses from KITTI-format txt file with optional step interval.
        Supported formats:
        - 12 floats: row-major 3x4 matrix (R11 R12 R13 Tx ... R33 Tz)
        - 13 floats: timestamp + 12 floats
        - 7 floats: tx ty tz qx qy qz qw
        - 8 floats: timestamp tx ty tz qx qy qz qw
        - 9 floats: timestamp tx ty tz qx qy qz qw index (TUM with index/ID)
        Returns:
            poses (dict): {index: 4x4 matrix}
            times (dict): {index: timestamp} or None if no timestamps found
        """
        from scipy.spatial.transform import Rotation
        
        try:
            with open(file_name, 'r') as f:
                lines = f.readlines()
            poses = {}
            times = {}
            count = 0
            format_type = "Unknown"
            has_times = False

            # Check detecting format (optional, just to set has_times flag correctly)
            for line in lines:
                if line.strip().startswith("#") or not line.strip():
                    continue
                split = [float(i) for i in line.strip().split() if i]
                if len(split) in [13, 8, 9]:
                    has_times = True
                break

            for cnt, line in enumerate(lines):
                if line.strip().startswith("#") or not line.strip():
                    continue
                    
                if (cnt % step) != 0:
                    continue
                
                try:
                    line_split = [float(i) for i in line.strip().split() if i]
                    n_vals = len(line_split)
                    if n_vals == 0: continue

                    P = np.eye(4)
                    timestamp = None
                    
                    if n_vals == 12:
                         # 3x4 matrix
                         P[:3, :] = np.array(line_split).reshape(3, 4)
                         timestamp = cnt
                         format_type = "KITTI 12"
                    elif n_vals == 13:
                         # timestamp + 3x4 matrix
                         timestamp = line_split[0]
                         P[:3, :] = np.array(line_split[1:]).reshape(3, 4)
                         format_type = "KITTI 13 (Timestamp)"
                    elif n_vals == 7:
                         # tx ty tz qx qy qz qw
                         t = line_split[:3]
                         q = line_split[3:]
                         # Scipy expects [x, y, z, w]
                         P[:3, :3] = Rotation.from_quat(q).as_matrix()
                         P[:3, 3] = t
                         timestamp = cnt
                         format_type = "TUM 7"
                    elif n_vals == 8:
                         # timestamp tx ty tz qx qy qz qw
                         timestamp = line_split[0]
                         t = line_split[1:4]
                         q = line_split[4:]
                         P[:3, :3] = Rotation.from_quat(q).as_matrix()
                         P[:3, 3] = t
                         format_type = "TUM 8 (Timestamp)"
                    elif n_vals == 9:
                         # timestamp tx ty tz qx qy qz qw index
                         # Ignore the 9th value (index)
                         timestamp = line_split[0]
                         t = line_split[1:4]
                         q = line_split[4:8]
                         P[:3, :3] = Rotation.from_quat(q).as_matrix()
                         P[:3, 3] = t
                         format_type = "TUM 9 (Timestamp + Index)"
                    else:
                        raise ValueError(f"Invalid number of values ({n_vals}) in line {cnt + 1}")
                    
                    poses[count] = P
                    if has_times:
                        times[count] = timestamp
                    count += 1
                        
                except ValueError as e:
                    print(f"Error parsing line {cnt + 1} in {file_name}: {e}")
                    continue
            
            if not poses:
                raise ValueError(f"No valid poses found in {file_name}")

            # Normalize timestamps if they look like absolute epochs (> 10000s)
            # This helps matching with GT which typically starts at 0.0
            if has_times and times:
                min_t = min(times.values())
                if min_t > 10000.0:
                    print(f"Normalizing timestamps by subtracting {min_t} (detected absolute epochs)")
                    for k in times:
                        times[k] -= min_t
            
            return poses, (times if has_times else {}), format_type


        except Exception as e:
            raise Exception(f"Failed to load poses from {file_name}: {e}")

    @staticmethod
    def associate_poses_by_timestamp(poses_gt, times_gt, poses_result, times_result, max_diff=0.01):
        """
        Associate poses based on closest timestamp matching.
        Returns re-indexed poses_gt, poses_result (0..N) containing only matched pairs.
        """
        sorted_keys_gt = sorted(times_gt.keys())
        sorted_keys_result = sorted(times_result.keys())
        
        matched_gt = {}
        matched_result = {}
        out_idx = 0
        
        # Simple nearest neighbor search. 
        # Since timestamps are likely sorted, we could optimize, but O(N*M) is fine for typically small N, M (<50k).
        # Actually linearly scanning is better since both sorted.
        
        gt_timestamps = np.array([times_gt[k] for k in sorted_keys_gt])
        gt_keys = np.array(sorted_keys_gt)
        
        for r_key in sorted_keys_result:
            t_res = times_result[r_key]
            
            # Find index in gt_timestamps closest to t_res
            # np.searchsorted finds insertion point.
            # We want nearest.
            idx = np.searchsorted(gt_timestamps, t_res)
            
            candidates = []
            if idx < len(gt_timestamps):
                candidates.append(idx)
            if idx > 0:
                candidates.append(idx - 1)
            
            best_idx = -1
            min_diff = float('inf')
            
            for c in candidates:
                diff = abs(gt_timestamps[c] - t_res)
                if diff < min_diff:
                    min_diff = diff
                    best_idx = c
            
            if best_idx != -1 and min_diff < max_diff:
                # Match found
                gt_key_matched = gt_keys[best_idx]
                matched_gt[out_idx] = poses_gt[gt_key_matched]
                matched_result[out_idx] = poses_result[r_key]
                out_idx += 1
                
        return matched_gt, matched_result

    @staticmethod
    def trajectory_distances(poses):
        """Compute distances w.r.t. frame 0."""
        dist = [0]
        keys = sorted(poses.keys())
        for i in range(len(keys) - 1):
            P1, P2 = poses[keys[i]], poses[keys[i + 1]]
            dx, dy, dz = P1[0, 3] - P2[0, 3], P1[1, 3] - P2[1, 3], P1[2, 3] - P2[2, 3]
            dist.append(dist[i] + np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
        return dist

    @staticmethod
    def rotation_error(pose_error):
        """Compute rotation error in radians."""
        a, b, c = pose_error[0, 0], pose_error[1, 1], pose_error[2, 2]
        d = 0.5 * (a + b + c - 1.0)
        return np.arccos(max(min(d, 1.0), -1.0))

    @staticmethod
    def translation_error(pose_error):
        """Compute translation error in meters."""
        dx, dy, dz = pose_error[0, 3], pose_error[1, 3], pose_error[2, 3]
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    @staticmethod
    def last_frame_from_segment_length(dist, first_frame, length):
        """Find frame index for segment of specified length."""
        for i in range(first_frame, len(dist)):
            if dist[i] > dist[first_frame] + length:
                return i
        return -1

    def calc_sequence_errors(self, poses_gt, poses_result):
        """Calculate relative translation and rotation errors."""
        err = []
        dist = self.trajectory_distances(poses_gt)
        for first_frame in range(0, len(poses_gt), self.step_size):
            for length in self.lengths:
                last_frame = self.last_frame_from_segment_length(dist, first_frame, length)
                if last_frame == -1 or last_frame not in poses_result or first_frame not in poses_result:
                    continue
                pose_delta_gt = np.linalg.inv(poses_gt[first_frame]) @ poses_gt[last_frame]
                pose_delta_result = np.linalg.inv(poses_result[first_frame]) @ poses_result[last_frame]
                pose_error = np.linalg.inv(pose_delta_result) @ pose_delta_gt
                r_err = self.rotation_error(pose_error)
                t_err = self.translation_error(pose_error)
                err.append([first_frame, r_err / length, t_err / length, length])
        return err

    @staticmethod
    def compute_overall_err(seq_err):
        """Compute average t_rel and r_rel."""
        if not seq_err:
            return 0, 0
        t_err = sum(item[2] for item in seq_err) / len(seq_err)
        r_err = sum(item[1] for item in seq_err) / len(seq_err)
        return t_err, r_err

    def compute_segment_error(self, seq_errs):
        """Calculate average errors for different segments."""
        segment_errs = {length: [] for length in self.lengths}
        avg_segment_errs = {}
        for err in seq_errs:
            length = err[3]
            t_err, r_err = err[2], err[1]
            segment_errs[length].append([t_err, r_err])
        for length in self.lengths:
            if segment_errs[length]:
                avg_t_err = np.mean(np.asarray(segment_errs[length])[:, 0])
                avg_r_err = np.mean(np.asarray(segment_errs[length])[:, 1])
                avg_segment_errs[length] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[length] = []
        return avg_segment_errs

    @staticmethod
    def compute_ATE(gt, pred):
        """Compute Absolute Trajectory Error (RMSE)."""
        errors = []
        for i in pred:
            gt_xyz = gt[i][:3, 3]
            pred_xyz = pred[i][:3, 3]
            errors.append(np.sqrt(np.sum((gt_xyz - pred_xyz) ** 2)))
        return np.sqrt(np.mean(np.asarray(errors) ** 2)) if errors else 0

    def compute_RPE(self, gt, pred):
        """Compute Relative Pose Error (translation and rotation)."""
        trans_errors, rot_errors = [], []
        keys = sorted(pred.keys())[:-1]
        for i in keys:
            if i + 1 in pred and i + 1 in gt:
                gt_rel = np.linalg.inv(gt[i]) @ gt[i + 1]
                pred_rel = np.linalg.inv(pred[i]) @ pred[i + 1]
                rel_err = np.linalg.inv(gt_rel) @ pred_rel
                trans_errors.append(self.translation_error(rel_err))
                rot_errors.append(self.rotation_error(rel_err))
        return np.mean(trans_errors) if trans_errors else 0, np.mean(rot_errors) if rot_errors else 0

    def compute_translational_rmse(self, gt, pred):
        """Compute Translational RMSE (same as ATE for consistency)."""
        return self.compute_ATE(gt, pred)

    @staticmethod
    def scale_optimization(gt, pred):
        """Optimize scaling factor for predicted poses."""
        pred_updated = copy.deepcopy(pred)
        xyz_pred = np.array([pred[i][:3, 3] for i in pred])
        xyz_ref = np.array([gt[i][:3, 3] for i in gt])
        scale = scale_lse_solver(xyz_pred, xyz_ref)
        for i in pred_updated:
            pred_updated[i][:3, 3] *= scale
        return pred_updated

    @staticmethod
    def compute_total_distance(poses):
        """Compute total trajectory distance."""
        keys = sorted(poses.keys())
        dist = 0.0
        for i in range(1, len(keys)):
            dist += np.linalg.norm(poses[keys[i]][:3, 3] - poses[keys[i - 1]][:3, 3])
        return dist

    @staticmethod
    def compute_drift(gt, pred):
        """Compute final drift (Euclidean distance at last frame)."""
        keys = sorted(gt.keys())
        if keys and keys[-1] in pred:
            return np.linalg.norm(gt[keys[-1]][:3, 3] - pred[keys[-1]][:3, 3])
        return 0.0

    @staticmethod
    def write_result(f, seq, errs):
        """Write evaluation metrics to file."""
        t_rel, r_rel, ate, rpe_trans, rpe_rot, trans_rmse, gt_dist, pred_dist, drift, t_rel_100 = errs
        drift_pct = (drift / gt_dist * 100) if gt_dist > 0 else 0
        lines = [
            f"Sequence: \t {seq}\n",
            f"t_rel (%): \t {t_rel * 100:.3f}\n",
            f"r_rel (deg/100m): \t {r_rel / np.pi * 180 * 100:.3f}\n",
            f"ATE (m): \t {ate:.3f}\n",
            f"Translational RMSE (m): \t {trans_rmse:.3f}\n",
            f"RPE trans (m): \t {rpe_trans:.3f}\n",
            f"RPE rot (deg): \t {rpe_rot * 180 / np.pi:.3f}\n",
            f"GT Distance (m): \t {gt_dist:.3f}\n",
            f"Pred Distance (m): \t {pred_dist:.3f}\n",
            f"Drift (m): \t {drift:.3f}\n",
            f"Drift (%): \t {drift_pct:.3f}\n",
            f"t_rel_100m (%): \t {t_rel_100 * 100:.3f}\n\n"
        ]
        f.writelines(lines)
        return [t_rel * 100, r_rel / np.pi * 180 * 100, ate, rpe_trans, rpe_rot * 180 / np.pi, trans_rmse, gt_dist,
                pred_dist, drift, drift_pct, t_rel_100 * 100]

    def eval(self, gt_dir, result_dir, alignment=None, seqs=None, eval_seqs="", method_name='', file_name_plot='',
             step=1):
        """Evaluate sequences and return metrics."""
        self.gt_dir = gt_dir
        error_dir = os.path.join(result_dir, "errors")
        self.plot_path_dir = os.path.join(result_dir, "plot_path")
        self.plot_error_dir = os.path.join(result_dir, "plot_error")
        result_txt = os.path.join(result_dir, "result.txt")
        os.makedirs(error_dir, exist_ok=True)
        os.makedirs(self.plot_path_dir, exist_ok=True)
        os.makedirs(self.plot_error_dir, exist_ok=True)
        with open(result_txt, 'w') as f:
            result_file = os.path.join(result_dir, f"{eval_seqs}.txt")
            gt_file = os.path.join(gt_dir, f"{eval_seqs[:2]}.txt")
            if not os.path.exists(result_file):
                print(f"Pose file {result_file} not found")
                return []
            try:
                # Load Results (step=1 usually unless pre-sparsing logic applied by user externally, but let's assume we read all)
                # Then we match to GT using 'step' or timestamps.
                # Load Results
                poses_result, times_result, res_fmt = self.load_poses_from_txt(result_file, step=1)
                print(f"Processing Method: {method_name} | Sequence: {eval_seqs}")
                print(f"  Result File: {result_file} | Format: {res_fmt}")

                
                # Load GT
                # If we rely on index matching with step, we pass step here.
                # If we rely on timestamp matching, we might want to load ALL GT (step=1) and then search?
                # Optimization: Load GT with step=1 to ensure we have all candidates if using timestamps.
                # But if defaulting to index, we need `step` application.
                
                # Strategy: Load GT with step=1 first.
                # Strategy: Load GT with step=1 first.
                poses_gt_raw, times_gt_raw, gt_fmt = self.load_poses_from_txt(gt_file, step=1)
                
                # Print GT info only once per GT file
                if not hasattr(KittiEvalOdom, 'printed_gt_files'):
                    KittiEvalOdom.printed_gt_files = set()
                if gt_file not in KittiEvalOdom.printed_gt_files:
                    print(f"  GT File: {gt_file} | Format: {gt_fmt}")
                    KittiEvalOdom.printed_gt_files.add(gt_file)

                
                # Determine matching strategy
                if times_result and times_gt_raw:
                    print(f"Found timestamps in both Result and GT for {eval_seqs}. matching by timestamp...")
                    poses_gt, poses_result = self.associate_poses_by_timestamp(poses_gt_raw, times_gt_raw, poses_result, times_result)
                    # If step > 1 specified, maybe we should subsample the *matched* result?
                    # User: "apply the step so only processs one frame out of 5"
                    # If matched produces dense keys 0..M, we can just take 0, 5, 10...
                    if step > 1:
                        poses_gt = {k: v for k, v in poses_gt.items() if (k % step) == 0}
                        poses_result = {k: v for k, v in poses_result.items() if (k % step) == 0}
                        # Re-key again to be sequential?
                        # `calc_sequence_errors` uses gaps in keys to determine distance, but iteration is on step_size.
                        # It iterates keys: 0, 10, 20... based on step_size (10).
                        # If we sparsify here, keys become 0, 5, 10... 
                        # If step=5 and step_size=10, we check key 0, key 10.
                        # Matches logic.
                        
                else:
                    # Fallback to Index Matching
                    # Logic: Result index i corresponds to GT index i * step.
                    print(f"Matching by Frame Index (step={step}) for {eval_seqs}...")
                    poses_gt = {}
                    poses_result_fixed = {}
                    
                    # We assume poses_result keys are 0..N
                    # We want poses_gt to have corresponding keys.
                    # Wait, standard `load_poses_from_txt(..., step=step)` used to return keys 0..M where 0->GT[0], 1->GT[step].
                    # And poses_result(step=1) returns keys 0..M. 
                    # So key 0 matches key 0.
                    # Let's replicate that behavior using the raw loaded data:
                    
                    valid_matches = 0
                    for res_idx in sorted(poses_result.keys()):
                        gt_idx = res_idx * step
                        if gt_idx in poses_gt_raw:
                            poses_gt[res_idx] = poses_gt_raw[gt_idx]
                            poses_result_fixed[res_idx] = poses_result[res_idx]
                            valid_matches += 1
                            
                    poses_result = poses_result_fixed
                    
                if not poses_result or not poses_gt:
                    print(f"No common poses found for {eval_seqs}")
                    return []
                idx_0 = sorted(poses_result.keys())[0]
                pred_0, gt_0 = poses_result[idx_0], poses_gt[idx_0]
                for cnt in poses_result:
                    poses_result[cnt] = np.linalg.inv(pred_0) @ poses_result[cnt]
                    poses_gt[cnt] = np.linalg.inv(gt_0) @ poses_gt[cnt]
                if alignment == "scale":
                    poses_result = self.scale_optimization(poses_gt, poses_result)
                elif alignment in ["scale_7dof", "7dof", "6dof"]:
                    xyz_gt = np.array([poses_gt[cnt][:3, 3] for cnt in poses_result]).T
                    xyz_gt = np.array([poses_gt[cnt][:3, 3] for cnt in poses_gt if cnt in poses_result]).T
                    xyz_result = np.array([poses_result[cnt][:3, 3] for cnt in poses_result]).T
                    r, t, scale = umeyama_alignment(xyz_result, xyz_gt, alignment != "6dof")
                    align_transformation = np.eye(4)
                    align_transformation[:3, :3], align_transformation[:3, 3] = r, t
                    for cnt in poses_result:
                        poses_result[cnt][:3, 3] *= scale
                        if alignment in ["7dof", "6dof"]:
                            poses_result[cnt] = align_transformation @ poses_result[cnt]
                seq_err = self.calc_sequence_errors(poses_gt, poses_result)
                self.save_sequence_errors(seq_err, os.path.join(error_dir, f"{eval_seqs}.txt"))
                t_rel, r_rel = self.compute_overall_err(seq_err)
                ate = self.compute_ATE(poses_gt, poses_result)
                trans_rmse = self.compute_translational_rmse(poses_gt, poses_result)
                rpe_trans, rpe_rot = self.compute_RPE(poses_gt, poses_result)
                gt_dist = self.compute_total_distance(poses_gt)
                pred_dist = self.compute_total_distance(poses_result)
                drift = self.compute_drift(poses_gt, poses_result)
                pos_result = np.array([poses_result[k][:3, 3] for k in sorted(poses_result.keys())])
                pos_gt = np.array([poses_gt[k][:3, 3] for k in sorted(poses_gt.keys()) if k in poses_result])
                self.plot_trajectory(poses_gt, poses_result, eval_seqs, pos_gt, pos_result, method_name, file_name_plot)
                avg_segment_errs_res = self.compute_segment_error(seq_err)
                self.plot_error(avg_segment_errs_res, eval_seqs)
                
                # Extract 100m error
                t_rel_100 = 0.0
                if 100 in avg_segment_errs_res and avg_segment_errs_res[100]:
                    t_rel_100 = avg_segment_errs_res[100][0]

                metrics = self.write_result(f, eval_seqs,
                                            [t_rel, r_rel, ate, rpe_trans, rpe_rot, trans_rmse, gt_dist, pred_dist,
                                             drift, t_rel_100])
                return metrics
            except Exception as e:
                print(f"Error processing sequence {eval_seqs}: {e}")
                return []

    @staticmethod
    def save_sequence_errors(err, file_name):
        """Save sequence errors to file."""
        with open(file_name, 'w') as f:
            for item in err:
                f.write(" ".join(map(str, item)) + "\n")

    def plot_trajectory(self, poses_gt, poses_result, seq, pos_gt, pos_result, method_name, file_name_plot):
        """Plot ground truth and predicted trajectories in fixed high resolution for publication."""
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        
        
        dpi = 300
        # Use plt.figure directly to match plot_gt.py exactly
        fig = plt.figure(figsize=(10, 6), dpi=dpi)
        
        # Rename method for legend
        method_name_dict = {
            "orb_feature_based_vo": "Feature Based VO",
            "optical_flow_based_vo": "Optical Flow VO",
            "mono": "Mono SLAM",
            "stereo": "Stereo SLAM",
            "DPVO": "DPVO",
            "droid_slam": "Droid SLAM",
            "features": "Features",
            "optical_flow": "Optical Flow",
            "orb_slam": "ORB-SLAM",
            "rtab_GFTT _BRIEF": "RTAB-Map (GFTT+BRIEF)"
        }
        method_name = method_name_dict.get(method_name, method_name)

        # Plot Ground Truth and Estimated Trajectory
        for label, poses in [("Ground Truth", poses_gt), (method_name, poses_result)]:
            pos_xz = np.array([[pose[0, 3], pose[2, 3]] for pose in [poses[k] for k in sorted(poses.keys())]])
            plt.plot(pos_xz[:, 0], pos_xz[:, 1], label=label, linewidth=2.5)

        # Aesthetic tuning
        plt.legend(loc="best", fontsize=16, framealpha=0.8)
        plt.xlabel('x (m)', fontsize=18)
        plt.ylabel('z (m)', fontsize=18)
        plt.title(f"Trajectory Comparison {method_name} Seq {seq}", fontsize=20)
        plt.axis('equal')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Use tight_layout for better spacing
        plt.tight_layout()

        # Save figure with fixed resolution and consistent layout
        save_path = os.path.join(self.plot_path_dir, file_name_plot)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        # Optionally save 3D interactive plot
        self.plot_3d_trajectories_interactive(
            pos_result,
            pos_gt,
            title=f"3D Trajectory Comparison for Sequence {seq}",
            file_name=os.path.join(self.plot_path_dir, file_name_plot.split('.')[0] + ".html")
        )

    def plot_error(self, avg_segment_errs, seq):
        """Plot translation and rotation errors per segment length."""
        fontsize = 10
        plot_x = self.lengths
        plot_y = [avg_segment_errs[length][0] * 100 if avg_segment_errs[length] else 0 for length in self.lengths]
        plt.figure(figsize=(5, 5))
        plt.plot(plot_x, plot_y, "bs-", label="t_rel (%)")
        plt.xlabel('Path Length (m)', fontsize=fontsize)
        plt.ylabel('Translation Error (%)', fontsize=fontsize)
        plt.legend(loc="upper right", prop={'size': fontsize})
        plt.savefig(os.path.join(self.plot_error_dir, f"trans_err_{seq}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()
        plot_y = [avg_segment_errs[length][1] / np.pi * 180 * 100 if avg_segment_errs[length] else 0 for length in
                  self.lengths]
        plt.figure(figsize=(5, 5))
        plt.plot(plot_x, plot_y, "bs-", label="r_rel (deg/100m)")
        plt.xlabel('Path Length (m)', fontsize=fontsize)
        plt.ylabel('Rotation Error (deg/100m)', fontsize=fontsize)
        plt.legend(loc="upper right", prop={'size': fontsize})
        plt.savefig(os.path.join(self.plot_error_dir, f"rot_err_{seq}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_folders_in_dir(dir_path):
    """Get a list of folders in directory."""
    # Hardcoded fallback as per previous version, or we could actually scan
    return ["StoringResults/eval_matrix/droid_slam_indoor", "StoringResults/eval_matrix/droid_slam_outdoor"]

if __name__ == "__main__":
    # Headers in the exact order requested
    headers = ["Method", "Alignment", "Sequence", "ATE (m)", "Trans RMSE (m)", "RPE trans (m)", "t_rel (%)",
               "r_rel (deg/100m)", "RPE rot (deg)", "GT Dist (m)", "Pred Dist (m)", "Drift (m)", "Drift (%)", "t_rel @ 100m (%)", "FPS",
               "Total Time (s)", "Avg CPU Overall (%)", "Avg RAM (GB)", "Avg GPU Mem (%)", "Avg GPU (%)"]

    results_table = []

    parser = argparse.ArgumentParser(description='KITTI VO evaluation with repetitions and computational metrics')
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--result', type=str, default='../../StoringResults', help='Root directory of result folders (override)')
    parser.add_argument('--gt_dir', type=str, default='dataset/', help='Ground truth poses directory (override)')
    parser.add_argument('--seqs', nargs="+", type=int, default=[1, 3, 9], help='List of base sequence numbers to evaluate (override)')
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory for output plots and Excel')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    methods_to_run = []

    if args.config:
        try:
            config = load_config(args.config)
            if 'eval_config' in config:
                methods_to_run = config['eval_config']
            else:
                print("Error: 'eval_config' key not found in config file.")
        except Exception as e:
            print(f"Error loading config file: {e}")
            exit(1)
    else:
        # Fallback to legacy discovery mode if no config provided
        print("No config file provided. Falling back to directory discovery.")
        dir_list = get_folders_in_dir(args.result)
        print("Detected method folders:", dir_list)
        
        alignments = ['scale_7dof', "6dof", "7dof", 'scale']
        # Default legacy behavior
        for folder in dir_list:
            method_name = os.path.basename(folder)
            gt_path = args.gt_dir + ("Tum_odom/gt_poses/" if method_name == "droid_slam_indoor" else "kitti_odom/gt_poses/")
            methods_to_run.append({
                'method_name': method_name,
                'result_dir': os.path.join(args.result, 'eval_matrix', method_name),  # Note: logic slightly different in loop below, adjusting to match
                'gt_dir': gt_path,
                'alignments': alignments,
                'sequences': args.seqs,
                'computational_metrics_optional': False
            })

    for method_config in methods_to_run:
        method_name = method_config.get('method_name', 'Unknown')
        # Handle path differences: config might provide full path or relative
        # In legacy code, dir_list had relative paths like "StoringResults/eval_matrix/droid_slam_indoor"
        # and then os.path.basename was used.
        # Here we trust the config to provide 'result_dir' path to the eval_matrix folder container or specific folder.
        
        # Let's support the config providing the base path to the results
        base_result_dir = method_config.get('result_dir')
        if not base_result_dir:
             # Try to construct if missing (legacy support fallback)
             base_result_dir = os.path.join(args.result, 'eval_matrix', method_name)

        gt_path = method_config.get('gt_dir', args.gt_dir)
        alignments = method_config.get('alignments', ['scale_7dof'])
        sequences = method_config.get('sequences', args.seqs)
        comp_optional = method_config.get('computational_metrics_optional', False)
        step_config = method_config.get('step', 1)
        
        # Normalize steps to a list matching sequences
        if not isinstance(step_config, list):
             steps_list = [step_config] * len(sequences)
        else:
             steps_list = step_config
             if len(steps_list) != len(sequences):
                  print(f"Warning: 'step' list length ({len(steps_list)}) does not match 'sequences' length ({len(sequences)}) for method {method_name}. Using first step for all.")
                  steps_list = [steps_list[0]] * len(sequences)
        
        seq_step_map = dict(zip(sequences, steps_list))
        
        # Expand sequences to include repetitions (e.g., 01_1, 01_2, 01_3)
        # Assuming repeat 1 for now as per original code loop `range(1, 2)`
        seq_repeats = [(f"{seq:02}", i) for seq in sequences for i in range(1, 2)]

        # Adjust paths. 
        # Logic in original: eval_base_path = os.path.join(args.result, 'eval_matrix', method_name)
        # If config provides full path, use it.
        eval_base_path = base_result_dir
        # Computational path assumption: replace 'eval_matrix' with 'Computational_matrix'
        # This is a bit brittle, but consistent with the repository structure implied
        if 'eval_matrix' in eval_base_path:
            comp_base_path = eval_base_path.replace('eval_matrix', 'Computational_matrix')
        else:
             # Fallback or assume sibling directory
             comp_base_path = os.path.join(os.path.dirname(eval_base_path), 'Computational_matrix', method_name)

        if not os.path.exists(eval_base_path):
            print(f"Missing eval folder: {eval_base_path} for {method_name}")
            continue
            
        # Optional: check comp folder existence only if not optional
        if not comp_optional and not os.path.exists(comp_base_path):
             print(f"Missing computational folder: {comp_base_path} for {method_name}")
             continue

        for align in alignments:
            for seq, repeat in seq_repeats:
                seq_str = f"{seq}_{repeat}"
                eval_path = os.path.join(eval_base_path, f"{seq_str}.txt")
                comp_path = os.path.join(comp_base_path, f"{seq_str}.txt")
                
                if not os.path.exists(eval_path):
                    print(f"Missing eval file for {seq_str} in {method_name}")
                    continue
                
                if not comp_optional and not os.path.exists(comp_path):
                    print(f"Missing comp file for {seq_str} in {method_name}")
                    continue

                eval_tool = KittiEvalOdom()
                try:
                    eval_metrics = eval_tool.eval(
                        gt_path,
                        os.path.dirname(eval_path),
                        alignment=align,
                        seqs=[int(seq)],
                        eval_seqs=seq_str,
                        method_name=str(method_name),
                        file_name_plot=f"{align}_seq_{seq_str}_3D_Plot.png",
                        step=seq_step_map[int(seq)]
                    )
                    
                    # Handle computational metrics
                    if os.path.exists(comp_path):
                        comp_metrics = parse_computational_metrics(comp_path)
                    elif comp_optional:
                        # Fill with zeros/defaults
                        comp_metrics = {
                            'total_time': 0.0, 'frames_processed': 1, 'fps': 0.0, # frames_processed > 0 to pass check
                            'avg_cpu': 0.0, 'avg_cpu_overall': 0.0, 'avg_gpu': 0.0,
                            'avg_gpu_mem': 0.0, 'avg_gpu_power': 0.0, 'avg_ram': 0.0,
                            'monitor_duration': 0.0
                        }
                    else:
                        comp_metrics = {'frames_processed': 0} # Trigger skip

                    if eval_metrics and comp_metrics.get('frames_processed', 0) > 0:
                        combined_row = [
                            method_name,
                            align,
                            seq_str,
                            f"{eval_metrics[2]:.3f}",  # ATE (m)
                            f"{eval_metrics[5]:.3f}",  # Trans RMSE (m)
                            f"{eval_metrics[3]:.3f}",  # RPE trans (m)
                            f"{eval_metrics[0]:.3f}",  # t_rel (%)
                            f"{eval_metrics[1]:.3f}",  # r_rel (deg/100m)
                            f"{eval_metrics[4]:.3f}",  # RPE rot (deg)
                            f"{eval_metrics[6]:.3f}",  # GT Dist (m)
                            f"{eval_metrics[7]:.3f}",  # Pred Dist (m)
                            f"{eval_metrics[8]:.3f}",  # Drift (m)
                            f"{eval_metrics[9]:.3f}",  # Drift (%)
                            f"{eval_metrics[10]:.3f}", # t_rel @ 100m (%)
                            f"{comp_metrics.get('fps', 0):.3f}",
                            f"{comp_metrics.get('total_time', 0):.3f}",
                            f"{comp_metrics.get('avg_cpu_overall', 0):.3f}",
                            f"{comp_metrics.get('avg_ram', 0):.3f}",
                            f"{comp_metrics.get('avg_gpu_mem', 0):.3f}",
                            f"{comp_metrics.get('avg_gpu', 0):.3f}"
                        ]
                        results_table.append(combined_row)
                except Exception as e:
                    print(f"Error processing {method_name} with alignment {align} for seq {seq_str}: {e}")

    if results_table:
        # Save detailed results
        df_detailed = pd.DataFrame(results_table, columns=headers)
        excel_detailed_file = output_dir / "vo_evaluation_results_detailed.xlsx"
        df_detailed.to_excel(excel_detailed_file, index=False)
        print(f"\nDetailed results saved to: {excel_detailed_file}")

        averaged_results = []
        unique_methods = set(row[0] for row in results_table)
        
        # Recalculate unique alignments/sequences found in results for averaging
        # (Since config might have diverse sets, we blindly avg over what we have)
        
        # But we need to group by Method + Alignment + BaseSeq
        # Extract unique combos
        combos = set((row[0], row[1], row[2].split('_')[0]) for row in results_table)
        
        for method, align, base_seq in combos:
             seq_data = [row for row in results_table if
                         row[0] == method and row[1] == align and row[2].startswith(base_seq + "_")]
             
             if seq_data:
                avg_row = [method, align, base_seq]
                for i in range(3, len(headers)):
                    values = [float(row[i]) for row in seq_data]
                    avg_row.append(f"{np.mean(values):.3f}")
                averaged_results.append(avg_row)

        print("\nAveraged Results:")
        print(tabulate(averaged_results, headers=headers, tablefmt="grid"))
        df_averaged = pd.DataFrame(averaged_results, columns=headers)
        excel_averaged_file = output_dir / "vo_evaluation_results_averaged.xlsx"
        df_averaged.to_excel(excel_averaged_file, index=False)
        print(f"\nAveraged results saved to: {excel_averaged_file}")
