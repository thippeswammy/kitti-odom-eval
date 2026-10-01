#!/usr/bin/env python3
"""
thirdparty/kitti-odom-eval/eval_infrastrucer_run.py -- run the official
KITTI-odometry-benchmark metric suite (odometry.py's KittiEvalOdom: ATE,
RPE, t_rel/r_rel segment errors, drift, trajectory/error plots) against an
InfraStructer DatabaseViewer run recorded from a --kitti source
(e.g. --kitti /media/.../2011_10_03_drive_0027_sync --db-output
output_2011_10_03_drive_0027_sync_<timestamp>.db).

This is NOT a reimplementation of the GPS/IMU -> cam2 ground-truth TF or
the DB pose decoding: both are imported from
tools/OdomScaleEval/odom_scale_eval.py, which already mirrors
DatabaseViewer's --kitti oxts->cam2 composition
(KittiRawFormat.cpp's buildOxtsToCam2()) and Node.pose's blob layout
exactly. This script only adds: a Node-id frame-range slice over the DB
trajectory, and a bridge from the matched (estimated, ground-truth)
pose pairs into the KITTI pose-file format odometry.py's KittiEvalOdom
API consumes, so the existing benchmark implementation (not a second,
independently re-derived one) produces the metrics.

Usage:
  eval_infrastrucer_run.py --db <output_..._sync.db> --kitti-raw <2011_10_03_drive_0027_sync dir> \\
                            [--calib-dir <dir>] [--frame-start <id>] [--frame-end <id>] \\
                            [--max-time-diff 0.05] [--align 7dof] [--step 1] \\
                            [--out-dir <dir>]
"""
import argparse
import math
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_ODOM_SCALE_EVAL_DIR = os.path.join(_REPO_ROOT, "tools", "OdomScaleEval")

sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _ODOM_SCALE_EVAL_DIR)

from odometry import KittiEvalOdom  # noqa: E402
import odom_scale_eval as ose  # noqa: E402


def load_db_trajectory_range(db_path, frame_start, frame_end):
    """Node id is 1-based and contiguous (one row per processed frame), so
    slicing ose.load_db_trajectory()'s id-ordered output by index is the
    same as filtering by Node.id -- done this way (rather than adding a
    WHERE clause / re-deriving the pose-blob decode here) so the actual
    decode stays the single copy in odom_scale_eval.py."""
    stamps, poses = ose.load_db_trajectory(db_path)
    n = len(stamps)
    start_idx = 0 if frame_start is None else max(0, frame_start - 1)
    end_idx = n if frame_end is None else min(n, frame_end)
    if start_idx >= end_idx:
        raise RuntimeError("empty frame range [%s, %s] over %d DB frames" % (frame_start, frame_end, n))
    return stamps[start_idx:end_idx], poses[start_idx:end_idx]


def write_kitti_pose_file(path, poses):
    with open(path, "w") as f:
        for T in poses:
            vals = T[:3, :4].reshape(-1)
            f.write(" ".join("%.9e" % v for v in vals) + "\n")


def rpy_from_rotation_deg(R):
    """roll/pitch/yaw (degrees) from a 3x3 rotation matrix, ZYX
    (R = Rz(yaw) @ Ry(pitch) @ Rx(roll)) -- the inverse of this same file's
    GT builder (odom_scale_eval.rot_from_rpy()) and of Transform.cpp's own
    eigenGetTransformation()/eigenGetTranslationAndEulerAngles() convention
    (both ZYX), so est and GT angles are directly comparable."""
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _wrap_deg(d):
    """Wrap an angle (degrees) into (-180, 180] -- needed since yaw differences
    near the +/-180 deg branch cut would otherwise look like a ~360 deg error."""
    return (d + 180.0) % 360.0 - 180.0


def flatten_to_se2(T):
    """Projects a 4x4 SE(3) pose onto the ground plane: keeps x/y and yaw,
    drops z and roll/pitch (z=0, rotation = pure Rz(yaw)).

    Use when the VO backend only observes planar motion (--se2): this run's
    own Odom/Strategy=2 (F2F, monocular, DepthSource=0) reports z and
    roll/pitch pinned at exactly 0 for every single frame -- a real
    near-planar PnP degeneracy, not noise -- while GT (real road camber/
    elevation) has small but genuine out-of-plane variation. Comparing the
    two in full SE(3) (odometry.py's ATE/RPE, the --rpy-plot roll/pitch
    curves) then counts GT's real-but-unobservable-by-this-VO out-of-plane
    motion as VO error. Flattening BOTH trajectories the same way before
    scoring removes that unfair penalty and evaluates what the VO strategy
    can actually be judged on: x, y, yaw (4DOF/SE(2))."""
    _, _, yaw_deg = rpy_from_rotation_deg(T[:3, :3])
    yaw = math.radians(yaw_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    T2 = np.eye(4)
    T2[:3, :3] = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    T2[0, 3] = T[0, 3]
    T2[1, 3] = T[1, 3]
    T2[2, 3] = 0.0
    return T2


def report_rpy(matched_est, matched_gt, matched_stamps, out_path, show):
    """Prints per-axis roll/pitch/yaw error stats and -- unless matplotlib is
    unavailable -- saves a GT-vs-VO angle plot plus a per-axis error-over-time
    plot to out_path, so orientation drift is visible, not just the
    translation-only ATE/RPE odometry.py reports."""
    est_rpy = np.array([rpy_from_rotation_deg(T[:3, :3]) for T in matched_est])
    gt_rpy = np.array([rpy_from_rotation_deg(T[:3, :3]) for T in matched_gt])
    err = _wrap_deg(est_rpy - gt_rpy)

    print()
    print("Orientation (roll/pitch/yaw) error, VO vs GT, over %d matched frames:" % len(matched_est))
    axis_names = ("roll", "pitch", "yaw")
    for i, name in enumerate(axis_names):
        a = err[:, i]
        print("  %-5s  mean_abs=%.3f deg  rms=%.3f deg  max_abs=%.3f deg"
              % (name, np.mean(np.abs(a)), math.sqrt(float(np.mean(a ** 2))), np.max(np.abs(a))))

    if out_path is None and not show:
        return

    try:
        import matplotlib
        matplotlib.use("Agg" if not show else matplotlib.get_backend())
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib not available -- skipping roll/pitch/yaw plot", file=sys.stderr)
        return

    t0 = matched_stamps[0]
    rel_time = np.asarray(matched_stamps) - t0

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    for i, name in enumerate(axis_names):
        ax = axes[i]
        ax.plot(rel_time, gt_rpy[:, i], color="black", linewidth=1.3, linestyle="--", label="GT")
        ax.plot(rel_time, est_rpy[:, i], color="tab:red", linewidth=1.3, label="VO (Node.pose)")
        ax.set_ylabel("%s (deg)" % name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    ax_err = axes[3]
    for i, name in enumerate(axis_names):
        ax_err.plot(rel_time, np.abs(err[:, i]), linewidth=1.2, label="%s error" % name)
    ax_err.set_yscale("log")
    ax_err.set_xlabel("elapsed time (s)")
    ax_err.set_ylabel("abs error (deg, log)")
    ax_err.grid(True, alpha=0.3, which="both")
    ax_err.legend(loc="best", fontsize=8)

    fig.suptitle("VO vs GT orientation (cam2/base_link frame)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path:
        fig.savefig(out_path, dpi=150)
        print("Saved roll/pitch/yaw comparison plot to %s" % out_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_xy_topdown(matched_est, matched_gt, out_path, show):
    """Top-down trajectory comparison in the x-y plane (not x-z).

    odometry.py's own plot_trajectory() always plots pose[0,3] vs pose[2,3]
    (x vs z), which is the right pair for raw-KITTI-camera-convention poses
    where z is the dominant forward axis. In this codebase's actual
    base_link/cam2 frame (see GPS.h / CameraModel::opticalRotation()), the
    ground-plane motion lands almost entirely in x/y instead -- z stays
    near-degenerate (see eval_infrastrucer_run.py's analysis in the repo's
    session notes). Plotting x vs z therefore makes a reasonably-tracking
    VO run look like a flat, totally-diverged line. This is the actual
    ground-plane pair for this frame."""
    try:
        import matplotlib
        matplotlib.use("Agg" if not show else matplotlib.get_backend())
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib not available -- skipping x-y top-down plot", file=sys.stderr)
        return

    # matched_{gt,est} are in absolute world coordinates (GT's Mercator
    # projection in particular is on the order of 1e5-1e6 m, not relative to
    # the sequence) -- normalize each trajectory to its own first pose, same
    # as KittiEvalOdom.eval()'s own pred_0/gt_0 identity-rebasing, so both
    # start at the origin and are actually comparable on one plot.
    gt0_inv = np.linalg.inv(matched_gt[0])
    est0_inv = np.linalg.inv(matched_est[0])
    gt_xy = np.array([(gt0_inv @ T)[:2, 3] for T in matched_gt])
    est_xy = np.array([(est0_inv @ T)[:2, 3] for T in matched_est])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="black", linewidth=1.5, linestyle="--", label="GT")
    ax.plot(est_xy[:, 0], est_xy[:, 1], color="tab:red", linewidth=1.5, label="VO (Node.pose)")
    ax.scatter(*gt_xy[0], color="green", marker="o", s=60, zorder=3, label="start")
    ax.scatter(*gt_xy[-1], color="black", marker="x", s=60, zorder=3, label="GT end")
    ax.scatter(*est_xy[-1], color="tab:red", marker="x", s=60, zorder=3, label="VO end")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Top-down (x-y) trajectory -- actual ground plane for this codebase's base_link/cam2 frame")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print("Saved x-y top-down comparison plot to %s" % out_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


_DEFAULT_DB = os.path.join(_REPO_ROOT, "output_2011_10_03_drive_0027_sync_20260929_181127.db")
_DEFAULT_KITTI_RAW = "/media/thippe/Download/ubuntu/2011_10_03_drive_0027_sync"
_DEFAULT_FRAME_START = 0
_DEFAULT_FRAME_END = 4000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=_DEFAULT_DB,
                     help="InfraStructer output .db (Node.pose = raw VO trajectory); default: %(default)s")
    ap.add_argument("--kitti-raw", default=_DEFAULT_KITTI_RAW,
                     help="KITTI raw sequence dir, e.g. 2011_10_03_drive_0027_sync; default: %(default)s")
    ap.add_argument("--calib-dir", help="Dir with calib_imu_to_velo.txt etc. (default: auto-detected, see odom_scale_eval.find_calib_dir)")
    ap.add_argument("--frame-start", type=int, default=_DEFAULT_FRAME_START,
                     help="First Node.id (1-based, inclusive) to evaluate (default: %(default)s)")
    ap.add_argument("--frame-end", type=int, default=_DEFAULT_FRAME_END,
                     help="Last Node.id (1-based, inclusive) to evaluate; clamped to the DB's actual last "
                          "frame if it exceeds it (default: %(default)s)")
    ap.add_argument("--max-time-diff", type=float, default=0.05,
                     help="Max seconds between a DB pose and its matched GT pose (default 0.05)")
    ap.add_argument("--align", choices=["none", "scale", "scale_7dof", "7dof", "6dof"], default="6dof",
                     help="KittiEvalOdom alignment mode for ATE/RPE, applied on top of --se2's x/y/yaw-only "
                          "poses (default: 6dof -- full rotation+translation Umeyama alignment, scale FIXED "
                          "at 1 so a real scale error still shows up in ATE; use 'none' for the raw "
                          "KITTI-benchmark convention of no alignment beyond the frame-0 identity rebase)")
    ap.add_argument("--step", type=int, default=1, help="KittiEvalOdom segment-error step_size passthrough (default 1)")
    ap.add_argument("--out-dir", default=os.path.join(_THIS_DIR, "result", "infrastrucer_run"),
                     help="Where to write the bridged KITTI pose files + odometry.py's own result/errors/plots")
    ap.add_argument("--no-rpy-plot", action="store_true",
                     help="Skip the roll/pitch/yaw VO-vs-GT comparison plot (printed stats are always shown)")
    ap.add_argument("--show-rpy", action="store_true", help="Also open the roll/pitch/yaw plot in an interactive window")
    ap.add_argument("--no-xy-plot", action="store_true",
                     help="Skip the x-y top-down VO-vs-GT trajectory plot (the actual ground plane for this "
                          "codebase's frame -- odometry.py's own plot is x-z, see module docstring)")
    ap.add_argument("--show-xy", action="store_true", help="Also open the x-y top-down plot in an interactive window")
    ap.add_argument("--se2", action=argparse.BooleanOptionalAction, default=True,
                     help="Evaluate as SE(2)/4DOF (x, y, yaw) instead of full SE(3): flattens both GT and VO "
                          "to the ground plane (z=0, rotation=pure yaw) before computing ATE/RPE/t_rel/r_rel "
                          "through the same odometry.py KittiEvalOdom pipeline. Default ON: this run's own "
                          "Odom/Strategy=2 (F2F, monocular, DepthSource=0) reports z/roll/pitch pinned at "
                          "exactly 0 every frame, so scoring against full-SE(3) GT (which has real road "
                          "camber/elevation) otherwise counts GT's genuine but VO-unobservable out-of-plane "
                          "motion as VO error. Pass --no-se2 for the full SE(3)/6DOF comparison.")
    args = ap.parse_args()

    calib_dir = args.calib_dir or ose.find_calib_dir(args.kitti_raw)
    if not calib_dir:
        print("error: could not auto-detect calibration directory -- pass --calib-dir", file=sys.stderr)
        return 1

    print("Ground truth: %s (calib: %s)" % (args.kitti_raw, calib_dir))
    gt_stamps, gt_poses = ose.build_gt_trajectory(args.kitti_raw, calib_dir)
    print("Loaded %d GT poses (GPS+IMU -> cam2 frame)" % len(gt_poses))

    est_stamps, est_poses = load_db_trajectory_range(args.db, args.frame_start, args.frame_end)
    print("Loaded %d DB poses for Node.id range [%d, %d]"
          % (len(est_poses), args.frame_start, args.frame_end))

    matched_est, matched_gt, matched_stamps = ose.match_by_timestamp(
        est_stamps, est_poses, gt_stamps, gt_poses, args.max_time_diff)
    if len(matched_est) < 3:
        print("error: only %d/%d DB frames matched GT within %.3fs"
              % (len(matched_est), len(est_poses), args.max_time_diff), file=sys.stderr)
        return 1
    print("Matched %d/%d DB frames to GT (max_time_diff=%.3fs)"
          % (len(matched_est), len(est_poses), args.max_time_diff))

    if args.se2:
        print("--se2: flattening both GT and VO to the ground plane (z=0, rotation=yaw-only) before scoring")
        matched_gt = [flatten_to_se2(T) for T in matched_gt]
        matched_est = [flatten_to_se2(T) for T in matched_est]

    # SE(2) and SE(3) runs are nested under separate subdirs of --out-dir so
    # re-running with/without --se2 never overwrites the other mode's pose
    # files/metrics/plots -- both are often wanted side by side (full SE(3)
    # vs the fair planar-only comparison).
    out_dir = os.path.join(args.out_dir, "se2" if args.se2 else "se3")

    rpy_plot_path = None if args.no_rpy_plot else os.path.join(out_dir, "rpy_comparison.png")
    if rpy_plot_path:
        os.makedirs(out_dir, exist_ok=True)
    report_rpy(matched_est, matched_gt, matched_stamps, rpy_plot_path, args.show_rpy)

    if not args.no_xy_plot or args.show_xy:
        os.makedirs(out_dir, exist_ok=True)
        xy_plot_path = None if args.no_xy_plot else os.path.join(out_dir, "xy_topdown.png")
        plot_xy_topdown(matched_est, matched_gt, xy_plot_path, args.show_xy)

    gt_dir = os.path.join(out_dir, "gt")
    result_dir = os.path.join(out_dir, "result")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    seq_str = "00_1"  # KittiEvalOdom.eval() reads gt_dir/<seq[:2]>.txt and result_dir/<seq>.txt
    write_kitti_pose_file(os.path.join(gt_dir, "00.txt"), matched_gt)
    write_kitti_pose_file(os.path.join(result_dir, seq_str + ".txt"), matched_est)

    evaluator = KittiEvalOdom()
    metrics = evaluator.eval(
        gt_dir=gt_dir,
        result_dir=result_dir,
        alignment=None if args.align == "none" else args.align,
        eval_seqs=seq_str,
        file_name_plot="%s_plot.png" % seq_str,
        method_name="infrastrucer_vo",
        step=args.step,
    )
    if not metrics:
        print("error: KittiEvalOdom.eval() produced no metrics -- see messages above", file=sys.stderr)
        return 1

    t_rel, r_rel, ate, rpe_trans, rpe_rot, trans_rmse, gt_dist, pred_dist, drift, drift_pct, t_rel_100 = metrics
    print()
    print("Sequence                : %s (Node.id [%d, %d], alignment=%s, mode=%s)"
          % (os.path.basename(args.kitti_raw), args.frame_start, args.frame_end, args.align,
             "SE(2)/4DOF (x,y,yaw)" if args.se2 else "SE(3)/6DOF"))
    print("t_rel (%%)               : %.3f" % t_rel)
    print("r_rel (deg/100m)        : %.3f" % r_rel)
    print("ATE (m)                 : %.3f" % ate)
    print("Translational RMSE (m)  : %.3f" % trans_rmse)
    print("RPE trans (m)           : %.3f" % rpe_trans)
    print("RPE rot (deg)           : %.3f" % rpe_rot)
    print("GT distance (m)         : %.3f" % gt_dist)
    print("Pred distance (m)       : %.3f" % pred_dist)
    print("Drift (m / %%)           : %.3f / %.3f" % (drift, drift_pct))
    print("t_rel @ 100m (%%)        : %.3f" % t_rel_100)
    print()
    print("Full results/errors/plots under: %s" % out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
