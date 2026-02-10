import os
import csv

from sensor_msgs.msg import Imu

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageFilter
from rosbag2_py._storage import StorageOptions, ConverterOptions
from sensor_msgs.msg import CameraInfo, NavSatFix, NavSatStatus
from sensor_msgs.msg import Image
from geometry_msgs.msg import QuaternionStamped

import pymap3d as pm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import bisect

# --- CONFIGURATION ---
BAG_PATH = "/home/tippeswamy/Bag_Files/T2"

IMAGE_TOPIC = '/ouster/nearir_image' 
CALIB_TOPIC = '/camera/color/camera_info'
IMU_TOPIC = '/ouster/imu'
GNSS_FIX_TOPIC = '/gnss/fix'
GNSS_HEADING_TOPIC = '/gnss/heading'

OUTPUT_DIR = "./AMR10_1"
CALIB_FILE_PATH = os.path.join(OUTPUT_DIR, "calib.txt")
TIMESTAMP_FILE_PATH = os.path.join(OUTPUT_DIR, 'times.txt')
IMU_FILE_PATH = os.path.join(OUTPUT_DIR, 'imu_data.csv')
POSES_FILE_PATH = os.path.join(OUTPUT_DIR, 'poses.txt')

# --- Refactored Constants ---
FIXED_PITCH_DEG = 0.0
FIXED_ROLL_DEG = 0.0
# OFFSET_BASELINK_FROM_GNSS: Vector from GNSS Antenna to BaseLink in Body Frame (X-Fwd, Y-Left, Z-Up)
# Based on AMR10.urdf: Lidar/Antenna is at (0.12, 0.477, 0.57) relative to base_link
# So base_link is at (-0.12, -0.477, -0.57) relative to Antenna.
OFFSET_BASELINK_FROM_GNSS = np.array([-0.12, -0.477, -0.57])

R_ros_to_kitti = np.array([
    [0, -1, 0],
    [0,  0, -1],
    [1,  0,  0]
])

# ---------------------

def initialize_reader(bag_path):
    """Initialize and return a configured SequentialReader.
    
    This function determines the appropriate storage ID based on the provided
    `bag_path`, which can be a file or directory. It checks the file extension to
    identify whether the storage ID should be 'mcap' or 'sqlite3'. After
    determining the storage options, it attempts to open the reader with the
    specified options. If an error occurs during the opening process, it logs the
    error and returns None.
    
    Args:
        bag_path (str): The path to the bag file or directory.
    
    Returns:
        SequentialReader or None: A configured SequentialReader instance or None if an
            error occurs during opening.
    """
    reader = SequentialReader()
    found_storage_id = 'sqlite3'
    if os.path.isfile(bag_path):
        if bag_path.endswith('.mcap'):
            found_storage_id = 'mcap'
        elif bag_path.endswith('.db3'):
            found_storage_id = 'sqlite3'
    elif os.path.isdir(bag_path):
        files = os.listdir(bag_path)
        if any(f.endswith('.mcap') for f in files):
            found_storage_id = 'mcap'
        elif any(f.endswith('.db3') for f in files):
            found_storage_id = 'sqlite3'

    print(f"Opening bag '{bag_path}' with storage_id: {found_storage_id}")
    storage_options = StorageOptions(uri=bag_path, storage_id=found_storage_id)
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening bag file: {e}")
        return None
    return reader


def extract_data(bag_path, image_topic, output_dir, timestamp_file):
    """Extracts images and timestamps from a ROS bag file.
    
    This function initializes a reader for the specified ROS bag file  located at
    `bag_path`. It creates a directory for storing extracted  images and sets a
    filter for the specified `image_topic`. As it  reads messages from the bag, it
    deserializes them into images,  saves them to the output directory, and writes
    relative timestamps  to the specified `timestamp_file`. The function returns a
    list of  absolute timestamps for the extracted images.
    
    Args:
        bag_path: The path to the ROS bag file.
        image_topic: The topic from which to extract images.
        output_dir: The directory where images will be saved.
        timestamp_file: The file where relative timestamps will be written.
    """
    reader = initialize_reader(bag_path)
    if reader is None: return None
    image_folder = os.path.join(output_dir, 'image_0')
    os.makedirs(image_folder, exist_ok=True)
    topic_filter = StorageFilter(topics=[image_topic])
    reader.set_filter(topic_filter)
    bridge = CvBridge()
    image_count = 0
    first_timestamp_sec = None
    abs_timestamps = []
    with open(timestamp_file, 'w') as ts_f:
        while reader.has_next():
            topic, data, timestamp_ns = reader.read_next()
            msg = deserialize_message(data, Image)
            t_abs_sec = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            abs_timestamps.append(t_abs_sec)
            if first_timestamp_sec is None:
                first_timestamp_sec = t_abs_sec
                t_rel_sec = 0.0
            else:
                t_rel_sec = t_abs_sec - first_timestamp_sec
            ts_f.write(f"{t_rel_sec:.10e}\n")
            try:
                cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as e:
                continue
            filename = os.path.join(image_folder, f"{image_count:06d}.png")
            cv2.imwrite(filename, cv_image)
            image_count += 1
    return abs_timestamps


def extract_calib(bag_path, calib_topic, output_file):
    """Extracts calibration data from a bag file and writes it to an output file."""
    reader = initialize_reader(bag_path)
    if reader is None: return
    topic_filter = StorageFilter(topics=[calib_topic])
    reader.set_filter(topic_filter)
    if not reader.has_next(): return
    _, data, _ = reader.read_next()
    msg = deserialize_message(data, CameraInfo)
    P = np.array(msg.p).reshape((3, 4))
    P_formatted = ' '.join([f"{x:.12e}" for x in P.flatten()])
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(f"P0: {P_formatted}\n")
        P_zero_placeholder = '0.000000000000e+00 ' * 12
        f.write(f"P1: {P_zero_placeholder.strip()}\n")
        f.write(f"P2: {P_zero_placeholder.strip()}\n")
        f.write(f"P3: {P_zero_placeholder.strip()}\n")
        R0_rect_identity = "1.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00"
        f.write(f"R0_rect: {R0_rect_identity}\n")
        Tr_zero_placeholder = '0.000000000000e+00 ' * 12
        f.write(f"Tr_velo_to_cam: {Tr_zero_placeholder.strip()}\n")


def extract_imu(bag_path, imu_topic, output_csv_file):
    """Extracts IMU data from a bag file and writes it to a CSV file."""
    reader = initialize_reader(bag_path)
    if reader is None: return
    topic_filter = StorageFilter(topics=[imu_topic])
    reader.set_filter(topic_filter)
    if not reader.has_next(): return
    os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
    header = ['header_stamp_sec', 'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
              'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'linear_accel_x', 'linear_accel_y', 'linear_accel_z']
    with open(output_csv_file, 'w', newline='') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(header)
        while reader.has_next():
            topic, data, _ = reader.read_next()
            msg = deserialize_message(data, Imu)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            o, w, a = msg.orientation, msg.angular_velocity, msg.linear_acceleration
            writer.writerow([f"{t:.10f}", o.x, o.y, o.z, o.w, w.x, w.y, w.z, a.x, a.y, a.z])


def extract_gnss_ground_truth(bag_path, fix_topic, heading_topic, output_poses_file, image_abs_timestamps=None):
    """Extract GNSS ground truth data and generate poses.
    
    This function reads GNSS and heading data from a ROS bag file, processes the
    data to convert coordinates from latitude, longitude, and altitude to East-
    North-Up (ENU) format, and interpolates the poses based on image timestamps or
    GNSS timestamps. It also normalizes the poses to a gravity-aligned frame and
    saves the resulting poses to a specified output file.
    
    Args:
        bag_path (str): The path to the ROS bag file containing GNSS and heading data.
        fix_topic (str): The topic name for GNSS fix messages.
        heading_topic (str): The topic name for heading messages.
        output_poses_file (str): The file path where the output poses will be saved.
        image_abs_timestamps (list?): A list of absolute timestamps for images to align poses with.
    """
    print(f"\n--- Starting GNSS Ground Truth Extraction (Refactored Gravity-Aligned Logic) ---")
    
    reader = initialize_reader(bag_path)
    if reader is None: return
    reader.set_filter(StorageFilter(topics=[fix_topic, heading_topic]))
    
    gnss_data = []      # (t, lat, lon, alt)
    heading_data = []   # (t, qx, qy, qz, qw)
    
    print("Reading GNSS/Heading data...")
    count = 0
    while reader.has_next():
        topic, data, _ = reader.read_next()
        
        if topic == fix_topic:
            msg = deserialize_message(data, NavSatFix)
            if msg.status.status >= NavSatStatus.STATUS_FIX:
                t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                gnss_data.append((t, msg.latitude, msg.longitude, msg.altitude))
        
        elif topic == heading_topic:
            msg = deserialize_message(data, QuaternionStamped)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            heading_data.append((t, msg.quaternion.x, msg.quaternion.y, msg.quaternion.z, msg.quaternion.w))
        
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} GNSS/Heading messages...", end='\r')
            
    print(f"\nRead {len(gnss_data)} GNSS fixes and {len(heading_data)} heading samples.")
    
    if not gnss_data or not heading_data:
        print("Error: Not enough GNSS or Heading data found.")
        return

    gnss_times = np.array([x[0] for x in gnss_data])
    gnss_vals = np.array([x[1:] for x in gnss_data])
    
    heading_times = np.array([x[0] for x in heading_data])
    heading_quats = np.array([x[1:] for x in heading_data]) # (x, y, z, w)

    # 3. Coordinate Conversion (LLA -> ENU)
    lat0, lon0, alt0 = gnss_data[0][1], gnss_data[0][2], gnss_data[0][3]
    print(f"Global Origin (First Fix): Lat={lat0:.8f}, Lon={lon0:.8f}, Alt={alt0:.2f}")
    
    enu_vals = []
    for lat, lon, alt in gnss_vals:
        e, n, u = pm.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
        enu_vals.append([e, n, u])
    enu_vals = np.array(enu_vals)

    # 4. Interpolation & Rotation Rebuilding
    unique_heading_times, unique_indices = np.unique(heading_times, return_index=True)
    unique_heading_quats = heading_quats[unique_indices]
    
    # Extract YAW from bag quaternions, then rebuild with FIXED_PITCH/ROLL
    r_obj = R.from_quat(unique_heading_quats)
    euler_angles = r_obj.as_euler('zyx', degrees=True) # Yaw, Pitch, Roll
    yaws = euler_angles[:, 0]
    
    new_eulers = np.zeros_like(euler_angles)
    new_eulers[:, 0] = yaws              # Yaw from Bag
    new_eulers[:, 1] = FIXED_PITCH_DEG   # Fixed Pitch
    new_eulers[:, 2] = FIXED_ROLL_DEG    # Fixed Roll
    
    fixed_rotations = R.from_euler('zyx', new_eulers, degrees=True)
    
    # Slerp the FIXED rotations
    slerp = Slerp(unique_heading_times, fixed_rotations)
    
    # Use image timestamps if available, else use GNSS timestamps
    target_times = image_abs_timestamps or gnss_times
    raw_poses = []
    
    for t_img in target_times:
        # Interpolate Position (GNSS Antenna)
        idx = bisect.bisect_left(gnss_times, t_img)
        if idx == 0:
            pos_gnss = enu_vals[0]
        elif idx >= len(gnss_times):
            pos_gnss = enu_vals[-1]
        else:
            t_low, t_high = gnss_times[idx-1], gnss_times[idx]
            p_low, p_high = enu_vals[idx-1], enu_vals[idx]
            ratio = (t_img - t_low) / (t_high - t_low) if (t_high - t_low) > 1e-6 else 0
            pos_gnss = p_low + (p_high - p_low) * ratio
            
        # Interpolate Rotation (Body Frame)
        t_clamp = max(min(t_img, unique_heading_times[-1]), unique_heading_times[0])
        rot_body_r = slerp([t_clamp])[0]
        rot_body = rot_body_r.as_matrix()
        
        # Apply BaseLink Offset (Antenna -> BaseLink)
        # pos_bl = pos_gnss + R_body * Offset_bl_from_GNSS
        offset_world = rot_body @ OFFSET_BASELINK_FROM_GNSS
        pos_bl = pos_gnss + offset_world
        
        # Convert Rotation to KITTI Optical (Z-Fwd)
        rot_optical = rot_body @ R_ros_to_kitti.T
        
        # Construct 4x4 Global Pose (in Optical Frame orientation)
        T = np.eye(4)
        T[:3, :3] = rot_optical
        T[:3, 3] = pos_bl
        raw_poses.append(T)

    # 5. Normalize Poses (Map to Local Frame)
    if not raw_poses:
        print("Error: No poses generated.")
        return

    print("Normalizing poses to Gravity-Aligned First Frame (Flat Map)...")
    
    # Re-calculate first frame's body orientation for normalization
    t0_img = target_times[0]
    t0_clamp = max(min(t0_img, unique_heading_times[-1]), unique_heading_times[0])
    rot_body_0 = slerp([t0_clamp])[0]
    yaw_0 = rot_body_0.as_euler('zyx', degrees=True)[0]
    
    # Construct Flat Body Rotation for T0 (Yaw only)
    rot_body_flat_0 = R.from_euler('zyx', [yaw_0, 0.0, 0.0], degrees=True).as_matrix()
    rot_opt_flat_0 = rot_body_flat_0 @ R_ros_to_kitti.T
    
    # Construct Reference Pose (T0_ref)
    T0_ref = np.eye(4)
    T0_ref[:3, :3] = rot_opt_flat_0
    T0_ref[:3, 3] = raw_poses[0][:3, 3] # First camera position
    
    T0_ref_inv = np.linalg.inv(T0_ref)
    
    os.makedirs(os.path.dirname(output_poses_file), exist_ok=True)
    with open(output_poses_file, 'w') as f_out:
        for i, T in enumerate(raw_poses):
            T_local = T0_ref_inv @ T
            pose_str = ' '.join([f"{x:.12e}" for x in T_local[:3, :].flatten()])
            
            origin_t = target_times[0]
            t_rel = target_times[i] - origin_t
            f_out.write(f"{t_rel:.10e} {pose_str}\n")

    print(f"--- GNSS Extraction Complete! Saved {len(raw_poses)} poses to: {output_poses_file} ---")


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract data from ROS2 bag to KITTI format.")
    parser.add_argument('--images', action='store_true', help="Extract Images")
    parser.add_argument('--calib', action='store_true', help="Extract Calibration")
    parser.add_argument('--imu', action='store_true', help="Extract IMU")
    parser.add_argument('--gnss', action='store_true', help="Extract GNSS Ground Truth")
    parser.add_argument('--all', action='store_true', help="Extract All")
    
    args = parser.parse_args()
    run_all = args.all or not (args.images or args.calib or args.imu or args.gnss)

    rclpy.init(args=None)
    cached_timestamps = None

    if run_all or args.calib:
        extract_calib(BAG_PATH, CALIB_TOPIC, CALIB_FILE_PATH)
    if run_all or args.images:
        cached_timestamps = extract_data(BAG_PATH, IMAGE_TOPIC, OUTPUT_DIR, TIMESTAMP_FILE_PATH)
    if run_all or args.imu:
        extract_imu(BAG_PATH, IMU_TOPIC, IMU_FILE_PATH)
    if run_all or args.gnss:
        extract_gnss_ground_truth(BAG_PATH, GNSS_FIX_TOPIC, GNSS_HEADING_TOPIC, POSES_FILE_PATH, 
                                  image_abs_timestamps=cached_timestamps)
    rclpy.shutdown()

