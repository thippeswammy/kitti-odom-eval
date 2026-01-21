# KITTI Odometry Evaluation Toolbox

An enhanced evaluation toolbox for KITTI Odometry benchmark, enabling robust parsing of various pose formats and automatic timestamp alignment.

## Features

- **Robust Parsing**: Supports multiple pose file formats:
  - **KITTI**: 12 floats (3x4 matrix)
  - **KITTI with Timestamp**: 13 floats (Timestamp + 12 floats)
  - **TUM**: 7 floats (tx ty tz qx qy qz qw)
  - **TUM with Timestamp**: 8 floats (Timestamp + 7 floats)
  - **TUM with Index**: 9 floats (Timestamp + 7 floats + index/id)
- **Automatic Timestamp Normalization**: Automatically detects absolute Unix timestamps (e.g., `1.7e9`) and normalizes them to start at `0.0` to match Ground Truth files.
- **Comment Handling**: Automatically ignores comment lines starting with `#`.
- **Method & Format Logging**: clearly logs detected file formats and processing details.

## Requirements

- Python 3.8 or higher
- NumPy
- Pandas
- Plotly
- PyYAML
- SciPy

## Usage

The primary script is `odometry.py`, which is driven by a YAML configuration file.

### Running Evaluation

```bash
python3 odometry.py --config config.yaml
```

### Configuration File (`config.yaml`)

Define your evaluation tasks in `config.yaml`:

```yaml
eval_config:
  - method_name: "droid_slam"
    result_dir: "result/droid_slam"     # Directory containing result files (00.txt, 01.txt...)
    gt_dir: "dataset/buggy_odom/gt_poses/" # Directory containing GT files
    alignments: ["7dof"]                # Alignment options: scale, 6dof, 7dof, scale_7dof
    sequences: [0, 1]                   # Sequences to evaluate
    step: 1                             # Frame step (optional)
    computational_metrics_optional: true
```

### Supported Result formats

Your result files (e.g., `result/droid_slam/00.txt`) can be in any of the following formats:

1. **Standard KITTI (12 floats)**:

   ```
   r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
   ```

2. **KITTI with Timestamp (13 floats)**:

   ```
   timestamp r11 r12 r13 tx ...
   ```

3. **TUM Format (7 floats)**:

   ```
   tx ty tz qx qy qz qw
   ```

4. **TUM with Timestamp (8 floats)**:

   ```
   timestamp tx ty tz qx qy qz qw
   ```

5. **TUM with Index (9 floats) - *New***:

   ```
   timestamp tx ty tz qx qy qz qw index
   ```

   *Note: Timestamps can be absolute (Unix epoch) or relative. The tool will normalize them automatically.*

## License

Released under the MIT license.
