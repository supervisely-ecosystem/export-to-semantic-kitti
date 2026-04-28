<div align="center" markdown>

<img align="center" src="https://github.com/supervisely-ecosystem/export-to-semantic-kitti/releases/download/v0.0.1/export_semantic_kitti_poster.png">

# Export to SemanticKITTI

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/export-to-semantic-kitti)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/export-to-semantic-kitti)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/export-to-semantic-kitti.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/export-to-semantic-kitti.png)](https://supervise.ly)

</div>

## Overview

The **Export to SemanticKITTI** application allows you to export point cloud episode projects from Supervisely to SemanticKITTI format. This format is widely used for autonomous driving datasets and semantic segmentation tasks.

The application converts point cloud episodes and their point-level segmentation annotations into semantic labels, following the SemanticKITTI data structure.

### Key Features

- Export Supervisely point cloud episodes projects/datasets to SemanticKITTI format
- Point clouds are saved in binary `.bin` format (x, y, z, intensity as float32)
- Semantic labels are saved in binary `.label` format (uint32: lower 16 bits = semantic class, upper 16 bits = instance ID)
- Automatic class mapping from Supervisely classes to SemanticKITTI label IDs
- Direct export of point-level segmentation annotations

### Output Format

The export creates a directory structure following SemanticKITTI conventions:

```
output_directory/
└── sequences/
    ├── 00/                      # First dataset (sequence)
    │   ├── velodyne/           # Point cloud files
    │   │   ├── 000000.bin
    │   │   ├── 000001.bin
    │   │   └── ...
    │   └── labels/             # Semantic label files
    │       ├── 000000.label
    │       ├── 000001.label
    │       └── ...
    ├── 01/                      # Second dataset (if present)
    └── ...
```

### File Formats

**Point Cloud Files (`.bin`):**

- Binary format: N x 4 array of float32 values
- Columns: x, y, z, intensity
- Row-major order

**Label Files (`.label`):**

- Binary format: N x 1 array of uint32 values
- Each value encodes: `(instance_id << 16) | semantic_label`
- Lower 16 bits: semantic class ID (0 = unlabeled)
- Upper 16 bits: instance ID (for distinguishing different objects of same class)

## How to Run

To run the application, follow these steps:

**Option 1. Supervisely Ecosystem:**

1. Find the application in the Ecosystem
2. Choose the necessary project or dataset and press `Run` button

**Option 2: Project/Dataset context menu:**

1. Go to Point Cloud Episode project or dataset you want to export
2. Right-click on the project or dataset and choose `Download as` -> `Export to SemanticKITTI`
3. Press `Run` button in the modal window

## Output

The app creates a task in the `workspace tasks` list. Once the app is finished, you will see a download link to the resulting tar archive.

The archive will be saved to `Files` in:

`Current Team` -> `Files` -> `/tmp/supervisely/export/Export Point Clouds to SemanticKITTI/<task_id>/<project_id>_<project_name>.tar`

## Limitations

- **Only Point Cloud Episodes projects are supported** (single point cloud projects are not supported)
- Only point-level segmentation annotations are exported (3D Cuboid annotations are ignored)
- Points without segmentation labels will have label 0 (unlabeled)
- Each dataset becomes a separate sequence in the output
