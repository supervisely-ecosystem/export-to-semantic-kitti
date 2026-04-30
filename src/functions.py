from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import supervisely as sly
from supervisely.convert.pointcloud_episodes.semantic_kitti.semantic_kitti_helper import (
    SEMANTIC_KITTI_LABEL_MAP,
)
from tqdm import tqdm

# Create name -> ID mapping from SDK's ID -> (name, color) mapping
SEMANTIC_KITTI_NAME_TO_ID = {name.lower(): id for id, (name, _) in SEMANTIC_KITTI_LABEL_MAP.items()}


def get_progress(
    total: int,
    message: str = "Processing...",
    is_size: bool = False,
) -> tuple:
    if sly.is_production():
        progress = sly.Progress(message, total, is_size=is_size)
        progress_cb = progress.iters_done_report
    else:
        progress = tqdm(
            total=total, desc=message, unit="B" if is_size else "it", unit_scale=is_size
        )
        progress_cb = progress.update
    return progress, progress_cb


def download_project(
    api: sly.Api,
    project: sly.Project,
    app_data: Path,
    dataset_id: int = None,
):
    local_path = Path(app_data).joinpath(project.name)
    if not local_path.exists():
        local_path.mkdir(parents=True, exist_ok=True)
    local_path = local_path.as_posix()
    if dataset_id is not None:
        dataset_ids = [dataset_id]
        nested_datasets = api.dataset.get_list(project.id, parent_id=dataset_id)
        dataset_ids.extend([d.id for d in nested_datasets])
    else:
        dataset_ids = None

    sly.fs.mkdir(local_path, remove_content_if_exists=True)

    sly.download_pointcloud_episode_project(
        api,
        project.id,
        local_path,
        download_pointclouds_info=True,
        dataset_ids=dataset_ids,
        log_progress=True,
    )
    sly_project = sly.PointcloudEpisodeProject(local_path, sly.OpenMode.READ)

    return sly_project


def create_class_mapping(
    meta: sly.ProjectMeta,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create mapping between Supervisely class names and SemanticKITTI label IDs.
    Uses case-insensitive matching for standard classes.
    Custom classes get IDs starting from 100.
    """
    name_to_id = {}
    id_to_name = {}
    custom_classes = []

    for obj_class in meta.obj_classes:
        class_name = obj_class.name
        class_lower = class_name.lower()

        if class_lower in SEMANTIC_KITTI_NAME_TO_ID:
            label_id = SEMANTIC_KITTI_NAME_TO_ID[class_lower]
            name_to_id[class_name] = label_id
            id_to_name[label_id] = class_name
            sly.logger.info(f"Mapped '{class_name}' to standard ID {label_id}")
        else:
            custom_classes.append(class_name)

    if custom_classes:
        next_custom_id = 100
        used_ids = set(SEMANTIC_KITTI_NAME_TO_ID.values())

        for class_name in custom_classes:
            while next_custom_id in used_ids or next_custom_id in id_to_name:
                next_custom_id += 1
                if next_custom_id > 251:
                    sly.logger.error(
                        f"Too many custom classes! Cannot assign ID for '{class_name}'"
                    )
                    break

            if next_custom_id <= 251:
                name_to_id[class_name] = next_custom_id
                id_to_name[next_custom_id] = class_name
                sly.logger.info(f"Custom class '{class_name}' mapped to ID {next_custom_id}")
                next_custom_id += 1

    if "unlabeled" not in name_to_id and 0 not in id_to_name:
        name_to_id["unlabeled"] = 0
        id_to_name[0] = "unlabeled"

    standard_count = len(meta.obj_classes) - len(custom_classes)
    sly.logger.info(
        f"Mapped {len(name_to_id)} classes: {standard_count} standard, {len(custom_classes)} custom"
    )

    return name_to_id, id_to_name


def get_labels_from_annotation(
    num_points: int,
    episode_ann: sly.PointcloudEpisodeAnnotation,
    frame_index: int,
    class_mapping: Dict[str, int],
    obj_to_instance_id: Dict[int, int],
) -> np.ndarray:

    labels = np.zeros(num_points, dtype=np.uint32)
    figures = episode_ann.get_figures_on_frame(frame_index)

    sly.logger.debug(f"Frame {frame_index}: processing {len(figures)} figures")

    for fig in figures:
        if fig is None or fig.parent_object is None:
            continue

        parent_obj = fig.parent_object
        class_name = parent_obj.obj_class.name
        semantic_label = class_mapping.get(class_name, None)

        if semantic_label is None:
            sly.logger.error(
                f"Frame {frame_index}: Class '{class_name}' not found in mapping. Skipping."
            )
            continue

        obj_key = parent_obj.key()
        if obj_key not in obj_to_instance_id:
            obj_to_instance_id[obj_key] = len(obj_to_instance_id) + 1
        instance_id = obj_to_instance_id[obj_key]

        geometry = fig.geometry
        if isinstance(geometry, sly.geometry.pointcloud.Pointcloud):
            point_indices = geometry.indices
            if point_indices is not None and len(point_indices) > 0:
                label_value = (instance_id << 16) | semantic_label
                try:
                    labels[point_indices] = label_value
                    sly.logger.debug(
                        f"  {class_name} (ID={semantic_label}, instance={instance_id}): {len(point_indices)} points"
                    )
                except (IndexError, TypeError) as e:
                    sly.logger.warning(
                        f"Frame {frame_index}: Failed to assign labels for {class_name}: {e}"
                    )
        else:
            geom_type = type(geometry).__name__
            sly.logger.debug(f"  {class_name} ({geom_type}): skipped (not Pointcloud geometry)")

    labeled_count = np.count_nonzero(labels)
    sly.logger.info(
        f"Frame {frame_index} summary: {labeled_count}/{num_points} points labeled ({100*labeled_count/num_points:.1f}%)"
    )

    return labels


def save_pointcloud_bin(points: np.ndarray, output_path: Path):
    if points.shape[1] == 3:
        intensity = np.zeros((points.shape[0], 1), dtype=np.float32)
        points = np.hstack([points, intensity])

    points = points.astype(np.float32)
    points.tofile(output_path)


def save_labels_bin(labels: np.ndarray, output_path: Path):
    labels = labels.astype(np.uint32)
    labels.tofile(output_path)


def write_semantic_kitti_dataset(
    output_dir: Path,
    dataset_name: str,
    items_data: List[Tuple[np.ndarray, np.ndarray, str]],
):

    seq_dir = output_dir / "sequences" / dataset_name
    velodyne_dir = seq_dir / "velodyne"
    labels_dir = seq_dir / "labels"

    velodyne_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    progress, progress_cb = get_progress(
        len(items_data), f"Writing SemanticKITTI sequence {dataset_name}..."
    )

    for idx, (points, labels, item_name) in enumerate(items_data):
        frame_id = f"{idx:06d}"
        bin_path = velodyne_dir / f"{frame_id}.bin"
        save_pointcloud_bin(points, bin_path)

        label_path = labels_dir / f"{frame_id}.label"
        save_labels_bin(labels, label_path)

        progress_cb(1)

    if sly.is_development():
        progress.close()

    sly.logger.info(f"Saved {len(items_data)} frames to sequence {dataset_name}")


def process_dataset(
    project: sly.Project,
    dataset_fs: sly.PointcloudEpisodeDataset,
    meta: sly.ProjectMeta,
    class_mapping: Dict[str, int],
) -> List[Tuple[np.ndarray, np.ndarray, str]]:

    items_names = dataset_fs.get_items_names()
    items_data = []

    progress, progress_cb = get_progress(
        len(items_names), f"Processing dataset {dataset_fs.name}..."
    )
    episode_ann = sly.PointcloudEpisodeAnnotation.load_json_file(dataset_fs.get_ann_path(), meta)

    obj_to_instance_id = {}

    for item_name in items_names:
        pcd_path = dataset_fs.get_item_path(item_name)

        points = sly.pointcloud.read(pcd_path)
        num_points = points.shape[0]

        frame_index = dataset_fs.get_frame_idx(item_name)
        labels = get_labels_from_annotation(
            num_points, episode_ann, frame_index, class_mapping, obj_to_instance_id
        )

        items_data.append((points, labels, item_name))

        progress_cb(1)

    if sly.is_development():
        progress.close()

    sly.logger.info(
        f"Processed {len(items_data)} frames with {len(obj_to_instance_id)} unique objects"
    )

    return items_data
