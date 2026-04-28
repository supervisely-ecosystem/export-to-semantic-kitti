import re
from pathlib import Path
from typing import Union, Dict, List, Tuple

import numpy as np
import supervisely as sly
from tqdm import tqdm

import src.globals as g


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

    if project.type != str(sly.ProjectType.POINT_CLOUD_EPISODES):
        raise ValueError(
            f"Only Point Cloud Episodes projects are supported. Got: {project.type}"
        )

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
    Returns: (name_to_id, id_to_name) dictionaries
    """
    name_to_id = {}
    id_to_name = {}

    for idx, obj_class in enumerate(meta.obj_classes, start=1):
        name_to_id[obj_class.name] = idx
        id_to_name[idx] = obj_class.name

    name_to_id["unlabeled"] = 0
    id_to_name[0] = "unlabeled"

    return name_to_id, id_to_name


def get_labels_from_annotation(
    num_points: int,
    episode_ann: sly.PointcloudEpisodeAnnotation,
    frame_index: int,
    class_mapping: Dict[str, int],
    obj_to_instance_id: Dict[int, int],
) -> np.ndarray:
    """
    Extract semantic labels from Supervisely point cloud episode annotation.

    Args:
        num_points: Total number of points in the point cloud
        episode_ann: PointcloudEpisodeAnnotation object
        frame_index: Frame index in the episode
        class_mapping: Dictionary mapping class names to label IDs
        obj_to_instance_id: Dictionary mapping object keys to instance IDs

    Returns:
        Array of label IDs for each point (uint32)
    """
    labels = np.zeros(num_points, dtype=np.uint32)

    figures = episode_ann.get_figures_on_frame(frame_index)

    for fig in figures:
        if fig is None or fig.parent_object is None:
            continue

        parent_obj = fig.parent_object
        class_name = parent_obj.obj_class.name
        semantic_label = class_mapping.get(class_name, 0)

        obj_key = parent_obj.key()
        if obj_key not in obj_to_instance_id:
            obj_to_instance_id[obj_key] = len(obj_to_instance_id) + 1
        instance_id = obj_to_instance_id[obj_key]

        geometry = fig.geometry
        point_indices = None

        if hasattr(geometry, "indices") and geometry.indices is not None:
            point_indices = geometry.indices
        elif hasattr(geometry, "point_indices") and geometry.point_indices is not None:
            point_indices = geometry.point_indices

        if point_indices is not None and len(point_indices) > 0:
            # SemanticKITTI format: lower 16 bits = semantic, upper 16 bits = instance
            label_value = (instance_id << 16) | semantic_label
            try:
                labels[point_indices] = label_value
            except (IndexError, TypeError) as e:
                sly.logger.warning(
                    f"Failed to assign labels for figure on frame {frame_index}: {e}"
                )
                continue

    labeled_count = np.count_nonzero(labels)
    if num_points > 0:
        sly.logger.debug(
            f"Frame {frame_index}: Labeled {labeled_count} / {num_points} points ({100*labeled_count/num_points:.1f}%)"
        )

    return labels


def save_pointcloud_bin(points: np.ndarray, output_path: Path):
    """
    Save point cloud in SemanticKITTI binary format.
    Format: N x 4 array of (x, y, z, intensity) as float32
    """
    # Ensure we have 4 columns (x, y, z, intensity)
    if points.shape[1] == 3:
        intensity = np.zeros((points.shape[0], 1), dtype=np.float32)
        points = np.hstack([points, intensity])

    points = points.astype(np.float32)
    points.tofile(output_path)


def save_labels_bin(labels: np.ndarray, output_path: Path):
    """
    Save labels in SemanticKITTI binary format.
    Format: N x 1 array of uint32 values
    """
    labels = labels.astype(np.uint32)
    labels.tofile(output_path)


def handle_exception(exc: Exception, api: sly.Api, task_id: int):
    from supervisely.io.exception_handlers import (
        handle_exception as sly_handle_exception,
    )

    handled_exc = sly_handle_exception(exc)
    if handled_exc is not None:
        api.task.set_output_error(task_id, handled_exc.title, handled_exc.message)
        err_msg = handled_exc.get_message_for_exception()
        sly.logger.error(err_msg, exc_info=True)
    else:
        err_msg = repr(exc)
        if len(err_msg) > 255:
            err_msg = err_msg[:252] + "..."
        title = "Error occurred"
        api.task.set_output_error(task_id, title, err_msg)
        sly.logger.error(f"{repr(exc)}", exc_info=True)


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
    episode_ann = sly.PointcloudEpisodeAnnotation.load_json_file(
        dataset_fs.get_ann_path(), meta
    )

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
