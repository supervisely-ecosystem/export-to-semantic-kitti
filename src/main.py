from pathlib import Path

import src.functions as f
import src.globals as g
import supervisely as sly


class MyExport(sly.app.Export):
    def process(self, context: sly.app.Export.Context):
        project = g.api.project.get_info_by_id(id=context.project_id)

        sly_project = f.download_project(g.api, project, g.app_data, context.dataset_id)
        meta = sly_project.meta

        output_dir = Path(g.app_data).joinpath(
            f"{project.id}_{project.name}_semantickitti"
        )
        output_dir.mkdir(exist_ok=True)

        class_mapping, id_to_name = f.create_class_mapping(meta)


        # Process each dataset as a separate sequence
        for seq_idx, dataset_fs in enumerate(sly_project.datasets):
            seq_name = f"{seq_idx:02d}"

            items_data = f.process_dataset(project, dataset_fs, meta, class_mapping)
            f.write_semantic_kitti_dataset(output_dir, seq_name, items_data)

        sly.logger.info(f"Export completed.")
        return output_dir.as_posix()


def main():
    try:
        app = MyExport()
        app.run()
    except Exception as e:
        f.handle_exception(e, g.api, g.task_id)
    finally:
        if not sly.is_development():
            sly.fs.remove_dir(g.app_data)


if __name__ == "__main__":
    sly.main_wrapper("main", main, log_for_agent=False)
