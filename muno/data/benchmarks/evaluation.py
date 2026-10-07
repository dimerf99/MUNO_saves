import csv
import json
from pathlib import Path

import torch

from muno.utils.metrics import compute_metrics
from muno.utils.metrics_physical import compute_physical_metrics


def prefix_metrics(metrics, prefix):
    return {
        f"{prefix}/{name}": value
        for name, value in metrics.items()
    }


def filter_physical_metric_configs(metric_configs, task_name):
    selected_configs = []

    for config in metric_configs or []:
        tasks = config.get("tasks", "all")

        if tasks == "all" or task_name in tasks:
            clean_config = {
                key: value
                for key, value in config.items()
                if key != "tasks"
            }
            selected_configs.append(clean_config)

    return selected_configs


def compute_batch_metrics(pred, target, metrics_config, task_name):
    results = {}

    metric_names = metrics_config.get("names", [])
    if metric_names:
        results.update(
            compute_metrics(
                pred,
                target,
                metric_names=metric_names,
            )
        )

    physical_configs = filter_physical_metric_configs(
        metrics_config.get("physical", []),
        task_name,
    )
    if physical_configs:
        results.update(
            compute_physical_metrics(
                pred,
                target,
                metric_configs=physical_configs,
            )
        )

    return results


def set_trainer_eval_mode(trainer):
    model = trainer.model
    if model is None:
        raise RuntimeError("Model has not been declared before evaluation.")
    model.eval()


def predict_batch(trainer, sample, data_processor=None):
    assert isinstance(sample, dict), "Sample has to be passed as dict."

    for key in sample.keys():
        sample[key]["x"] = sample[key]["x"].to(trainer.device)
        sample[key]["y"] = sample[key]["y"].to(trainer.device)

        if "mask" in sample[key]:
            sample[key]["mask"] = sample[key]["mask"].to(trainer.device)

    if data_processor is not None:
        sample = data_processor.preprocess(sample, training=False)

    model = trainer.model
    if model is None:
        raise RuntimeError("Model has not been declared before evaluation.")

    pred = model({key: sample[key]["x"] for key in sample})

    if data_processor is not None:
        pred, sample = data_processor.postprocess(
            pred,
            sample,
            training=False,
        )

    target = {
        key: sample[key]["y"]
        for key in sample
    }

    return pred, target


def evaluate_loader(
    trainer,
    loader,
    data_processor,
    metrics_config,
    task_metadata,
):
    metric_sums = {}
    metrics_count = {}

    set_trainer_eval_mode(trainer)

    with torch.no_grad():
        for sample in loader:
            pred, target = predict_batch(
                trainer,
                sample,
                data_processor=data_processor,
            )

            for task_idx in pred.keys():
                task_name = task_metadata[task_idx].get("name", f"task_{task_idx}")

                batch_metrics = compute_batch_metrics(
                    pred[task_idx],
                    target[task_idx],
                    metrics_config=metrics_config,
                    task_name=task_name,
                )

                batch_size = int(target[task_idx].shape[0])

                for name, value in batch_metrics.items():
                    metric_key = f"{task_name}/{name}"
                    metric_sums[metric_key] = metric_sums.get(metric_key, 0.0) + float(value) * batch_size
                    metrics_count[metric_key] = metrics_count.get(metric_key, 0) + batch_size

    if not metric_sums:
        return {}

    return {
        metric_key: metric_sums[metric_key] / metrics_count[metric_key]
        for metric_key in metric_sums.keys()
    }


def evaluate_multitask_loaders(
    trainer,
    loader,
    data_processor,
    task_metadata,
    metrics_config,
    split_name,
):
    task_metrics = evaluate_loader(
        trainer=trainer,
        loader=loader,
        data_processor=data_processor,
        metrics_config=metrics_config,
        task_metadata=task_metadata,
    )

    return prefix_metrics(
        task_metrics,
        split_name,
    )


def save_metrics(metrics, output_dir, filename_stem):
    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    json_path = metrics_dir / f"{filename_stem}.json"
    csv_path = metrics_dir / f"{filename_stem}.csv"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])

        for name, value in metrics.items():
            writer.writerow([name, value])

    print(f"metrics saved to: {json_path}")
    print(f"metrics saved to: {csv_path}")
