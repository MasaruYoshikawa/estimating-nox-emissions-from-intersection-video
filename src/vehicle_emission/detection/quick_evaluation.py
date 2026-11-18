from pathlib import Path

from ultralytics import RTDETR

if __name__ == "__main__":
    # 1. Load your newly trained model
    project_root = Path(
        "D:/Development/Estimating-Vehicle-Emissions-from-Intersection-Traffic-Video-main"
    )

    # Automatically find the latest training run directory
    runs_dir = project_root / "runs"
    finetune_dirs = [
        d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("rtdetr_finetune")
    ]
    latest_run = max(finetune_dirs, key=lambda d: d.stat().st_mtime)

    model_path = latest_run / "weights" / "best.pt"
    print(f"Loading model from latest run: {model_path}")
    yaml_path = project_root / "data" / "train_dataset" / "coco_train.yaml"

    model = RTDETR(model_path)

    # 2. Run validation
    print("Evaluating fine-tuned model...")
    metrics = model.val(data=str(yaml_path), split="val")

    # 3. Ultralytics stores the F1 scores per class in the metrics object
    # It returns a 1D array of optimal F1 scores for each evaluated class.
    f1_scores = metrics.box.f1
    mean_f1 = f1_scores.mean()

    print("\n--- Results ---")
    print(f"Mean F1 Score: {mean_f1:.4f}")
    # Iterate safely over the classes that were actually evaluated
    for i, class_idx in enumerate(metrics.box.ap_class_index):
        class_name = metrics.names[class_idx]
        print(f"F1 Score for {class_name}: {f1_scores[i]:.4f}")
