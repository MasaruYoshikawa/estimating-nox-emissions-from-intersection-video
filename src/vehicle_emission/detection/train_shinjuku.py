from pathlib import Path

from ultralytics import RTDETR


def train_shinjuku_model() -> None:
    project_root = Path(__file__).resolve().parents[3]
    dataset_yaml = project_root / "data" / "shinjuku_train_dataset" / "shinjuku_data.yaml"

    if not dataset_yaml.exists():
        print(f"Error: {dataset_yaml} not found. Please ensure dataset conversion is complete.")
        return

    print("=======================================================")
    print("Starting RT-DETR Fine-Tuning for Shinjuku Intersection")
    print("=======================================================")
    print(f"Dataset config: {dataset_yaml}")

    # Start from our best Sukiyabashi checkpoint to leverage learned vehicle representations
    prev_model = project_root / "runs" / "rtdetr_finetune-4" / "weights" / "best.pt"
    initial_weights = str(prev_model) if prev_model.exists() else "rtdetr-l.pt"
    print(f"Initial weights: {initial_weights}")

    model = RTDETR(initial_weights)

    # Train and save into a dedicated new run: runs/rtdetr_shinjuku
    model.train(
        data=str(dataset_yaml),
        epochs=50,
        imgsz=640,
        batch=4,
        project=str(project_root / "runs"),
        name="rtdetr_shinjuku",
        save=True,
        plots=True,
        workers=2,
    )

    print("\nTraining completed successfully!")
    print(
        "New model saved in: "
        f"{project_root / 'runs' / 'rtdetr_shinjuku' / 'weights' / 'best.pt'}"
    )


if __name__ == "__main__":
    train_shinjuku_model()
