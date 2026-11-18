import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def bbox_iou_cxcywh(box1, box2):
    """Calculate IoU for two bounding boxes in normalized [cx, cy, w, h] format."""
    b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    b1_area = box1[2] * box1[3]
    b2_area = box2[2] * box2[3]

    return inter_area / (b1_area + b2_area - inter_area + 1e-6)


def main():
    project_root = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(description="Evaluate YOLOv5 baseline model.")
    parser.add_argument(
        "--model",
        type=str,
        default=str(
            project_root
            / "src"
            / "vehicle_emission"
            / "baselines-VehicleOrientation"
            / "best-fp16.tflite"
        ),
        help="Path to the pre-trained YOLOv5 model.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}.")
        print("Please provide the exact filename using the --model argument.")
        return

    print(f"Loading YOLOv5 model from {model_path}...")
    try:
        # TFLite models cannot be loaded directly via torch.hub.load's 'custom' endpoint
        # because it internally uses torch.load() which crashes with "invalid load key".
        # Instead, we find the downloaded YOLOv5 repository and use its DetectMultiBackend.
        hub_dir = Path(torch.hub.get_dir())
        yolov5_dir = next(
            (
                p
                for p in hub_dir.rglob("ultralytics_yolov5_*")
                if (p / "models" / "common.py").exists()
            ),
            None,
        )

        if not yolov5_dir:
            raise RuntimeError("Could not locate ultralytics/yolov5 in PyTorch Hub cache.")

        sys.path.insert(0, str(yolov5_dir))
        from models.common import AutoShape, DetectMultiBackend

        # TFLite executes via its own runtime on the CPU, so we bypass PyTorch's native .cuda() call
        device = (
            torch.device("cpu")
            if model_path.suffix == ".tflite"
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        backend_model = DetectMultiBackend(
            str(model_path), device=device, dnn=False, data=None, fp16=False
        )
        model = AutoShape(backend_model)

        if hasattr(model, "eval"):
            model.eval()
    except Exception as e:
        print(f"Failed to load YOLOv5 model: {e}")
        if "No module named" in str(e):
            missing_module = str(e).split("'")[1] if "'" in str(e) else "required dependencies"
            print(f"\nHint: YOLOv5 requires '{missing_module}'.")
            print(f"Please install it by running: pip install {missing_module}")
        return

    # 1. Map Model Classes
    print(f"Detected Model Classes: {model.names}")
    local_names = {0: "bus", 1: "car", 2: "heavy truck", 3: "light truck"}

    # Dynamic mapping logic to catch common class names
    yolo_to_local = {}
    # TFLite sometimes returns names as a list instead of a dictionary, handle both
    names_dict = (
        model.names
        if isinstance(model.names, dict)
        else {i: name for i, name in enumerate(model.names)}
    )
    for idx, name in names_dict.items():
        name_lower = name.lower()
        if "bus" in name_lower:
            yolo_to_local[idx] = 0
        elif "car" in name_lower or "sedan" in name_lower or "suv" in name_lower:
            yolo_to_local[idx] = 1
        elif "light" in name_lower and "truck" in name_lower:
            yolo_to_local[idx] = 3
        elif "heavy" in name_lower and "truck" in name_lower:
            yolo_to_local[idx] = 2
        elif "truck" in name_lower:
            yolo_to_local[idx] = 2  # Default generic truck to heavy truck

    if not yolo_to_local:
        print(
            "\nWarning: No class names matched automatically. Defaulting to exact index matches (0->0, 1->1, etc)."
        )
        print(
            "Please edit the 'yolo_to_local' dictionary in this script to properly map the classes."
        )

        # TODO: Replace the numbers below with the actual class IDs from your Japanese YOLOv5 model.
        # Map them to our project's local IDs: 0: "bus", 1: "car", 2: "heavy truck", 3: "light truck"
        yolo_to_local = {
            0: 0,  # e.g., if Model ID 0 is Bus -> map to Local ID 0
            1: 1,  # e.g., if Model ID 1 is Car -> map to Local ID 1
            2: 2,  # e.g., if Model ID 2 is Heavy Truck -> map to Local ID 2
            3: 3,  # e.g., if Model ID 3 is Light Truck -> map to Local ID 3
        }
    else:
        print(f"Mapped Model IDs to Local IDs: {yolo_to_local}")

    # 2. Evaluate on the Test Dataset
    test_images_dir = project_root / "data" / "train_dataset" / "test" / "images"
    test_labels_dir = project_root / "data" / "train_dataset" / "test" / "labels"

    tp_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    fp_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    fn_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    image_paths = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))

    # Test only 20 images for quick evaluation
    image_paths = image_paths[:20]
    print(f"\nEvaluating {len(image_paths)} images in the test set...")

    total_raw_preds = 0
    raw_class_counts = {}

    for img_path in tqdm(image_paths, desc="Processing test images"):
        image = Image.open(img_path).convert("RGB")
        w_img, h_img = image.size

        # TFLite rigidly requires 640x640. Squashing destroys aspect ratio.
        # We use manual letterboxing (padding with gray bars) to preserve vehicle shapes.
        scale = 640.0 / max(w_img, h_img)
        new_w, new_h = int(w_img * scale), int(h_img * scale)
        image_resized = image.resize((new_w, new_h))

        image_padded = Image.new("RGB", (640, 640), (114, 114, 114))
        pad_w, pad_h = (640 - new_w) // 2, (640 - new_h) // 2
        image_padded.paste(image_resized, (pad_w, pad_h))

        with torch.no_grad():
            results = model(image_padded, size=640)

        # xyxy format gives absolute pixels: [x1, y1, x2, y2, conf, cls]
        predictions = results.xyxy[0].cpu().numpy()

        pred_objs = []
        for pred in predictions:
            x1, y1, x2, y2, conf, cls_id = pred
            cls_id = int(cls_id)

            total_raw_preds += 1
            raw_class_counts[cls_id] = raw_class_counts.get(cls_id, 0) + 1

            if conf > 0.45 and cls_id in yolo_to_local:
                # Un-pad the bounding box coordinates back to the original image scale
                x1_orig = (x1 - pad_w) / scale
                y1_orig = (y1 - pad_h) / scale
                x2_orig = (x2 - pad_w) / scale
                y2_orig = (y2 - pad_h) / scale

                # Normalize to 0.0 - 1.0 against original image dimensions
                cx = (x1_orig + x2_orig) / 2.0 / w_img
                cy = (y1_orig + y2_orig) / 2.0 / h_img
                w_norm = (x2_orig - x1_orig) / w_img
                h_norm = (y2_orig - y1_orig) / h_img

                pred_objs.append(
                    {
                        "class": yolo_to_local[cls_id],
                        "box": np.array([cx, cy, w_norm, h_norm]),
                        "matched": False,
                    }
                )

        # Load Ground Truth
        label_path = test_labels_dir / (img_path.stem + ".txt")
        gt_objs = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Convert normalized [x_min, y_min, x_max, y_max] from Roboflow to [cx, cy, w, h]
                        x_min, y_min, x_max, y_max = map(float, parts[1:5])
                        cx = (x_min + x_max) / 2.0
                        cy = (y_min + y_max) / 2.0
                        w_norm = x_max - x_min
                        h_norm = y_max - y_min
                        gt_objs.append(
                            {
                                "class": int(parts[0]),
                                "box": np.array([cx, cy, w_norm, h_norm]),
                                "matched": False,
                            }
                        )

        # Greedy Match
        for gt in gt_objs:
            best_iou, best_idx = 0, -1
            for i, pred in enumerate(pred_objs):
                if not pred["matched"] and pred["class"] == gt["class"]:
                    iou = bbox_iou_cxcywh(gt["box"], pred["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i

            if best_iou > 0.5:
                pred_objs[best_idx]["matched"] = True
                gt["matched"] = True
                tp_counts[gt["class"]] += 1
            else:
                fn_counts[gt["class"]] += 1

        for pred in pred_objs:
            if not pred["matched"]:
                fp_counts[pred["class"]] += 1

    print("\n--- Diagnostic Info ---")
    print(f"Total raw predictions before filtering: {total_raw_preds}")
    print(f"Detected raw class IDs: {raw_class_counts}")
    if sum(tp_counts.values()) == 0:
        print(
            "\nWARNING: 0 True Positives detected! This usually means your 'yolo_to_local' dictionary"
        )
        print(
            "is not mapping the model's raw class IDs to our project IDs (0: bus, 1: car, 2: heavy truck, 3: light truck)."
        )
        print(
            "Look at the 'Detected raw class IDs' above and update the 'yolo_to_local' mapping in the script!"
        )

    # Calculate Metrics
    print("\n--- Baseline Results (YOLOv5 Model) ---")
    f1_list = []
    for cls_idx in range(4):
        tp, fp, fn = tp_counts[cls_idx], fp_counts[cls_idx], fn_counts[cls_idx]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_list.append(f1)
        print(
            f"F1 Score for {local_names[cls_idx]}: {f1:.4f} (P: {precision:.4f}, R: {recall:.4f})"
        )

    print(f"\nMean F1 Score: {np.mean(f1_list):.4f}")


if __name__ == "__main__":
    main()
