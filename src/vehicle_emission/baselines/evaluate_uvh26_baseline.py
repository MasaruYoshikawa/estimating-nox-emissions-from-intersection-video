import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


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
    baselines_dir = project_root / "src" / "vehicle_emission" / "baselines"
    repo_dir = baselines_dir / "RT-DETR"
    rtdetrv2_dir = repo_dir / "rtdetrv2_pytorch"

    # 1. Ensure the official repository is cloned
    if not rtdetrv2_dir.exists():
        print("Cloning the official RT-DETR repository...")
        os.system(f"git clone https://github.com/lyuwenyu/RT-DETR.git {repo_dir}")

    # 2. Add the official codebase to the Python path
    sys.path.insert(0, str(rtdetrv2_dir))

    try:
        from src.core import YAMLConfig
    except ImportError as e:
        print(f"Failed to import from official RT-DETR repo: {e}")
        print(
            "Please ensure you have installed its requirements: pip install -r src/vehicle_emission/baselines/RT-DETR/rtdetrv2_pytorch/requirements.txt"
        )
        return

    # 3. Load the UVH-26 Model
    config_path = baselines_dir / "rtdetrv2_x.yaml"
    weight_path = baselines_dir / "UVH-26-MV-RT-DETRv2-X.pth"

    print(f"Loading official architecture from {config_path.name}...")
    cfg = YAMLConfig(str(config_path))

    # Force num_classes to 15 to match the UVH-26 pre-trained checkpoint shape
    cfg.yaml_cfg["num_classes"] = 15
    if hasattr(cfg, "global_cfg"):
        cfg.global_cfg["num_classes"] = 15

    # Manually resolve the 'model_variants' block which isn't natively supported by RT-DETR's YAMLConfig
    if "active_model_variant" in cfg.yaml_cfg and "model_variants" in cfg.yaml_cfg:
        variant = cfg.yaml_cfg["active_model_variant"]

        # Auto-switch to r101vd if loading the -X model to ensure architecture capacity matches
        if "v2-x" in weight_path.name.lower() and "r101vd" in cfg.yaml_cfg["model_variants"]:
            variant = "r101vd"

        print(f"Applying YAML overrides for variant: {variant}...")
        overrides = cfg.yaml_cfg["model_variants"].get(variant, {})
        for k, v in overrides.items():
            if k not in cfg.yaml_cfg:
                cfg.yaml_cfg[k] = {}
            if isinstance(v, dict) and isinstance(cfg.yaml_cfg[k], dict):
                cfg.yaml_cfg[k].update(v)
            else:
                cfg.yaml_cfg[k] = v

            if hasattr(cfg, "global_cfg"):
                if k not in cfg.global_cfg:
                    cfg.global_cfg[k] = {}
                if isinstance(v, dict) and isinstance(cfg.global_cfg[k], dict):
                    cfg.global_cfg[k].update(v)
                else:
                    cfg.global_cfg[k] = v

    model = cfg.model

    print(f"Loading weights from {weight_path.name}...")
    state_dict = torch.load(weight_path, map_location="cpu")
    if "ema" in state_dict:
        state_dict = state_dict["ema"]["module"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # 4. Map UVH-26 Indian Traffic Classes (0-indexed) to your Project's 4 classes
    # Project: 0: bus, 1: car, 2: heavy truck, 3: light truck
    uvh_to_local = {
        0: 1,  # Hatchback -> car
        1: 1,  # Sedan -> car
        2: 1,  # SUV -> car
        3: 1,  # MUV -> car
        4: 0,  # Bus -> bus
        5: 2,  # Truck -> heavy truck
        8: 3,  # LCV (Light Commercial) -> light truck
        9: 0,  # Mini-bus -> bus
        10: 1,  # Tempo-traveller -> car
        12: 1,  # Van -> car
    }
    local_names = {0: "bus", 1: "car", 2: "heavy truck", 3: "light truck"}

    # 5. Evaluate on the Test Dataset
    test_images_dir = project_root / "data" / "train_dataset" / "test" / "images"
    test_labels_dir = project_root / "data" / "train_dataset" / "test" / "labels"

    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    tp_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    fp_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    fn_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    image_paths = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    print(f"\nEvaluating {len(image_paths)} images in the test set...")

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        img_tensor = transforms(image).unsqueeze(0)
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        with torch.no_grad():
            outputs = model(img_tensor)

        logits, boxes = outputs["pred_logits"][0], outputs["pred_boxes"][0]
        scores, labels = logits.sigmoid().max(dim=-1)

        # Filter predictions
        keep = scores > 0.45
        scores, labels, boxes = scores[keep], labels[keep], boxes[keep]

        pred_objs = [
            {
                "class": uvh_to_local[label.item()],
                "box": box.cpu().numpy(),
                "matched": False,
            }
            for label, box in zip(labels, boxes, strict=False)
            if label.item() in uvh_to_local
        ]

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
                        w = x_max - x_min
                        h = y_max - y_min
                        gt_objs.append(
                            {
                                "class": int(parts[0]),
                                "box": np.array([cx, cy, w, h]),
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
                        best_iou, best_idx = iou, i

            if best_iou > 0.5:
                pred_objs[best_idx]["matched"] = gt["matched"] = True
                tp_counts[gt["class"]] += 1
            else:
                fn_counts[gt["class"]] += 1

        for pred in pred_objs:
            if not pred["matched"]:
                fp_counts[pred["class"]] += 1

    # Calculate Metrics
    print("\n--- Baseline Results (UVH-26 Model) ---")
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
