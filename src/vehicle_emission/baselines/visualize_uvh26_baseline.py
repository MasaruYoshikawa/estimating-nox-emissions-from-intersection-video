import sys
from pathlib import Path

import cv2
import torch
import torchvision.transforms as T
from PIL import Image


def main():
    project_root = Path(__file__).resolve().parents[3]
    baselines_dir = project_root / "src" / "vehicle_emission" / "baselines"
    repo_dir = baselines_dir / "RT-DETR"
    rtdetrv2_dir = repo_dir / "rtdetrv2_pytorch"

    if not rtdetrv2_dir.exists():
        print("RT-DETR repo not found. Please run evaluate_uvh26_baseline.py first to clone it.")
        return

    # Add the official codebase to the Python path
    sys.path.insert(0, str(rtdetrv2_dir))
    try:
        from src.core import YAMLConfig
    except ImportError:
        print("Failed to import YAMLConfig. Ensure requirements are installed.")
        return

    # Load the UVH-26 Model configuration
    config_path = baselines_dir / "rtdetrv2_x.yaml"
    weight_path = baselines_dir / "UVH-26-MV-RT-DETRv2-X.pth"

    print(f"Loading official architecture from {config_path.name}...")
    cfg = YAMLConfig(str(config_path))

    # Force num_classes to 15 to match the UVH-26 pre-trained checkpoint shape
    cfg.yaml_cfg["num_classes"] = 15
    if hasattr(cfg, "global_cfg"):
        cfg.global_cfg["num_classes"] = 15

    # Apply model variants
    if "active_model_variant" in cfg.yaml_cfg and "model_variants" in cfg.yaml_cfg:
        variant = cfg.yaml_cfg["active_model_variant"]
        if "v2-x" in weight_path.name.lower() and "r101vd" in cfg.yaml_cfg["model_variants"]:
            variant = "r101vd"

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

    # Load weights
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

    # Class mapping
    uvh_to_local = {0: 1, 1: 1, 2: 1, 3: 1, 4: 0, 5: 2, 8: 3, 9: 0, 10: 1, 12: 1}
    local_names = {0: "bus", 1: "car", 2: "heavy truck", 3: "light truck"}

    # 1. Grab the first image from the test dataset
    test_images_dir = project_root / "data" / "train_dataset" / "test" / "images"
    test_labels_dir = project_root / "data" / "train_dataset" / "test" / "labels"

    image_paths = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    if not image_paths:
        print(f"No images found in {test_images_dir}")
        return

    img_path = image_paths[0]
    label_path = test_labels_dir / (img_path.stem + ".txt")
    print(f"\nProcessing {img_path.name}...")

    # Load image for OpenCV drawing
    img_cv = cv2.imread(str(img_path))
    h_img, w_img, _ = img_cv.shape

    # 2. Draw Ground Truth (Green)
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_min, y_min, x_max, y_max = map(float, parts[1:5])

                    # Format exported from Roboflow was normalized [x_min, y_min, x_max, y_max]
                    x1 = int(x_min * w_img)
                    y1 = int(y_min * h_img)
                    x2 = int(x_max * w_img)
                    y2 = int(y_max * h_img)

                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cls_name = local_names.get(cls_id, f"cls_{cls_id}")
                    cv2.putText(
                        img_cv,
                        f"GT: {cls_name}",
                        (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

    # 3. Run Inference & Draw Predictions (Red)
    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    image_pil = Image.open(img_path).convert("RGB")
    img_tensor = transforms(image_pil).unsqueeze(0)
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    with torch.no_grad():
        outputs = model(img_tensor)

    logits, boxes = outputs["pred_logits"][0], outputs["pred_boxes"][0]
    scores, labels = logits.sigmoid().max(dim=-1)

    keep = scores > 0.45
    scores, labels, boxes = scores[keep], labels[keep], boxes[keep]

    for score, label, box in zip(scores, labels, boxes, strict=False):
        l_item = label.item()
        if l_item in uvh_to_local:
            cls_name = local_names[uvh_to_local[l_item]]
            conf = score.item()

            cx, cy, w_box, h_box = box.cpu().numpy()
            x1 = int((cx - w_box / 2) * w_img)
            y1 = int((cy - h_box / 2) * h_img)
            x2 = int((cx + w_box / 2) * w_img)
            y2 = int((cy + h_box / 2) * h_img)

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # Draw prediction text slightly offset so it doesn't overlap the GT text
            cv2.putText(
                img_cv,
                f"Pred: {cls_name} ({conf:.2f})",
                (x2 + 5, y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

    # 4. Save the Visualization
    out_dir = project_root / "figures"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "uvh26_baseline_visualization.jpg"

    cv2.imwrite(str(out_file), img_cv)
    print(f"Successfully saved comparison image to {out_file}")


if __name__ == "__main__":
    main()
