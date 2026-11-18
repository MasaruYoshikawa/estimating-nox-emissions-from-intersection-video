from pathlib import Path

import cv2
from ultralytics import RTDETR


def visualize_comparison(
    image_path: str | Path,
    gt_label_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
):
    # Load the original image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    h, w, _ = img.shape

    # 1. Draw Ground Truth Boxes (Green)
    # Assuming Roboflow exported annotations in YOLO format (.txt)
    if Path(gt_label_path).exists():
        with open(gt_label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])

                    print(
                        f"[DEBUG GT] class={cls_id}, val1={x_center}, val2={y_center}, val3={width}, val4={height}"
                    )

                    # Change this to False to test if your dataset is actually [x_min, y_min, x_max, y_max]
                    IS_YOLO_CENTER_FORMAT = False

                    if IS_YOLO_CENTER_FORMAT:
                        # YOLO format: [x_center, y_center, width, height] normalized
                        x1 = int((x_center - width / 2) * w)
                        y1 = int((y_center - height / 2) * h)
                        x2 = int((x_center + width / 2) * w)
                        y2 = int((y_center + height / 2) * h)
                    else:
                        # Normalized min/max format: [x_min, y_min, x_max, y_max]
                        x1 = int(x_center * w)
                        y1 = int(y_center * h)
                        x2 = int(width * w)
                        y2 = int(height * h)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        img,
                        f"GT: {cls_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

    # 2. Draw RT-DETR Prediction Boxes (Red)
    model = RTDETR(str(model_path))
    results = model(img, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            img,
            f"Pred: {cls_id} ({conf:.2f})",
            (x2 + 5, y1 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    # Save the output overlay image
    cv2.imwrite(str(output_path), img)
    print(f"Successfully saved comparison image to {output_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    # Target the Roboflow exported dataset directly
    train_dir = project_root / "data" / "manual_annotations" / "train"
    images_dir = train_dir / "images"
    labels_dir = train_dir / "labels"

    model_file = "rtdetr-l.pt"  # Replace with your fine-tuned .pt file path if available
    out_file = project_root / "figures" / "gt_vs_pred_sample.jpg"

    # Auto-find the first paired image and label to prevent dimension mismatch
    first_image = next(images_dir.glob("*.jpg"), None)
    if first_image:
        label_file = labels_dir / (first_image.stem + ".txt")
        if label_file.exists():
            print(
                f"Testing with matched pair:\nImage: {first_image.name}\nLabel: {label_file.name}"
            )
            visualize_comparison(first_image, label_file, model_file, out_file)
        else:
            print(f"Found image {first_image.name} but no matching text file at {label_file}")
    else:
        print(f"No .jpg images found in {images_dir}")
