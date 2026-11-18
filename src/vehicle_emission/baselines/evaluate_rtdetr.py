import argparse
from pathlib import Path

from ultralytics import RTDETR


def main():
    project_root = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(description="Evaluate pre-trained RT-DETR model.")
    parser.add_argument(
        "--model",
        type=str,
        default=str(project_root / "src" / "vehicle_emission" / "baselines" / "rtdetr-x.pt"),
        help="Path to the pre-trained RT-DETR model.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(project_root / "data" / "train_dataset" / "data.yaml"),
        help="Path to the dataset YAML file. Must define 'test' split.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)

    # Ultralytics explicitly checks for a '.pt' extension and will throw an AssertionError for '.pth'
    # if model_path.suffix == '.pth':
    #     pt_path = model_path.with_suffix('.pt')
    #     if model_path.exists():
    #         print(f"Notice: Ultralytics requires a '.pt' extension. Renaming {model_path.name} to {pt_path.name}...")
    #         model_path.rename(pt_path)
    #     model_path = pt_path

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    if not data_path.exists():
        print(f"Error: Dataset YAML not found at {data_path}. Ensure it is configured correctly.")
        return

    print(f"Loading model from {model_path}...")
    try:
        model = RTDETR(str(model_path))
    except AttributeError as e:
        print("\n--- Model Loading Error ---")
        print(f"Direct loading failed: {e}")
        print(
            "Ultralytics expected a full PyTorch model object, but found a dictionary (state_dict)."
        )
        print("Attempting to reconstruct the Ultralytics model and inject the weights...")

        import torch
        import yaml
        from ultralytics.nn.tasks import RTDETRDetectionModel

        # Load the checkpoint dictionary
        ckpt = torch.load(model_path, map_location="cpu")
        raw_state_dict = ckpt.get("ema") or ckpt.get("model") or ckpt

        # Clean up keys if they were saved with an extra 'model.' prefix
        state_dict = {}
        for k, v in raw_state_dict.items():
            if k.startswith("model.model."):
                state_dict[k[6:]] = v
            else:
                state_dict[k] = v

        # Determine class count and names from data.yaml
        nc = 80
        names = {i: f"class_{i}" for i in range(nc)}
        try:
            with open(data_path) as f:
                data_info = yaml.safe_load(f)
            nc = data_info.get("nc", len(data_info.get("names", [])))
            if "names" in data_info:
                names_info = data_info["names"]
                if isinstance(names_info, list):
                    names = {i: name for i, name in enumerate(names_info)}
                elif isinstance(names_info, dict):
                    names = names_info
        except Exception as parse_e:
            print(
                f"Notice: Could not parse data.yaml for class info, defaulting to nc=80 ({parse_e})"
            )

        # Infer architecture from filename
        arch_yaml = "rtdetr-l.yaml"
        lower_name = model_path.name.lower()
        if "v2-x" in lower_name:
            arch_yaml = "rtdetrv2_x.yaml"
        elif "v2-l" in lower_name:
            arch_yaml = "rtdetrv2_l.yaml"
        elif "x" in lower_name:
            arch_yaml = "rtdetr-x.yaml"

        # Use the local YAML file if the user downloaded it
        local_yaml = model_path.parent / arch_yaml
        if local_yaml.exists():
            arch_yaml = str(local_yaml)

        print(f"Initializing {arch_yaml} architecture (nc={nc})...")
        try:
            raw_model = RTDETRDetectionModel(arch_yaml, nc=nc)
        except (KeyError, TypeError, AttributeError) as yaml_e:
            print("\n--- Architecture Parsing Error ---")
            print(f"Failed to parse {arch_yaml}: {yaml_e}")
            print(
                "The YAML you provided is in the official RT-DETRv2 format, but the `ultralytics`"
            )
            print(
                "package requires a YOLO-style YAML format (with 'backbone' and 'head' layer lists)."
            )
            print(
                "Because this checkpoint was trained with the official Paddle/PyTorch RT-DETR repository,"
            )
            print("it cannot be evaluated using the Ultralytics library directly.")
            print(
                "Recommendation: Evaluate it using the official repo's tools/eval.py script, or switch"
            )
            print("to a model natively trained with Ultralytics (e.g., rtdetr-l.pt).")
            return

        print("Injecting state_dict weights into the architecture...")
        try:
            raw_model.load_state_dict(state_dict)
            raw_model.names = names

            fixed_pt_path = model_path.with_name(model_path.stem + "_fixed.pt")
            print(
                f"Weights successfully injected! Saving fixed Ultralytics checkpoint to {fixed_pt_path.name}..."
            )

            torch.save({"model": raw_model, "names": names, "train_args": {}}, fixed_pt_path)

            # Reload the newly fixed model using standard Ultralytics flow
            model = RTDETR(str(fixed_pt_path))
        except Exception as inject_e:
            print(f"\nFailed to load weights: {inject_e}")
            print("The checkpoint's layer keys do not match the Ultralytics architecture.")
            print(
                "This likely means the model was trained using the official RT-DETR repository or another framework."
            )
            return

    print(f"Evaluating model on dataset {data_path} (test split)...")
    # Run validation specifically on the test dataset split
    metrics = model.val(data=str(data_path), split="test")

    print("\n--- Results ---")

    # Output the F1 score per the thesis requirements
    if hasattr(metrics.box, "f1"):
        f1_scores = metrics.box.f1
        mean_f1 = f1_scores.mean() if len(f1_scores) > 0 else 0
        print(f"Mean F1 Score: {mean_f1:.4f}")

        ap_class_idx = (
            metrics.box.ap_class_index
            if hasattr(metrics.box, "ap_class_index")
            else range(len(f1_scores))
        )
        for i, class_idx in enumerate(ap_class_idx):
            class_name = (
                model.names[class_idx]
                if model.names and class_idx in model.names
                else f"Class {class_idx}"
            )
            f1_val = f1_scores[i]
            print(f"F1 Score for {class_name}: {f1_val:.4f}")
    else:
        print("F1 scores not directly available on the metrics object.")


if __name__ == "__main__":
    main()
