import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.fftpack import dct
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, Dataset
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

SEED = 42
IMAGE_SIZE = 384
THRESHOLD = 0.50
LORA_RANK = 8
LORA_ALPHA = 16.0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_DIRECTORIES = {"non_glaucoma": 0, "glaucoma": 1}
THREE_CLASS_NAMES = {0: "Non-Glaucoma", 1: "Suspect", 2: "Glaucoma"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_external_dataset(destination):
    import kagglehub

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".kaggle_complete"
    if marker.exists():
        return destination
    downloaded = kagglehub.dataset_download(
        "jeremypoveda/dataset-unified-fundus-images-for-glaucoma",
        output_dir=str(destination),
    )
    marker.write_text(str(downloaded), encoding="utf-8")
    return destination


def find_test_root(root):
    root = Path(root)
    candidates = [root] if root.name.lower() == "test" else []
    candidates.extend(path for path in root.rglob("test") if path.is_dir())
    for candidate in candidates:
        children = {path.name.lower().replace("-", "_").replace(" ", "_"): path for path in candidate.iterdir() if path.is_dir()}
        if "glaucoma" in children and "non_glaucoma" in children:
            return candidate
    raise FileNotFoundError(f"Could not locate a test directory containing glaucoma and non_glaucoma folders under {root}")


def build_external_index(test_root, audit_file):
    rows = []
    invalid = []
    children = {
        path.name.lower().replace("-", "_").replace(" ", "_"): path
        for path in Path(test_root).iterdir()
        if path.is_dir()
    }
    for folder_name, label in CLASS_DIRECTORIES.items():
        folder = children.get(folder_name)
        if folder is None:
            raise FileNotFoundError(Path(test_root) / folder_name)
        files = sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        for path in tqdm(files, desc=f"Auditing {folder_name}", unit="image"):
            try:
                with Image.open(path) as image:
                    image.verify()
                rows.append({
                    "image_path": str(path.resolve()),
                    "filename": path.name,
                    "label": label,
                    "label_name": "Glaucoma" if label == 1 else "Non-Glaucoma",
                    "source_folder": folder_name,
                })
            except Exception as error:
                invalid.append({"image_path": str(path), "error": str(error)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No readable external test images were found.")
    if frame["label"].nunique() != 2:
        raise RuntimeError("The external test directory must contain both glaucoma and non_glaucoma images.")
    pd.DataFrame(invalid).to_csv(audit_file, index=False)
    return frame


def normalized_image(image):
    return image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)


def read_hdf5_image(path, index):
    with h5py.File(path, "r") as handle:
        array = handle["images"][int(index)]
    return Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")


def development_image_sources(root):
    root = Path(root)
    fives_manifest = root / "data" / "processed" / "fives" / "manifest.csv"
    odir_manifest = root / "data" / "processed" / "odir" / "manifest.csv"
    brset_manifest = root / "data" / "processed" / "brset" / "manifest.csv"
    smdg_manifest = root / "data" / "processed" / "smdg" / "manifest.csv"
    for manifest_path, column in ((fives_manifest, "image_path"), (odir_manifest, "image_path")):
        if manifest_path.exists():
            frame = pd.read_csv(manifest_path)
            if column in frame:
                for value in frame[column].dropna().astype(str):
                    path = Path(value)
                    if path.is_file():
                        yield str(path), Image.open(path).convert("RGB")
    for manifest_path, h5_path in (
        (brset_manifest, root / "data" / "processed" / "brset" / "brset_384.h5"),
        (smdg_manifest, root / "data" / "processed" / "smdg" / "smdg_384.h5"),
    ):
        if manifest_path.exists() and h5_path.exists():
            frame = pd.read_csv(manifest_path)
            if "h5_index" in frame:
                with h5py.File(h5_path, "r") as handle:
                    for row in frame.itertuples(index=False):
                        array = handle["images"][int(getattr(row, "h5_index"))]
                        yield f"{h5_path}::{getattr(row, 'h5_index')}", Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")


def perceptual_hash(image):
    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    transformed = dct(dct(gray, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    values = transformed.flatten()[1:]
    median = np.median(values)
    bits = values > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hash_hex(value):
    return hashlib.sha256(value).hexdigest()


def phash_bands(value):
    return tuple((value >> shift) & mask for shift, mask in ((51, 0x1FFF), (38, 0x1FFF), (25, 0x1FFF), (12, 0x1FFF), (0, 0x0FFF)))


def audit_development_overlap(root, external_frame, output_path, phash_distance=4):
    exact_index = defaultdict(list)
    phash_values = []
    phash_sources = []
    development_count = 0
    for source, image in tqdm(development_image_sources(root), desc="Indexing development images", unit="image"):
        try:
            normalized = normalized_image(image)
            pixels = np.asarray(normalized, dtype=np.uint8).tobytes()
            digest = hash_hex(pixels)
            p_hash = perceptual_hash(normalized)
            exact_index[digest].append(source)
            phash_values.append(p_hash)
            phash_sources.append(source)
            development_count += 1
        except Exception:
            continue
        finally:
            image.close()
    band_index = [defaultdict(list) for _ in range(5)]
    for index, value in enumerate(phash_values):
        for band, part in enumerate(phash_bands(value)):
            band_index[band][part].append(index)
    exact_matches = []
    perceptual_matches = []
    external_hashes = {}
    for row in tqdm(external_frame.itertuples(index=False), total=len(external_frame), desc="Comparing external images", unit="image"):
        path = Path(row.image_path)
        try:
            with Image.open(path) as source_image:
                normalized = normalized_image(source_image)
            pixel_bytes = np.asarray(normalized, dtype=np.uint8).tobytes()
            digest = hash_hex(pixel_bytes)
            external_hashes[str(path)] = digest
            if digest in exact_index:
                exact_matches.append({"external_image": str(path), "development_matches": exact_index[digest]})
            external_phash = perceptual_hash(normalized)
            candidates = set()
            for band, part in enumerate(phash_bands(external_phash)):
                candidates.update(band_index[band].get(part, []))
            best = []
            for candidate_index in candidates:
                distance = (external_phash ^ phash_values[candidate_index]).bit_count()
                if distance <= phash_distance:
                    best.append((distance, phash_sources[candidate_index]))
            if best:
                best.sort(key=lambda item: item[0])
                perceptual_matches.append({
                    "external_image": str(path),
                    "minimum_hamming_distance": int(best[0][0]),
                    "development_matches": [source for _, source in best[:10]],
                })
        except Exception:
            continue
    report = {
        "development_images_indexed": development_count,
        "external_test_images_indexed": len(external_hashes),
        "exact_standardized_pixel_matches": len(exact_matches),
        "perceptual_candidate_matches": len(perceptual_matches),
        "perceptual_hash_hamming_threshold": int(phash_distance),
        "exact_matches": exact_matches,
        "perceptual_candidates": perceptual_matches,
        "patient_level_independence": "not_verifiable_from_external_dataset_because_patient_metadata_is_not_available_in_the_external_test_folder",
        "interpretation": "This is an image-level screening audit only. A zero match count does not prove patient-level or source-level independence.",
    }
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


class ExternalDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as source:
            image = normalized_image(source)
        tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        return tensor, int(row["label"]), index


class LoRAWeight(nn.Module):
    def __init__(self, out_features, in_features, rank, alpha):
        super().__init__()
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, original):
        return original + self.scale * (self.lora_B @ self.lora_A)


class SharedEncoder(nn.Module):
    def __init__(self, feature_module, norm_module):
        super().__init__()
        self.features = feature_module
        self.norm = norm_module

    def spatial_features(self, images):
        return self.norm(self.features(images))

    def forward(self, images):
        return self.spatial_features(images).mean(dim=(1, 2))


class ClientHead(nn.Module):
    def __init__(self, outputs=3):
        super().__init__()
        self.model = nn.Sequential(nn.LayerNorm(768), nn.Dropout(0.20), nn.Linear(768, outputs))

    def forward(self, features):
        return self.model(features)


def build_model(root, device):
    encoder_path = Path(root) / "checkpoints" / "fives_segmentation" / "retinal_swinv2_encoder.pt"
    best_path = Path(root) / "checkpoints" / "federated" / "best_federated.pt"
    last_path = Path(root) / "checkpoints" / "federated" / "last_federated.pt"
    for path in (encoder_path, best_path, last_path):
        if not path.exists():
            raise FileNotFoundError(f"Required checkpoint is missing: {path}")
    last_checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    run_configuration = last_checkpoint.get("configuration", {})
    expected_rounds = int(run_configuration.get("rounds", 30))
    completed_round = int(last_checkpoint.get("round", 0))
    if completed_round != expected_rounds:
        raise RuntimeError(
            f"Federated training is incomplete: {completed_round} of {expected_rounds} rounds are saved. "
            "Complete the second script before running external validation."
        )
    if int(run_configuration.get("lora_rank", LORA_RANK)) != LORA_RANK or float(run_configuration.get("lora_alpha", LORA_ALPHA)) != LORA_ALPHA:
        raise RuntimeError("Federated checkpoint LoRA configuration does not match the paper protocol.")
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    if checkpoint.get("configuration") != run_configuration:
        raise RuntimeError("Best and final federated checkpoints do not belong to the same training configuration.")
    if int(checkpoint.get("round", 0)) != int(checkpoint.get("best_round", 0)):
        raise RuntimeError("The selected checkpoint is not identified as its own best round.")
    backbone = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
    encoder_state = torch.load(encoder_path, map_location="cpu", weights_only=False)
    backbone.features.load_state_dict(encoder_state["encoder_state_dict"], strict=True)
    for parameter in backbone.features.parameters():
        parameter.requires_grad = False
    for parameter in backbone.norm.parameters():
        parameter.requires_grad = False
    registered_blocks = 0
    for module in backbone.features.modules():
        attention = getattr(module, "attn", None)
        if attention is None or not hasattr(attention, "qkv") or not hasattr(attention, "proj"):
            continue
        for projection in (attention.qkv, attention.proj):
            if isinstance(projection, nn.Linear):
                parametrize.register_parametrization(
                    projection,
                    "weight",
                    LoRAWeight(projection.out_features, projection.in_features, LORA_RANK, LORA_ALPHA),
                )
                projection.parametrizations.weight.original.requires_grad = False
        registered_blocks += 1
    encoder = SharedEncoder(backbone.features, backbone.norm).to(device)
    named_parameters = dict(encoder.named_parameters())
    lora_state = checkpoint["global_lora"]
    expected_names = {name for name in named_parameters if "lora_A" in name or "lora_B" in name}
    if set(lora_state) != expected_names:
        missing = sorted(expected_names.difference(lora_state))
        extra = sorted(set(lora_state).difference(expected_names))
        raise RuntimeError(f"LoRA checkpoint keys do not match this model. Missing={missing[:5]}, extra={extra[:5]}")
    with torch.no_grad():
        for name, value in lora_state.items():
            named_parameters[name].copy_(value.to(device))
    head = ClientHead(3).to(device)
    head.load_state_dict(checkpoint["client_heads"]["SMDG"], strict=True)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    for parameter in head.parameters():
        parameter.requires_grad = False
    encoder.eval()
    head.eval()
    return encoder, head, checkpoint, registered_blocks


def run_inference(encoder, head, frame, device, batch_size, workers):
    loader = DataLoader(
        ExternalDataset(frame),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    labels = []
    glaucoma_scores = []
    three_class_predictions = []
    indices = []
    with torch.inference_mode():
        for images, targets, row_indices in tqdm(loader, desc="External inference", unit="batch"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = head(encoder(images))
            probabilities = torch.softmax(logits.float(), dim=1)
            labels.extend(targets.numpy().tolist())
            glaucoma_scores.extend(probabilities[:, 2].cpu().numpy().tolist())
            three_class_predictions.extend(probabilities.argmax(dim=1).cpu().numpy().tolist())
            indices.extend(row_indices.numpy().tolist())
    order = np.argsort(np.asarray(indices))
    return (
        np.asarray(labels, dtype=np.int64)[order],
        np.asarray(glaucoma_scores, dtype=np.float64)[order],
        np.asarray(three_class_predictions, dtype=np.int64)[order],
    )


def calculate_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    metrics = {
        "n_images": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None,
        "average_precision": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }
    return y_pred, matrix, metrics


def save_evaluation_figures(y_true, y_score, y_pred, matrix, metrics, output_dir):
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.8, 5.8))
    display = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(display, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks([0, 1], ["Non-glaucoma", "Glaucoma"])
    axis.set_yticks([0, 1], ["Non-glaucoma", "Glaucoma"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("External glaucoma confusion matrix")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center", color="black")
    figure.tight_layout()
    figure.savefig(output_dir / "external_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    if metrics["auroc"] is not None:
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
        figure, axis = plt.subplots(figsize=(6.8, 5.8))
        axis.plot(false_positive_rate, true_positive_rate, linewidth=2, label=f"AUROC = {metrics['auroc']:.4f}")
        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.set_title("External glaucoma ROC curve")
        axis.legend(frameon=False)
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(output_dir / "external_roc_curve.png", dpi=300, bbox_inches="tight")
        plt.close(figure)
        precision_values, recall_values, _ = precision_recall_curve(y_true, y_score)
        figure, axis = plt.subplots(figsize=(6.8, 5.8))
        axis.plot(recall_values, precision_values, linewidth=2, label=f"AP = {metrics['average_precision']:.4f}")
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_title("External glaucoma precision-recall curve")
        axis.legend(frameon=False)
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(output_dir / "external_precision_recall_curve.png", dpi=300, bbox_inches="tight")
        plt.close(figure)
    metric_names = ["accuracy", "precision", "sensitivity", "specificity", "f1", "auroc"]
    metric_values = [metrics[name] for name in metric_names if metrics[name] is not None]
    displayed_names = [name.upper() if name == "auroc" else name.capitalize() for name in metric_names if metrics[name] is not None]
    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    bars = axis.bar(displayed_names, metric_values)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("External glaucoma evaluation")
    axis.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, metric_values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    figure.tight_layout()
    figure.savefig(output_dir / "external_metric_summary.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def generate_gradcam(encoder, head, frame, y_true, y_pred, y_score, device, output_dir):
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    positive_true_positive = np.flatnonzero((y_true == 1) & (y_pred == 1))
    if len(positive_true_positive) == 0:
        raise RuntimeError("Grad-CAM requires at least one correctly predicted glaucoma image.")
    selected_index = int(np.random.default_rng(SEED).choice(positive_true_positive))
    row = frame.iloc[selected_index]
    with Image.open(row["image_path"]) as source:
        original = normalized_image(source)
    display_image = np.asarray(original, dtype=np.float32) / 255.0
    input_tensor = TF.normalize(TF.to_tensor(original), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)
    activation_holder = {}

    def capture_activation(module, inputs, output):
        activation_holder["activation"] = output

    hook = encoder.norm.register_forward_hook(capture_activation)
    encoder.eval()
    head.eval()
    with torch.enable_grad():
        encoder.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        feature_vector = encoder(input_tensor)
        logits = head(feature_vector)
        predicted_class = int(logits.argmax(dim=1).item())
        target_logit = logits[0, predicted_class]
        activation = activation_holder["activation"]
        gradients = torch.autograd.grad(target_logit, activation, retain_graph=False, create_graph=False)[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        heatmap = F.relu((activation * weights).sum(dim=-1))
        heatmap = heatmap[0].detach()
        heatmap = heatmap - heatmap.min()
        maximum = heatmap.max()
        if maximum > 0:
            heatmap = heatmap / maximum
        heatmap = F.interpolate(heatmap[None, None], size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)[0, 0]
        probabilities = torch.softmax(logits.float(), dim=1)[0].detach().cpu().numpy()
    hook.remove()
    heatmap_array = heatmap.cpu().numpy()
    color_map = colormaps["jet"](heatmap_array)[..., :3]
    overlay = np.clip(0.58 * display_image + 0.42 * color_map, 0, 1)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(display_image)
    axes[0].set_title(f"External image\nTrue: {row['label_name']}")
    axes[1].imshow(heatmap_array, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title(f"Predicted: {THREE_CLASS_NAMES[predicted_class]}\nP(glaucoma)={probabilities[2]:.3f}")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    output_path = output_dir / "external_gradcam_example.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {
        "image_path": row["image_path"],
        "true_label": row["label_name"],
        "predicted_three_class": THREE_CLASS_NAMES[predicted_class],
        "glaucoma_probability": float(probabilities[2]),
        "gradcam_target": "predicted_class_logit",
        "figure": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.environ.get("FAIRLORA_ROOT", "./fairlora_retinal_project"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--allow-image-overlap", action="store_true")
    parser.add_argument("--skip-gradcam", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--phash-distance", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Batch size must be positive and workers cannot be negative.")
    set_seed(SEED)
    root = Path(args.root).resolve()
    raw_external_root = Path(args.data_root).resolve() if args.data_root else root / "data" / "raw" / "external_glaucoma"
    if not args.skip_download and args.data_root is None:
        download_external_dataset(raw_external_root)
    test_root = find_test_root(raw_external_root)
    output_root = root / "reports" / "external_validation"
    figure_root = output_root / "figures"
    output_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    frame = build_external_index(test_root, output_root / "external_invalid_images.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    overlap_report = audit_development_overlap(
        root, frame, output_root / "external_overlap_audit.json", args.phash_distance
    )
    if overlap_report["development_images_indexed"] == 0:
        raise RuntimeError("No development images were available for the required external-overlap audit.")
    image_overlap_detected = bool(
        overlap_report["exact_standardized_pixel_matches"]
        or overlap_report["perceptual_candidate_matches"]
    )
    if image_overlap_detected and not args.allow_image_overlap:
        raise RuntimeError(
            "Potential image overlap with the development cohorts was detected. Review "
            f"{output_root / 'external_overlap_audit.json'}; use --allow-image-overlap only "
            "when the overlap has been reviewed and the evaluation is not described as independent."
        )
    encoder, head, checkpoint, attention_blocks = build_model(root, device)
    y_true, y_score, three_class_prediction = run_inference(
        encoder, head, frame, device, args.batch_size, args.workers
    )
    y_pred, matrix, metrics = calculate_metrics(y_true, y_score, THRESHOLD)
    prediction_frame = frame.copy()
    prediction_frame["glaucoma_probability"] = y_score
    prediction_frame["three_class_prediction"] = three_class_prediction
    prediction_frame["three_class_prediction_name"] = [THREE_CLASS_NAMES[int(value)] for value in three_class_prediction]
    prediction_frame["binary_prediction"] = y_pred
    prediction_frame["binary_prediction_name"] = ["Glaucoma" if value == 1 else "Non-Glaucoma" for value in y_pred]
    prediction_frame.to_csv(output_root / "external_predictions.csv", index=False)
    metrics.update({
        "decision_rule": "glaucoma_softmax_probability_greater_than_or_equal_to_fixed_threshold",
        "threshold_tuned_on_external_data": False,
        "federated_best_round": int(checkpoint["round"]),
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "registered_attention_blocks": int(attention_blocks),
        "external_test_directory": str(test_root),
        "external_class_counts": {str(key): int(value) for key, value in frame["label_name"].value_counts().items()},
        "image_overlap_detected": image_overlap_detected,
        "patient_level_independence": "not_verifiable_without_external_patient_or_source_metadata",
        "source_level_independence": "not_verifiable_without_external_source_metadata",
    })
    (output_root / "external_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(output_root / "external_metrics.csv", index=False)
    save_evaluation_figures(y_true, y_score, y_pred, matrix, metrics, figure_root)
    gradcam_metadata = None
    if not args.skip_gradcam:
        gradcam_metadata = generate_gradcam(
            encoder, head, frame, y_true, y_pred, y_score, device, figure_root
        )
    summary = {
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
        "overlap_audit": overlap_report,
        "gradcam": gradcam_metadata,
        "predictions_file": str(output_root / "external_predictions.csv"),
        "metrics_file": str(output_root / "external_metrics.json"),
        "figures_directory": str(figure_root),
        "patient_level_independence": "not_verifiable_without_external_patient_or_source_metadata",
        "source_level_independence": "not_verifiable_without_external_source_metadata",
    }
    (output_root / "external_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
