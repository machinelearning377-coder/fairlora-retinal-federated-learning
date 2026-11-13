import argparse
import json
import os
import random
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from torchvision.transforms import ColorJitter, InterpolationMode
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

SEED = 42
IMAGE_SIZE = 384
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ODIR_LABELS = ["N", "D", "C", "A", "H", "M", "O"]
BRSET_LABELS = [
    "diabetic_retinopathy",
    "macular_edema",
    "scar",
    "nevus",
    "amd",
    "vascular_occlusion",
    "hypertensive_retinopathy",
    "drusens",
    "hemorrhage",
    "retinal_detachment",
    "myopic_fundus",
    "increased_cup_disc",
    "other",
]
DATASETS = {
    "odir": "andrewmvd/ocular-disease-recognition-odir5k",
    "fives": "nikitamanaenkov/fundus-image-dataset-for-vessel-segmentation",
    "brset": "tanzinabdul/fundus-patientwise-split",
    "smdg": "deathtrooper/multichannel-glaucoma-benchmark-dataset",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_file(root, filename):
    target = filename.lower()
    matches = [p for p in Path(root).rglob("*") if p.is_file() and p.name.lower() == target]
    if not matches:
        raise FileNotFoundError(f"Could not locate {filename} under {root}")
    return sorted(matches, key=lambda p: len(p.parts))[0]


def find_directory(root, dirname):
    target = dirname.lower()
    matches = [p for p in Path(root).rglob("*") if p.is_dir() and p.name.lower() == target]
    if not matches:
        raise FileNotFoundError(f"Could not locate directory {dirname} under {root}")
    return sorted(matches, key=lambda p: len(p.parts))[0]


def download_dataset(handle, destination):
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError("Install kagglehub before downloading datasets: pip install kagglehub") from error
    destination = Path(destination)
    marker = destination / ".kaggle_complete"
    if marker.exists():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = kagglehub.dataset_download(handle, output_dir=str(destination))
    marker.write_text(str(downloaded), encoding="utf-8")
    return destination


def save_rgb(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS).save(
            destination, format="JPEG", quality=95
        )


def save_mask(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        mask = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
        array = (np.asarray(mask) > 127).astype(np.uint8) * 255
        Image.fromarray(array).save(destination, format="PNG")


def resolve_smdg_image(value, raw_root, image_root, name_index, stem_index):
    if pd.isna(value):
        return None
    normalized = str(value).replace("\\", "/")
    candidate = Path(normalized)
    direct = Path(raw_root) / normalized
    if direct.is_file():
        return direct
    if candidate.name in name_index:
        return name_index[candidate.name]
    if candidate.stem in stem_index:
        return stem_index[candidate.stem]
    return None


def group_split(frame, group_values, seed=SEED, train_fraction=0.8):
    groups = pd.Series(group_values, index=frame.index).astype(str)
    unique_groups = groups.unique().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    train_count = int(round(len(unique_groups) * train_fraction))
    train_groups = set(unique_groups[:train_count])
    return np.where(groups.isin(train_groups), "train", "val")


def harmonize_sex(value, source="generic"):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"male", "m"}:
        return 0.0
    if text in {"female", "f"}:
        return 1.0
    if source == "brset":
        if text in {"1", "1.0"}:
            return 0.0
        if text in {"2", "2.0"}:
            return 1.0
    return np.nan


def harmonize_smdg_sex(row):
    sex = row.get("sex", np.nan)
    if pd.notna(sex):
        text = str(sex).strip().lower()
        if text in {"m", "male"}:
            return 0.0
        if text in {"f", "female"}:
            return 1.0
    gender = row.get("gender", np.nan)
    if pd.notna(gender):
        try:
            value = int(float(gender))
        except (TypeError, ValueError, OverflowError):
            return np.nan
        if value in {0, 1}:
            return float(value)
    return np.nan


def add_fairness_groups(frame, age_column, sex_column, sex_source):
    output = frame.copy()
    output["fair_age"] = pd.to_numeric(output[age_column], errors="coerce")
    output["fair_sex_group"] = output[sex_column].apply(lambda value: harmonize_sex(value, sex_source))
    threshold = output.loc[output["split"] == "train", "fair_age"].dropna().median()
    if pd.isna(threshold):
        raise ValueError(f"No training ages are available for {sex_source}.")
    output["fair_age_group"] = np.where(
        output["fair_age"].isna(), np.nan, (output["fair_age"] >= float(threshold)).astype(float)
    )
    return output, float(threshold)


def prepare_fives(raw_root, processed_root, split_root):
    raw_root = Path(raw_root)
    processed_root = Path(processed_root) / "fives"
    image_output = processed_root / "images"
    mask_output = processed_root / "masks"
    image_output.mkdir(parents=True, exist_ok=True)
    mask_output.mkdir(parents=True, exist_ok=True)
    records = []
    for original_split in ("train", "test"):
        split_directory = find_directory(raw_root, original_split)
        image_directory = find_directory(split_directory, "Original")
        mask_directory = find_directory(split_directory, "Ground truth")
        images = sorted(p for p in image_directory.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
        for source in tqdm(images, desc=f"FIVES {original_split}", unit="image"):
            mask_source = mask_directory / source.name
            if not mask_source.exists():
                continue
            sample_id = f"{original_split}__{source.stem}"
            image_path = image_output / f"{sample_id}.jpg"
            mask_path = mask_output / f"{sample_id}.png"
            save_rgb(source, image_path)
            save_mask(mask_source, mask_path)
            match = re.search(r"_([ADGN])$", source.stem.upper())
            disease_code = match.group(1) if match else "UNK"
            records.append({
                "sample_id": sample_id,
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "original_split": original_split,
                "disease_code": disease_code,
            })
    manifest = pd.DataFrame(records)
    if manifest.empty or manifest["sample_id"].duplicated().any():
        raise RuntimeError("FIVES image-mask manifest is empty or contains duplicate sample IDs.")
    if (manifest["original_split"] == "train").sum() != 600 or (manifest["original_split"] == "test").sum() != 200:
        raise RuntimeError("The downloaded FIVES source does not match the paper's 600 development and 200 official-test pairs.")
    rng = np.random.default_rng(SEED)
    development = manifest.index[manifest["original_split"] == "train"].to_numpy()
    rng.shuffle(development)
    manifest["split"] = "test"
    manifest.loc[development[:540], "split"] = "train"
    manifest.loc[development[540:], "split"] = "val"
    manifest.to_csv(processed_root / "manifest.csv", index=False)
    Path(split_root).mkdir(parents=True, exist_ok=True)
    manifest.to_csv(Path(split_root) / "fives_segmentation_split.csv", index=False)
    return manifest


def prepare_odir(raw_root, processed_root, split_root):
    raw_root = Path(raw_root)
    metadata_path = find_file(raw_root, "full_df.csv")
    image_root = find_directory(raw_root, "preprocessed_images")
    frame = pd.read_csv(metadata_path)
    required = {"filename", "ID", "Patient Age", "Patient Sex", *ODIR_LABELS, "G"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"ODIR metadata is missing columns: {sorted(missing)}")
    frame = frame[pd.to_numeric(frame["G"], errors="coerce").fillna(0).eq(0)].copy().reset_index(drop=True)
    output_root = Path(processed_root) / "odir"
    image_output = output_root / "images"
    image_output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, row in tqdm(frame.iterrows(), total=len(frame), desc="ODIR preprocessing", unit="image"):
        source = image_root / str(row["filename"])
        if not source.is_file():
            continue
        output_path = image_output / f"{index:06d}_{Path(str(row['filename'])).stem}.jpg"
        try:
            save_rgb(source, output_path)
        except Exception:
            continue
        record = {
            "image_path": str(output_path.resolve()),
            "filename": output_path.name,
            "patient_id": row["ID"],
            "age": row["Patient Age"],
            "sex": row["Patient Sex"],
        }
        for label in ODIR_LABELS:
            record[label] = pd.to_numeric(row[label], errors="coerce")
        records.append(record)
    result = pd.DataFrame(records)
    if result.empty:
        raise RuntimeError("No valid ODIR images were processed.")
    group_values = result["patient_id"].astype(str)
    missing_patient = result["patient_id"].isna() | group_values.str.lower().isin({"", "nan", "none"})
    group_values.loc[missing_patient] = "IMAGE::" + result.loc[missing_patient, "filename"].astype(str)
    result["fl_group_id"] = group_values.astype(str)
    result["split"] = group_split(result, result["fl_group_id"])
    result, age_threshold = add_fairness_groups(result, "age", "sex", "odir")
    result.to_csv(output_root / "manifest.csv", index=False)
    Path(split_root).mkdir(parents=True, exist_ok=True)
    result.to_csv(Path(split_root) / "odir.csv", index=False)
    return result, age_threshold


def write_hdf5_images(frame, output_file, source_column, description):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    valid_records = []
    with h5py.File(temporary, "w") as handle:
        dataset = handle.create_dataset(
            "images",
            shape=(0, IMAGE_SIZE, IMAGE_SIZE, 3),
            maxshape=(None, IMAGE_SIZE, IMAGE_SIZE, 3),
            dtype="uint8",
            chunks=(1, IMAGE_SIZE, IMAGE_SIZE, 3),
            compression="lzf",
        )
        for _, row in tqdm(frame.iterrows(), total=len(frame), desc=description, unit="image"):
            source = Path(str(row[source_column]))
            try:
                with Image.open(source) as image:
                    array = np.asarray(
                        image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS),
                        dtype=np.uint8,
                    )
            except Exception:
                continue
            new_index = dataset.shape[0]
            dataset.resize((new_index + 1, IMAGE_SIZE, IMAGE_SIZE, 3))
            dataset[new_index] = array
            record = row.to_dict()
            record["h5_index"] = new_index
            valid_records.append(record)
    temporary.replace(output_file)
    return pd.DataFrame(valid_records)


def prepare_brset(raw_root, processed_root, split_root):
    raw_root = Path(raw_root)
    split_root_raw = find_directory(raw_root, "dataset_split")
    frames = []
    image_index = {}
    for split_name in ("train", "val", "test"):
        split_directory = split_root_raw / split_name
        csv_path = split_directory / f"{split_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        current = pd.read_csv(csv_path)
        current["original_split"] = split_name
        frames.append(current)
        image_directory = split_directory / "images"
        for image_path in image_directory.iterdir():
            if image_path.is_file():
                image_index[image_path.stem] = image_path
    frame = pd.concat(frames, ignore_index=True)
    required = {"image_id", "patient_id", "patient_age", "patient_sex", *BRSET_LABELS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"BRSET metadata is missing columns: {sorted(missing)}")
    frame["source_path"] = frame["image_id"].astype(str).map(lambda value: str(image_index.get(value, "")))
    frame = frame[frame["source_path"] != ""].reset_index(drop=True)
    output_root = Path(processed_root) / "brset"
    h5_path = output_root / "brset_384.h5"
    result = write_hdf5_images(frame, h5_path, "source_path", "BRSET 384x384")
    if result.empty:
        raise RuntimeError("No valid BRSET images were processed.")
    group_values = result["patient_id"].astype(str)
    missing_patient = result["patient_id"].isna() | group_values.str.lower().isin({"", "nan", "none"})
    group_values.loc[missing_patient] = "IMAGE::" + result.loc[missing_patient, "image_id"].astype(str)
    result["fl_group_id"] = group_values.astype(str)
    result["split"] = group_split(result, result["fl_group_id"])
    result, age_threshold = add_fairness_groups(result, "patient_age", "patient_sex", "brset")
    result.to_csv(output_root / "manifest.csv", index=False)
    Path(split_root).mkdir(parents=True, exist_ok=True)
    result.to_csv(Path(split_root) / "brset.csv", index=False)
    return result, age_threshold


def prepare_smdg(raw_root, processed_root, split_root):
    raw_root = Path(raw_root)
    metadata_path = find_file(raw_root, "metadata - standardized.csv")
    image_root = find_directory(raw_root, "full-fundus")
    frame = pd.read_csv(metadata_path)
    required = {"names", "fundus", "patient_id", "age", "types", "sex"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"SMDG metadata is missing columns: {sorted(missing)}")
    names = frame["names"].astype(str)
    frame = frame[~names.str.startswith(("OIA-ODIR", "FIVES"), na=False)].copy().reset_index(drop=True)
    files = [p for p in image_root.rglob("*") if p.is_file()]
    name_index = {p.name: p for p in files}
    stem_index = {p.stem: p for p in files}
    frame["source_path"] = frame["fundus"].apply(
        lambda value: resolve_smdg_image(value, raw_root, image_root, name_index, stem_index)
    )
    frame = frame[frame["source_path"].notna()].copy().reset_index(drop=True)
    frame["source_path"] = frame["source_path"].astype(str)
    frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
    output_root = Path(processed_root) / "smdg"
    h5_path = output_root / "smdg_384.h5"
    result = write_hdf5_images(frame, h5_path, "source_path", "SMDG-19 384x384")
    if result.empty:
        raise RuntimeError("No valid SMDG-19 images were processed.")
    result["source_dataset"] = result["names"].astype(str).str.extract(r"^([^-]+)", expand=False).fillna("UNKNOWN")
    patient = result["patient_id"].astype(str)
    missing_patient = result["patient_id"].isna() | patient.str.lower().isin({"", "nan", "none"})
    result["fl_group_id"] = result["source_dataset"].astype(str) + "::" + patient
    result.loc[missing_patient, "fl_group_id"] = (
        result.loc[missing_patient, "source_dataset"].astype(str)
        + "::IMAGE::"
        + result.loc[missing_patient, "names"].astype(str)
    )
    result["split"] = group_split(result, result["fl_group_id"])
    if "gender" in result.columns:
        numeric_gender = pd.to_numeric(result["gender"], errors="coerce").notna()
        papila_rows = result["names"].astype(str).str.startswith("PAPILA")
        if bool((numeric_gender & ~papila_rows).any()):
            raise RuntimeError("Numeric SMDG gender values were found outside PAPILA records.")
    result["fair_sex_group"] = result.apply(harmonize_smdg_sex, axis=1)
    result["fair_age"] = pd.to_numeric(result["age"], errors="coerce")
    age_threshold = result.loc[result["split"] == "train", "fair_age"].dropna().median()
    if pd.isna(age_threshold):
        raise ValueError("No training ages are available for SMDG-19.")
    result["fair_age_group"] = np.where(
        result["fair_age"].isna(), np.nan, (result["fair_age"] >= float(age_threshold)).astype(float)
    )
    result.to_csv(output_root / "manifest.csv", index=False)
    Path(split_root).mkdir(parents=True, exist_ok=True)
    result.to_csv(Path(split_root) / "smdg.csv", index=False)
    return result, float(age_threshold)


def save_group_split_audit(results, split_root):
    rows = []
    for name, frame, group_column in results:
        train_groups = set(frame.loc[frame["split"] == "train", group_column].astype(str))
        val_groups = set(frame.loc[frame["split"] == "val", group_column].astype(str))
        rows.append({
            "client": name,
            "train_images": int((frame["split"] == "train").sum()),
            "validation_images": int((frame["split"] == "val").sum()),
            "group_overlap": len(train_groups.intersection(val_groups)),
        })
    pd.DataFrame(rows).to_csv(Path(split_root) / "group_split_audit.csv", index=False)


class FIVESDataset(Dataset):
    def __init__(self, frame, training=False):
        self.frame = frame.reset_index(drop=True)
        self.training = training
        self.jitter = ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08, hue=0.02)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
        with Image.open(row["mask_path"]) as mask:
            mask = mask.convert("L")
        if self.training:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            angle = random.uniform(-10.0, 10.0)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)
            if random.random() < 0.5:
                image = self.jitter(image)
        image_tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        mask_array = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return image_tensor, mask_tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, value):
        return self.block(value)


class SwinV2VesselSegmenter(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.features = backbone.features
        self.projections = nn.ModuleList([
            nn.Conv2d(96, 96, 1),
            nn.Conv2d(192, 96, 1),
            nn.Conv2d(384, 96, 1),
            nn.Conv2d(768, 96, 1),
        ])
        self.fusion = ConvBlock(384, 192)
        self.up_blocks = nn.ModuleList([
            ConvBlock(192, 160),
            ConvBlock(160, 128),
            ConvBlock(128, 96),
            ConvBlock(96, 64),
            ConvBlock(64, 48),
        ])
        self.detail = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            ConvBlock(80, 48),
            nn.Conv2d(48, 1, 1),
        )

    @staticmethod
    def to_nchw(value):
        return value.permute(0, 3, 1, 2).contiguous()

    def forward(self, image):
        value = self.features[0](image)
        value = self.features[1](value)
        feature1 = self.to_nchw(value)
        value = self.features[2](value)
        value = self.features[3](value)
        feature2 = self.to_nchw(value)
        value = self.features[4](value)
        value = self.features[5](value)
        feature3 = self.to_nchw(value)
        value = self.features[6](value)
        value = self.features[7](value)
        feature4 = self.to_nchw(value)
        target_size = feature4.shape[-2:]
        projected = [self.projections[0](feature1)]
        for projection, feature in zip(self.projections[1:], (feature2, feature3, feature4)):
            projected.append(F.interpolate(projection(feature), size=target_size, mode="bilinear", align_corners=False))
        value = self.fusion(torch.cat([
            F.interpolate(projected[0], size=target_size, mode="bilinear", align_corners=False),
            projected[1],
            projected[2],
            projected[3],
        ], dim=1))
        for block in self.up_blocks:
            value = F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False)
            value = block(value)
        detail = self.detail(image)
        if value.shape[-2:] != detail.shape[-2:]:
            value = F.interpolate(value, size=detail.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(torch.cat((value, detail), dim=1))


def segmentation_counts(logits, masks):
    predictions = logits.sigmoid() >= 0.5
    targets = masks >= 0.5
    return {
        "tp": int((predictions & targets).sum().item()),
        "tn": int(((~predictions) & (~targets)).sum().item()),
        "fp": int((predictions & (~targets)).sum().item()),
        "fn": int(((~predictions) & targets).sum().item()),
    }


def segmentation_metrics_from_counts(counts):
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    return {
        "dice": float((2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)),
        "iou": float((tp + 1e-7) / (tp + fp + fn + 1e-7)),
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "precision": float(tp / max(tp + fp, 1)),
    }


def segmentation_metrics(logits, masks):
    return segmentation_metrics_from_counts(segmentation_counts(logits, masks))


def run_segmentation(root, epochs=30, batch_size=8, workers=4):
    root = Path(root)
    split_file = root / "splits" / "fives_segmentation_split.csv"
    run_directory = root / "checkpoints" / "fives_segmentation"
    report_directory = root / "reports" / "segmentation"
    run_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(split_file)
    train_frame = frame[frame["split"] == "train"].reset_index(drop=True)
    validation_frame = frame[frame["split"] == "val"].reset_index(drop=True)
    test_frame = frame[frame["split"] == "test"].reset_index(drop=True)
    if len(train_frame) != 540 or len(validation_frame) != 60 or len(test_frame) != 200:
        raise RuntimeError("FIVES split must contain 540 training, 60 validation, and 200 official-test pairs.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    use_amp = device.type == "cuda"
    train_loader = DataLoader(
        FIVESDataset(train_frame, True), batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=use_amp, persistent_workers=workers > 0, drop_last=False
    )
    validation_loader = DataLoader(
        FIVESDataset(validation_frame, False), batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=use_amp, persistent_workers=workers > 0
    )
    test_loader = DataLoader(
        FIVESDataset(test_frame, False), batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=use_amp, persistent_workers=workers > 0
    )
    positive_pixels = 0
    total_pixels = 0
    for mask_path in tqdm(train_frame["mask_path"], desc="FIVES training-mask audit", unit="mask"):
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8) > 127
        positive_pixels += int(mask.sum())
        total_pixels += int(mask.size)
    negative_pixels = total_pixels - positive_pixels
    if positive_pixels == 0 or negative_pixels == 0:
        raise RuntimeError("FIVES training masks do not contain both vessel and background pixels.")
    positive_weight = torch.tensor([negative_pixels / positive_pixels], dtype=torch.float32, device=device)
    backbone = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
    model = SwinV2VesselSegmenter(backbone).to(device)
    encoder_parameters = list(model.features.parameters())
    decoder_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("features.")]
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": 1e-5},
        {"params": decoder_parameters, "lr": 3e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    best_dice = -1.0
    best_checkpoint = run_directory / "best_segmentation.pt"
    last_checkpoint = run_directory / "last_segmentation.pt"
    history = []
    for epoch in range(1, epochs + 1):
        freeze_encoder = epoch <= 2
        for parameter in model.features.parameters():
            parameter.requires_grad = not freeze_encoder
        model.train()
        train_loss_sum = 0.0
        train_metrics_sum = {key: 0.0 for key in ("dice", "iou", "accuracy", "sensitivity", "specificity", "precision")}
        train_batches = 0
        progress = tqdm(train_loader, desc=f"Segmentation epoch {epoch}/{epochs}", unit="batch")
        for images, masks in progress:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(images)
                bce = F.binary_cross_entropy_with_logits(logits.float(), masks.float(), pos_weight=positive_weight)
                probabilities = logits.float().sigmoid()
                intersection = (probabilities * masks.float()).sum(dim=(1, 2, 3))
                denominator = probabilities.sum(dim=(1, 2, 3)) + masks.float().sum(dim=(1, 2, 3))
                dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
                loss = 0.5 * bce + 0.5 * dice_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.item())
            batch_metrics = segmentation_metrics(logits.detach(), masks)
            for key in train_metrics_sum:
                train_metrics_sum[key] += batch_metrics[key]
            train_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}", dice=f"{batch_metrics['dice']:.4f}")
        model.eval()
        validation_loss_sum = 0.0
        validation_metrics_sum = {key: 0.0 for key in train_metrics_sum}
        validation_batches = 0
        with torch.no_grad():
            for images, masks in tqdm(validation_loader, desc="FIVES validation", unit="batch", leave=False):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    logits = model(images)
                    bce = F.binary_cross_entropy_with_logits(logits.float(), masks.float(), pos_weight=positive_weight)
                    probabilities = logits.float().sigmoid()
                    intersection = (probabilities * masks.float()).sum(dim=(1, 2, 3))
                    denominator = probabilities.sum(dim=(1, 2, 3)) + masks.float().sum(dim=(1, 2, 3))
                    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
                    loss = 0.5 * bce + 0.5 * dice_loss
                validation_loss_sum += float(loss.item())
                metrics = segmentation_metrics(logits, masks)
                for key in validation_metrics_sum:
                    validation_metrics_sum[key] += metrics[key]
                validation_batches += 1
        scheduler.step()
        validation_dice = validation_metrics_sum["dice"] / max(validation_batches, 1)
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_batches, 1),
            "validation_loss": validation_loss_sum / max(validation_batches, 1),
        }
        row.update({f"train_{key}": value / max(train_batches, 1) for key, value in train_metrics_sum.items()})
        row.update({f"validation_{key}": value / max(validation_batches, 1) for key, value in validation_metrics_sum.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(report_directory / "segmentation_history.csv", index=False)
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_validation_dice": best_dice,
            "history": history,
            "architecture": "torchvision_swin_v2_t",
            "image_size": IMAGE_SIZE,
        }
        torch.save(state, last_checkpoint)
        if validation_dice > best_dice:
            best_dice = validation_dice
            state["best_validation_dice"] = best_dice
            torch.save(state, best_checkpoint)
        print(json.dumps(row, indent=2))
    selected = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model_state_dict"], strict=True)
    encoder_file = run_directory / "retinal_swinv2_encoder.pt"
    torch.save({
        "encoder_state_dict": model.features.state_dict(),
        "architecture": "torchvision_swin_v2_t",
        "pretraining": "ImageNet1K_V1",
        "input_size": IMAGE_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "adaptation": "FIVES retinal-vessel segmentation",
    }, encoder_file)
    model.eval()
    test_counts = {key: 0 for key in ("tp", "tn", "fp", "fn")}
    per_image_metrics = []
    test_offset = 0
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="FIVES official test", unit="batch"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(images)
            batch_counts = segmentation_counts(logits, masks)
            for key in test_counts:
                test_counts[key] += batch_counts[key]
            for sample_index in range(images.shape[0]):
                image_counts = segmentation_counts(
                    logits[sample_index:sample_index + 1],
                    masks[sample_index:sample_index + 1],
                )
                per_image_metrics.append({
                    "sample_id": str(test_frame.iloc[test_offset]["sample_id"]),
                    **image_counts,
                    **segmentation_metrics_from_counts(image_counts),
                })
                test_offset += 1
    test_metrics = {
        **test_counts,
        **segmentation_metrics_from_counts(test_counts),
        "test_images": int(len(test_frame)),
        "aggregation": "global_pixel_counts_over_all_official_test_images",
    }
    pd.DataFrame(per_image_metrics).to_csv(report_directory / "fives_official_test_per_image.csv", index=False)
    (report_directory / "fives_official_test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print(json.dumps({"selected_epoch": selected["epoch"], "official_test": test_metrics}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.environ.get("FAIRLORA_ROOT", "./fairlora_retinal_project"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-segmentation", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    set_seed(SEED)
    root = Path(args.root).resolve()
    raw_root = root / "data" / "raw"
    processed_root = root / "data" / "processed"
    split_root = root / "splits"
    for path in (raw_root, processed_root, split_root):
        path.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        for key, handle in DATASETS.items():
            destination = raw_root / key
            print(f"Downloading {key}: {handle}")
            download_dataset(handle, destination)
    fives = prepare_fives(raw_root / "fives", processed_root, split_root)
    odir, odir_age = prepare_odir(raw_root / "odir", processed_root, split_root)
    brset, brset_age = prepare_brset(raw_root / "brset", processed_root, split_root)
    smdg, smdg_age = prepare_smdg(raw_root / "smdg", processed_root, split_root)
    save_group_split_audit([
        ("ODIR", odir, "fl_group_id"),
        ("BRSET", brset, "fl_group_id"),
        ("SMDG-19", smdg, "fl_group_id"),
    ], split_root)
    summary = {
        "fives": {"samples": len(fives), "train": int((fives["split"] == "train").sum()), "validation": int((fives["split"] == "val").sum()), "official_test": int((fives["split"] == "test").sum())},
        "odir": {"samples": len(odir), "training_age_median": odir_age},
        "brset": {"samples": len(brset), "training_age_median": brset_age},
        "smdg": {"samples": len(smdg), "training_age_median": smdg_age},
    }
    (root / "data_preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.skip_segmentation:
        run_segmentation(root, args.epochs, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
