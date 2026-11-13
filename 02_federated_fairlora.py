import argparse
import json
import math
import os
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, Dataset, Subset
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
SMDG_MAP = {0: 0, -1: 1, 1: 2}
CLIENTS = ("ODIR", "BRSET", "SMDG")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def demographic_value(value):
    if pd.isna(value):
        return -1
    return int(float(value))


def prepare_image(image, training, jitter):
    if training:
        if random.random() < 0.5:
            image = TF.hflip(image)
        if random.random() < 0.4:
            angle = random.uniform(-7.0, 7.0)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
        if random.random() < 0.4:
            image = jitter(image)
    return TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)


class ClientDataset(Dataset):
    def __init__(self, frame, client, root, training=False):
        self.frame = frame.reset_index(drop=True)
        self.client = client
        self.root = Path(root)
        self.training = training
        self.jitter = ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08, hue=0.02)
        self.h5 = None
        if client == "BRSET":
            self.h5_path = self.root / "data" / "processed" / "brset" / "brset_384.h5"
        elif client == "SMDG":
            self.h5_path = self.root / "data" / "processed" / "smdg" / "smdg_384.h5"
        else:
            self.h5_path = None

    def __len__(self):
        return len(self.frame)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["h5"] = None
        return state

    def get_h5(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r")
        return self.h5

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        if self.client == "ODIR":
            with Image.open(row["image_path"]) as source:
                image = source.convert("RGB")
            target = np.nan_to_num(row[ODIR_LABELS].to_numpy(dtype=np.float32), nan=0.0)
        elif self.client == "BRSET":
            array = self.get_h5()["images"][int(row["h5_index"])]
            image = Image.fromarray(array, mode="RGB")
            target = np.nan_to_num(row[BRSET_LABELS].to_numpy(dtype=np.float32), nan=0.0)
        else:
            array = self.get_h5()["images"][int(row["h5_index"])]
            image = Image.fromarray(array, mode="RGB")
            target = SMDG_MAP[int(row["types"])]
        image_tensor = prepare_image(image, self.training, self.jitter)
        if self.client == "SMDG":
            target_tensor = torch.tensor(target, dtype=torch.long)
        else:
            target_tensor = torch.tensor(target, dtype=torch.float32)
        sex = torch.tensor(demographic_value(row.get("fair_sex_group", np.nan)), dtype=torch.long)
        age = torch.tensor(demographic_value(row.get("fair_age_group", np.nan)), dtype=torch.long)
        return image_tensor, target_tensor, sex, age


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
    def __init__(self, outputs):
        super().__init__()
        self.model = nn.Sequential(nn.LayerNorm(768), nn.Dropout(0.20), nn.Linear(768, outputs))

    def forward(self, features):
        return self.model(features)


def build_encoder(encoder_path, rank, alpha, device):
    backbone = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
    checkpoint = torch.load(encoder_path, map_location="cpu", weights_only=False)
    backbone.features.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    for parameter in backbone.features.parameters():
        parameter.requires_grad = False
    for parameter in backbone.norm.parameters():
        parameter.requires_grad = False
    attention_blocks = 0
    for module in backbone.features.modules():
        attention = getattr(module, "attn", None)
        if attention is None or not hasattr(attention, "qkv") or not hasattr(attention, "proj"):
            continue
        for projection in (attention.qkv, attention.proj):
            if isinstance(projection, nn.Linear):
                parametrize.register_parametrization(
                    projection,
                    "weight",
                    LoRAWeight(projection.out_features, projection.in_features, rank, alpha),
                )
                projection.parametrizations.weight.original.requires_grad = False
        attention_blocks += 1
    encoder = SharedEncoder(backbone.features, backbone.norm).to(device)
    trainable = [parameter for name, parameter in encoder.named_parameters() if "lora_A" in name or "lora_B" in name]
    if not trainable:
        raise RuntimeError("No LoRA parameters were registered in the SwinV2-T attention layers.")
    unexpected = [name for name, parameter in encoder.named_parameters() if parameter.requires_grad and "lora_A" not in name and "lora_B" not in name]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable encoder parameters: {unexpected[:10]}")
    lora_parameter_count = sum(parameter.numel() for parameter in trainable)
    frozen_base_parameter_count = sum(
        parameter.numel()
        for name, parameter in encoder.named_parameters()
        if "lora_A" not in name and "lora_B" not in name
    )
    print(json.dumps({
        "attention_blocks": attention_blocks,
        "trainable_lora_parameters": lora_parameter_count,
        "frozen_base_encoder_parameters": frozen_base_parameter_count,
        "trainable_fraction_percent": 100.0 * lora_parameter_count / frozen_base_parameter_count,
        "parameter_reduction_percent": 100.0 * (1.0 - lora_parameter_count / frozen_base_parameter_count),
    }, indent=2))
    return encoder


def lora_parameters(encoder):
    return [parameter for name, parameter in encoder.named_parameters() if "lora_A" in name or "lora_B" in name]


def get_lora_state(encoder):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in encoder.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }


def load_lora_state(encoder, state):
    current = dict(encoder.named_parameters())
    expected = {name for name in current if "lora_A" in name or "lora_B" in name}
    if set(state) != expected:
        missing = sorted(expected.difference(state))
        extra = sorted(set(state).difference(expected))
        raise RuntimeError(f"LoRA checkpoint keys do not match. Missing={missing[:5]}, extra={extra[:5]}")
    with torch.no_grad():
        for name, value in state.items():
            current[name].copy_(value.to(current[name].device))


def build_task_weights(client_frames, device):
    odir_train = client_frames["ODIR"].query("split == 'train'")
    brset_train = client_frames["BRSET"].query("split == 'train'")
    smdg_train = client_frames["SMDG"].query("split == 'train'")

    def multilabel_weights(frame, labels):
        values = np.nan_to_num(frame[labels].to_numpy(dtype=np.float32), nan=0.0)
        positives = values.sum(axis=0)
        negatives = len(values) - positives
        weights = np.clip(negatives / np.maximum(positives, 1.0), 1.0, 20.0)
        return torch.tensor(weights, dtype=torch.float32, device=device)

    smdg_targets = smdg_train["types"].astype(int).map(SMDG_MAP)
    class_counts = smdg_targets.value_counts().reindex([0, 1, 2], fill_value=0).to_numpy(dtype=np.float64)
    class_weights = np.sqrt(len(smdg_targets) / (3.0 * np.maximum(class_counts, 1.0)))
    class_weights = class_weights / class_weights.mean()
    return {
        "ODIR": multilabel_weights(odir_train, ODIR_LABELS),
        "BRSET": multilabel_weights(brset_train, BRSET_LABELS),
        "SMDG": torch.tensor(class_weights, dtype=torch.float32, device=device),
    }


def task_loss(client, logits, target, weights):
    if client in {"ODIR", "BRSET"}:
        return F.binary_cross_entropy_with_logits(logits, target.float(), pos_weight=weights[client])
    return F.cross_entropy(logits, target.long(), weight=weights[client])


def risk_and_outcome(client, logits, target):
    if client == "ODIR":
        probabilities = torch.sigmoid(logits[:, 1:])
        risk = 1.0 - torch.prod(1.0 - probabilities, dim=1)
        outcome = (target[:, 1:].sum(dim=1) > 0).float()
        valid = torch.ones_like(outcome, dtype=torch.bool)
    elif client == "BRSET":
        probabilities = torch.sigmoid(logits)
        risk = 1.0 - torch.prod(1.0 - probabilities, dim=1)
        outcome = (target.sum(dim=1) > 0).float()
        valid = torch.ones_like(outcome, dtype=torch.bool)
    else:
        probabilities = torch.softmax(logits, dim=1)
        risk = probabilities[:, 2]
        outcome = (target == 2).float()
        valid = target != 1
    return risk, outcome, valid


def differentiable_gap(values, groups, mask):
    group_zero = mask & (groups == 0)
    group_one = mask & (groups == 1)
    if group_zero.sum().item() < 2 or group_one.sum().item() < 2:
        return values.new_tensor(0.0)
    return torch.abs(values[group_zero].mean() - values[group_one].mean())


def one_attribute_fairness(score, outcome, valid, groups):
    usable = valid & (groups >= 0)
    dp = differentiable_gap(score, groups, usable)
    positive = usable & (outcome > 0.5)
    negative = usable & (outcome <= 0.5)
    tpr_gap = differentiable_gap(score, groups, positive)
    fpr_gap = differentiable_gap(score, groups, negative)
    eo = 0.5 * (tpr_gap + fpr_gap)
    return dp + eo


def fairness_loss(client, logits, target, sex, age):
    score, outcome, valid = risk_and_outcome(client, logits.float(), target)
    sex_penalty = one_attribute_fairness(score, outcome, valid, sex)
    age_penalty = one_attribute_fairness(score, outcome, valid, age)
    return 0.5 * (sex_penalty + age_penalty)


def numpy_group_gap(score, group, mask):
    group_zero = mask & (group == 0)
    group_one = mask & (group == 1)
    if group_zero.sum() < 2 or group_one.sum() < 2:
        return float("nan")
    return float(abs(score[group_zero].mean() - score[group_one].mean()))


def fairness_metrics_numpy(score, outcome, valid, group):
    usable = valid & (group >= 0)
    dp = numpy_group_gap(score, group, usable)
    positive = usable & (outcome > 0.5)
    negative = usable & (outcome <= 0.5)
    tpr_gap = numpy_group_gap(score, group, positive)
    fpr_gap = numpy_group_gap(score, group, negative)
    eo = float("nan") if np.isnan(tpr_gap) or np.isnan(fpr_gap) else 0.5 * (tpr_gap + fpr_gap)
    return {"dp": dp, "eo": eo}


@torch.no_grad()
def validate_client(client, encoder, head, loader, device, amp_dtype, use_amp):
    encoder.eval()
    head.eval()
    targets = []
    predictions = []
    risks = []
    outcomes = []
    valid_values = []
    sexes = []
    ages = []
    for images, target, sex, age in tqdm(loader, desc=f"{client} validation", leave=False, unit="batch"):
        images = images.to(device, non_blocking=True)
        target_device = target.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = head(encoder(images))
        risk, outcome, valid = risk_and_outcome(client, logits.float(), target_device)
        risks.append(risk.cpu().numpy())
        outcomes.append(outcome.cpu().numpy())
        valid_values.append(valid.cpu().numpy())
        sexes.append(sex.numpy())
        ages.append(age.numpy())
        if client in {"ODIR", "BRSET"}:
            prediction = (torch.sigmoid(logits.float()) >= 0.5).int().cpu().numpy()
            truth = target.int().numpy()
        else:
            prediction = logits.argmax(dim=1).cpu().numpy()
            truth = target.numpy()
        targets.append(truth)
        predictions.append(prediction)
    y_true = np.concatenate(targets, axis=0)
    y_pred = np.concatenate(predictions, axis=0)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    label_accuracy = float((y_true == y_pred).mean()) if client in {"ODIR", "BRSET"} else float(accuracy_score(y_true, y_pred))
    risk = np.concatenate(risks)
    outcome = np.concatenate(outcomes)
    valid = np.concatenate(valid_values).astype(bool)
    sex = np.concatenate(sexes)
    age = np.concatenate(ages)
    sex_metrics = fairness_metrics_numpy(risk, outcome, valid, sex)
    age_metrics = fairness_metrics_numpy(risk, outcome, valid, age)
    if client in {"ODIR", "BRSET"}:
        matrix = confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1), labels=[0, 1])
        matrix_type = "aggregated_multilabel_binary_decisions"
    else:
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        matrix_type = "image_level_three_class"
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "accuracy": label_accuracy,
        "sex_dp": sex_metrics["dp"],
        "sex_eo": sex_metrics["eo"],
        "age_dp": age_metrics["dp"],
        "age_eo": age_metrics["eo"],
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_type": matrix_type,
    }


def train_local_client(client, encoder, client_configuration, global_state, head_state, task_weights, device, amp_dtype, use_amp, scaler, round_number, local_epochs, fairness_batch_size, fair_every, lambda_fairness, lora_lr, head_lr, weight_decay, workers):
    load_lora_state(encoder, global_state)
    configuration = client_configuration[client]
    train_loader = configuration["train_loader"]
    fairness_loader = configuration["fairness_loader"]
    head = ClientHead(configuration["outputs"]).to(device)
    head.load_state_dict(head_state, strict=True)
    encoder.train()
    head.train()
    optimizer = torch.optim.AdamW([
        {"params": lora_parameters(encoder), "lr": lora_lr},
        {"params": head.parameters(), "lr": head_lr},
    ], weight_decay=weight_decay)
    fairness_iterator = iter(fairness_loader) if fairness_loader is not None else None
    task_total = 0.0
    fairness_total = 0.0
    task_steps = 0
    fairness_steps = 0
    for local_epoch in range(1, local_epochs + 1):
        progress = tqdm(train_loader, desc=f"{client} round {round_number} epoch {local_epoch}/{local_epochs}", unit="batch")
        for step, (images, target, sex, age) in enumerate(progress):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = head(encoder(images))
                prediction_loss = task_loss(client, logits.float(), target, task_weights)
                fairness_penalty = prediction_loss.new_tensor(0.0)
                if fairness_iterator is not None and step % fair_every == 0:
                    try:
                        fair_images, fair_target, fair_sex, fair_age = next(fairness_iterator)
                    except StopIteration:
                        fairness_iterator = iter(fairness_loader)
                        fair_images, fair_target, fair_sex, fair_age = next(fairness_iterator)
                    fair_images = fair_images.to(device, non_blocking=True)
                    fair_target = fair_target.to(device, non_blocking=True)
                    fair_sex = fair_sex.to(device, non_blocking=True)
                    fair_age = fair_age.to(device, non_blocking=True)
                    fair_logits = head(encoder(fair_images))
                    fairness_penalty = fairness_loss(client, fair_logits, fair_target, fair_sex, fair_age)
                    fairness_total += float(fairness_penalty.detach().item())
                    fairness_steps += 1
                loss = prediction_loss + lambda_fairness * fairness_penalty
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_parameters(encoder) + list(head.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
            task_total += float(prediction_loss.detach().item())
            task_steps += 1
            progress.set_postfix(task=f"{prediction_loss.item():.4f}", fair=f"{fairness_penalty.item():.4f}")
    result = {
        "head_state": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()},
        "lora_state": get_lora_state(encoder),
        "task_loss": task_total / max(task_steps, 1),
        "fair_loss": fairness_total / max(fairness_steps, 1),
    }
    return result


def weighted_fedavg(states, sample_counts):
    total = float(sum(sample_counts))
    if total <= 0:
        raise ValueError("Federated clients have no training samples.")
    averaged = {}
    for key in states[0]:
        value = torch.zeros_like(states[0][key])
        for state, count in zip(states, sample_counts):
            value.add_(state[key], alpha=float(count) / total)
        averaged[key] = value
    return averaged


def create_client_configuration(root, batch_size, fairness_batch_size, workers):
    split_root = Path(root) / "splits"
    frames = {
        "ODIR": pd.read_csv(split_root / "odir.csv"),
        "BRSET": pd.read_csv(split_root / "brset.csv"),
        "SMDG": pd.read_csv(split_root / "smdg.csv"),
    }
    outputs = {"ODIR": len(ODIR_LABELS), "BRSET": len(BRSET_LABELS), "SMDG": 3}
    configurations = {}
    for client in CLIENTS:
        train_frame = frames[client][frames[client]["split"] == "train"].reset_index(drop=True)
        validation_frame = frames[client][frames[client]["split"] == "val"].reset_index(drop=True)
        if train_frame.empty or validation_frame.empty:
            raise RuntimeError(f"{client} has an empty training or validation partition.")
        train_dataset = ClientDataset(train_frame, client, root, True)
        validation_dataset = ClientDataset(validation_frame, client, root, False)
        usable_fairness = train_frame["fair_sex_group"].notna() | train_frame["fair_age_group"].notna()
        fairness_indices = np.flatnonzero(usable_fairness.to_numpy()).tolist()
        fairness_loader = None
        if fairness_indices:
            fairness_loader = DataLoader(
                Subset(train_dataset, fairness_indices), batch_size=fairness_batch_size, shuffle=True,
                num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                drop_last=False,
            )
        configurations[client] = {
            "train_loader": DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0, drop_last=False,
            ),
            "validation_loader": DataLoader(
                validation_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
            ),
            "fairness_loader": fairness_loader,
            "outputs": outputs[client],
            "train_size": len(train_frame),
            "head_state": None,
        }
    return frames, configurations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.environ.get("FAIRLORA_ROOT", "./fairlora_retinal_project"))
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fairness-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    set_seed(SEED)
    root = Path(args.root).resolve()
    encoder_path = root / "checkpoints" / "fives_segmentation" / "retinal_swinv2_encoder.pt"
    if not encoder_path.exists():
        raise FileNotFoundError(f"Run 01_data_download_and_segmentation.py first. Missing: {encoder_path}")
    run_directory = root / "checkpoints" / "federated"
    checkpoint_directory = run_directory / "rounds"
    report_directory = root / "reports" / "federated"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        for path in (
            run_directory / "last_federated.pt",
            run_directory / "best_federated.pt",
            report_directory / "federated_history.csv",
            report_directory / "best_round_validation_summary.json",
        ):
            path.unlink(missing_ok=True)
        for path in checkpoint_directory.glob("round_*.pt"):
            path.unlink()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    use_amp = device.type == "cuda"
    encoder = build_encoder(encoder_path, 8, 16.0, device)
    frames, client_configuration = create_client_configuration(root, args.batch_size, args.fairness_batch_size, args.workers)
    task_weights = build_task_weights(frames, device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    initial_lora = get_lora_state(encoder)
    client_heads = {}
    for offset, client in enumerate(CLIENTS):
        torch.manual_seed(SEED + offset)
        head = ClientHead(client_configuration[client]["outputs"])
        client_heads[client] = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    history = []
    best_mean_macro_f1 = -1.0
    best_round = 0
    start_round = 1
    last_checkpoint = run_directory / "last_federated.pt"
    best_checkpoint = run_directory / "best_federated.pt"
    if last_checkpoint.exists() and not args.fresh:
        saved = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        if saved.get("configuration", {}).get("rounds") != args.rounds or saved.get("configuration", {}).get("local_epochs") != args.local_epochs:
            raise RuntimeError("Existing checkpoint configuration differs from this run. Use --fresh for a new run.")
        initial_lora = saved["global_lora"]
        client_heads = saved["client_heads"]
        best_mean_macro_f1 = float(saved["best_mean_macro_f1"])
        best_round = int(saved["best_round"])
        start_round = int(saved["round"]) + 1
        history_path = report_directory / "federated_history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict("records")
    configuration = {
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "fairness_coefficient": 0.05,
        "fairness_batch_interval": 4,
        "task_batch_size": args.batch_size,
        "fairness_batch_size": args.fairness_batch_size,
        "optimizer": "AdamW",
        "lora_learning_rate": 2e-4,
        "head_learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "pretrained_encoder": str(encoder_path),
        "client_label_counts": {"ODIR": 7, "BRSET": 13, "SMDG": 3},
    }
    (run_directory / "federated_config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    for round_number in range(start_round, args.rounds + 1):
        print(f"Federated round {round_number}/{args.rounds}")
        local_states = []
        local_counts = []
        round_row = {"round": round_number}
        for client in CLIENTS:
            local = train_local_client(
                client=client,
                encoder=encoder,
                client_configuration=client_configuration,
                global_state=initial_lora,
                head_state=client_heads[client],
                task_weights=task_weights,
                device=device,
                amp_dtype=amp_dtype,
                use_amp=use_amp,
                scaler=scaler,
                round_number=round_number,
                local_epochs=args.local_epochs,
                fairness_batch_size=args.fairness_batch_size,
                fair_every=4,
                lambda_fairness=0.05,
                lora_lr=2e-4,
                head_lr=5e-4,
                weight_decay=1e-4,
                workers=args.workers,
            )
            client_heads[client] = local["head_state"]
            local_states.append(local["lora_state"])
            local_counts.append(client_configuration[client]["train_size"])
            round_row[f"{client}_task_loss"] = local["task_loss"]
            round_row[f"{client}_fair_loss"] = local["fair_loss"]
        initial_lora = weighted_fedavg(local_states, local_counts)
        load_lora_state(encoder, initial_lora)
        macro_values = []
        round_validation_details = {}
        for client in CLIENTS:
            head = ClientHead(client_configuration[client]["outputs"]).to(device)
            head.load_state_dict(client_heads[client], strict=True)
            metrics = validate_client(
                client, encoder, head, client_configuration[client]["validation_loader"], device, amp_dtype, use_amp
            )
            for key, value in metrics.items():
                if key not in {"confusion_matrix", "confusion_matrix_type"}:
                    round_row[f"{client}_{key}"] = value
            round_validation_details[client] = {
                key: value for key, value in metrics.items()
            }
            macro_values.append(metrics["macro_f1"])
            print(f"{client}: macro-F1={metrics['macro_f1']:.4f}, micro-F1={metrics['micro_f1']:.4f}, accuracy={metrics['accuracy']:.4f}")
            print(f"{client}: sex DP={metrics['sex_dp']:.4f}, sex EO={metrics['sex_eo']:.4f}, age DP={metrics['age_dp']:.4f}, age EO={metrics['age_eo']:.4f}")
        mean_macro_f1 = float(np.mean(macro_values))
        round_row["mean_client_macro_f1"] = mean_macro_f1
        history.append(round_row)
        pd.DataFrame(history).to_csv(report_directory / "federated_history.csv", index=False)
        checkpoint_state = {
            "round": round_number,
            "global_lora": initial_lora,
            "client_heads": client_heads,
            "best_mean_macro_f1": max(best_mean_macro_f1, mean_macro_f1),
            "best_round": best_round if mean_macro_f1 <= best_mean_macro_f1 else round_number,
            "configuration": configuration,
        }
        torch.save(checkpoint_state, checkpoint_directory / f"round_{round_number:02d}.pt")
        torch.save(checkpoint_state, last_checkpoint)
        if mean_macro_f1 > best_mean_macro_f1:
            best_mean_macro_f1 = mean_macro_f1
            best_round = round_number
            checkpoint_state["best_mean_macro_f1"] = best_mean_macro_f1
            checkpoint_state["best_round"] = best_round
            torch.save(checkpoint_state, best_checkpoint)
            best_summary = {
                "selected_round": int(best_round),
                "mean_client_macro_f1": float(best_mean_macro_f1),
                "selection_rule": "maximum_mean_client_validation_macro_f1",
                "clients": round_validation_details,
            }
            (report_directory / "best_round_validation_summary.json").write_text(
                json.dumps(best_summary, indent=2), encoding="utf-8"
            )
        print(f"Mean client Macro-F1: {mean_macro_f1:.4f}; best round: {best_round}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not best_checkpoint.exists():
        raise RuntimeError("No best federated checkpoint was created.")
    print(json.dumps({"best_round": best_round, "best_mean_client_macro_f1": best_mean_macro_f1, "checkpoint": str(best_checkpoint)}, indent=2))


if __name__ == "__main__":
    main()
