# -*- coding: utf-8 -*-
"""
Easy LoRA dataset preparation backend.

This module is intentionally independent from the training implementation.
It prepares a Kohya/TrainTrain-compatible image + .txt dataset and can
optionally start the existing TrainTrain LoRA trainer.

Design goals:
- beginner-first UI
- deterministic, local-only tag filtering
- no LLM dependency
- preserve originals
- expose only uncertain items for manual review
- use the current WD14 v3 ONNX tagger
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
WD_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
WD_DEFAULT_GENERAL_THRESHOLD = 0.35
WD_DEFAULT_CHARACTER_THRESHOLD = 0.85
WD_IMAGE_SIZE = 448


# ---------------------------------------------------------------------------
# Tag classification
# ---------------------------------------------------------------------------

# WD tagger categories used by SmilingWolf's current tagger:
# 0 general, 1 artist, 3 copyright, 4 character, 5 meta, 9 rating.
# The official tagger UI explicitly separates rating/general/character tags.
WD_CATEGORY_NAMES = {
    0: "general",
    1: "artist",
    2: "unknown",
    3: "copyright",
    4: "character",
    5: "meta",
    9: "rating",
}

CLOTHING_RE = re.compile(
    r"(?<![A-Za-z0-9])(dress|skirt|shirt|blouse|jacket|coat|uniform|suit|hoodie|sweater|cardigan|"
    r"necktie|bowtie|tie|ribbon|scarf|glove|shoe|sock|boot|sandal|hat|cap|beret|"
    r"swimwear|bikini|bra|stockings|pantyhose|shorts|pants|jeans|apron|armor|"
    r"kimono|yukata|cloak|cape|belt|corset|lingerie|bodysuit|leotard|sleeve|"
    r"button|collar|pajama|robe|cuff|vest|tank_top|crop_top|camisole|"
    r"school_uniform|sailor|maid|military_uniform|workout_clothes)(?![A-Za-z0-9])", re.I
)

HAIR_RE = re.compile(
    r"(?<![A-Za-z0-9])(hair|bangs|ponytail|twintails|twin_tails|braid|braided|ahoge|pigtail|"
    r"bun|undercut|sideburn|hairband|hairpin|hair_ribbon|hair_ornament|"
    r"drill_hair|hime_cut|bob_cut|pixie_cut|very_long_hair|short_hair|long_hair)(?![A-Za-z0-9])", re.I
)

APPEARANCE_RE = re.compile(
    r"(?<![A-Za-z0-9])(hair|eyes?|eyebrow|eyelashes|face|skin|freckles|blush|makeup|"
    r"lipstick|fangs|horns|animal_ears|ears|tail|wings|scar|mole|"
    r"pointy_ears|facial_mark|beard|mustache|dark_skin|pale_skin|"
    r"heterochromia|two-tone_skin|colored_skin)(?![A-Za-z0-9])", re.I
)

POSE_RE = re.compile(
    r"(?<![A-Za-z0-9])(standing|sitting|kneeling|lying|walking|running|jumping|crouching|"
    r"leaning|bending|arms_up|arms_crossed|hand_on_hip|looking_at_viewer|"
    r"looking_away|looking_back|from_above|from_below|upper_body|full_body|"
    r"cowboy_shot|portrait|close-up|closeup|profile|side_view|"
    r"dynamic_pose|fighting_stance|squatting|crossed_legs|on_back|on_stomach)(?![A-Za-z0-9])", re.I
)

EXPRESSION_RE = re.compile(
    r"(?<![A-Za-z0-9])(smile|smiling|grin|grinning|angry|sad|crying|tears|blush|surprised|"
    r"open_mouth|closed_eyes|wink|expressionless|confused|embarrassed|"
    r"nervous|serious|laughing|sleeping|pout|frown)(?![A-Za-z0-9])", re.I
)

BACKGROUND_RE = re.compile(
    r"(?<![A-Za-z0-9])(indoors|outdoors|sky|cloud|city|street|classroom|school|room|bedroom|"
    r"forest|park|beach|mountain|ocean|river|snow|rain|sunset|night|day|"
    r"building|architecture|window|wall|floor|road|tree|flower|field|"
    r"nature|landscape|garden|underwater|space|desert|bridge|railway|"
    r"store|cafe|restaurant|office|laboratory|hospital|shrine|temple)(?![A-Za-z0-9])", re.I
)

STYLE_RE = re.compile(
    r"(?<![A-Za-z0-9])(anime|manga|comic|cartoon|chibi|realistic|photorealistic|illustration|"
    r"watercolor|oil_painting|sketch|lineart|flat_color|cel_shading|"
    r"monochrome|greyscale|pixel_art|3d|render|digital_painting|"
    r"traditional_media|surreal|abstract|painted|pastel|ink|charcoal|"
    r"rough_sketch|concept_art|flat_chest|style)(?![A-Za-z0-9])", re.I
)

OBJECT_RE = re.compile(
    r"(?<![A-Za-z0-9])(sword|knife|gun|rifle|pistol|bow|arrow|weapon|bag|backpack|purse|"
    r"handbag|phone|smartphone|book|food|drink|cup|bottle|car|vehicle|"
    r"bicycle|motorcycle|chair|table|computer|laptop|camera|microphone|"
    r"instrument|flower|ball|umbrella|shield|staff|wand|axe|hammer|"
    r"scabbard|lantern|door|window|furniture|vehicle|robot|mecha)(?![A-Za-z0-9])", re.I
)

QUALITY_RE = re.compile(
    r"(?<![A-Za-z0-9])(masterpiece|best_quality|best quality|highres|high_resolution|"
    r"absurdres|very_high_resolution|ultra_detailed|detailed|lowres|"
    r"blurry|watermark|signature|text|jpeg_artifacts|compression|"
    r"scan|official_art|translated|language|artist_name)(?![A-Za-z0-9])", re.I
)


@dataclass
class TagItem:
    name: str
    category: int
    score: float

    @property
    def category_name(self) -> str:
        return WD_CATEGORY_NAMES.get(self.category, "unknown")


def classify_tag(tag: str, category: int) -> str:
    if category == 4:
        return "character"
    if category == 3:
        return "copyright"
    if category == 1:
        return "artist"
    if category == 5:
        return "meta"
    if category == 9:
        return "rating"

    t = tag.lower()

    # More specific categories first.
    if QUALITY_RE.search(t):
        return "quality"
    if CLOTHING_RE.search(t):
        return "clothing"
    if HAIR_RE.search(t):
        return "hair"
    if APPEARANCE_RE.search(t):
        return "appearance"
    if STYLE_RE.search(t):
        return "style"
    if OBJECT_RE.search(t):
        return "object"
    if BACKGROUND_RE.search(t):
        return "background"
    if EXPRESSION_RE.search(t):
        return "expression"
    if POSE_RE.search(t):
        return "pose"
    return "other"


PRESETS = {
    "キャラクター": {
        "remove": {
            "character", "copyright", "appearance", "hair", "clothing", "quality"
        },
        "label": "キャラクター本体を学習。外見・髪・衣装をcaptionから外します。",
        "rank": 16,
        "steps_per_image": 50,
    },
    "衣装": {
        "remove": {"clothing", "quality"},
        "label": "衣装を学習。キャラクターや背景はcaptionに残します。",
        "rank": 16,
        "steps_per_image": 50,
    },
    "画風": {
        "remove": {"style", "artist", "quality"},
        "label": "画風を学習。画像内容を説明するタグはcaptionに残します。",
        "rank": 16,
        "steps_per_image": 40,
    },
    "オブジェクト": {
        "remove": {"object", "quality"},
        "label": "対象オブジェクトを学習。周囲の人物・背景はcaptionに残します。",
        "rank": 16,
        "steps_per_image": 50,
    },
}


# ---------------------------------------------------------------------------
# WD14 ONNX tagger
# ---------------------------------------------------------------------------

class WD14Tagger:
    def __init__(
        self,
        repo_id: str = WD_REPO,
        general_threshold: float = WD_DEFAULT_GENERAL_THRESHOLD,
        character_threshold: float = WD_DEFAULT_CHARACTER_THRESHOLD,
    ):
        self.repo_id = repo_id
        self.general_threshold = float(general_threshold)
        self.character_threshold = float(character_threshold)
        self.session = None
        self.input_name = None
        self.channels_last = True
        self.tags: list[str] = []
        self.categories: list[int] = []
        self.model_target_size = WD_IMAGE_SIZE
        self.provider = "not loaded"

    def _model_dir(self) -> Path:
        safe = self.repo_id.replace("/", "__")
        # Keep the model under the TrainTrain extension rather than ~/.cache
        # so a copied Colab repository can be made self-contained.
        try:
            from trainer import trainer as trainer_module
            root = Path(trainer_module.path_root)
        except Exception:
            root = Path(__file__).resolve().parents[1]
        p = root / "models" / "wd14" / safe
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _ensure(self):
        if self.session is not None:
            return

        try:
            import numpy as np
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RuntimeError(
                "WD14の依存関係がありません。"
                "Install後にWebUIを再起動してください。"
            ) from e

        model_dir = self._model_dir()
        model_path = hf_hub_download(
            self.repo_id,
            "model.onnx",
            local_dir=str(model_dir),
        )
        labels_path = hf_hub_download(
            self.repo_id,
            "selected_tags.csv",
            local_dir=str(model_dir),
        )

        with open(labels_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        self.tags = [row["name"] for row in rows]
        self.categories = [int(row["category"]) for row in rows]

        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        if not providers:
            providers = available

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.provider = "/".join(self.session.get_providers())

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        if len(shape) == 4:
            # Current WD14 v3 is NHWC [N,448,448,3].
            if shape[-1] == 3:
                self.channels_last = True
            elif shape[1] == 3:
                self.channels_last = False
            else:
                self.channels_last = True
            if isinstance(shape[1], int):
                self.model_target_size = shape[1]
            elif isinstance(shape[2], int):
                self.model_target_size = shape[2]

        _ = np  # imported for clear dependency verification

    def _preprocess(self, path: str):
        import numpy as np

        image = Image.open(path).convert("RGB")
        side = max(image.width, image.height)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        image = canvas.resize(
            (self.model_target_size, self.model_target_size),
            Image.Resampling.BICUBIC,
        )
        arr = np.asarray(image, dtype=np.float32) / 255.0
        # Model config specifies mean/std = 0.5/0.5.
        arr = (arr - 0.5) / 0.5
        if self.channels_last:
            return arr[None, ...]
        return np.transpose(arr, (2, 0, 1))[None, ...]

    def _decode_probs(self, probs) -> list[list[TagItem]]:
        import numpy as np

        output = []
        probs = np.asarray(probs)
        for row in probs:
            result: list[TagItem] = []
            for i, score in enumerate(row):
                if i >= len(self.tags):
                    break
                category = self.categories[i]
                if category == 9:  # rating
                    continue
                threshold = (
                    self.character_threshold
                    if category == 4
                    else self.general_threshold
                )
                if float(score) >= threshold:
                    result.append(TagItem(self.tags[i], category, float(score)))
            result.sort(key=lambda x: -x.score)
            output.append(result)
        return output

    def tag_image(self, path: str) -> list[TagItem]:
        self._ensure()
        batch = self._preprocess(path)
        probs = self.session.run(None, {self.input_name: batch})[0]
        return self._decode_probs(probs)[0]

    def tag_images(self, paths: list[Path], batch_size: int = 8) -> list[list[TagItem]]:
        """Batch WD14 inference. Current WD14 v3 supports a dynamic batch dimension."""
        import numpy as np

        self._ensure()
        all_results: list[list[TagItem]] = []
        for start in range(0, len(paths), max(1, int(batch_size))):
            chunk = paths[start:start + batch_size]
            batch = np.concatenate([self._preprocess(str(p)) for p in chunk], axis=0)
            probs = self.session.run(None, {self.input_name: batch})[0]
            all_results.extend(self._decode_probs(probs))
        return all_results


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def _safe_name(value: str) -> str:
    value = re.sub(r"[^\w\- .\u3040-\u30ff\u3400-\u9fff]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value).strip("._")
    return value or "my_lora"


def _extension_root() -> Path:
    try:
        from trainer import trainer as trainer_module
        return Path(trainer_module.path_root)
    except Exception:
        return Path(__file__).resolve().parents[1]


def dataset_root() -> Path:
    root = _extension_root() / "easy_datasets"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _extract_zip_secure(zip_path: Path, destination: Path) -> Path:
    extract_root = destination / "_zip_extract"
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (extract_root / member.filename).resolve()
            if not str(target).startswith(str(extract_root.resolve())):
                raise RuntimeError("安全上の理由で、このZIPは展開できません。")
        zf.extractall(extract_root)

    return extract_root


def collect_input_images(files, folder_path: str | None) -> list[Path]:
    temp_root = Path(tempfile.mkdtemp(prefix="traintrain_easy_"))
    collected: list[Path] = []

    try:
        if files:
            for item in files:
                src = Path(getattr(item, "name", item))
                if src.suffix.lower() == ".zip":
                    extracted = _extract_zip_secure(src, temp_root)
                    collected.extend(_iter_images(extracted))
                elif src.suffix.lower() in IMAGE_EXTS:
                    collected.append(src)

        if folder_path and folder_path.strip():
            p = Path(folder_path.strip()).expanduser()
            if not p.exists():
                raise RuntimeError(f"指定フォルダが見つかりません: {p}")
            if p.is_dir():
                collected.extend(_iter_images(p))
            elif p.is_file() and p.suffix.lower() == ".zip":
                extracted = _extract_zip_secure(p, temp_root)
                collected.extend(_iter_images(extracted))
            else:
                raise RuntimeError("画像フォルダまたはZIPを指定してください。")

        # Deduplicate by content hash, not only filename.
        unique: list[Path] = []
        seen: set[str] = set()
        for p in collected:
            try:
                digest = hashlib.sha1(p.read_bytes()).hexdigest()
            except Exception:
                continue
            if digest not in seen:
                seen.add(digest)
                unique.append(p)

        if not unique:
            raise RuntimeError("画像が見つかりませんでした。")

        # Copy to a stable temporary area so caller can safely remove uploads.
        stable_dir = temp_root / "images"
        stable_dir.mkdir(exist_ok=True)
        result = []
        for i, src in enumerate(unique):
            suffix = src.suffix.lower()
            name = _safe_name(src.stem)[:80] or f"image_{i:04d}"
            dst = stable_dir / f"{i:04d}_{name}{suffix}"
            shutil.copy2(src, dst)
            result.append(dst)

        return result
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def _category_counts(items: list[TagItem]) -> Counter:
    c = Counter()
    for item in items:
        c[classify_tag(item.name, item.category)] += 1
    return c


def _filtered_caption(
    items: list[TagItem],
    remove_categories: set[str],
    manual_keep: set[str] | None = None,
    manual_remove: set[str] | None = None,
) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    manual_keep = manual_keep or set()
    manual_remove = manual_remove or set()

    kept: list[str] = []
    removed: list[str] = []
    review: list[tuple[str, str]] = []

    seen = set()
    for item in items:
        if item.name in seen:
            continue
        seen.add(item.name)
        cat = classify_tag(item.name, item.category)

        if item.name in manual_remove:
            removed.append(item.name)
            continue
        if item.name in manual_keep:
            kept.append(item.name)
            continue
        if cat in remove_categories:
            removed.append(item.name)
        else:
            kept.append(item.name)

        # Unknowns are kept, but are surfaced in review.
        if cat == "other":
            review.append((item.name, "未分類のため保持"))

    return kept, removed, review


def _image_probe(path: Path) -> dict:
    try:
        with Image.open(path) as im:
            w, h = im.size
            return {
                "ok": True,
                "width": w,
                "height": h,
                "megapixels": round((w * h) / 1_000_000, 2),
                "small": min(w, h) < 512,
                "mode": im.mode,
            }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def prepare_dataset(
    files,
    folder_path: str,
    dataset_name: str,
    preset_name: str,
    trigger: str,
    general_threshold: float,
    character_threshold: float,
    manual_keep_text: str = "",
    manual_remove_text: str = "",
):
    images = collect_input_images(files, folder_path)
    preset = PRESETS[preset_name]
    dataset_name = _safe_name(dataset_name)
    root = dataset_root() / dataset_name

    if root.exists():
        shutil.rmtree(root)
    original_dir = root / "original"
    prepared_dir = root / "prepared"
    original_caption_dir = root / "captions_original"
    meta_dir = root / "meta"
    for d in (original_dir, prepared_dir, original_caption_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    manual_keep = {
        x.strip() for x in manual_keep_text.split(",") if x.strip()
    }
    manual_remove = {
        x.strip() for x in manual_remove_text.split(",") if x.strip()
    }

    tagger = WD14Tagger(
        general_threshold=float(general_threshold),
        character_threshold=float(character_threshold),
    )

    table = []
    review_counter = Counter()
    removed_counter = Counter()
    kept_counter = Counter()
    warning_counter = Counter()
    all_tag_presence = Counter()

    tag_results = tagger.tag_images(images, batch_size=8)

    for idx, (src, items) in enumerate(zip(images, tag_results)):
        # Stable filename in output.
        out_name = f"{idx:04d}_{src.stem}{src.suffix.lower()}"
        dst_image = original_dir / out_name
        shutil.copy2(src, dst_image)

        probe = _image_probe(src)
        if not probe["ok"]:
            warning_counter["読み込み失敗"] += 1
            continue
        if probe.get("small"):
            warning_counter["短辺512未満"] += 1

        for item in items:
            all_tag_presence[item.name] += 1

        # Preserve raw detector output and scores for auditing.
        with open(original_caption_dir / f"{Path(out_name).stem}.json", "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "tag": x.name,
                        "category": x.category_name,
                        "category_id": x.category,
                        "score": round(x.score, 5),
                    }
                    for x in items
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )

        kept, removed, review = _filtered_caption(
            items,
            preset["remove"],
            manual_keep=manual_keep,
            manual_remove=manual_remove,
        )

        if not kept:
            warning_counter["caption空"] += 1

        # Keep raw text as a separate convenience file too.
        with open(
            original_caption_dir / f"{Path(out_name).stem}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(", ".join(x.name for x in items))

        prepared_image = prepared_dir / out_name
        shutil.copy2(src, prepared_image)

        # Important: do NOT put the trigger into the txt here.
        # TrainTrain's dataset.py prepends lora_trigger_word itself.
        with open(
            prepared_dir / f"{Path(out_name).stem}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(", ".join(kept))

        for x in kept:
            kept_counter[x] += 1
        for x in removed:
            removed_counter[x] += 1
        for x, reason in review:
            review_counter[x] += 1

        table.append(
            [
                out_name,
                ", ".join(x.name for x in items),
                ", ".join(removed),
                ", ".join(kept),
                "; ".join(x for x, _ in review[:8]),
                " / ".join(
                    f"{k}:{v}" for k, v in sorted(_category_counts(items).items())
                ),
            ]
        )

    # Dataset-level review is deliberately informational.
    common_review = [
        [tag, count, round(count / max(len(images), 1), 2)]
        for tag, count in review_counter.most_common(30)
    ]

    stats = {
        "dataset": dataset_name,
        "preset": preset_name,
        "description": preset["label"],
        "trigger": trigger.strip(),
        "images": len(images),
        "tagger_provider": tagger.provider,
        "general_threshold": float(general_threshold),
        "character_threshold": float(character_threshold),
        "removed_tag_types": len(removed_counter),
        "kept_tag_types": len(kept_counter),
        "review_tag_types": len(review_counter),
        "warnings": dict(warning_counter),
        "common_review_tags": common_review,
        "top_removed": removed_counter.most_common(30),
        "top_kept": kept_counter.most_common(30),
        "output": str(prepared_dir),
    }

    with open(meta_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # A compact CSV is convenient for manual inspection.
    with open(meta_dir / "review.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image", "raw_tags", "removed_tags", "prepared_caption", "review", "categories"]
        )
        writer.writerows(table)

    steps = auto_steps(len(images), preset_name)
    summary = (
        f"準備完了：{len(images)}枚\n"
        f"プリセット：{preset_name}\n"
        f"WD14：{tagger.provider}\n"
        f"自動削除：{len(removed_counter)}種類\n"
        f"要確認：{len(review_counter)}種類\n"
        f"警告：{sum(warning_counter.values())}件\n"
        f"推奨初期step：約{steps}"
    )

    return {
        "prepared_dir": str(prepared_dir),
        "summary": summary,
        "table": table[:100],
        "stats_json": json.dumps(stats, ensure_ascii=False, indent=2),
        "review_table": common_review,
        "image_count": len(images),
        "auto_steps": steps,
    }


def auto_steps(image_count: int, preset_name: str) -> int:
    per_image = PRESETS[preset_name]["steps_per_image"]
    # A beginner-safe starting point, not a guarantee of optimal training.
    raw = max(400, int(image_count) * per_image)
    rounded = int(round(raw / 50.0) * 50)
    return max(400, min(1800, rounded))


# ---------------------------------------------------------------------------
# TrainTrain integration
# ---------------------------------------------------------------------------

def build_train_values(
    trainer_module,
    prepared_dir: str,
    trigger: str,
    output_name: str,
    preset_name: str,
    image_size: int,
    steps: int,
):
    configs = list(trainer_module.all_configs)
    if not configs:
        raise RuntimeError("TrainTrainの設定定義を取得できません。")

    defaults = [cfg[3] for cfg in configs]
    first = list(defaults)
    second = list(defaults)

    # Keep TrainTrain's existing defaults, only overriding beginner-facing
    # parameters. This avoids guessing at the many advanced knobs.
    overrides = {
        "network_type": "lierla",
        "network_rank": str(PRESETS[preset_name]["rank"]),
        "network_alpha": "8",
        "lora_data_directory": prepared_dir,
        "lora_trigger_word": trigger.strip(),
        "image_size(height, width)": str(image_size),
        "train_iterations": int(steps),
        "train_batch_size": 1,
        "train_learning_rate": "1e-4",
        "train_optimizer": "AdamW",
        "train_lr_scheduler": "cosine",
        "use_gradient_checkpointing": True,
        "image_buckets_step": "256",
        "image_min_length": 512,
        "image_max_ratio": 2,
        "train_snr_gamma": 5,
        "train_seed": -1,
        "train_model_precision": "fp16",
        "train_lora_precision": "fp32",
        "train_VAE_precision": "fp32",
        "image_shuffle_tags": True,
        "train_self_reg": 0,
    }

    index = {cfg[0].split("(")[0]: i for i, cfg in enumerate(configs)}
    for key, value in overrides.items():
        if key in index:
            first[index[key]] = value
            second[index[key]] = value

    # No prompts/images for standard LoRA.
    return first + second + ["", "", "", None, None]


def start_training(
    train_module,
    trainer_module,
    prepared_dir: str,
    trigger: str,
    output_name: str,
    preset_name: str,
    image_size: int,
    steps: int,
    model: str,
    vae: str,
    te: str,
):
    if not prepared_dir or not os.path.isdir(prepared_dir):
        return "先に「自動で学習準備」を実行してください。"
    if not model:
        return "学習モデルを選択してください。"

    values = build_train_values(
        trainer_module,
        prepared_dir,
        trigger,
        output_name,
        preset_name,
        int(image_size),
        int(steps),
    )
    try:
        result = train_module.train(False, "LoRA", model, vae, te, *values)
    except Exception as e:
        return f"学習開始に失敗しました: {type(e).__name__}: {e}"
    return str(result)


def guess_image_size(model_name: str) -> int:
    s = (model_name or "").lower()
    if any(x in s for x in ("sdxl", "pony", "flux", "z-image", "anima", "krea")):
        return 1024
    return 512


# ---------------------------------------------------------------------------
# Easy UI
# ---------------------------------------------------------------------------

def build_easy_tab(
    train_module,
    trainer_module,
    model_choices=None,
    vae_choices=None,
    te_choices=None,
    default_model="",
    gradio_module=None,
):
    """Build the beginner-first tab inside the existing TrainTrain Blocks."""
    gr = gradio_module
    if gr is None:
        import gradio as gr

    model_choices = list(model_choices or [])
    vae_choices = list(vae_choices or ["None"])
    te_choices = list(te_choices or ["None"])

    if default_model and default_model not in model_choices:
        model_choices = [default_model] + model_choices

    default_size = guess_image_size(default_model)

    css = """
    .tt-easy-wrap { max-width: 1100px; margin: 0 auto; }
    .tt-easy-title { font-size: 30px; font-weight: 700; margin: 2px 0 0 0; }
    .tt-easy-sub { opacity: 0.72; margin-bottom: 10px; }
    .tt-easy-card {
        border: 1px solid var(--border-color-primary);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .tt-easy-step { font-size: 18px; font-weight: 700; }
    .tt-easy-note { opacity: 0.72; font-size: 0.92em; }
    .tt-easy-primary button { min-height: 52px; font-size: 17px; font-weight: 700; }
    """

    def _safe_output_name(name: str):
        return _safe_name(name)

    def _preset_text(name: str):
        p = PRESETS.get(name, PRESETS["キャラクター"])
        remove = " / ".join(sorted(p["remove"]))
        return f'{p["label"]}\n\n自動削除カテゴリ: `{remove}`'

    def _default_steps(name: str):
        return int(PRESETS.get(name, PRESETS["キャラクター"])["steps_per_image"] * 10)

    def _prepare_ui(
        files,
        folder_path,
        dataset_name,
        preset_name,
        trigger,
        general_threshold,
        character_threshold,
        manual_keep,
        manual_remove,
        auto_start,
        model,
        vae,
        te,
        output_name,
        image_size,
    ):
        result = prepare_dataset(
            files,
            folder_path,
            dataset_name,
            preset_name,
            trigger,
            general_threshold,
            character_threshold,
            manual_keep,
            manual_remove,
        )
        train_status = "準備だけ完了しました。"
        if auto_start:
            train_status = start_training(
                train_module,
                trainer_module,
                result["prepared_dir"],
                trigger,
                output_name or dataset_name,
                preset_name,
                int(image_size),
                int(result["auto_steps"]),
                model,
                vae,
                te,
            )
        return (
            result["prepared_dir"],
            result["summary"],
            result["table"],
            result["stats_json"],
            result["review_table"],
            gr.update(value=result["auto_steps"]),
            train_status,
        )

    gr.HTML(f"<style>{css}</style>")
    with gr.Column(elem_classes=["tt-easy-wrap"]):
        gr.Markdown("## Easy LoRA", elem_classes=["tt-easy-title"])
        gr.Markdown(
            "画像を入れて、何を学習するか選ぶだけ。"
            "WD14タグ生成 → 不要タグ整理 → 学習用Dataset作成まで自動で行います。",
            elem_classes=["tt-easy-sub"],
        )

        with gr.Group(elem_classes=["tt-easy-card"]):
            gr.Markdown("### ① 画像を入れる", elem_classes=["tt-easy-step"])
            gr.Markdown(
                "複数画像、ZIP、またはColab/Google Drive上の画像フォルダを指定できます。",
                elem_classes=["tt-easy-note"],
            )
            files = gr.File(
                label="画像 / ZIP",
                file_count="multiple",
                file_types=["image", ".zip"],
                type="filepath",
            )
            folder_path = gr.Textbox(
                label="画像フォルダ（任意）",
                placeholder="/content/drive/MyDrive/LoRA/my_character",
            )
            with gr.Row():
                dataset_name = gr.Textbox(
                    label="データセット名",
                    value="my_lora",
                )
                trigger = gr.Textbox(
                    label="トリガーワード",
                    value="my_character",
                )

        with gr.Group(elem_classes=["tt-easy-card"]):
            gr.Markdown("### ② 何を学習するか選ぶ", elem_classes=["tt-easy-step"])
            preset = gr.Radio(
                choices=list(PRESETS.keys()),
                value="キャラクター",
                label="LoRAの種類",
            )
            preset_help = gr.Markdown(_preset_text("キャラクター"))
            preset.change(_preset_text, [preset], [preset_help])

            with gr.Accordion("細かい設定（通常は変更不要）", open=False):
                with gr.Row():
                    general_threshold = gr.Slider(
                        0.10, 0.70, value=WD_DEFAULT_GENERAL_THRESHOLD, step=0.01,
                        label="WD14 一般タグしきい値",
                    )
                    character_threshold = gr.Slider(
                        0.50, 0.95, value=WD_DEFAULT_CHARACTER_THRESHOLD, step=0.01,
                        label="WD14 キャラクタータグしきい値",
                    )
                manual_keep = gr.Textbox(
                    label="必ず残すタグ（カンマ区切り）",
                    placeholder="smile, looking_at_viewer",
                )
                manual_remove = gr.Textbox(
                    label="必ず削除するタグ（カンマ区切り）",
                    placeholder="1girl, solo",
                )

        with gr.Group(elem_classes=["tt-easy-card"]):
            gr.Markdown("### ③ 自動準備", elem_classes=["tt-easy-step"])
            with gr.Row():
                prepare = gr.Button(
                    "自動で学習準備",
                    variant="primary",
                    elem_classes=["tt-easy-primary"],
                )
                auto_start = gr.Checkbox(
                    label="準備後、そのまま学習を開始",
                    value=False,
                )
            summary = gr.Textbox(label="結果", lines=4, interactive=False)

        with gr.Group(elem_classes=["tt-easy-card"]):
            gr.Markdown("### ④ 自動チェック", elem_classes=["tt-easy-step"])
            gr.Markdown(
                "自動削除できないタグは勝手に消さず、要確認として残します。",
                elem_classes=["tt-easy-note"],
            )
            table = gr.Dataframe(
                headers=[
                    "画像", "WD14タグ", "自動削除", "学習用caption", "要確認", "カテゴリ"
                ],
                datatype=["str"] * 6,
                interactive=False,
                wrap=True,
                height=360,
            )
            review_table = gr.Dataframe(
                headers=["未分類タグ", "出現枚数", "出現率"],
                datatype=["str", "number", "number"],
                interactive=False,
                height=220,
            )
            stats = gr.Code(label="詳細統計", language="json")

        with gr.Group(elem_classes=["tt-easy-card"]):
            gr.Markdown("### ⑤ そのまま学習（任意）", elem_classes=["tt-easy-step"])
            prepared_dir = gr.Textbox(
                label="準備済みDataset",
                interactive=False,
            )
            with gr.Row():
                model = gr.Dropdown(
                    choices=model_choices,
                    value=default_model if default_model in model_choices else (model_choices[0] if model_choices else None),
                    label="モデル",
                    allow_custom_value=True,
                )
                vae = gr.Dropdown(
                    choices=vae_choices,
                    value=vae_choices[0] if vae_choices else "None",
                    label="VAE",
                    allow_custom_value=True,
                )
                te = gr.Dropdown(
                    choices=te_choices,
                    value=te_choices[0] if te_choices else "None",
                    label="Text Encoder",
                    allow_custom_value=True,
                )
            with gr.Row():
                output_name = gr.Textbox(
                    label="出力LoRA名",
                    value="my_lora",
                )
                image_size = gr.Dropdown(
                    choices=[512, 768, 1024, 1280],
                    value=default_size,
                    label="解像度",
                )
                steps = gr.Slider(
                    minimum=400,
                    maximum=1800,
                    step=50,
                    value=_default_steps("キャラクター"),
                    label="学習step（自動設定）",
                )
            start = gr.Button("学習開始", variant="primary")
            train_result = gr.Textbox(label="学習結果", lines=5, interactive=False)

        # Convenience: dataset name becomes the suggested output name/trigger,
        # but never overwrites fields once the user types something else.
        def _sync_dataset_name(name):
            v = _safe_output_name(name)
            return gr.update(value=v), gr.update(value=v)

        dataset_name.blur(
            _sync_dataset_name,
            [dataset_name],
            [output_name, trigger],
        )

        prepare.click(
            _prepare_ui,
            [
                files, folder_path, dataset_name, preset, trigger,
                general_threshold, character_threshold,
                manual_keep, manual_remove, auto_start,
                model, vae, te, output_name, image_size,
            ],
            [
                prepared_dir, summary, table, stats, review_table, steps, train_result
            ],
        )

        start.click(
            lambda p, tr, out, pr, size, st, m, v, t: start_training(
                train_module, trainer_module, p, tr, out, pr, int(size), int(st), m, v, t
            ),
            [prepared_dir, trigger, output_name, preset, image_size, steps, model, vae, te],
            [train_result],
        )
