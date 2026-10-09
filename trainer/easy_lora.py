# -*- coding: utf-8 -*-
"""
Easy LoRA - 「画像を入れてボタンを押すだけ」のLoRA学習準備バックエンド。

TrainTrainの学習エンジンとは独立しており、Kohya/TrainTrain互換の
「画像 + .txt」データセットを作り、必要ならそのままTrainTrainの学習を開始します。

設計方針
- 初心者が触る項目は「画像」「LoRAの種類」「モデル」の3つだけ
- タグの選別はLLMを使わず、ローカルで決定的に行う（同じ入力なら同じ結果）
- 元画像は絶対に書き換えない（original/ に保存）
- 判断の理由を全部残す（meta/tag_decisions.csv）
- 迷うものは消さずに残し、「確認してね」として表示する
"""

from __future__ import annotations

import csv
import contextlib
import gc
import hashlib
import html as html_lib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
# そのままコピーしてよい形式（それ以外はPNGに変換する）
PASSTHROUGH_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

WD_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
WD_DEFAULT_GENERAL_THRESHOLD = 0.35
WD_DEFAULT_CHARACTER_THRESHOLD = 0.85
WD_AUTO_THRESHOLD_FLOOR = 0.25        # 自動調整で下げてよい下限
WD_GENERAL_FLOOR = 0.20               # 推論結果として保存する下限（しきい値の自動調整用）
WD_CHARACTER_FLOOR = 0.50
WD_MIN_TAGS_PER_IMAGE = 8             # これを下回るならしきい値を自動で下げる
WD_IMAGE_SIZE = 448

MAX_SIDE = 2048          # これより大きい画像は縮小して準備用フォルダに入れる
MIN_SIDE_HARD = 256      # これ未満は学習に使えないので自動除外
MIN_SIDE_WARN = 512      # これ未満は警告
NEAR_DUP_MAX_BITS = 6    # 知覚ハッシュ(240bit)の差がこれ以下なら「ほぼ同一」候補
NEAR_DUP_MAX_MAD = 4.0   # 16x16グレースケールの平均絶対差がこれ以下
ZIP_MAX_BYTES = 8 * 1024 ** 3

PREVIEW_MAX = 24         # プレビューに出す枚数
TABLE_MAX_TAGS = 80


# ---------------------------------------------------------------------------
# タグの分類
# ---------------------------------------------------------------------------

# SmilingWolf WD taggerのカテゴリ番号
WD_CATEGORY_NAMES = {
    0: "general",
    1: "artist",
    2: "unknown",
    3: "copyright",
    4: "character",
    5: "meta",
    9: "rating",
}

CATEGORY_LABELS_JA = {
    "character": "キャラ名",
    "copyright": "作品名",
    "artist": "作者名",
    "meta": "メタ情報",
    "rating": "年齢区分",
    "subject": "人数・主題",
    "hair": "髪",
    "expression": "表情",
    "appearance": "外見(目・肌・体など)",
    "clothing": "衣装・小物",
    "style": "画風・技法",
    "object": "物",
    "background": "背景",
    "pose": "ポーズ・構図",
    "quality": "品質・メタ",
    "defect": "画像の欠点(透かし等)",
    "other": "未分類",
    "unknown": "未分類",
}

_SUBJECT_RE = re.compile(
    r"^(\d+\+?(girl|boy|other)s?|multiple_(girls|boys|others)|solo|solo_focus|"
    r"no_humans|(male|female)_focus|everyone)$"
)

# 画像そのものの欠点。キャプションに「残す」ことで、LoRAが透かしや文字を
# 学習してしまうのを防ぎ、生成時にネガティブプロンプトで消せるようにする。
_DEFECT = (
    "watermark signature text artist_name username web_address jpeg_artifacts "
    "lowres blurry logo dated copyright_name character_name patreon_username "
    "twitter_username pixiv_id"
).split()

# 画像の内容を表さないメタ情報・品質タグ。キャプションには不要。
_QUALITY = (
    "masterpiece best_quality high_quality highres absurdres incredibly_absurdres "
    "huge_filesize commentary commentary_request translated translation_request "
    "bad_id bad_pixiv_id bad_link official_art scan artbook"
).split()

_HAIR = (
    "hair hairstyle bangs ponytail twintails twin_tails braid braids braided ahoge "
    "pigtails bun buns hairband hairpin hairclip sidelocks undercut sideburns drills "
    "bob_cut hime_cut pixie_cut mohawk dreadlocks afro topknot"
).split()

_EXPRESSION = (
    "smile smiling grin angry sad crying tears surprised open_mouth closed_mouth "
    "closed_eyes eyes_closed one_eye_closed half-closed_eyes wink expressionless "
    "confused embarrassed nervous serious laughing sleeping pout frown blush "
    "tongue tongue_out parted_lips clenched_teeth teeth happy annoyed scared shy "
    "jitome smirk furrowed_brow light_smile :d ;d :o :3 ^_^"
).split()

_APPEARANCE = (
    "eyes eyebrows eyelashes face skin freckles makeup lipstick fangs horns ears "
    "tail wings scar mole beard mustache tanlines tanned dark_skin pale_skin "
    "facial_mark forehead_mark heterochromia pupils halo colored_skin breasts "
    "flat_chest thighs abs muscular slim skinny curvy petite"
).split()

_CLOTHING = (
    "dress skirt miniskirt shirt t-shirt blouse jacket coat uniform suit hoodie "
    "sweater cardigan necktie bowtie tie ribbon bow scarf gloves glove shoes socks "
    "boots sandals hat cap beret swimsuit swimwear bikini bra panties underwear "
    "stockings thighhighs pantyhose kneehighs legwear shorts pants jeans apron armor "
    "kimono yukata cloak cape capelet poncho shawl belt corset lingerie bodysuit "
    "leotard sleeves sleeveless buttons collar pajamas robe cuffs vest tank_top "
    "crop_top camisole sailor_collar serafuku maid headdress headband headphones "
    "glasses sunglasses earrings necklace choker bracelet jewelry mask helmet crown "
    "tiara pendant brooch hood turtleneck off_shoulder bare_shoulders midriff "
    "clothes clothing outfit costume frills lace trim strap straps zipper pocket "
    "pockets eyepatch footwear barefoot armband"
).split()

_STYLE = (
    "anime anime_coloring manga comic cartoon chibi realistic photorealistic "
    "illustration sketch lineart flat_color cel_shading monochrome greyscale "
    "spot_color pixel_art 3d render digital_painting traditional_media surreal "
    "abstract ink ink_wash_painting charcoal rough_sketch concept_art "
    "retro_artstyle style halftone limited_palette sepia painterly film_grain "
    "chromatic_aberration vignetting"
).split()

_OBJECT = (
    "sword katana knife dagger gun rifle pistol bow_(weapon) arrow weapon weapons "
    "shield staff wand axe spear hammer bag backpack handbag purse phone smartphone "
    "cellphone book books cup teacup bottle food drink cake sweets fruit umbrella car "
    "motor_vehicle vehicle bicycle motorcycle chair table desk bed couch sofa "
    "computer laptop camera microphone guitar instrument piano ball lantern door "
    "furniture robot mecha cat dog bird horse rabbit fish animal stuffed_toy"
).split()

_BACKGROUND = (
    "background indoors outdoors sky cloud clouds city cityscape street classroom "
    "school room bedroom forest park beach mountain ocean sea river lake snow rain "
    "sunset sunrise night day sunlight moon stars building architecture window wall "
    "floor road tree trees flower flowers petals grass field nature landscape garden "
    "underwater space desert bridge railway store shop cafe restaurant office "
    "hospital shrine temple stairs fence bokeh depth_of_field lens_flare light "
    "shadow sunbeam scenery water horizon"
).split()

_POSE = (
    "standing sitting kneeling lying walking running jumping crouching leaning "
    "bending squatting arms_up arms_behind_back arms_behind_head arm_up arms_crossed "
    "crossed_arms hand_on_hip hands_on_hips hand_up hands_up hand_on_own_face "
    "hand_in_pocket looking_at_viewer looking_away looking_back looking_to_the_side "
    "looking_up looking_down from_above from_below from_behind from_side upper_body "
    "full_body cowboy_shot portrait close-up profile dynamic_pose fighting_stance "
    "crossed_legs on_back on_stomach on_side spread_legs pose peace_sign v waving "
    "pointing reaching holding head_tilt dutch_angle facing_viewer feet_out_of_frame "
    "head_out_of_frame lower_body hand hands arms legs fingers"
).split()


def _padded(tag: str) -> str:
    return "_" + tag.lower().strip().replace(" ", "_") + "_"


def _has(padded_tag: str, words: Iterable[str]) -> bool:
    """タグ中に「単語として」含まれているか（hair_ribbon に ribbon, など）。"""
    return any(f"_{w}_" in padded_tag for w in words)


@dataclass
class TagItem:
    name: str          # WD14の生の名前（アンダースコア区切り）
    category: int
    score: float

    @property
    def category_name(self) -> str:
        return WD_CATEGORY_NAMES.get(self.category, "unknown")


def classify_tag(tag: str, category: int) -> str:
    """タグを内容カテゴリに分類する。WD14のカテゴリ番号を優先し、generalは名前から推定。"""
    native = WD_CATEGORY_NAMES.get(category)
    if native in ("character", "copyright", "artist", "meta", "rating"):
        return native

    key = tag.lower().strip().replace(" ", "_")
    p = _padded(key)

    if _SUBJECT_RE.match(key):
        return "subject"
    if key.startswith("blurry_"):          # blurry_background 等は構図情報
        return "background"
    if _has(p, _DEFECT):
        return "defect"
    if _has(p, _QUALITY):
        return "quality"
    if _has(p, _HAIR):
        return "hair"
    if _has(p, _EXPRESSION):
        return "expression"
    if _has(p, _APPEARANCE):
        return "appearance"
    if "(weapon)" in key:
        return "object"
    if _has(p, _CLOTHING):
        return "clothing"
    if "_(medium)" in key or "_(style)" in key or _has(p, _STYLE):
        return "style"
    if _has(p, _OBJECT):
        return "object"
    if _has(p, _BACKGROUND):
        return "background"
    if _has(p, _POSE):
        return "pose"
    return "other"


# ---------------------------------------------------------------------------
# プリセット（何を学習するか）
# ---------------------------------------------------------------------------
# remove        : そのカテゴリのタグは常にキャプションから外す
# remove_common : 「全画像のうち ratio 以上に出てくる」タグだけ外す。
#                 常に出る髪色・目の色などは「そのキャラの特徴」なので外し、
#                 たまにしか出ない髪型・衣装は「変化する要素」として残す。
# 画像が少なく(MIN_IMAGES_FOR_RATIO未満)出現率が当てにならない時は、
# remove_common のカテゴリも常に外す。

MIN_IMAGES_FOR_RATIO = 4
ALWAYS_REMOVE = {"quality", "meta", "rating"}

PRESETS = {
    "キャラクター": {
        "short": "人物・キャラの見た目を覚えさせる",
        "label": (
            "キャラクター本体を学習します。名前・作品名は外し、"
            "どの画像にも共通する髪色や目の色などは「そのキャラの特徴」として"
            "キャプションから外します。画像ごとに変わる髪型・衣装・背景・ポーズは残します。"
        ),
        "remove": {"character", "copyright"},
        "remove_common": {"hair": 0.3, "appearance": 0.3, "clothing": 0.4},
        "rank": 16,
        "steps_per_image": 50,
    },
    "衣装": {
        "short": "特定の服・コスチュームを覚えさせる",
        "label": (
            "衣装を学習します。多くの画像に共通する服のタグを外し、"
            "キャラクター・髪・背景・ポーズは残します。"
        ),
        "remove": set(),
        "remove_common": {"clothing": 0.25},
        "rank": 16,
        "steps_per_image": 50,
    },
    "画風": {
        "short": "絵柄・塗り・雰囲気を覚えさせる",
        "label": (
            "画風を学習します。画風や作者名のタグを外し、"
            "画像の内容を説明するタグ(人物・服・背景など)は残します。"
        ),
        "remove": {"style", "artist"},
        "remove_common": {},
        "rank": 16,
        "steps_per_image": 40,
    },
    "オブジェクト": {
        "short": "特定の物・アイテム・マスコットを覚えさせる",
        "label": (
            "対象の物を学習します。物に関するタグと、ほぼ全画像に共通する未分類タグを外し、"
            "周囲の人物・背景は残します。"
        ),
        "remove": {"object"},
        "remove_common": {"other": 0.6},
        "rank": 16,
        "steps_per_image": 50,
    },
}
DEFAULT_PRESET = "キャラクター"


# ---------------------------------------------------------------------------
# タグ表記の整形
# ---------------------------------------------------------------------------

# 顔文字タグは _ を空白に変えない（kohya の wd14 tagger と同じ扱い）
_KAOMOJI = {
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<",
    "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
}


def to_caption_tag(tag: str) -> str:
    """学習用の表記へ。long_hair -> long hair（顔文字はそのまま）。"""
    return tag if tag in _KAOMOJI else tag.replace("_", " ")


def _norm_key(tag: str) -> str:
    """ユーザー入力のタグを、内部キー(アンダースコア区切り小文字)に揃える。"""
    t = tag.strip().lower()
    return t if t in _KAOMOJI else t.replace(" ", "_")


def _parse_tag_list(text: str) -> set[str]:
    # 全角カンマ・改行も区切りとして扱う
    parts = re.split(r"[,\u3001\uff0c\n]+", text or "")
    return {_norm_key(x) for x in parts if x.strip()}


# ---------------------------------------------------------------------------
# WD14 ONNX タガー
# ---------------------------------------------------------------------------

class WD14Tagger:
    """SmilingWolf WD14 v3 (ONNX)。モデルは初回のみHugging Faceからダウンロード。

    重要:
    - 入力は BGR・0〜255のfloat32・正規化なし（公式の推論コードと同じ）。
    - タグ付けが終わったら close() でGPUメモリを必ず解放する
      （そうしないと、続く学習でVRAMが足りなくなる）。
    """

    def __init__(self, repo_id: str = WD_REPO):
        self.repo_id = repo_id
        self.session = None
        self.input_name = None
        self.channels_last = True
        self.tags: list[str] = []
        self.categories: list[int] = []
        self.model_target_size = WD_IMAGE_SIZE
        self.provider = "not loaded"

    def _model_dir(self) -> Path:
        safe = self.repo_id.replace("/", "__")
        p = _extension_root() / "models" / "wd14" / safe
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _ensure(self):
        if self.session is not None:
            return
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RuntimeError(
                "タグ付けに必要なライブラリ(onnxruntime / huggingface_hub)が見つかりません。"
                "WebUIを再起動して、拡張機能のインストールが完了するのを待ってください。"
            ) from e

        model_dir = self._model_dir()
        try:
            model_path = hf_hub_download(self.repo_id, "model.onnx", local_dir=str(model_dir))
            labels_path = hf_hub_download(self.repo_id, "selected_tags.csv", local_dir=str(model_dir))
        except Exception as e:
            raise RuntimeError(
                "タグ付けモデル(WD14)をダウンロードできませんでした。"
                "インターネット接続を確認して、もう一度実行してください。"
                f"（詳細: {type(e).__name__}）"
            ) from e

        with open(labels_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.tags = [row["name"] for row in rows]
        self.categories = [int(row["category"]) for row in rows]

        try:
            ort.set_default_logger_severity(3)
        except Exception:
            pass
        available = ort.get_available_providers()
        wanted = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        try:
            self.session = ort.InferenceSession(model_path, providers=wanted or available)
        except Exception:
            # GPU版の初期化に失敗した場合はCPUで続行
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.provider = "/".join(self.session.get_providers())

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        if len(shape) == 4:
            if shape[-1] == 3:
                self.channels_last = True
            elif shape[1] == 3:
                self.channels_last = False
            for idx in ((1, 2) if self.channels_last else (2, 3)):
                if isinstance(shape[idx], int):
                    self.model_target_size = shape[idx]
                    break

    def close(self):
        """ONNXセッションを破棄してGPUメモリを解放する。"""
        self.session = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _preprocess(self, path):
        import numpy as np

        image = Image.open(path)
        image = ImageOps.exif_transpose(image)
        # 透明部分は白で塗る（黒になると別物として認識されるため）
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            bg.alpha_composite(rgba)
            image = bg
        image = image.convert("RGB")

        side = max(image.width, image.height)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        size = self.model_target_size
        resample = Image.Resampling.BOX if side > size else Image.Resampling.BICUBIC
        canvas = canvas.resize((size, size), resample)

        arr = np.asarray(canvas, dtype=np.float32)   # 0..255
        arr = arr[:, :, ::-1]                         # RGB -> BGR
        arr = np.ascontiguousarray(arr)
        if self.channels_last:
            return arr[None, ...]
        return np.transpose(arr, (2, 0, 1))[None, ...]

    def _decode_probs(self, probs) -> list[list[TagItem]]:
        """確率を TagItem のリストへ。しきい値は後段で掛けるので、低めの下限だけ適用する。"""
        import numpy as np

        probs = np.asarray(probs)
        cats = np.asarray(self.categories[: probs.shape[1]])
        floors = np.where(cats == 4, WD_CHARACTER_FLOOR, WD_GENERAL_FLOOR)
        out: list[list[TagItem]] = []
        for row in probs:
            r = row[: len(cats)]
            idx = np.where((r >= floors) & (cats != 9))[0]
            items = [TagItem(self.tags[i], int(cats[i]), float(r[i])) for i in idx]
            items.sort(key=lambda x: -x.score)
            out.append(items)
        return out

    def tag_images(self, paths, batch_size: int = 8,
                   progress: Optional[Callable[[float, str], None]] = None) -> list[list[TagItem]]:
        import numpy as np

        self._ensure()
        results: list[list[TagItem]] = []
        total = len(paths)
        step = max(1, int(batch_size))
        for start in range(0, total, step):
            chunk = paths[start:start + step]
            batch = np.concatenate([self._preprocess(str(p)) for p in chunk], axis=0)
            probs = self.session.run(None, {self.input_name: batch})[0]
            results.extend(self._decode_probs(probs))
            if progress:
                progress(min(1.0, (start + len(chunk)) / total), f"タグ付け中 {min(total, start + step)}/{total}")
        return results


# ---------------------------------------------------------------------------
# パス・入力の収集
# ---------------------------------------------------------------------------

def _safe_name(value: str) -> str:
    value = re.sub(r"[^\w\- .\u3040-\u30ff\u3400-\u9fff]+", "_", value or "", flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value).strip("._")
    return value or "my_lora"


def _extension_root() -> Path:
    try:
        from trainer import trainer as trainer_module
        return Path(trainer_module.path_root)
    except Exception:
        pass
    try:
        from traintrain.trainer import trainer as trainer_module
        return Path(trainer_module.path_root)
    except Exception:
        return Path(__file__).resolve().parents[1]


def dataset_root() -> Path:
    root = _extension_root() / "easy_datasets"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _is_junk(path: Path) -> bool:
    return any(part.startswith(".") or part == "__MACOSX" for part in path.parts)


def _iter_images(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not _is_junk(p.relative_to(root)):
            yield p


def _extract_zip_secure(zip_path: Path, destination: Path) -> Path:
    extract_root = destination / f"_zip_{zip_path.stem[:40]}_{int(time.time() * 1000) % 100000}"
    extract_root.mkdir(parents=True, exist_ok=True)
    base = extract_root.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        total = 0
        for member in zf.infolist():
            target = (extract_root / member.filename).resolve()
            try:
                target.relative_to(base)
            except ValueError:
                raise RuntimeError("安全上の理由で、このZIPは展開できません。")
            total += member.file_size
            if total > ZIP_MAX_BYTES:
                raise RuntimeError("ZIPの展開後サイズが大きすぎます(8GB超)。")
        zf.extractall(extract_root)
    return extract_root


def collect_input_images(files, folder_path: str | None) -> tuple[list[Path], Path]:
    """アップロード/ZIP/フォルダから画像を集める。内容が完全に同じ画像は1枚にまとめる。

    戻り値: (画像パスのリスト, 後で削除すべき一時フォルダ)
    """
    temp_root = Path(tempfile.mkdtemp(prefix="traintrain_easy_"))
    collected: list[Path] = []
    try:
        for item in (files or []):
            src = Path(getattr(item, "name", item))
            if src.suffix.lower() == ".zip":
                collected.extend(_iter_images(_extract_zip_secure(src, temp_root)))
            elif src.suffix.lower() in IMAGE_EXTS:
                collected.append(src)

        if folder_path and folder_path.strip():
            p = Path(folder_path.strip()).expanduser()
            if not p.exists():
                raise RuntimeError(f"指定したフォルダが見つかりません: {p}")
            if p.is_dir():
                collected.extend(_iter_images(p))
            elif p.is_file() and p.suffix.lower() == ".zip":
                collected.extend(_iter_images(_extract_zip_secure(p, temp_root)))
            else:
                raise RuntimeError("画像フォルダまたはZIPファイルを指定してください。")

        unique: list[Path] = []
        seen: set[str] = set()
        for p in collected:
            try:
                h = hashlib.sha1()
                with open(p, "rb") as f:                  # 1MBずつ読む（大きな画像でもメモリを食わない）
                    for block in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(block)
                digest = h.hexdigest()
            except Exception:
                continue
            if digest not in seen:
                seen.add(digest)
                unique.append(p)

        if not unique:
            raise RuntimeError(
                "画像が見つかりませんでした。画像ファイル(jpg/png/webp等)かZIPを入れてください。"
            )
        return unique, temp_root
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# 画像のチェックと正規化
# ---------------------------------------------------------------------------

@dataclass
class ImageInfo:
    src: Path
    stem: str                      # 出力ファイル名（拡張子なし）
    width: int = 0
    height: int = 0
    ok: bool = True
    reason: str = ""               # 除外理由
    exif_rotated: bool = False
    thumb: object = None           # 16x16 グレースケール (numpy)
    bits: object = None            # 知覚ハッシュ (numpy bool)


def _probe_image(src: Path, stem: str) -> ImageInfo:
    """画像を開いて、サイズ・向き・知覚ハッシュを調べる。"""
    import numpy as np

    info = ImageInfo(src=src, stem=stem)
    try:
        with Image.open(src) as im:
            im.load()
            orientation = im.getexif().get(0x0112, 1)
            info.exif_rotated = orientation not in (None, 0, 1)
            im2 = ImageOps.exif_transpose(im)
            info.width, info.height = im2.size
            if im2.mode in ("RGBA", "LA") or (im2.mode == "P" and "transparency" in im2.info):
                rgba = im2.convert("RGBA")
                bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                bg.alpha_composite(rgba)
                gray = bg.convert("L")
            else:
                gray = im2.convert("L")
            thumb = np.asarray(gray.resize((16, 16), Image.Resampling.BOX), dtype=np.float32)
            info.thumb = thumb
            info.bits = (thumb[:, :-1] > thumb[:, 1:]).reshape(-1)
    except Exception as e:  # 壊れた画像など
        info.ok = False
        info.reason = f"読み込めない画像 ({type(e).__name__})"
        return info

    if min(info.width, info.height) < MIN_SIDE_HARD:
        info.ok = False
        info.reason = f"小さすぎる({info.width}x{info.height}、短辺{MIN_SIDE_HARD}未満)"
    return info


def _find_near_duplicates(infos: list[ImageInfo]) -> None:
    """ほぼ同一(再圧縮・縮小しただけ)の画像を、2枚目以降を除外扱いにする。"""
    import numpy as np

    kept_idx: list[int] = []
    kept_bits = []
    for i, info in enumerate(infos):
        if not info.ok:
            continue
        if kept_bits:
            diffs = (np.stack(kept_bits) != info.bits).sum(axis=1)
            for j in np.where(diffs <= NEAR_DUP_MAX_BITS)[0]:
                other = infos[kept_idx[int(j)]]
                aspect = abs(info.width / info.height - other.width / other.height)
                mad = float(np.abs(info.thumb - other.thumb).mean())
                if aspect <= 0.02 and mad <= NEAR_DUP_MAX_MAD:
                    info.ok = False
                    info.reason = f"{other.stem} とほぼ同じ画像"
                    break
        if info.ok:
            kept_idx.append(i)
            kept_bits.append(info.bits)


def _write_prepared_image(info: ImageInfo, dst_dir: Path) -> Path:
    """学習用フォルダへ画像を出力。向きの補正・縮小・形式変換が必要な時だけ再保存する。"""
    ext = info.src.suffix.lower()
    needs_convert = (
        ext not in PASSTHROUGH_EXTS
        or info.exif_rotated
        or max(info.width, info.height) > MAX_SIDE
    )
    if not needs_convert:
        with Image.open(info.src) as im:
            # パレット/特殊モードはそのままだと学習側で扱いにくい
            needs_convert = im.mode not in ("RGB", "RGBA", "L")
    if not needs_convert:
        dst = dst_dir / f"{info.stem}{ext}"
        shutil.copy2(info.src, dst)
        return dst

    with Image.open(info.src) as im:
        im.load()
        im = ImageOps.exif_transpose(im)
        if im.mode == "P":
            im = im.convert("RGBA" if "transparency" in im.info else "RGB")
        elif im.mode not in ("RGB", "RGBA", "L"):
            im = im.convert("RGB")
        if max(im.size) > MAX_SIDE:
            scale = MAX_SIDE / max(im.size)
            im = im.resize(
                (max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                Image.Resampling.LANCZOS,
            )
        dst = dst_dir / f"{info.stem}.png"
        im.save(dst, format="PNG")
    return dst


# ---------------------------------------------------------------------------
# タグの選別
# ---------------------------------------------------------------------------

@dataclass
class TagDecision:
    tag: str
    category: str
    count: int
    ratio: float
    keep: bool
    reason: str


def select_items(items: list[TagItem], general_threshold: float,
                 character_threshold: float) -> list[TagItem]:
    """生の推論結果にしきい値を掛ける。"""
    out = []
    for it in items:
        th = character_threshold if it.category in (1, 3, 4) else general_threshold
        if it.score >= th:
            out.append(it)
    return out


def auto_general_threshold(per_image: list[list[TagItem]], start: float,
                           character_threshold: float) -> tuple[float, str]:
    """タグが少なすぎる時だけ、しきい値を少しずつ下げる（上げることはしない）。"""
    if not per_image:
        return start, ""
    th = round(start, 2)

    def avg(t):
        return sum(len(select_items(x, t, character_threshold)) for x in per_image) / len(per_image)

    note = ""
    while avg(th) < WD_MIN_TAGS_PER_IMAGE and th - 0.05 >= WD_AUTO_THRESHOLD_FLOOR - 1e-9:
        th = round(th - 0.05, 2)
        note = f"タグが少なかったため、しきい値を {start:.2f} → {th:.2f} に自動で下げました。"
    return th, note


def decide_tags(per_image_sets: list[set[str]], tag_category: dict[str, str], preset: dict,
                manual_keep: set[str], manual_remove: set[str]) -> dict[str, TagDecision]:
    """データセット全体の出現率を見て、各タグを残すか外すかを決める。"""
    n = max(1, len(per_image_sets))
    counts: Counter = Counter()
    for s in per_image_sets:
        counts.update(s)

    use_ratio = len(per_image_sets) >= MIN_IMAGES_FOR_RATIO
    decisions: dict[str, TagDecision] = {}
    for tag, cnt in counts.items():
        cat = tag_category[tag]
        ratio = cnt / n
        keep, reason = True, "残す"

        if tag in manual_remove:
            keep, reason = False, "手動指定で削除"
        elif tag in manual_keep:
            keep, reason = True, "手動指定で残す"
        elif cat in ALWAYS_REMOVE:
            keep, reason = False, "画像の内容を表さないタグ"
        elif cat in preset["remove"]:
            keep, reason = False, f"この種類では不要({CATEGORY_LABELS_JA.get(cat, cat)})"
        elif cat in preset["remove_common"]:
            threshold = preset["remove_common"][cat]
            if not use_ratio:
                keep, reason = False, f"画像が少ないため{CATEGORY_LABELS_JA.get(cat, cat)}は一律で削除"
            elif ratio >= threshold:
                keep = False
                reason = (f"{ratio:.0%}の画像に共通 → 学習対象の特徴とみなして削除"
                          f"({CATEGORY_LABELS_JA.get(cat, cat)})")
            else:
                reason = f"{ratio:.0%}の画像にだけ出る → 変化する要素として残す"
        decisions[tag] = TagDecision(tag, cat, cnt, ratio, keep, reason)
    return decisions


# ---------------------------------------------------------------------------
# トリガーワード・ステップ数
# ---------------------------------------------------------------------------

def resolve_trigger(trigger: str, dataset_name: str = "") -> str:
    """トリガーワード。基本は不要なので、空欄なら「なし」（空文字）のまま。

    dataset_name は以前の自動生成の名残で、互換性のために引数だけ残している。
    """
    return (trigger or "").strip()


def auto_steps(image_count: int, preset_name: str) -> int:
    """最初に試すための学習ステップ数。最適とは限らない。"""
    per_image = PRESETS.get(preset_name, PRESETS[DEFAULT_PRESET])["steps_per_image"]
    raw = max(400, int(image_count) * per_image)
    rounded = int(round(raw / 50.0) * 50)
    return max(400, min(1800, rounded))


# ---------------------------------------------------------------------------
# データセット作成
# ---------------------------------------------------------------------------

def _make_previews(prepared_dir: Path, captions: dict[str, str], name: str) -> list[tuple[str, str]]:
    """ブラウザ表示用の軽いサムネイル。Gradioが必ず配信できる一時フォルダに置く。"""
    out_dir = Path(tempfile.gettempdir()) / "easylora_preview" / _safe_name(name)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    gallery = []
    for i, (fname, caption) in enumerate(list(captions.items())[:PREVIEW_MAX]):
        try:
            with Image.open(prepared_dir / fname) as im:
                im = ImageOps.exif_transpose(im)
                if im.mode in ("RGBA", "LA", "P"):
                    rgba = im.convert("RGBA")
                    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                    bg.alpha_composite(rgba)
                    im = bg
                im = im.convert("RGB")
                im.thumbnail((512, 512))
                p = out_dir / f"{i:03d}.jpg"
                im.save(p, quality=85)
            short = caption if len(caption) <= 160 else caption[:157] + "..."
            gallery.append((str(p), short or "(キャプションなし)"))
        except Exception:
            continue
    return gallery


def _make_zip(prepared_dir: Path, name: str) -> Optional[str]:
    try:
        out_dir = Path(tempfile.gettempdir()) / "easylora_zip"
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / f"{_safe_name(name)}_dataset"
        return shutil.make_archive(str(base), "zip", str(prepared_dir))
    except Exception:
        return None


def prepare_dataset(
    files,
    folder_path: str,
    dataset_name: str,
    preset_name: str,
    trigger: str = "",
    general_threshold: float = WD_DEFAULT_GENERAL_THRESHOLD,
    character_threshold: float = WD_DEFAULT_CHARACTER_THRESHOLD,
    manual_keep_text: str = "",
    manual_remove_text: str = "",
    auto_threshold: bool = True,
    progress: Optional[Callable[[float, str], None]] = None,
    free_mem: bool = False,
) -> dict:
    """画像の取り込み → チェック → WD14タグ付け → タグ選別 → データセット出力。"""

    def report(frac: float, desc: str):
        if progress:
            try:
                progress(frac, desc)
            except Exception:
                pass

    preset = PRESETS.get(preset_name) or PRESETS[DEFAULT_PRESET]
    preset_name = preset_name if preset_name in PRESETS else DEFAULT_PRESET
    dataset_name = _safe_name(dataset_name)
    trigger = resolve_trigger(trigger, dataset_name)
    manual_keep = _parse_tag_list(manual_keep_text)
    manual_remove = _parse_tag_list(manual_remove_text)

    memory_note = ""
    if free_mem:
        report(0.01, "メモリを空けています（WebUIのモデルを解放中）")
        memory_note = free_memory(True)["note"]

    report(0.02, "画像を集めています")
    sources, temp_root = collect_input_images(files, folder_path)
    tagger = None
    try:
        # --- 出力先の用意（前回の結果は1世代だけ退避して残す） -----------------
        root = dataset_root() / dataset_name
        backup = root.with_name(root.name + "__prev")
        if root.exists():
            shutil.rmtree(backup, ignore_errors=True)
            root.rename(backup)
        original_dir = root / "original"
        prepared_dir = root / "prepared"
        original_caption_dir = root / "captions_original"
        meta_dir = root / "meta"
        for d in (original_dir, prepared_dir, original_caption_dir, meta_dir):
            d.mkdir(parents=True, exist_ok=True)

        # --- 画像チェック -------------------------------------------------------
        infos: list[ImageInfo] = []
        for i, src in enumerate(sources):
            stem = f"{i:04d}_{_safe_name(src.stem)[:60]}"
            infos.append(_probe_image(src, stem))
            report(0.02 + 0.13 * (i + 1) / len(sources), f"画像をチェック中 {i + 1}/{len(sources)}")
            try:
                shutil.copy2(src, original_dir / f"{stem}{src.suffix.lower()}")  # 元データ保護
            except Exception:
                pass
        _find_near_duplicates(infos)

        accepted = [x for x in infos if x.ok]
        excluded = [x for x in infos if not x.ok]
        if not accepted:
            raise RuntimeError(
                "学習に使える画像がありませんでした（すべて小さすぎる/壊れている/重複でした）。"
            )

        prepared_paths: list[Path] = []
        for info in accepted:
            prepared_paths.append(_write_prepared_image(info, prepared_dir))

        # --- WD14タグ付け -------------------------------------------------------
        tagger = WD14Tagger()
        raw_results = tagger.tag_images(
            prepared_paths, batch_size=8,
            progress=lambda f, d: report(0.15 + 0.65 * f, d),
        )
        provider = tagger.provider
        tagger.close()          # 学習の前にVRAMを空ける
        tagger = None
        gc.collect()

        report(0.82, "タグを整理しています")
        threshold_used = float(general_threshold)
        threshold_note = ""
        if auto_threshold:
            threshold_used, threshold_note = auto_general_threshold(
                raw_results, float(general_threshold), float(character_threshold)
            )
        selected = [select_items(r, threshold_used, float(character_threshold)) for r in raw_results]

        # 重複タグを除いたセット（出現率の計算用）とカテゴリ表
        tag_category: dict[str, str] = {}
        per_image_sets: list[set[str]] = []
        for items in selected:
            names = set()
            for it in items:
                names.add(it.name)
                tag_category.setdefault(it.name, classify_tag(it.name, it.category))
            per_image_sets.append(names)

        decisions = decide_tags(per_image_sets, tag_category, preset, manual_keep, manual_remove)

        # --- キャプション出力 -----------------------------------------------------
        image_rows = []
        captions: dict[str, str] = {}
        empty_captions = 0
        total_kept = 0
        for info, dst, raw, items in zip(accepted, prepared_paths, raw_results, selected):
            kept, removed, seen = [], [], set()
            for it in items:                      # スコア順
                if it.name in seen:
                    continue
                seen.add(it.name)
                (kept if decisions[it.name].keep else removed).append(it.name)

            caption = ", ".join(to_caption_tag(t) for t in kept)
            captions[dst.name] = caption
            if not kept:
                empty_captions += 1
            total_kept += len(kept)

            # 注意: トリガーワードはここには書かない。
            # TrainTrainのdataset.pyが lora_trigger_word を先頭に自動で付けるため。
            (prepared_dir / f"{dst.stem}.txt").write_text(caption, encoding="utf-8")

            # 生の推論結果（スコア付き）も監査用に保存
            with open(original_caption_dir / f"{info.stem}.json", "w", encoding="utf-8") as f:
                json.dump(
                    [{"tag": x.name, "category": x.category_name,
                      "score": round(x.score, 5)} for x in raw],
                    f, ensure_ascii=False, indent=2,
                )
            (original_caption_dir / f"{info.stem}.txt").write_text(
                ", ".join(to_caption_tag(x.name) for x in items), encoding="utf-8",
            )
            image_rows.append([
                dst.name, caption,
                ", ".join(to_caption_tag(t) for t in removed),
            ])

        report(0.92, "レポートを作っています")
        n_used = len(accepted)
        small = [x for x in accepted if min(x.width, x.height) < MIN_SIDE_WARN]
        avg_tags = total_kept / n_used if n_used else 0.0

        # タグ一覧表（出現数の多い順）
        ordered = sorted(decisions.values(), key=lambda d: (-d.count, d.tag))
        tag_rows = [
            [to_caption_tag(d.tag), CATEGORY_LABELS_JA.get(d.category, d.category),
             d.count, f"{d.ratio:.0%}", "残す" if d.keep else "削除", d.reason]
            for d in ordered[:TABLE_MAX_TAGS]
        ]

        # 「未分類なのにほぼ全画像に出ている残ったタグ」＝そのキャラ/物の特徴かもしれない
        suspicious = [
            d for d in ordered
            if d.keep and d.category == "other" and d.ratio >= 0.7 and n_used >= MIN_IMAGES_FOR_RATIO
        ][:8]

        with open(meta_dir / "tag_decisions.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tag", "category", "count", "ratio", "decision", "reason"])
            for d in ordered:
                w.writerow([to_caption_tag(d.tag), d.category, d.count, round(d.ratio, 3),
                            "keep" if d.keep else "remove", d.reason])
        with open(meta_dir / "excluded.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "reason"])
            for x in excluded:
                w.writerow([x.src.name, x.reason])
        with open(meta_dir / "review.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "prepared_caption", "removed_tags"])
            w.writerows(image_rows)

        stats = {
            "dataset": dataset_name,
            "preset": preset_name,
            "trigger": trigger,
            "images_input": len(infos),
            "images_used": n_used,
            "excluded": [[x.src.name, x.reason] for x in excluded],
            "small_images": len(small),
            "exif_rotated": sum(1 for x in accepted if x.exif_rotated),
            "tagger_provider": provider,
            "general_threshold": threshold_used,
            "character_threshold": float(character_threshold),
            "threshold_note": threshold_note,
            "avg_tags_per_image": round(avg_tags, 1),
            "empty_captions": empty_captions,
            "kept_tag_types": sum(1 for d in decisions.values() if d.keep),
            "removed_tag_types": sum(1 for d in decisions.values() if not d.keep),
            "suspicious_common_tags": [to_caption_tag(d.tag) for d in suspicious],
            "output": str(prepared_dir),
        }
        (meta_dir / "summary.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        steps = auto_steps(n_used, preset_name)
        gallery = _make_previews(prepared_dir, captions, dataset_name)
        zip_path = _make_zip(prepared_dir, dataset_name)
        report(1.0, "完了")

        return {
            "prepared_dir": str(prepared_dir),
            "trigger": trigger,
            "stats": stats,
            "checkup": build_checkup(stats, preset_name),
            "gallery": gallery,
            "tag_rows": tag_rows,
            "image_rows": image_rows[:100],
            "stats_json": json.dumps(stats, ensure_ascii=False, indent=2),
            "zip_path": zip_path,
            "image_count": n_used,
            "auto_steps": steps,
            "memory_note": memory_note,
        }
    finally:
        if tagger is not None:
            tagger.close()
        shutil.rmtree(temp_root, ignore_errors=True)


def build_checkup(stats: dict, preset_name: str) -> str:
    """初心者向けの診断。✅=問題なし ⚠️=確認推奨 ℹ️=お知らせ"""
    n = stats["images_used"]
    lines: list[str] = []

    if n < 8:
        lines.append(f"⚠️ **画像が{n}枚しかありません。** 8〜10枚でも学習はできますが、"
                     "15〜30枚あると安定します。画風・衣装はもう少し多めがおすすめです。")
    elif n < 15:
        lines.append(f"ℹ️ 画像は{n}枚です。学習はできます。15枚以上あるとさらに安定します。")
    else:
        lines.append(f"✅ 画像は{n}枚です。十分な枚数です。")

    for name, reason in stats["excluded"][:6]:
        lines.append(f"ℹ️ 自動で除外: `{name}` — {reason}")
    if len(stats["excluded"]) > 6:
        lines.append(f"ℹ️ ほかにも{len(stats['excluded']) - 6}枚を自動で除外しました（`meta/excluded.csv`）。")

    if stats["small_images"]:
        lines.append(f"⚠️ 短辺{MIN_SIDE_WARN}px未満の画像が{stats['small_images']}枚あります。"
                     "ぼやけて学習される可能性があります。")
    if stats["exif_rotated"]:
        lines.append(f"ℹ️ {stats['exif_rotated']}枚はスマホ等の回転情報を反映して向きを直しました。")

    if stats["empty_captions"]:
        lines.append(f"⚠️ タグが1つも残らなかった画像が{stats['empty_captions']}枚あります。")
    avg = stats["avg_tags_per_image"]
    if avg < 5:
        lines.append(f"⚠️ 1枚あたりのタグが平均{avg}個と少なめです。"
                     "「詳細設定」でWD14のしきい値を下げてみてください。")
    else:
        lines.append(f"✅ 1枚あたり平均{avg}個のタグが残りました。")
    if stats["threshold_note"]:
        lines.append(f"ℹ️ {stats['threshold_note']}")

    if stats["suspicious_common_tags"]:
        tags = ", ".join(f"`{t}`" for t in stats["suspicious_common_tags"])
        lines.append(f"⚠️ ほぼ全画像に共通する未分類タグが残っています: {tags}。"
                     "学習したい対象の特徴なら、「詳細設定 → 必ず削除するタグ」に入れてください。")

    if "CUDA" not in stats["tagger_provider"] and n > 100:
        lines.append("ℹ️ タグ付けはCPUで動きました。枚数が多いと時間がかかります。")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# メモリの解放
# ---------------------------------------------------------------------------
# Google Colab は、メモリを使い切るとログも出さずにプロセスを強制終了する。
# WebUIで画像を生成した後は、モデルがメモリ(RAM/VRAM)に残っているので、
# 準備・学習の前にできる限り空けておく。

def _memory_snapshot() -> dict:
    """このプロセスのメモリ使用量(RSS)と、マシン全体の空きメモリ。取得できなければ空の辞書。"""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "rss": psutil.Process().memory_info().rss / 1024 ** 3,
            "avail": vm.available / 1024 ** 3,
            "total": vm.total / 1024 ** 3,
        }
    except Exception:
        pass
    try:                                    # psutil が無い環境（Linuxのみ）
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) * 1024
        rss = 0
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    rss = int(line.split()[1]) * 1024
        return {"rss": rss / 1024 ** 3, "avail": info["MemAvailable"] / 1024 ** 3,
                "total": info["MemTotal"] / 1024 ** 3}
    except Exception:
        return {}


def _fmt_gb(value: float) -> str:
    return f"{value:.1f}GB" if value >= 1 else f"{value * 1024:.0f}MB"


def free_memory(unload_webui_model: bool = True) -> dict:
    """可能な限りメモリを空ける。失敗しても例外は出さず、できた範囲で続行する。

    unload_webui_model=True の時は、WebUIが読み込んでいるモデルも解放する
    （次に画像を生成する時に、WebUIが自動で読み込み直す）。
    戻り値: {"before": GB, "after": GB, "avail": GB, "total": GB, "note": 画面用の説明}
    """
    before = _memory_snapshot()

    if unload_webui_model:
        try:
            from modules import sd_models
            unload = getattr(sd_models, "unload_model_weights", None)
            if unload is not None:
                unload()
            cache = getattr(sd_models, "checkpoints_loaded", None)   # A1111のチェックポイントキャッシュ
            if hasattr(cache, "clear"):
                cache.clear()
        except Exception as e:
            print(f"[Easy LoRA] WebUIのモデル解放をスキップ: {type(e).__name__}: {e}")
        try:
            from backend import memory_management as mm              # Forge系
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception:
            pass
        try:
            from modules import devices                              # A1111系
            devices.torch_gc()
        except Exception:
            pass

    for _ in range(3):
        gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    try:                                    # glibcが保持している空きメモリをOSへ返す（Linux/Colab）
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    after = _memory_snapshot()
    note = ""
    if before and after:
        note = (f"メモリ使用量 {_fmt_gb(before['rss'])} → {_fmt_gb(after['rss'])}"
                f"（マシンの空き {_fmt_gb(after['avail'])} / {_fmt_gb(after['total'])}）")
        print(f"[Easy LoRA] {note}")
    return {"before": before.get("rss"), "after": after.get("rss"),
            "avail": after.get("avail"), "total": after.get("total"), "note": note}


# ---------------------------------------------------------------------------
# TrainTrain のメッセージを日本語にする
# ---------------------------------------------------------------------------

_TRAIN_MESSAGES = [
    (re.compile(r"^Stopped\. Successfully created to (.+)$"),
     lambda m: f"学習を途中で止めて、ここまでの結果を保存しました: {m.group(1)}"),
    (re.compile(r"^Stopped$"), lambda m: "学習を途中で停止しました（保存はしていません）"),
    (re.compile(r"^Successfully created to (.+)$"), lambda m: f"LoRAを保存しました: {m.group(1)}"),
    (re.compile(r"^File exist!$"), lambda m: "同じ名前のLoRAファイルが既にあります"),
    (re.compile(r"^No Model Selected\.?$"), lambda m: "モデルが選択されていません"),
    (re.compile(r"^No data!?$"),
     lambda m: "学習に使える画像がありません（画像が空、小さすぎる、または解像度の条件に合っていない可能性があります）"),
    (re.compile(r"^Test mode$"), lambda m: "テストモードのため、学習は行いませんでした"),
    (re.compile(r"^Not save copy$"), lambda m: "コピーは保存しませんでした"),
    (re.compile(r"^Preset saved$"), lambda m: "プリセットを保存しました"),
    (re.compile(r"^Added to Queue$"), lambda m: "キューに追加しました"),
    (re.compile(r"^Duplicated LoRA name! Could not add to queue\.?$"),
     lambda m: "同じ名前のLoRAが既にキューにあるため、追加できませんでした"),
    (re.compile(r"^(.+) can only be trained from inside Forge Neo$"),
     lambda m: f"{m.group(1)} はForge Neoの中でのみ学習できます"),
]

_ERROR_HINTS = [
    (re.compile(r"out of memory|CUDA error: out of memory|OOM", re.I),
     "GPUメモリ(VRAM)が足りませんでした。「詳細設定」で学習解像度を下げて、もう一度お試しください。"),
    (re.compile(r"No such file|FileNotFoundError|not found", re.I),
     "必要なファイルが見つかりませんでした。モデルやVAE・Text Encoderの指定を確認してください。"),
]


def translate_train_message(text: str) -> str:
    """TrainTrainが返す英語のメッセージを日本語にする。未知のメッセージは原文のまま残す。"""
    out_lines = []
    for line in str(text or "").splitlines() or [""]:
        s = line.strip()
        translated = None
        for pattern, fn in _TRAIN_MESSAGES:
            m = pattern.match(s)
            if m:
                translated = fn(m)
                break
        if translated is None and s.startswith("Error:"):
            detail = s[len("Error:"):].strip()
            translated = f"エラーが発生しました: {detail}"
            for pattern, hint in _ERROR_HINTS:
                if pattern.search(detail):
                    translated += f"\n{hint}"
                    break
        out_lines.append(translated if translated is not None else line)
    return "\n".join(out_lines)


def find_saved_lora(result: str, save_dir: Optional[str], name: str, started: float) -> Optional[str]:
    """学習で保存されたLoRAのパスを探す。

    完了時は「名前.safetensors」、途中で止めて保存した時は「名前_123steps.safetensors」になるので、
    TrainTrainのメッセージ(Successfully created to ...)から取るのが確実。
    """
    m = re.search(r"Successfully created to (.+?\.safetensors)", str(result or ""))
    if m and os.path.isfile(m.group(1).strip()):
        return m.group(1).strip()
    if save_dir and os.path.isdir(save_dir):
        candidates = [
            p for p in Path(save_dir).glob(f"{name}*.safetensors")
            if p.stat().st_mtime >= started - 2
        ]
        if candidates:
            return str(max(candidates, key=lambda p: p.stat().st_mtime))
    return None


def prepare_download(path: str) -> Optional[str]:
    """ダウンロード用にコピーを作る。LoRAの保存先がGradioの配信範囲外でも確実に取れるようにする。"""
    try:
        out_dir = Path(tempfile.gettempdir()) / "easylora_download"
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / Path(path).name
        shutil.copy2(path, dst)
        return str(dst)
    except Exception as e:
        print(f"[Easy LoRA] ダウンロード用コピーに失敗: {e}")
        return None


# ---------------------------------------------------------------------------
# モデルの自動ダウンロード
# ---------------------------------------------------------------------------

@dataclass
class CompanionFile:
    """モデル本体とは別に必要なファイル（分割配布のモデルのVAE・Text Encoder）。"""
    kind: str             # "vae" | "text_encoder"
    label: str            # 画面表示用
    repo: str
    path: str

    @property
    def filename(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.path}"


# Anima は本体(diffusion model)だけの分割配布。VAEとText Encoderも要る。
ANIMA_COMPANIONS = (
    CompanionFile("vae", "VAE", "circlestone-labs/Anima", "split_files/vae/qwen_image_vae.safetensors"),
    CompanionFile("text_encoder", "Text Encoder", "circlestone-labs/Anima",
                  "split_files/text_encoders/qwen_3_06b_base.safetensors"),
)


@dataclass
class CatalogModel:
    label: str            # ドロップダウンに出す名前
    repo: str             # Hugging Face のリポジトリ
    path: str             # リポジトリ内のファイルパス
    size_gb: float        # 目安（画面表示用。実際のサイズはダウンロード時に取得）
    companions: tuple = ()        # 一緒に必要なVAE・Text Encoder

    @property
    def needs_modules(self) -> bool:
        return bool(self.companions)

    @property
    def filename(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.path}"


MODEL_CATALOG = [
    CatalogModel(
        label="★ Illustrious-XL v2.0（約6.9GB・無ければ自動ダウンロード）",
        repo="OnomaAIResearch/Illustrious-XL-v2.0",
        path="Illustrious-XL-v2.0.safetensors",
        size_gb=6.94,
    ),
    CatalogModel(
        label="★ Anima base v1.0（約4.2GB・無ければ自動ダウンロード）",
        repo="circlestone-labs/Anima",
        path="split_files/diffusion_models/anima-base-v1.0.safetensors",
        size_gb=4.18,
        companions=ANIMA_COMPANIONS,
    ),
]
CATALOG_BY_LABEL = {m.label: m for m in MODEL_CATALOG}


def _norm_stem(name: str) -> str:
    """ファイル名の表記ゆれをなくす。anima_baseV10 と anima-base-v1.0 を同じとみなすため。"""
    return re.sub(r"[^a-z0-9]", "", Path(name).stem.lower())


def checkpoint_dir() -> Path:
    """モデルを保存するフォルダ（WebUIのチェックポイントフォルダ）。"""
    try:
        from modules import sd_models
        p = getattr(sd_models, "model_path", None)
        if p:
            return Path(p)
    except Exception:
        pass
    try:
        from modules import shared
        p = getattr(shared.cmd_opts, "ckpt_dir", None)
        if p:
            return Path(p)
    except Exception:
        pass
    p = _extension_root() / "models" / "checkpoints"
    return p


def find_installed_model(spec: CatalogModel) -> Optional[Path]:
    """既にあるモデルを探す。名前の表記ゆれ(記号・大文字小文字)は無視する。"""
    wanted = _norm_stem(spec.filename)
    exts = {".safetensors", ".ckpt", ".sft"}
    candidates: list[Path] = []
    try:
        from modules import sd_models
        for info in list(getattr(sd_models, "checkpoints_list", {}).values()):
            fn = getattr(info, "filename", None)
            if fn:
                candidates.append(Path(fn))
    except Exception:
        pass
    root = checkpoint_dir()
    if root.is_dir():
        candidates.extend(p for p in root.rglob("*") if p.suffix.lower() in exts)
    for p in candidates:
        if p.suffix.lower() in exts and _norm_stem(p.name) == wanted and p.is_file():
            if is_valid_safetensors(p):
                return p
    return None


def is_valid_safetensors(path) -> bool:
    """ヘッダだけを見て、途切れた/壊れたファイルを弾く（全体は読まないので速い）。"""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return False
            header_len = int.from_bytes(raw, "little")
            if header_len <= 0 or header_len > 100 * 1024 * 1024 or 8 + header_len > size:
                return False
            header = json.loads(f.read(header_len))
        end = 0
        for k, v in header.items():
            if k != "__metadata__":
                end = max(end, v["data_offsets"][1])
        return 8 + header_len + end <= size
    except Exception:
        return False


def download_file(url: str, dest: Path, progress_cb: Optional[Callable[[int, int, float], None]] = None,
                  retries: int = 5, chunk: int = 256 * 1024) -> Path:
    """URLをdestへ保存する。途中から再開でき、サイズが合わなければ再試行する。

    - 保存中は dest + ".part" に書き、完了してから本来の名前に変える（途中のファイルを使わせない）
    - 失敗しても .part は残すので、もう一度実行すると続きから再開する
    progress_cb(取得済みバイト, 全体バイト(不明なら0), 速度バイト/秒)
    """
    import requests

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    headers_base = {"User-Agent": "EasyLoRA/1.0"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers_base["Authorization"] = f"Bearer {token}"

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            headers = dict(headers_base)
            if have:
                headers["Range"] = f"bytes={have}-"
            with requests.get(url, headers=headers, stream=True, timeout=(15, 60),
                              allow_redirects=True) as r:
                if r.status_code in (401, 403):
                    raise RuntimeError(
                        "このモデルはログインが必要か、アクセスが許可されていません。"
                        "環境変数 HF_TOKEN にHugging Faceのトークンを設定してください。")
                if r.status_code == 404:
                    raise RuntimeError(f"モデルのURLが見つかりません（404）: {url}")
                if r.status_code == 416:               # 既に全部取得済み
                    total = have
                else:
                    r.raise_for_status()
                    if r.status_code == 200 and have:  # 再開を断られた → 最初から
                        have = 0
                        part.unlink()
                    if r.status_code == 206:
                        m = re.search(r"/(\d+)$", r.headers.get("Content-Range", ""))
                        total = int(m.group(1)) if m else have + int(r.headers.get("Content-Length", 0))
                    else:
                        total = int(r.headers.get("Content-Length", 0))
                    try:
                        free = shutil.disk_usage(dest.parent).free
                        if total and total - have > free * 0.98:
                            raise RuntimeError(
                                f"ディスクの空き容量が足りません（必要 約{(total - have) / 1024 ** 3:.1f}GB / "
                                f"空き {free / 1024 ** 3:.1f}GB）。")
                    except OSError:
                        pass

                    done, t0, t_last, speed = have, time.time(), 0.0, 0.0
                    if progress_cb:
                        progress_cb(done, total, 0.0)   # 開始を知らせる（表示の切り替えと、停止の判定を即座に効かせる）
                    with open(part, "ab" if have else "wb") as f:
                        for block in r.iter_content(chunk_size=chunk):
                            if not block:
                                continue
                            f.write(block)
                            done += len(block)
                            now = time.time()
                            if progress_cb and now - t_last >= 0.5:
                                speed = (done - have) / max(now - t0, 1e-6)
                                progress_cb(done, total, speed)
                                t_last = now
            size = part.stat().st_size
            if total and size != total:
                raise IOError(f"ダウンロードが途中で切れました（{size}/{total}バイト）")
            os.replace(part, dest)
            if progress_cb:
                progress_cb(size, size, 0.0)
            return dest
        except RuntimeError:
            raise
        except Exception as e:                          # ネットワークの一時的な失敗は再試行
            last_error = e
            print(f"[Easy LoRA] ダウンロード失敗({attempt}/{retries}): {type(e).__name__}: {e}")
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(
        "モデルのダウンロードに失敗しました。インターネット接続を確認して、もう一度実行してください"
        f"（途中まで保存してあるので、続きから再開します）。詳細: {type(last_error).__name__}: {last_error}")


def refresh_webui_checkpoints():
    """ダウンロードしたモデルをWebUIの一覧に反映する。"""
    try:
        from modules import sd_models
        sd_models.list_models()
    except Exception as e:
        print(f"[Easy LoRA] チェックポイント一覧の更新をスキップ: {type(e).__name__}: {e}")


def checkpoint_name_for(path: Path) -> str:
    """TrainTrainに渡すモデル名。WebUIの一覧にあればそのタイトル、無ければファイルのパス。"""
    try:
        from modules import sd_models
        target = os.path.abspath(str(path))
        for info in sd_models.checkpoints_list.values():
            if os.path.abspath(getattr(info, "filename", "")) == target:
                return info.title
    except Exception:
        pass
    return str(path)


def looks_like_anima(model_value: str) -> bool:
    """ドロップダウンで選ばれたモデルが Anima か。名前で判定する（animagine など別物は除く）。"""
    stem = _norm_stem(str(model_value or "").split(" [")[0])
    return bool(re.match(r"^anima(?!gine)", stem))


def companions_for(model_value: str) -> tuple:
    """このモデルに必要なVAE・Text Encoder。Anima以外は空（何もしない）。"""
    spec = CATALOG_BY_LABEL.get(model_value)
    if spec is not None:
        return spec.companions
    return ANIMA_COMPANIONS if looks_like_anima(model_value) else ()


def module_dirs(kind: str) -> list:
    """VAE / Text Encoder を置くフォルダ（先頭が、ダウンロードの保存先）。"""
    sub = "VAE" if kind == "vae" else "text_encoder"
    dirs: list = []
    try:
        from modules import paths, shared
        dirs.append(Path(paths.models_path) / sub)
        extra = getattr(shared.cmd_opts, "vae_dirs" if kind == "vae" else "text_encoder_dirs", None) or []
        dirs.extend(Path(x) for x in extra)
    except Exception:
        dirs.append(_extension_root() / "models" / sub)
    return dirs


def find_installed_companion(comp: CompanionFile) -> Optional[Path]:
    """既にあるVAE/Text Encoderを探す（名前の表記ゆれは無視。壊れたファイルは無視）。"""
    wanted = _norm_stem(comp.filename)
    for d in module_dirs(comp.kind):
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".safetensors", ".sft") \
                    and _norm_stem(p.name) == wanted and is_valid_safetensors(p):
                return p
    return None


def ensure_companions(companions, progress_cb: Optional[Callable[[CompanionFile, int, int, float], None]] = None,
                      refresh: bool = True) -> list:
    """必要なVAE・Text Encoderを揃える。無ければダウンロードする。戻り値はファイルのパスのリスト。"""
    result = []
    downloaded = False
    for comp in companions:
        found = find_installed_companion(comp)
        if found is None:
            dest = module_dirs(comp.kind)[0] / comp.filename
            download_file(comp.url, dest,
                          (lambda d, t, sp, c=comp: progress_cb(c, d, t, sp)) if progress_cb else None)
            if not is_valid_safetensors(dest):
                dest.unlink(missing_ok=True)
                raise RuntimeError(f"ダウンロードした{comp.label}のファイルが壊れていました。もう一度実行してください。")
            found = dest
            downloaded = True
        result.append(found)
    if downloaded and refresh:
        try:                                   # Forge Neo: VAE / Text Encoder の一覧を更新
            from modules_forge import main_entry
            main_entry.refresh_models()
        except Exception:
            pass
    return result


@contextlib.contextmanager
def use_webui_modules(paths: list):
    """学習の間だけ、WebUIの「VAE / Text Encoder」の選択を paths にする。終わったら元に戻す。

    TrainTrain は学習開始時に shared.opts.forge_additional_modules を読んでモデルを読み込む。
    ユーザーの選択を書き換えたままにしないよう、必ず元に戻す（Forge以外では何もしない）。
    """
    try:
        from modules import shared
        has = hasattr(shared.opts, "forge_additional_modules")
    except Exception:
        has = False
    if not has:
        yield False
        return

    def norm(items):
        return sorted(os.path.normpath(str(x)) for x in items)

    previous = list(shared.opts.forge_additional_modules)
    wanted = norm(paths)
    changed = norm(previous) != wanted
    if changed:
        shared.opts.set("forge_additional_modules", wanted)
    try:
        yield changed
    finally:
        if changed:
            shared.opts.set("forge_additional_modules", previous)
            try:       # 次の画像生成が、画面の選択どおりに読み込まれるようにする
                from modules_forge.main_entry import refresh_model_loading_parameters
                refresh_model_loading_parameters(refresh=True)
            except Exception:
                pass


def resolve_model(model_value: str, progress_cb: Optional[Callable[[int, int, float], None]] = None) -> str:
    """ドロップダウンの値を、TrainTrainに渡せるモデル名にする。

    ★付きの候補は、既にあればそれを使い、無ければダウンロードする。通常のモデルはそのまま返す。
    """
    spec = CATALOG_BY_LABEL.get(model_value)
    if spec is None:
        return model_value
    installed = find_installed_model(spec)
    if installed is None:
        dest = checkpoint_dir() / spec.filename
        download_file(spec.url, dest, progress_cb)
        if not is_valid_safetensors(dest):
            dest.unlink(missing_ok=True)
            raise RuntimeError("ダウンロードしたファイルが壊れていました。もう一度実行してください。")
        installed = dest
        refresh_webui_checkpoints()
    return checkpoint_name_for(installed)


# ---------------------------------------------------------------------------
# TrainTrain との連携
# ---------------------------------------------------------------------------

def guess_image_size(model_name: str) -> int:
    """モデル名（と、わかればファイルサイズ）から学習解像度を推定する。"""
    s = (model_name or "").lower()
    keys = ("sdxl", "xl", "pony", "illustrious", "noob", "flux", "z-image", "zimage",
            "anima", "krea", "sd3")
    if any(k in s for k in keys):
        return 1024
    # 名前でわからない時はファイルサイズで判定（SDXL系は約6GB以上、SD1.5は約2〜4GB）
    try:
        path = None
        if os.path.isfile(model_name):
            path = model_name
        else:
            from modules import sd_models  # A1111 / Forge
            info = sd_models.get_closet_checkpoint_match(model_name)
            path = getattr(info, "filename", None)
        if path and os.path.getsize(path) > 5.0e9:
            return 1024
    except Exception:
        pass
    return 512


def resolve_image_size(choice, model_name: str) -> int:
    c = str(choice if choice is not None else "").strip()
    return int(c) if c.isdigit() else guess_image_size(model_name)


def _pick_model_precision() -> str:
    """bf16が使えるGPUならbf16、そうでなければfp16（Colab無料のT4など）。"""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bf16"
    except Exception:
        pass
    return "fp16"


def _free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _unique_lora_name(trainer_module, base: str) -> str:
    """同名ファイルがあると TrainTrain が "File exist!" で止まるので、連番を付けて避ける。"""
    base = _safe_name(base)
    save_dir = getattr(trainer_module, "lora_dir", None)
    if not save_dir:
        return base
    name, n = base, 2
    while os.path.exists(os.path.join(save_dir, f"{name}.safetensors")):
        name = f"{base}_{n}"
        n += 1
    return name


def _config_index(configs) -> dict:
    """設定名 -> 位置。"image_size(height, width)" のような括弧付きの名前は、
    括弧付きのままでも、括弧より前だけでも引けるようにする。"""
    index = {}
    for i, cfg in enumerate(configs):
        full = cfg[0]
        index[full] = i
        index.setdefault(full.split("(")[0].strip(), i)
    return index


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
    preset = PRESETS.get(preset_name) or PRESETS[DEFAULT_PRESET]

    # TrainTrainの既定値をベースに、初心者向けの項目だけ上書きする。
    overrides = {
        "network_type": "lierla",
        "network_rank": str(preset["rank"]),
        "network_alpha": "8",
        "lora_data_directory": prepared_dir,
        "lora_trigger_word": (trigger or "").strip(),
        "image_size": str(int(image_size)),
        "train_iterations": int(steps),
        "train_batch_size": 1,
        "train_learning_rate": "1e-4",
        "train_optimizer": "AdamW",
        "train_lr_scheduler": "cosine",
        "save_lora_name": _safe_name(output_name),
        "use_gradient_checkpointing": True,
        "image_buckets_step": "256",
        "image_min_length": 512,
        "image_max_ratio": 2,
        "train_snr_gamma": 5,
        "train_seed": -1,
        "train_model_precision": _pick_model_precision(),
        "train_lora_precision": "fp32",
        "train_VAE_precision": "fp32",
        "image_shuffle_tags": True,
        "train_self_reg": 0,
    }

    index = _config_index(configs)
    for key, value in overrides.items():
        if key in index:
            first[index[key]] = value
            second[index[key]] = value
        else:
            # 黙って無視しない（TrainTrain側の設定名が変わった時に気づけるように）
            print(f"[Easy LoRA] 警告: TrainTrainに設定 '{key}' が見つかりません。既定値のまま学習します。")

    # prompts(3つ) + images(2つ) は標準LoRAでは使わない
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
    free_mem: bool = False,
) -> tuple[str, Optional[str]]:
    """学習を実行する。戻り値は (画面に出すメッセージ, 保存されたLoRAのパス or None)。"""
    if not prepared_dir or not os.path.isdir(prepared_dir):
        return "❌ 先に「準備」を実行してください（学習用データが見つかりません）。", None
    if not model:
        return "❌ モデルを選択してください。", None

    name = _unique_lora_name(trainer_module, output_name or "my_lora")
    started = time.time()

    def run(lora_name: str) -> str:
        values = build_train_values(
            trainer_module, prepared_dir, trigger, lora_name, preset_name,
            int(image_size), int(steps),
        )
        return str(train_module.train(False, "LoRA", model, vae or "None", te or "None", *values))

    if free_mem:
        free_memory(True)
    else:
        _free_gpu()
    try:
        result = run(name)
        if "File exist" in result:
            name = f"{name}_{time.strftime('%m%d%H%M')}"
            result = run(name)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (f"❌ 学習を開始できませんでした: {type(e).__name__}: {e}\n\n"
                "GPUメモリ(VRAM)が足りない場合は、「詳細設定」で学習解像度を下げるか、"
                "より軽いモデルを選んでください。"), None

    save_dir = getattr(trainer_module, "lora_dir", None)
    path = find_saved_lora(result, save_dir, name, started)
    translated = translate_train_message(result)
    if path:
        stem = Path(path).stem
        stopped = str(result).lstrip().startswith("Stopped")
        title = "### ⏹ 学習を途中で止めて、ここまでを保存しました" if stopped else "### ✅ 学習完了"
        if (trigger or "").strip():
            usage = f"プロンプトに `{trigger}` を入れて、`<lora:{stem}:1>` を追加"
        else:
            usage = f"プロンプトに `<lora:{stem}:1>` を追加するだけで使えます（トリガーワードなし）"
        message = (
            f"{title}\n\n"
            f"- 出力: `{Path(path).name}`\n"
            f"- 使い方: {usage}\n"
            f"- 効きすぎ/弱すぎる時は `:1` を `:0.6` や `:1.2` に変えて調整\n"
            f"- 下の「ダウンロード」ボタンで、このファイルを手元に保存できます\n\n"
            f"{translated}"
        )
        return message, path
    return f"### 学習は完了しませんでした\n\n{translated}", None


# ---------------------------------------------------------------------------
# 進捗表示（準備・学習の両方で使う）
# ---------------------------------------------------------------------------

class ProgressTracker:
    """作業スレッドが書き込み、UI側（ジェネレータ）が読む、スレッドセーフな進捗の入れ物。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = {"title": "", "frac": None, "detail": ""}

    def set(self, title: str, frac: Optional[float] = None, detail: str = ""):
        with self._lock:
            self._state = {"title": title, "frac": frac, "detail": detail}

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)


def render_progress(snap: dict) -> str:
    """進捗バーのHTML。frac が None の時は「処理中」の動くバーにする。"""
    if not snap or not snap.get("title"):
        return ""
    frac = snap.get("frac")
    pct = None if frac is None else max(0, min(100, int(float(frac) * 100)))
    fill_cls = "tt-prog-fill" + (" tt-prog-indeterminate" if pct is None else "")
    width = 100 if pct is None else pct
    pct_text = "" if pct is None else f"{pct}%"
    esc = html_lib.escape
    return (
        '<div class="tt-prog">'
        f'<div class="tt-prog-head"><b>{esc(snap["title"])}</b><span>{pct_text}</span></div>'
        f'<div class="tt-prog-track"><div class="{fill_cls}" style="width:{width}%"></div></div>'
        f'<div class="tt-prog-detail">{esc(snap.get("detail", ""))}</div>'
        '</div>'
    )


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds != seconds or seconds < 0:
        return "計算中"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}秒"
    if seconds < 3600:
        return f"{seconds // 60}分{seconds % 60:02d}秒"
    return f"{seconds // 3600}時間{(seconds % 3600) // 60:02d}分"


def _report_tqdm(bar, tracker: ProgressTracker, total_steps: int):
    """TrainTrainのtqdmの状態を、画面用の進捗に変換する。"""
    n, total = int(bar.n), bar.total
    if not total:
        return
    if int(total) == int(total_steps):          # 本番の学習ループ
        info = bar.format_dict
        rate = info.get("rate")
        remain = (total - n) / rate if rate else None
        desc = getattr(bar, "desc", "") or ""
        parts = [f"{min(n, total)} / {total} step"]
        m = re.search(r"Epoch: (\d+)", desc)
        if m:
            parts.append(f"Epoch {m.group(1)}")
        m = re.search(r"Loss EMA \* 1000: ([0-9.]+)", desc)
        if m:
            parts.append(f"Loss {m.group(1)}")
        parts.append(f"経過 {format_duration(info.get('elapsed'))}")
        parts.append(f"残り約 {format_duration(remain)}")
        tracker.set("学習中", min(1.0, n / total), " ・ ".join(parts))
    else:                                        # 学習前の画像のlatent化など
        tracker.set("学習の準備中（画像の前処理）", min(1.0, n / total), f"{n} / {total} 枚")


@contextlib.contextmanager
def hook_training_progress(train_module, tracker: ProgressTracker, total_steps: int,
                           on_loop: Optional[Callable[[int], None]] = None):
    """学習の間だけ train.py の tqdm を「進捗を記録する版」に差し替える。

    train.py 本体は書き換えない（TrainTrain本家の更新を取り込みやすくするため）。
    元のtqdmの動作（コンソール表示）はそのまま残る。
    """
    base = getattr(train_module, "tqdm", None)
    if base is None:
        yield
        return

    class TrackedTqdm(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._tt_report()

        def update(self, n=1):
            result = super().update(n)
            self._tt_report()
            return result

        def _tt_report(self):
            try:
                _report_tqdm(self, tracker, total_steps)
                if on_loop is not None and self.total and int(self.total) == int(total_steps):
                    on_loop(int(self.n))          # 学習ループに入った/1step進んだ
            except Exception:
                pass          # 進捗表示の失敗で学習を止めない

    train_module.tqdm = TrackedTqdm
    try:
        yield
    finally:
        train_module.tqdm = base


# ---------------------------------------------------------------------------
# ジョブ実行（状態をサーバー側に持つ）
# ---------------------------------------------------------------------------
# 以前は「処理の進捗をブラウザとの長い接続に流し続ける」方式だったため、ブラウザのタブを
# 長く離れて接続が切れると、処理が終わっても画面の進捗が止まったままになることがあった。
#
# 今は、処理は別スレッドで進め、状態(進捗・結果)はすべてサーバー側に保存する。
# 画面は短い問い合わせ(gr.Timer)で状態を取りに来るだけなので、
#   - タブを離れて接続が切れても、戻った時の次の問い合わせで最新の状態に追いつく
#   - ブラウザをリロードしても、進行中/完了済みの状態がそのまま復元される
# 処理自体もブラウザの接続とは無関係に最後まで進む。

class JobCancelled(RuntimeError):
    """ユーザーが停止を押したため、処理を中止した。"""


KEEP = object()        # 「この出力は変更しない」の印（Gradioではgr.update()に変換する）

UI_KEYS = ("prep_progress", "train_progress", "status", "checkup", "gallery", "tag_rows",
           "image_rows", "stats", "zip_file", "prepared_dir", "steps", "train_result",
           "stop_visible", "download", "active")


def format_prep_status(res: dict, preset_name: str, will_train: bool) -> str:
    st = res["stats"]
    lines = [
        "### ✅ 学習データの準備ができました",
        "",
        f"- 使う画像: **{st['images_used']}枚**"
        + (f"（{len(st['excluded'])}枚は自動で除外）" if st["excluded"] else ""),
        f"- 種類: **{preset_name}**",
    ]
    if res["trigger"]:
        lines.append(f"- トリガーワード: `{res['trigger']}` ← 画像を生成する時、プロンプトに入れます")
    lines.append(f"- おすすめ学習step: 約 **{res['auto_steps']}**")
    if res.get("memory_note"):
        lines.append(f"- メモリを空けました: {res['memory_note']}")
    if will_train:
        lines += ["", "▶ 続けて学習を開始します。"]
    else:
        lines += ["", "内容を確認して、よければ下の「準備済みデータで学習だけ実行」を押してください。"]
    return "\n".join(lines)


def compose_ui(snap: dict, seen: Optional[dict]) -> tuple:
    """サーバー側の状態から、画面に反映する値を作る。

    seen は「この画面に、どこまで送ったか」の記録。変わっていない出力は KEEP にして送らない
    （ギャラリーや表を毎秒送り直さないため）。seen が空(初回・リロード)なら全部送る。
    戻り値: (出力の辞書, 更新後のseen)
    """
    seen = dict(seen or {})
    out = {k: KEEP for k in UI_KEYS}
    running, stage = snap["running"], snap["stage"]

    bar = render_progress(snap["progress"]) if running else ""
    prep_html = bar if stage == "prep" else ""
    train_html = bar if stage == "train" else ""
    if prep_html != seen.get("prep_html"):
        out["prep_progress"], seen["prep_html"] = prep_html, prep_html
    if train_html != seen.get("train_html"):
        out["train_progress"], seen["train_html"] = train_html, train_html

    if snap["prep_version"] != seen.get("prep_version"):
        seen["prep_version"] = snap["prep_version"]
        po = snap["prep_out"]
        if po is not None:
            for key in ("status", "checkup", "gallery", "tag_rows", "image_rows", "stats",
                        "zip_file", "prepared_dir", "steps"):
                if key in po:
                    out[key] = po[key]

    if snap["train_version"] != seen.get("train_version"):
        seen["train_version"] = snap["train_version"]
        out["train_result"] = snap["train_msg"]
        out["stop_visible"] = snap["stop_visible"]
        out["download"] = snap["download"]

    changed = any(v is not KEEP for k, v in out.items() if k != "active")
    out["active"] = bool(running or changed)      # 変化があった回の、もう1回後に問い合わせを止める
    return out, seen


class JobRunner:
    """準備・学習を別スレッドで実行し、進捗と結果をサーバー側に保持する。画面の接続とは独立。"""

    def __init__(self, train_module, trainer_module):
        self.train_module = train_module
        self.trainer_module = trainer_module
        self._lock = threading.RLock()
        self.tracker = ProgressTracker()
        self.cancel = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.stage = ""                  # "prep" | "train"
        self.loop_started = False        # 学習ループに入ったか
        self._stop_sent = False
        self.prep_version = 0
        self.prep_out: Optional[dict] = None
        self.train_version = 0
        self.train_msg = ""
        self.stop_visible = False
        self.download: dict = {"visible": False}

    # ---- 画面から見える状態 ----------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self.running, "stage": self.stage,
                "progress": self.tracker.snapshot(),
                "prep_version": self.prep_version, "prep_out": self.prep_out,
                "train_version": self.train_version, "train_msg": self.train_msg,
                "stop_visible": self.stop_visible, "download": dict(self.download),
            }

    def _set_prep(self, out: dict):
        with self._lock:
            self.prep_out = out
            self.prep_version += 1

    def _set_train(self, msg: str, stop_visible: bool, download: Optional[dict] = None):
        with self._lock:
            self.train_msg = msg
            self.stop_visible = stop_visible
            self.download = download or {"visible": False}
            self.train_version += 1

    # ---- 開始・停止 ------------------------------------------------------
    def start(self, plan: dict) -> Optional[str]:
        """plan = {"prep": prepare_datasetの引数 or None, "train": 学習の引数 or None}。
        既に実行中なら、画面に出すメッセージを返す（開始しない）。"""
        with self._lock:
            if self.running:
                return "### ⚠️ すでに別の処理を実行中です\n\n終わるまでお待ちください。"
            self.running = True
            self.cancel.clear()
            self.loop_started = False
            self._stop_sent = False
            self.stage = "prep" if plan.get("prep") else "train"
            self.tracker.set("準備中" if plan.get("prep") else "学習の準備中", None, "開始しています")
            self.thread = threading.Thread(target=self._run, args=(plan,), daemon=True)
            self.thread.start()
        return None

    def request_stop(self) -> str:
        with self._lock:
            if not self.running:
                return "実行中の処理はありません。"
            self.cancel.set()
            if self.loop_started and not self._stop_sent:
                self._stop_sent = True
                self.train_module.stop_time(True)
                return "停止を要求しました。次の区切りで、ここまでの結果を保存して止まります。"
        return ("停止を要求しました。モデルのダウンロード中なら中止します（次回は続きから再開します）。"
                "モデルの読み込み中なら、読み込みが終わった時点で、保存せずに止まります。")

    def _on_loop_step(self, n: int):
        """学習ループが1stepごとに呼ぶ。読み込み中に押された停止は、ここで反映する。"""
        with self._lock:
            self.loop_started = True
            if self.cancel.is_set() and not self._stop_sent:
                self._stop_sent = True
                self.train_module.stop_time(False)   # 学習前の停止は、1stepだけのLoRAを残さない

    def join(self, timeout: Optional[float] = None):
        if self.thread is not None:
            self.thread.join(timeout)

    # ---- 本体 ------------------------------------------------------------
    def _run(self, plan: dict):
        try:
            prepared_dir = (plan.get("train") or {}).get("prepared_dir")
            steps = (plan.get("train") or {}).get("steps")
            if plan.get("prep"):
                res = self._do_prep(plan["prep"], will_train=bool(plan.get("train")))
                if res is None:
                    return
                prepared_dir, steps = res["prepared_dir"], res["auto_steps"]
            if plan.get("train"):
                self._do_train(plan["train"], prepared_dir, steps)
        except BaseException as e:           # noqa: BLE001  ここで必ず記録して、画面に出す
            self._set_train(f"### ❌ 予期しないエラー\n\n{_friendly_error(e)}", False)
        finally:
            with self._lock:
                self.running = False
                self.stage = ""
                self.tracker.set("")
                if self.stop_visible:
                    self.stop_visible = False
                    self.train_version += 1

    def _do_prep(self, kwargs: dict, will_train: bool) -> Optional[dict]:
        self._set_prep({"status": "", "checkup": "", "gallery": None})   # 前回の結果を消す
        self.tracker.set("準備中", None, "画像を集めています")
        try:
            res = prepare_dataset(
                **kwargs,
                progress=lambda frac, desc="": self.tracker.set("準備中", frac, desc),
            )
        except Exception as e:
            self._set_prep({"status": f"### ❌ 準備に失敗しました\n\n{_friendly_error(e)}"})
            return None
        self._set_prep({
            "status": format_prep_status(res, kwargs["preset_name"], will_train),
            "checkup": res["checkup"], "gallery": res["gallery"], "tag_rows": res["tag_rows"],
            "image_rows": res["image_rows"], "stats": res["stats_json"], "zip_file": res["zip_path"],
            "prepared_dir": res["prepared_dir"], "steps": res["auto_steps"],
        })
        return res

    def _check_cancel(self, done: int = 0, total: int = 0):
        """停止が押されていれば中断する。ただし、そのファイルのダウンロードが完了した瞬間は中断しない。"""
        if self.cancel.is_set() and not (total and done >= total):
            raise JobCancelled("中止しました")

    def _do_train(self, p: dict, prepared_dir: Optional[str], steps: Optional[int]):
        with self._lock:
            self.stage = "train"
            self.loop_started = False
        self.tracker.set("学習の準備中", None, "準備しています")
        self._set_train("", True)                                   # 結果を消して、停止ボタンを出す

        saved_path = None
        try:
            model = p["model"]

            def on_download(done, total, speed, name=""):
                self._check_cancel(done, total)
                gb = 1024 ** 3
                frac = (done / total) if total else None
                remain = ((total - done) / speed) if (total and speed) else None
                detail = name + (" ・ " if name else "")
                detail += f"{_fmt_gb(done / gb)} / {_fmt_gb(total / gb)}" if total else _fmt_gb(done / gb)
                if speed:
                    detail += f" ・ {speed / 1024 ** 2:.1f} MB/s ・ 残り約 {format_duration(remain)}"
                self.tracker.set("モデルをダウンロード中", frac, detail)

            # 1) モデル本体（★付きで無ければダウンロード）
            spec = CATALOG_BY_LABEL.get(model)
            if spec is not None:
                self.tracker.set("モデルを確認中", None, spec.filename)
            model_name = resolve_model(model, lambda d, t, s: on_download(d, t, s, spec.filename if spec else ""))
            self._check_cancel()

            # 2) Animaを選んだ時だけ、VAE と Text Encoder も揃える（無ければダウンロード）
            companions = companions_for(model)
            module_paths: list = []
            if companions:
                def on_companion(comp, done, total, speed):
                    self._check_cancel(done, total)
                    detail = comp.filename + " ・ " + _fmt_gb(done / 1024 ** 3)
                    if total:
                        detail += " / " + _fmt_gb(total / 1024 ** 3)
                    if speed:
                        detail += f" ・ {speed / 1024 ** 2:.1f} MB/s"
                    self.tracker.set(f"{comp.label}をダウンロード中",
                                     (done / total) if total else None, detail)
                self.tracker.set("VAE / Text Encoderを確認中", None, "Animaに必要なファイルを確認しています")
                module_paths = [str(x) for x in ensure_companions(companions, on_companion)]
            self._check_cancel()

            # 3) メモリを空ける（WebUIのモデルなど）。使用量の数値はコンソールにだけ出す
            self.tracker.set("メモリを空けています", None, "WebUIのモデルをメモリから解放しています")
            free_memory(bool(p.get("free_mem")))

            # 4) 学習。ここから「モデルを読み込み中」→「画像の前処理」→「学習中」と進む
            self._check_cancel()
            self.tracker.set("モデルを読み込み中", None, "モデルをメモリに読み込んでいます（数十秒〜数分かかります）")
            modules_ctx = use_webui_modules(module_paths) if module_paths else contextlib.nullcontext()
            with hook_training_progress(self.train_module, self.tracker, int(steps), self._on_loop_step), \
                    modules_ctx:
                message, saved_path = start_training(
                    self.train_module, self.trainer_module, prepared_dir,
                    resolve_trigger(p.get("trigger"), p.get("dataset_name", "")),
                    (p.get("output_name") or "").strip() or p.get("dataset_name"),
                    p["preset_name"], resolve_image_size(p["size_choice"], model),
                    int(steps), model_name, p.get("vae"), p.get("te"),
                )
        except JobCancelled:
            message = ("### ⏹ 中止しました\n\n学習は始めていません。"
                       "ダウンロード途中のファイルは残してあるので、次回は続きから再開します。")
        except Exception as e:
            message = f"### ❌ 学習に失敗しました\n\n{_friendly_error(e)}"

        download = {"visible": False}
        if saved_path:
            copy = prepare_download(saved_path)
            if copy:
                download = {"value": copy, "visible": True,
                            "label": f"⬇ LoRAをダウンロード（{Path(saved_path).name}）"}
        self._set_train(message, False, download)


_RUNNER: Optional[JobRunner] = None


def get_runner(train_module, trainer_module) -> JobRunner:
    """プロセスで1つだけのJobRunner。WebUIの「Reload UI」をしても、進行中の処理を引き継げる。"""
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = JobRunner(train_module, trainer_module)
    else:
        _RUNNER.train_module, _RUNNER.trainer_module = train_module, trainer_module
    return _RUNNER


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _friendly_error(e: Exception) -> str:
    import traceback
    traceback.print_exc()
    if isinstance(e, RuntimeError):
        return str(e)
    return (f"{type(e).__name__}: {e}\n\n"
            "詳しい内容はWebUIのコンソール(ログ)に出力されています。")


def preset_key_from_label(label: str) -> str:
    key = (label or "").split("｜")[0].strip()
    return key if key in PRESETS else DEFAULT_PRESET


def _preset_text(label: str) -> str:
    p = PRESETS[preset_key_from_label(label)]
    return p["label"]


def build_easy_tab(
    train_module,
    trainer_module,
    model_choices=None,
    vae_choices=None,
    te_choices=None,
    default_model="",
    gradio_module=None,
):
    """TrainTrainのBlocksの中に、初心者向けの「Easy LoRA」タブを作る。"""
    gr = gradio_module
    if gr is None:
        import gradio as gr

    local_models = list(model_choices or [])
    vae_choices = list(vae_choices or ["None"])
    te_choices = list(te_choices or ["None"])
    if default_model and default_model not in local_models:
        local_models = [default_model] + local_models
    # ★付きの候補（無ければ学習開始時に自動でダウンロード）を先頭に並べる
    catalog_labels = [m.label for m in MODEL_CATALOG]
    model_choices = catalog_labels + [m for m in local_models if m not in catalog_labels]
    if default_model in local_models:
        first_model = default_model
    elif local_models:
        first_model = local_models[0]
    else:
        first_model = catalog_labels[0]          # モデルが1つも無くても、そのまま始められる

    def mk(factory, *args, optional=None, **kwargs):
        """Gradioのバージョン差を吸収する。optional の引数は、そのバージョンが
        受け付けない場合（TypeError / ValueError）だけ外して作り直す。
        例: Gradio 3.41 の gr.File は type="filepath" を受け付けない。"""
        try:
            return factory(*args, **kwargs, **(optional or {}))
        except (TypeError, ValueError):
            return factory(*args, **kwargs)

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
    .tt-easy-primary button, button.tt-easy-primary { min-height: 56px; font-size: 17px; font-weight: 700; }
    .tt-prog { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 12px 16px; margin: 6px 0; }
    .tt-prog-head { display: flex; justify-content: space-between; font-size: 15px; margin-bottom: 8px; }
    .tt-prog-track { height: 12px; border-radius: 6px; background: var(--background-fill-secondary); overflow: hidden; }
    .tt-prog-fill { height: 100%; border-radius: 6px; background: var(--color-accent, #f97316); transition: width .5s ease; }
    .tt-prog-indeterminate { width: 40% !important; animation: tt-prog-slide 1.4s ease-in-out infinite; }
    @keyframes tt-prog-slide { 0% { margin-left: -40%; } 100% { margin-left: 100%; } }
    .tt-prog-detail { margin-top: 8px; opacity: 0.75; font-size: 0.92em; }
    /* Gradio 6.5.x applies opacity:.2 to pending HTML/Markdown nodes even when
       show_progress="hidden". Polling every second makes live progress blink.
       Keep only these frequently updated outputs fully opaque. */
    .tt-no-flicker .pending, .tt-no-flicker.pending { opacity: 1 !important; }
    """

    preset_labels = [f"{k}｜{v['short']}" for k, v in PRESETS.items()]

    # ---- イベントハンドラ ---------------------------------------------------
    # 処理はJobRunnerが別スレッドで進め、状態はサーバー側に持つ。ボタンは「開始」を伝えるだけで、
    # 画面は gr.Timer の短い問い合わせで状態を取りに来る（長い接続に依存しない）。
    runner = get_runner(train_module, trainer_module)
    NO = gr.update()
    use_timer = hasattr(gr, "Timer")

    def _result_versions(seen: dict) -> dict:
        """結果表示用タイマーが最後に反映した版数だけを保持する。"""
        return {key: seen.get(key, 0) for key in ("prep_version", "train_version")}

    def pack(up: dict, seen: dict) -> tuple:
        """開始ボタン用。開始時は画面全体をまとめて更新してよい。"""
        def val(key):
            return NO if up[key] is KEEP else up[key]
        stop = NO if up["stop_visible"] is KEEP else gr.update(visible=bool(up["stop_visible"]))
        steps_u = NO if up["steps"] is KEEP else gr.update(value=int(up["steps"]))
        download = NO if up["download"] is KEEP else gr.update(**up["download"])
        values = [val("prep_progress"), val("train_progress"), val("status"), val("checkup"),
                  val("gallery"), val("tag_rows"), val("image_rows"), val("stats"),
                  val("zip_file"), val("prepared_dir"), steps_u, val("train_result"),
                  stop, download, seen]
        if use_timer:
            # 画面全体を更新した直後の版数を、結果専用タイマーにも同期する。
            values.extend([
                _result_versions(seen),
                gr.update(active=bool(up["active"])),
                gr.update(active=False),
            ])
        return tuple(values)

    def poll(seen):
        """Timer がない古い Gradio 向けの互換ポーリング。"""
        up, new_seen = compose_ui(runner.snapshot(), seen)
        return pack(up, new_seen)

    def poll_progress(seen, result_seen):
        """毎秒更新するのは進捗バーだけ。結果コンポーネントは出力対象にしない。

        Gradio は値が変わらない出力に gr.update() を返しても、そのコンポーネントを
        イベントの出力対象として扱うバージョンがある。プレビューや設定欄をこの
        1秒タイマーの出力リストから外し、不要な pending / レイアウト変化を避ける。
        """
        snap = runner.snapshot()
        seen = dict(seen or {})
        result_seen = dict(result_seen or {})

        bar = render_progress(snap["progress"]) if snap["running"] else ""
        prep_html = bar if snap["stage"] == "prep" else ""
        train_html = bar if snap["stage"] == "train" else ""
        prep_value = prep_html if prep_html != seen.get("prep_html") else NO
        train_value = train_html if train_html != seen.get("train_html") else NO
        seen["prep_html"] = prep_html
        seen["train_html"] = train_html

        # 結果の版数が変わった時だけ、専用タイマーを一度起動する。
        # requested の印で、結果タイマーが処理するまで毎秒再起動するのを防ぐ。
        request_results = False
        for key in ("prep_version", "train_version"):
            current = snap[key]
            requested_key = f"_requested_{key}"
            if current != result_seen.get(key, 0) and current != seen.get(requested_key):
                seen[requested_key] = current
                request_results = True

        result_timer_update = gr.update(active=True) if request_results else NO
        if snap["running"]:
            seen["_timer_was_active"] = True
            timer_update = NO
        else:
            timer_update = (gr.update(active=False)
                            if seen.get("_timer_was_active", True) else NO)
            seen["_timer_was_active"] = False

        return prep_value, train_value, seen, result_timer_update, timer_update

    def poll_results(result_seen):
        """版数が変わった時だけ結果表示を更新し、結果タイマーを停止する。"""
        up, new_seen = compose_ui(runner.snapshot(), result_seen)

        def val(key):
            return NO if up[key] is KEEP else up[key]

        stop = NO if up["stop_visible"] is KEEP else gr.update(visible=bool(up["stop_visible"]))
        steps_u = NO if up["steps"] is KEEP else gr.update(value=int(up["steps"]))
        download = NO if up["download"] is KEEP else gr.update(**up["download"])
        values = [val("status"), val("checkup"), val("gallery"), val("tag_rows"),
                  val("image_rows"), val("stats"), val("zip_file"), val("prepared_dir"),
                  steps_u, val("train_result"), stop, download,
                  _result_versions(new_seen), gr.update(active=False)]
        return tuple(values)

    def _after_start(refusal):
        up, seen = compose_ui(runner.snapshot(), {})      # 全部を送り直す
        if refusal:
            up["status"] = refusal
        up["active"] = True                               # 問い合わせを再開する
        return pack(up, seen)

    def _prep_kwargs(files, folder_path, dataset_name, preset_label, trigger, general_threshold,
                     character_threshold, manual_keep, manual_remove, auto_th, free_mem):
        return dict(
            files=files, folder_path=folder_path, dataset_name=dataset_name,
            preset_name=preset_key_from_label(preset_label), trigger=trigger,
            general_threshold=general_threshold, character_threshold=character_threshold,
            manual_keep_text=manual_keep, manual_remove_text=manual_remove,
            auto_threshold=bool(auto_th), free_mem=bool(free_mem))

    def _train_kwargs(prepared_dir, trigger, dataset_name, output_name, preset_label,
                      size_choice, steps, model, vae, te, free_mem):
        return dict(
            prepared_dir=prepared_dir, trigger=trigger, dataset_name=dataset_name,
            output_name=output_name, preset_name=preset_key_from_label(preset_label),
            size_choice=size_choice, steps=steps, model=model, vae=vae, te=te,
            free_mem=bool(free_mem))

    def start_all(files, folder_path, dataset_name, preset_label, trigger, general_threshold,
                  character_threshold, manual_keep, manual_remove, auto_th, free_mem,
                  output_name, size_choice, steps, model, vae, te):
        plan = {
            "prep": _prep_kwargs(files, folder_path, dataset_name, preset_label, trigger,
                                 general_threshold, character_threshold, manual_keep,
                                 manual_remove, auto_th, free_mem),
            "train": _train_kwargs(None, trigger, dataset_name, output_name, preset_label,
                                   size_choice, steps, model, vae, te, free_mem),
        }
        return _after_start(runner.start(plan))

    def start_prep(files, folder_path, dataset_name, preset_label, trigger, general_threshold,
                   character_threshold, manual_keep, manual_remove, auto_th, free_mem):
        plan = {"prep": _prep_kwargs(files, folder_path, dataset_name, preset_label, trigger,
                                     general_threshold, character_threshold, manual_keep,
                                     manual_remove, auto_th, free_mem),
                "train": None}
        return _after_start(runner.start(plan))

    def start_train(prepared_dir, trigger, dataset_name, output_name, preset_label,
                    size_choice, steps, model, vae, te, free_mem):
        plan = {"prep": None,
                "train": _train_kwargs(prepared_dir, trigger, dataset_name, output_name,
                                       preset_label, size_choice, steps, model, vae, te, free_mem)}
        return _after_start(runner.start(plan))

    def _stop():
        try:
            return runner.request_stop()
        except Exception as e:
            return f"停止できませんでした: {e}"

    def _size_hint(choice, model):
        size = resolve_image_size(choice, model)
        auto = "（自動判定）" if not str(choice).strip().isdigit() else ""
        return f"学習解像度: **{size}px** {auto}"

    # ---- 画面 ---------------------------------------------------------------
    gr.HTML(f"<style>{css}</style>")
    with gr.Column(elem_classes=["tt-easy-wrap"]):
        gr.Markdown("## Easy LoRA", elem_classes=["tt-easy-title"])
        gr.Markdown(
            "画像を入れて、種類を選んで、ボタンを押すだけ。"
            "タグ付け・不要タグの整理・チェック・学習まで自動で行います。",
            elem_classes=["tt-easy-sub"],
        )

        with gr.Group(elem_classes=["tt-easy-card"]):
            gr.Markdown("### ① 画像を入れる", elem_classes=["tt-easy-step"])
            gr.Markdown("画像を何枚でもドラッグ。ZIPでもOKです（目安：10〜30枚）。",
                        elem_classes=["tt-easy-note"])
            files = mk(gr.File, label="画像 / ZIP", file_count="multiple",
                       file_types=["image", ".zip"], optional={"type": "filepath"})
            with gr.Accordion("Google Drive / Colab のフォルダから読み込む", open=False):
                folder_path = gr.Textbox(
                    label="画像フォルダのパス（任意）",
                    placeholder="/content/drive/MyDrive/LoRA/my_character",
                )

        with gr.Group(elem_classes=["tt-easy-card"]):
            gr.Markdown("### ② 何を学習する？", elem_classes=["tt-easy-step"])
            preset = gr.Radio(choices=preset_labels, value=preset_labels[0], label="LoRAの種類")
            preset_help = gr.Markdown(_preset_text(preset_labels[0]), elem_classes=["tt-easy-note"])
            preset.change(_preset_text, [preset], [preset_help])

        with gr.Group(elem_classes=["tt-easy-card"]):
            gr.Markdown("### ③ 学習に使うモデル", elem_classes=["tt-easy-step"])
            if not model_choices:
                gr.Markdown("⚠️ モデルが見つかりません。チェックポイントを配置してから再読み込みしてください。",
                            elem_classes=["tt-easy-note"])
            model = gr.Dropdown(choices=model_choices, value=first_model,
                                label="モデル（チェックポイント）", allow_custom_value=True)
            gr.Markdown(
                "★付きは、モデルが無ければ**学習を始める時に自動でダウンロード**します（Hugging Face）。"
                "Animaを選んだ時は、必要なVAEとText Encoderも、無ければ自動でダウンロードして使います"
                "（WebUI上部の選択は、学習の間だけ切り替えて、終わったら元に戻します）。",
                elem_classes=["tt-easy-note"])

        with gr.Row():
            run_all = gr.Button("🚀 おまかせで学習まで実行", variant="primary",
                                elem_classes=["tt-easy-primary"])
            run_prep = gr.Button("📦 準備だけ実行（タグ付け・チェックまで）",
                                 elem_classes=["tt-easy-primary"])

        prep_progress = gr.HTML(elem_classes=["tt-no-flicker"])
        train_progress = gr.HTML(elem_classes=["tt-no-flicker"])
        with gr.Row():
            # 学習中だけ表示する（開始時に表示、終了時に非表示にする）
            stop_btn = gr.Button("⏹ 学習を止めて、ここまでを保存", variant="stop", visible=False)
        stop_note = gr.Markdown()
        status = gr.Markdown(elem_classes=["tt-no-flicker"])
        checkup = gr.Markdown(elem_classes=["tt-no-flicker"])
        gallery = mk(gr.Gallery, label="学習用データのプレビュー（画像と、学習に使うタグ）",
                     optional={"columns": 4, "height": "auto"})
        train_result = gr.Markdown(elem_classes=["tt-no-flicker"])
        if hasattr(gr, "DownloadButton"):
            lora_download = gr.DownloadButton("⬇ LoRAをダウンロード", variant="primary", visible=False)
        else:                                    # 古いGradioにはDownloadButtonが無い
            lora_download = gr.File(label="完成したLoRA（クリックでダウンロード）", interactive=False, visible=False)

        with gr.Accordion("詳細設定（通常は変更不要）", open=False):
            with gr.Row():
                dataset_name = gr.Textbox(label="データセット名（半角英数字がおすすめ）", value="my_lora")
                trigger = gr.Textbox(
                    label="トリガーワード（基本は不要。空欄ならなし）", value="",
                    placeholder="使う場合だけ入力。例: mychar",
                )
                output_name = gr.Textbox(label="出力LoRA名（空欄ならデータセット名）", value="")
            with gr.Row():
                auto_threshold = gr.Checkbox(
                    label="タグが少ない時、WD14のしきい値を自動で下げる", value=True)
                free_mem = gr.Checkbox(
                    label="準備・学習の前に、WebUIのモデルをメモリから解放する（推奨。次の画像生成時に自動で再読み込み）",
                    value=True)
            with gr.Row():
                general_threshold = gr.Slider(
                    0.10, 0.70, value=WD_DEFAULT_GENERAL_THRESHOLD, step=0.01,
                    label="WD14 一般タグのしきい値（低いほどタグが増える）")
                character_threshold = gr.Slider(
                    0.50, 0.95, value=WD_DEFAULT_CHARACTER_THRESHOLD, step=0.01,
                    label="WD14 キャラ名タグのしきい値")
            manual_keep = gr.Textbox(label="必ず残すタグ（カンマ区切り）", placeholder="smile, looking at viewer")
            manual_remove = gr.Textbox(label="必ず削除するタグ（カンマ区切り）", placeholder="1girl, solo")
            with gr.Row():
                vae = gr.Dropdown(choices=vae_choices, value=vae_choices[0] if vae_choices else "None",
                                  label="VAE", allow_custom_value=True)
                te = gr.Dropdown(choices=te_choices, value=te_choices[0] if te_choices else "None",
                                 label="Text Encoder", allow_custom_value=True)
            with gr.Row():
                image_size = gr.Dropdown(choices=["自動", "512", "768", "1024", "1280"],
                                         value="自動", label="学習解像度")
                steps = gr.Slider(minimum=100, maximum=3000, step=50,
                                  value=auto_steps(20, DEFAULT_PRESET),
                                  label="学習step（準備のたびに画像枚数から自動設定）")
            size_hint = gr.Markdown(_size_hint("自動", first_model or ""), elem_classes=["tt-easy-note"])
            model.change(_size_hint, [image_size, model], [size_hint])
            image_size.change(_size_hint, [image_size, model], [size_hint])

        with gr.Accordion("結果の詳細（タグの判断理由・画像ごとのキャプション）", open=False):
            gr.Markdown(
                "どのタグを、なぜ残した/削除したかの一覧です。"
                "消しすぎ・残しすぎは「詳細設定」の「必ず残す/削除するタグ」で直せます。",
                elem_classes=["tt-easy-note"])
            tag_table = mk(gr.Dataframe,
                           headers=["タグ", "分類", "出現枚数", "出現率", "判定", "理由"],
                           datatype=["str", "str", "number", "str", "str", "str"],
                           interactive=False,
                           optional={"wrap": True, "height": 360})
            image_table = mk(gr.Dataframe,
                             headers=["画像", "学習用キャプション", "自動で外したタグ"],
                             datatype=["str"] * 3, interactive=False,
                             optional={"wrap": True, "height": 360})
            zip_file = mk(gr.File, label="学習用データセット(ZIP)", interactive=False)
            stats = gr.Code(label="詳細統計", language="json")

        with gr.Accordion("準備済みデータで学習だけ実行（キャプションを手で直した後など）", open=False):
            prepared_dir = gr.Textbox(label="準備済みデータのフォルダ", interactive=True)
            start = gr.Button("学習開始", variant="primary")

        # ---- 配線 ----------------------------------------------------------
        seen_state = gr.State({})                 # 進捗HTMLのポーリング位置
        if use_timer:
            result_seen_state = gr.State({"prep_version": 0, "train_version": 0})
            timer = gr.Timer(1.0, active=True)
            # 結果コンポーネントは、準備/学習の結果が変わった時に一度だけ更新する。
            result_timer = gr.Timer(0.25, active=False)
        else:
            result_seen_state = None
            timer = result_timer = None
        poll_outputs = [prep_progress, train_progress, status, checkup, gallery, tag_table,
                        image_table, stats, zip_file, prepared_dir, steps, train_result,
                        stop_btn, lora_download, seen_state]
        if use_timer:
            poll_outputs.extend([result_seen_state, timer, result_timer])

        prep_inputs = [files, folder_path, dataset_name, preset, trigger,
                       general_threshold, character_threshold, manual_keep, manual_remove,
                       auto_threshold, free_mem]
        all_inputs = prep_inputs + [output_name, image_size, steps, model, vae, te]
        train_inputs = [prepared_dir, trigger, dataset_name, output_name, preset,
                        image_size, steps, model, vae, te, free_mem]

        # queue=False: 開始・停止・問い合わせは一瞬で終わるので、キューを待たせない
        # show_progress="hidden": Gradio標準の進捗表示は使わない（専用の進捗バーに一本化）
        fast = {"queue": False, "show_progress": "hidden"}
        run_prep.click(start_prep, prep_inputs, poll_outputs, **fast)
        run_all.click(start_all, all_inputs, poll_outputs, **fast)
        start.click(start_train, train_inputs, poll_outputs, **fast)
        stop_btn.click(_stop, None, [stop_note], **fast)

        if use_timer:
            # 毎秒の出力対象を進捗バーだけに限定する。
            # ギャラリー・表・結果欄は result_timer で版数変更時に一度だけ更新する。
            timer.tick(
                poll_progress, [seen_state, result_seen_state],
                [prep_progress, train_progress, seen_state, result_timer, timer], **fast)
            result_timer.tick(
                poll_results, [result_seen_state],
                [status, checkup, gallery, tag_table, image_table, stats, zip_file,
                 prepared_dir, steps, train_result, stop_btn, lora_download,
                 result_seen_state, result_timer], **fast)
        else:
            # 古いGradio: ページを開いている間、1秒ごとに状態を取りに行く
            try:
                gr.context.Context.root_block.load(
                    poll, [seen_state], poll_outputs, every=1, show_progress="hidden")
            except Exception as e:
                print(f"[Easy LoRA] 進捗の自動更新を設定できませんでした: {type(e).__name__}: {e}")
