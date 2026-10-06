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
import gc
import hashlib
import json
import os
import re
import shutil
import tempfile
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
                digest = hashlib.sha1(p.read_bytes()).hexdigest()
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

def resolve_trigger(trigger: str, dataset_name: str) -> str:
    """トリガーワードが空なら、データセット名から自動で作る。"""
    t = (trigger or "").strip()
    if t:
        return t
    slug = re.sub(r"[^a-z0-9]+", "", (dataset_name or "").lower())[:12]
    if not slug:
        slug = hashlib.sha1((dataset_name or "lora").encode("utf-8")).hexdigest()[:6]
    return f"elora_{slug}"


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
) -> str:
    if not prepared_dir or not os.path.isdir(prepared_dir):
        return "❌ 先に「準備」を実行してください（学習用データが見つかりません）。"
    if not model:
        return "❌ モデルを選択してください。"

    name = _unique_lora_name(trainer_module, output_name or "my_lora")

    def run(lora_name: str) -> str:
        values = build_train_values(
            trainer_module, prepared_dir, trigger, lora_name, preset_name,
            int(image_size), int(steps),
        )
        return str(train_module.train(False, "LoRA", model, vae or "None", te or "None", *values))

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
                "VRAM不足の場合は、解像度(詳細設定)を下げるか、より軽いモデルを選んでください。")

    save_dir = getattr(trainer_module, "lora_dir", None)
    saved = save_dir and os.path.exists(os.path.join(save_dir, f"{name}.safetensors"))
    if saved:
        return (
            f"### ✅ 学習完了\n\n"
            f"- 出力: `{name}.safetensors`\n"
            f"- 使い方: プロンプトに `{trigger}` を入れて、`<lora:{name}:1>` を追加\n"
            f"- 効きすぎ/弱すぎる時は `:1` を `:0.6` や `:1.2` に変えて調整\n\n"
            f"（TrainTrainからのメッセージ: {result}）"
        )
    return f"### 学習が終了しました\n\nTrainTrainからのメッセージ: {result}"


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

    model_choices = list(model_choices or [])
    vae_choices = list(vae_choices or ["None"])
    te_choices = list(te_choices or ["None"])
    if default_model and default_model not in model_choices:
        model_choices = [default_model] + model_choices
    first_model = default_model if default_model in model_choices else (
        model_choices[0] if model_choices else None)

    def mk(factory, *args, optional=None, **kwargs):
        """Gradioのバージョン差を吸収する。optional の引数は、そのバージョンが
        受け付けない場合（TypeError / ValueError）だけ外して作り直す。
        例: Gradio 3.41 の gr.File は type="filepath" を受け付けない。"""
        try:
            return factory(*args, **kwargs, **(optional or {}))
        except (TypeError, ValueError):
            return factory(*args, **kwargs)

    # 進捗バー（古いGradioには無い）
    progress_default = gr.Progress() if hasattr(gr, "Progress") else None

    def make_progress(p):
        if p is None:
            return None
        return lambda frac, desc="": p(frac, desc=desc)

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
    """

    preset_labels = [f"{k}｜{v['short']}" for k, v in PRESETS.items()]

    # ---- イベントハンドラ ---------------------------------------------------
    def _run_prep(flag, files, folder_path, dataset_name, preset_label, trigger,
                  general_threshold, character_threshold, manual_keep, manual_remove,
                  auto_th, progress):
        preset_name = preset_key_from_label(preset_label)
        try:
            res = prepare_dataset(
                files, folder_path, dataset_name, preset_name, trigger,
                general_threshold, character_threshold, manual_keep, manual_remove,
                bool(auto_th), progress=make_progress(progress),
            )
        except Exception as e:
            msg = f"### ❌ 準備に失敗しました\n\n{_friendly_error(e)}"
            keep = gr.update()
            return (msg, "", keep, keep, keep, keep, keep, keep, keep, "0")

        st = res["stats"]
        lines = [
            "### ✅ 学習データの準備ができました",
            "",
            f"- 使う画像: **{st['images_used']}枚**"
            + (f"（{len(st['excluded'])}枚は自動で除外）" if st["excluded"] else ""),
            f"- 種類: **{preset_name}**",
            f"- トリガーワード: `{res['trigger']}` ← 画像を生成する時、プロンプトに入れます",
            f"- おすすめ学習step: 約 **{res['auto_steps']}**",
        ]
        if flag:
            lines += ["", "▶ 続けて学習を開始します。進行状況はWebUIのコンソール(ログ)に表示されます。"]
        else:
            lines += ["", "内容を確認して、よければ下の「準備済みデータで学習だけ実行」を押してください。"]
        return (
            "\n".join(lines), res["checkup"], res["gallery"], res["tag_rows"],
            res["image_rows"], res["stats_json"], res["zip_path"], res["prepared_dir"],
            gr.update(value=res["auto_steps"]), "1" if flag else "0",
        )

    def _prep_only(files, folder_path, dataset_name, preset_label, trigger,
                   general_threshold, character_threshold, manual_keep, manual_remove,
                   auto_th, progress=progress_default):
        return _run_prep(False, files, folder_path, dataset_name, preset_label, trigger,
                         general_threshold, character_threshold, manual_keep, manual_remove,
                         auto_th, progress)

    def _prep_and_train(files, folder_path, dataset_name, preset_label, trigger,
                        general_threshold, character_threshold, manual_keep, manual_remove,
                        auto_th, progress=progress_default):
        return _run_prep(True, files, folder_path, dataset_name, preset_label, trigger,
                         general_threshold, character_threshold, manual_keep, manual_remove,
                         auto_th, progress)

    def _train(prepared_dir, trigger, dataset_name, output_name, preset_label,
               size_choice, steps, model, vae, te):
        try:
            return start_training(
                train_module, trainer_module, prepared_dir,
                resolve_trigger(trigger, dataset_name),
                (output_name or "").strip() or dataset_name,
                preset_key_from_label(preset_label),
                resolve_image_size(size_choice, model),
                int(steps), model, vae, te,
            )
        except Exception as e:
            return f"### ❌ 学習に失敗しました\n\n{_friendly_error(e)}"

    def _train_if_pending(pending, *args):
        if pending != "1":
            return gr.update()
        return _train(*args)

    def _stop():
        try:
            train_module.stop_time(True)
            return "停止を要求しました。次の区切りで、ここまでの結果を保存して止まります。"
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

        with gr.Row():
            run_all = gr.Button("🚀 おまかせで学習まで実行", variant="primary",
                                elem_classes=["tt-easy-primary"])
            run_prep = gr.Button("📦 準備だけ実行（タグ付け・チェックまで）",
                                 elem_classes=["tt-easy-primary"])

        status = gr.Markdown()
        checkup = gr.Markdown()
        gallery = mk(gr.Gallery, label="学習用データのプレビュー（画像と、学習に使うタグ）",
                     optional={"columns": 4, "height": "auto"})
        train_result = gr.Markdown()

        with gr.Row():
            stop_btn = gr.Button("⏹ 学習を止めて、ここまでを保存")
            stop_note = gr.Markdown()

        with gr.Accordion("詳細設定（通常は変更不要）", open=False):
            with gr.Row():
                dataset_name = gr.Textbox(label="データセット名（半角英数字がおすすめ）", value="my_lora")
                trigger = gr.Textbox(
                    label="トリガーワード（空欄なら自動で決めます）", value="",
                    placeholder="例: elora_mychar",
                )
                output_name = gr.Textbox(label="出力LoRA名（空欄ならデータセット名）", value="")
            with gr.Row():
                auto_threshold = gr.Checkbox(
                    label="タグが少ない時、WD14のしきい値を自動で下げる", value=True)
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

        pending = gr.State("0")

        prep_inputs = [files, folder_path, dataset_name, preset, trigger,
                       general_threshold, character_threshold, manual_keep, manual_remove,
                       auto_threshold]
        prep_outputs = [status, checkup, gallery, tag_table, image_table, stats,
                        zip_file, prepared_dir, steps, pending]
        train_inputs = [prepared_dir, trigger, dataset_name, output_name, preset,
                        image_size, steps, model, vae, te]

        run_prep.click(_prep_only, prep_inputs, prep_outputs)
        run_all.click(_prep_and_train, prep_inputs, prep_outputs).then(
            _train_if_pending, [pending] + train_inputs, [train_result])
        start.click(_train, train_inputs, [train_result])
        stop_btn.click(_stop, None, [stop_note])
