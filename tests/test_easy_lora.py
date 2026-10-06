# -*- coding: utf-8 -*-
"""
Easy LoRA のテスト。WebUI・GPU・ネット接続は不要です。

    python -m unittest tests.test_easy_lora -v      # リポジトリのルートで実行

- WD14は「決め打ちの結果を返す偽タガー」に差し替えて、準備処理全体を通しで検証します。
- TrainTrainの設定定義は scripts/traintrain.py から実物を読み込んで検証します
  （設定名の照合バグを再発させないため）。
"""

import csv
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location("easy_lora_under_test", ROOT / "trainer" / "easy_lora.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


E = load_module()


# ---------------------------------------------------------------------------
# テスト用データ
# ---------------------------------------------------------------------------

def structured_image(seed: int, size=(640, 640)) -> Image.Image:
    """乱数ノイズではなく、縮小しても特徴が残る図形画像。"""
    rng = np.random.RandomState(seed)
    im = Image.new("RGB", size, tuple(int(x) for x in rng.randint(0, 255, 3)))
    d = ImageDraw.Draw(im)
    for _ in range(8):
        x0, y0 = rng.randint(0, size[0] - 100), rng.randint(0, size[1] - 100)
        x1, y1 = x0 + rng.randint(80, 300), y0 + rng.randint(80, 300)
        color = tuple(int(x) for x in rng.randint(0, 255, 3))
        (d.ellipse if rng.rand() > 0.5 else d.rectangle)([x0, y0, x1, y1], fill=color)
    return im


class FakeTagger:
    """決め打ちのタグを返す。画像の並び順(index)で内容を変える。"""
    provider = "FakeProvider"
    script = None  # テストごとに差し替える

    def __init__(self, *a, **k):
        self.closed = False

    def tag_images(self, paths, batch_size=8, progress=None):
        out = [FakeTagger.script(i, Path(p)) for i, p in enumerate(paths)]
        if progress:
            progress(1.0, "done")
        return out

    def close(self):
        self.closed = True


def T(name, cat, score):
    return E.TagItem(name, cat, score)


def character_script(i, path):
    items = [
        T("1girl", 0, 0.99), T("solo", 0, 0.98), T("long_hair", 0, 0.9),
        T("blue_eyes", 0, 0.8), T("blue_hair", 0, 0.8), T("watermark", 0, 0.5),
        T("smile", 0, 0.7), T("looking_at_viewer", 0, 0.7), T("simple_background", 0, 0.6),
        T("masterpiece", 0, 0.5), T("highres", 5, 0.9),
    ]
    if i % 2 == 0:
        items.append(T("school_uniform", 0, 0.8))      # 50% -> 共通とみなして削除
    if i in (0, 1):
        items.append(T("ponytail", 0, 0.7))            # 20% -> 残す
    if i == 2:
        items.append(T("hatsune_miku", 4, 0.95))       # キャラ名 -> 削除
    items.append(T("zzz_unknown_thing", 0, 0.45))
    return sorted(items, key=lambda x: -x.score)


# ---------------------------------------------------------------------------
# タグ分類
# ---------------------------------------------------------------------------

class TestClassify(unittest.TestCase):
    CASES = {
        "long_hair": "hair", "blue_hair": "hair", "ponytail": "hair",
        "hair_ribbon": "hair", "hair_between_eyes": "hair", "alternate_hairstyle": "hair",
        "blue_eyes": "appearance", "animal_ears": "appearance", "cat_tail": "appearance",
        "large_breasts": "appearance", "dark_skin": "appearance",
        "closed_eyes": "expression", "open_mouth": "expression", "smile": "expression",
        "blush": "expression",
        "school_uniform": "clothing", "white_shirt": "clothing", "bow": "clothing",
        "fingerless_gloves": "clothing", "glasses": "clothing", "alternate_costume": "clothing",
        "sword": "object", "bow_(weapon)": "object", "holding_umbrella": "object",
        "simple_background": "background", "white_background": "background",
        "blurry_background": "background", "outdoors": "background", "flower": "background",
        "standing": "pose", "looking_at_viewer": "pose", "upper_body": "pose",
        "1girl": "subject", "solo": "subject", "2girls": "subject", "multiple_girls": "subject",
        "6+girls": "subject", "no_humans": "subject",
        "watermark": "defect", "signature": "defect", "english_text": "defect",
        "jpeg_artifacts": "defect", "blurry": "defect",
        "masterpiece": "quality", "highres": "quality", "absurdres": "quality",
        "commentary_request": "quality",
        "monochrome": "style", "flat_color": "style", "watercolor_(medium)": "style",
        "zzz_unknown_thing": "other",
    }

    def test_general_tags(self):
        wrong = {t: (E.classify_tag(t, 0), want) for t, want in self.CASES.items()
                 if E.classify_tag(t, 0) != want}
        self.assertEqual(wrong, {}, f"分類ミス: {wrong}")

    def test_native_categories_win(self):
        self.assertEqual(E.classify_tag("hatsune_miku", 4), "character")
        self.assertEqual(E.classify_tag("vocaloid", 3), "copyright")
        self.assertEqual(E.classify_tag("long_hair", 4), "character")  # カテゴリ番号を優先

    def test_caption_format(self):
        self.assertEqual(E.to_caption_tag("long_hair"), "long hair")
        self.assertEqual(E.to_caption_tag("^_^"), "^_^")
        self.assertEqual(E._norm_key(" Long Hair "), "long_hair")
        self.assertEqual(E._parse_tag_list("smile, Looking At Viewer，solo\n1girl"),
                         {"smile", "looking_at_viewer", "solo", "1girl"})


# ---------------------------------------------------------------------------
# WD14前処理（過去のバグ: RGB+正規化だった）
# ---------------------------------------------------------------------------

class TestWD14Preprocess(unittest.TestCase):
    def test_bgr_0_255_no_normalization(self):
        tagger = E.WD14Tagger()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "red.png"
            Image.new("RGB", (200, 100), (255, 0, 0)).save(p)
            arr = tagger._preprocess(p)
        self.assertEqual(arr.shape, (1, 448, 448, 3))
        self.assertEqual(arr.dtype, np.float32)
        center = arr[0, 224, 224]
        self.assertEqual(tuple(center), (0.0, 0.0, 255.0))     # BGR順の赤
        self.assertEqual(tuple(arr[0, 0, 0]), (255.0, 255.0, 255.0))  # 余白は白

    def test_transparent_becomes_white(self):
        tagger = E.WD14Tagger()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.png"
            Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(p)
            arr = tagger._preprocess(p)
        self.assertTrue((arr == 255.0).all())

    def test_decode_applies_floor_and_skips_rating(self):
        tagger = E.WD14Tagger()
        tagger.tags = ["a", "b", "c", "d"]
        tagger.categories = [0, 4, 9, 0]
        items = tagger._decode_probs(np.array([[0.5, 0.4, 0.99, 0.1]]))[0]
        self.assertEqual([i.name for i in items], ["a"])  # b: キャラ下限0.5未満 / c: rating / d: 下限0.2未満


# ---------------------------------------------------------------------------
# タグ選別
# ---------------------------------------------------------------------------

class TestDecisions(unittest.TestCase):
    def _decide(self, preset, n=10, **kw):
        sets, cats = [], {}
        for i in range(n):
            s = {"blue_hair", "smile"}
            if i < 2:
                s.add("ponytail")
            if i % 2 == 0:
                s.add("school_uniform")
            sets.append(s)
        for t in set().union(*sets):
            cats[t] = E.classify_tag(t, 0)
        return E.decide_tags(sets, cats, E.PRESETS[preset], kw.get("keep", set()), kw.get("remove", set()))

    def test_character_uses_frequency(self):
        d = self._decide("キャラクター")
        self.assertFalse(d["blue_hair"].keep)        # 100% -> 特徴
        self.assertTrue(d["ponytail"].keep)          # 20% -> 変化する要素
        self.assertFalse(d["school_uniform"].keep)   # 50% >= 40%
        self.assertTrue(d["smile"].keep)

    def test_tiny_dataset_falls_back_to_always_remove(self):
        d = self._decide("キャラクター", n=3)
        self.assertFalse(d["ponytail"].keep)
        self.assertIn("画像が少ない", d["ponytail"].reason)

    def test_manual_overrides_win(self):
        d = self._decide("キャラクター", keep={"blue_hair"}, remove={"smile"})
        self.assertTrue(d["blue_hair"].keep)
        self.assertFalse(d["smile"].keep)

    def test_style_preset_keeps_content_tags(self):
        sets = [{"monochrome", "1girl", "school_uniform"}] * 5
        cats = {t: E.classify_tag(t, 0) for t in sets[0]}
        d = E.decide_tags(sets, cats, E.PRESETS["画風"], set(), set())
        self.assertFalse(d["monochrome"].keep)
        self.assertTrue(d["school_uniform"].keep)

    def test_defect_tags_are_kept_but_quality_removed(self):
        sets = [{"watermark", "highres", "masterpiece", "smile"}] * 5
        cats = {t: E.classify_tag(t, 0) for t in sets[0]}
        d = E.decide_tags(sets, cats, E.PRESETS["キャラクター"], set(), set())
        self.assertTrue(d["watermark"].keep)
        self.assertFalse(d["highres"].keep)
        self.assertFalse(d["masterpiece"].keep)

    def test_auto_threshold_only_lowers(self):
        few = [[T(f"t{i}", 0, 0.30) for i in range(10)] for _ in range(5)]   # 0.35だと0個
        th, note = E.auto_general_threshold(few, 0.35, 0.85)
        self.assertEqual(th, 0.30)
        self.assertTrue(note)
        many = [[T(f"t{i}", 0, 0.9) for i in range(20)] for _ in range(5)]
        th, note = E.auto_general_threshold(many, 0.35, 0.85)
        self.assertEqual((th, note), (0.35, ""))
        hopeless = [[T("a", 0, 0.21)] for _ in range(5)]
        th, _ = E.auto_general_threshold(hopeless, 0.35, 0.85)
        self.assertGreaterEqual(th, E.WD_AUTO_THRESHOLD_FLOOR)

    def test_trigger_and_steps(self):
        self.assertEqual(E.resolve_trigger("  myword ", "x"), "myword")
        self.assertEqual(E.resolve_trigger("", "My Char 01"), "elora_mychar01")
        self.assertTrue(E.resolve_trigger("", "あいう").startswith("elora_"))
        self.assertEqual(E.auto_steps(3, "キャラクター"), 400)
        self.assertEqual(E.auto_steps(20, "キャラクター"), 1000)
        self.assertEqual(E.auto_steps(500, "キャラクター"), 1800)


# ---------------------------------------------------------------------------
# 準備処理を通しで
# ---------------------------------------------------------------------------

class TestPrepare(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="easylora_test_"))
        self.ext_root = self.tmp / "ext"
        self.ext_root.mkdir()
        self._orig_root = E._extension_root
        self._orig_tagger = E.WD14Tagger
        E._extension_root = lambda: self.ext_root
        E.WD14Tagger = FakeTagger
        FakeTagger.script = staticmethod(character_script)

        self.src = self.tmp / "src"
        self.src.mkdir()
        for i in range(9):
            structured_image(i).save(self.src / f"img{i}.png")
        # 回転情報(EXIF)付き: 640x480 のまま保存 → 表示上は 480x640
        exif_img = structured_image(100, (640, 480))
        exif = Image.Exif()
        exif[0x0112] = 6
        exif_img.save(self.src / "img_exif.jpg", exif=exif)
        shutil.copy(self.src / "img0.png", self.src / "img0_copy.png")             # 完全に同じ
        structured_image(1).resize((576, 576)).save(self.src / "img1_near.jpg", quality=70)  # ほぼ同じ
        Image.new("RGB", (100, 100), (10, 20, 30)).save(self.src / "tiny.png")      # 小さすぎ

        self.tmp_before = set(os.listdir(tempfile.gettempdir()))

    def tearDown(self):
        E._extension_root = self._orig_root
        E.WD14Tagger = self._orig_tagger
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_prepare(self, **kw):
        args = dict(files=None, folder_path=str(self.src), dataset_name="test set",
                    preset_name="キャラクター", trigger="", general_threshold=0.35,
                    character_threshold=0.85)
        args.update(kw)
        return E.prepare_dataset(**args)

    def test_end_to_end(self):
        calls = []
        res = self.run_prepare(progress=lambda f, d: calls.append(f))
        root = self.ext_root / "easy_datasets" / "test_set"
        prepared = root / "prepared"

        # 画像の選別: 9 + exif = 10枚。tiny / near-dup が除外、完全一致は1枚にまとまる
        self.assertEqual(res["image_count"], 10)
        reasons = dict(res["stats"]["excluded"])
        self.assertEqual(len(reasons), 2)
        self.assertTrue(any("小さすぎ" in r for r in reasons.values()))
        self.assertTrue(any("ほぼ同じ" in r for r in reasons.values()))
        self.assertEqual(len(list(prepared.glob("*.txt"))), 10)
        self.assertEqual(len([p for p in prepared.iterdir() if p.suffix != ".txt"]), 10)

        # 元データ保護: 完全一致を除く12枚すべてが original/ に残る
        self.assertEqual(len(list((root / "original").iterdir())), 12)

        # EXIF回転が反映されて PNG に変換される
        exif_out = [p for p in prepared.iterdir() if "img_exif" in p.name and p.suffix != ".txt"][0]
        self.assertEqual(exif_out.suffix, ".png")
        with Image.open(exif_out) as im:
            self.assertEqual(im.size, (480, 640))

        # キャプション
        caps = {p.stem: p.read_text(encoding="utf-8") for p in prepared.glob("*.txt")}
        self.assertTrue(all("_" not in c for c in caps.values()), "アンダースコアは空白にする")
        self.assertTrue(all(res["trigger"] not in c for c in caps.values()), "トリガーはtxtに書かない")
        joined = "\n".join(caps.values())
        for must_be_gone in ("blue hair", "blue eyes", "long hair", "school uniform",
                             "hatsune miku", "masterpiece", "highres", "1girl, 1girl"):
            self.assertNotIn(must_be_gone, joined)
        for must_stay in ("1girl", "solo", "smile", "looking at viewer", "ponytail",
                          "watermark", "simple background", "zzz unknown thing"):
            self.assertIn(must_stay, joined)
        self.assertEqual(res["trigger"], "elora_testset")

        # 監査用ファイル
        for f in ("summary.json", "tag_decisions.csv", "excluded.csv", "review.csv"):
            self.assertTrue((root / "meta" / f).exists(), f)
        self.assertEqual(len(list((root / "captions_original").glob("*.json"))), 10)
        with open(root / "meta" / "tag_decisions.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        blue = [r for r in rows if r["tag"] == "blue hair"][0]
        self.assertEqual(blue["decision"], "remove")
        self.assertIn("共通", blue["reason"])

        # UI向け出力
        self.assertLessEqual(len(res["gallery"]), E.PREVIEW_MAX)
        self.assertTrue(all(os.path.exists(p) for p, _ in res["gallery"]))
        self.assertTrue(res["zip_path"] and os.path.exists(res["zip_path"]))
        self.assertIn("✅", res["checkup"])
        self.assertTrue(calls and calls[-1] == 1.0)
        self.assertEqual(res["auto_steps"], 500)

        # 一時フォルダを残さない
        leaked = [x for x in set(os.listdir(tempfile.gettempdir())) - self.tmp_before
                  if x.startswith("traintrain_easy_")]
        self.assertEqual(leaked, [])

    def test_rerun_keeps_one_backup(self):
        self.run_prepare()
        marker = self.ext_root / "easy_datasets" / "test_set" / "prepared" / "MARK.txt"
        marker.write_text("hand edited", encoding="utf-8")
        self.run_prepare()
        backup = self.ext_root / "easy_datasets" / "test_set__prev" / "prepared" / "MARK.txt"
        self.assertTrue(backup.exists(), "手で直した前回の結果は消さずに退避する")

    def test_auto_threshold_kicks_in(self):
        def sparse(i, path):
            return [T("smile", 0, 0.5)] + [T(f"tag{j}", 0, 0.28) for j in range(10)]
        FakeTagger.script = staticmethod(sparse)
        res = self.run_prepare(auto_threshold=True)
        self.assertLess(res["stats"]["general_threshold"], 0.35)
        self.assertIn("自動で下げ", res["stats"]["threshold_note"])
        res2 = self.run_prepare(auto_threshold=False)
        self.assertEqual(res2["stats"]["general_threshold"], 0.35)

    def test_zip_input_and_traversal_guard(self):
        import zipfile
        zp = self.tmp / "set.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            for i in range(5):
                zf.write(self.src / f"img{i}.png", f"folder/img{i}.png")
            zf.write(self.src / "img5.png", "__MACOSX/img5.png")     # ゴミは無視
        res = self.run_prepare(files=[str(zp)], folder_path="")
        self.assertEqual(res["image_count"], 5)

        evil = self.tmp / "evil.zip"
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr("../../escape.png", b"x")
        with self.assertRaises(RuntimeError):
            self.run_prepare(files=[str(evil)], folder_path="")
        self.assertFalse((self.tmp.parent / "escape.png").exists())

    def test_errors_are_readable(self):
        with self.assertRaises(RuntimeError) as cm:
            self.run_prepare(folder_path=str(self.tmp / "nope"))
        self.assertIn("見つかりません", str(cm.exception))
        empty = self.tmp / "empty"
        empty.mkdir()
        with self.assertRaises(RuntimeError) as cm:
            self.run_prepare(folder_path=str(empty))
        self.assertIn("画像が見つかりません", str(cm.exception))

    def test_tagger_released_even_on_failure(self):
        closed = []

        class Boom(FakeTagger):
            def tag_images(self, *a, **k):
                raise RuntimeError("boom")

            def close(self):
                closed.append(True)

        E.WD14Tagger = Boom
        with self.assertRaises(RuntimeError):
            self.run_prepare()
        self.assertTrue(closed)


# ---------------------------------------------------------------------------
# TrainTrain連携（実物の設定定義を使う）
# ---------------------------------------------------------------------------

def load_real_configs():
    src = (ROOT / "scripts" / "traintrain.py").read_text(encoding="utf-8")
    start = src.index("BLOCKID26=")
    end_marker = "trainer.all_configs ="
    end = src.index("\n", src.index(end_marker))
    trainer_stub = types.SimpleNamespace(
        OPTIMIZERS=["AdamW", "AdamW8bit", "AdaFactor", "Lion", "Prodigy"], all_configs=[])
    ns = {"trainer": trainer_stub}
    exec(src[start:end], ns)
    return trainer_stub


class TestTrainIntegration(unittest.TestCase):
    def setUp(self):
        self.stub = load_real_configs()
        self.tmp = tempfile.mkdtemp(prefix="easylora_lora_")
        self.stub.lora_dir = self.tmp

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def values(self, **kw):
        args = dict(prepared_dir="/data/prepared", trigger="elora_x", output_name="My Char",
                    preset_name="キャラクター", image_size=1024, steps=900)
        args.update(kw)
        return E.build_train_values(self.stub, **args)

    def get(self, vals, key, second=False):
        idx = E._config_index(self.stub.all_configs)[key]
        n = len(self.stub.all_configs)
        return vals[idx + (n if second else 0)]

    def test_layout_matches_train_main(self):
        vals = self.values()
        n = len(self.stub.all_configs)
        self.assertEqual(len(vals), 2 * n + 5)
        clen = n * (len(vals) // n)
        self.assertEqual(vals[clen:clen + 3], ["", "", ""])
        self.assertEqual(vals[clen + 3:], [None, None])

    def test_overrides_actually_applied(self):
        vals = self.values()
        # 過去のバグ: 括弧付きの設定名が照合できず、解像度が512のままだった
        self.assertEqual(self.get(vals, "image_size(height, width)"), "1024")
        self.assertEqual(self.get(vals, "image_size"), "1024")
        # 過去のバグ: 出力名が未設定で、いつも untitled になっていた
        self.assertEqual(self.get(vals, "save_lora_name"), "My_Char")
        self.assertEqual(self.get(vals, "lora_data_directory"), "/data/prepared")
        self.assertEqual(self.get(vals, "lora_trigger_word"), "elora_x")
        self.assertEqual(self.get(vals, "train_iterations"), 900)
        self.assertEqual(self.get(vals, "network_type"), "lierla")
        self.assertIn(self.get(vals, "train_model_precision"), ("fp16", "bf16"))
        self.assertTrue(self.get(vals, "image_shuffle_tags"))
        self.assertEqual(self.get(vals, "image_size", second=True), "1024")

    def test_untouched_settings_keep_defaults(self):
        vals = self.values()
        self.assertEqual(self.get(vals, "network_conv_rank"), "0")
        self.assertEqual(self.get(vals, "train_loss_function"), "MSE")

    def test_every_override_key_exists(self):
        index = E._config_index(self.stub.all_configs)
        import re
        src = (ROOT / "trainer" / "easy_lora.py").read_text(encoding="utf-8")
        block = src[src.index("overrides = {"):src.index("index = _config_index")]
        keys = re.findall(r'^\s+"([a-zA-Z_]+)":', block, re.M)
        self.assertGreater(len(keys), 15)
        missing = [k for k in keys if k not in index]
        self.assertEqual(missing, [], f"TrainTrainに存在しない設定: {missing}")

    def test_unique_name_avoids_file_exist(self):
        self.assertEqual(E._unique_lora_name(self.stub, "abc"), "abc")
        Path(self.tmp, "abc.safetensors").write_bytes(b"x")
        self.assertEqual(E._unique_lora_name(self.stub, "abc"), "abc_2")
        Path(self.tmp, "abc_2.safetensors").write_bytes(b"x")
        self.assertEqual(E._unique_lora_name(self.stub, "abc"), "abc_3")

    def test_start_training_flow(self):
        calls = []

        class FakeTrain:
            @staticmethod
            def train(*args):
                calls.append(args)
                name_idx = E._config_index(self_stub.all_configs)["save_lora_name"] + 5
                Path(self_dir, args[name_idx] + ".safetensors").write_bytes(b"x")
                return "Trained"

        self_stub, self_dir = self.stub, self.tmp
        prepared = Path(self.tmp) / "prepared"
        prepared.mkdir()
        msg = E.start_training(FakeTrain, self.stub, str(prepared), "elora_x", "mychar",
                               "キャラクター", 1024, 800, "model.safetensors", "None", "None")
        self.assertEqual(calls[0][:5], (False, "LoRA", "model.safetensors", "None", "None"))
        self.assertIn("✅", msg)
        self.assertIn("<lora:mychar:1>", msg)
        # 2回目は同名にならない
        msg2 = E.start_training(FakeTrain, self.stub, str(prepared), "elora_x", "mychar",
                                "キャラクター", 1024, 800, "model.safetensors", "None", "None")
        self.assertIn("<lora:mychar_2:1>", msg2)

    def test_start_training_guards(self):
        msg = E.start_training(None, self.stub, "/nonexistent", "t", "n", "キャラクター", 512, 400, "m", "", "")
        self.assertIn("先に", msg)
        msg = E.start_training(None, self.stub, self.tmp, "t", "n", "キャラクター", 512, 400, "", "", "")
        self.assertIn("モデル", msg)

    def test_guess_image_size(self):
        self.assertEqual(E.guess_image_size("animagineXL_v3.safetensors"), 1024)
        self.assertEqual(E.guess_image_size("ponyDiffusionV6.safetensors"), 1024)
        self.assertEqual(E.guess_image_size("anything-v5.safetensors [abc]"), 512)
        self.assertEqual(E.resolve_image_size("768", "x"), 768)
        self.assertEqual(E.resolve_image_size("自動", "sdxl_base"), 1024)


# ---------------------------------------------------------------------------
# UI（Gradioがある環境のみ）
# ---------------------------------------------------------------------------

@unittest.skipUnless(importlib.util.find_spec("gradio"), "gradio not installed")
class TestUI(unittest.TestCase):
    def test_build_tab(self):
        import gradio as gr
        stub = load_real_configs()
        with gr.Blocks() as demo:
            with gr.Tabs():
                with gr.Tab("Easy LoRA"):
                    E.build_easy_tab(
                        train_module=types.SimpleNamespace(train=lambda *a: "ok", stop_time=lambda s: None),
                        trainer_module=stub,
                        model_choices=["a.safetensors", "b_XL.safetensors"],
                        vae_choices=["None"], te_choices=["None"],
                        default_model="a.safetensors", gradio_module=gr)
        self.assertGreater(len(demo.fns), 5)

    def test_build_tab_without_models(self):
        import gradio as gr
        with gr.Blocks():
            E.build_easy_tab(types.SimpleNamespace(), load_real_configs(), gradio_module=gr)

    def test_preset_labels(self):
        for k, v in E.PRESETS.items():
            label = f"{k}｜{v['short']}"
            self.assertEqual(E.preset_key_from_label(label), k)
        self.assertEqual(E.preset_key_from_label("???"), E.DEFAULT_PRESET)


if __name__ == "__main__":
    unittest.main(verbosity=2)
