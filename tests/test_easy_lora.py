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
        # トリガーワードは基本的に不要: 空欄ならそのまま「なし」
        self.assertEqual(E.resolve_trigger("", "My Char 01"), "")
        self.assertEqual(E.resolve_trigger(None, "あいう"), "")
        self.assertEqual(E.resolve_trigger("   ", "x"), "")
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
        joined = "\n".join(caps.values())
        for must_be_gone in ("blue hair", "blue eyes", "long hair", "school uniform",
                             "hatsune miku", "masterpiece", "highres", "1girl, 1girl"):
            self.assertNotIn(must_be_gone, joined)
        for must_stay in ("1girl", "solo", "smile", "looking at viewer", "ponytail",
                          "watermark", "simple background", "zzz unknown thing"):
            self.assertIn(must_stay, joined)
        self.assertEqual(res["trigger"], "", "トリガーワードは既定でなし")

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

    def test_explicit_trigger_is_returned_but_not_written_to_txt(self):
        res = self.run_prepare(trigger=" mychar ")
        self.assertEqual(res["trigger"], "mychar")
        prepared = Path(res["prepared_dir"])
        self.assertTrue(all("mychar" not in p.read_text(encoding="utf-8") for p in prepared.glob("*.txt")))

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
        msg, path = E.start_training(FakeTrain, self.stub, str(prepared), "elora_x", "mychar",
                                     "キャラクター", 1024, 800, "model.safetensors", "None", "None")
        self.assertEqual(calls[0][:5], (False, "LoRA", "model.safetensors", "None", "None"))
        self.assertIn("✅", msg)
        self.assertIn("<lora:mychar:1>", msg)
        self.assertTrue(path.endswith("mychar.safetensors") and os.path.isfile(path))
        # 2回目は同名にならない
        msg2, path2 = E.start_training(FakeTrain, self.stub, str(prepared), "elora_x", "mychar",
                                       "キャラクター", 1024, 800, "model.safetensors", "None", "None")
        self.assertIn("<lora:mychar_2:1>", msg2)
        self.assertTrue(path2.endswith("mychar_2.safetensors"))


    def test_usage_message_without_trigger(self):
        class FakeTrain:
            @staticmethod
            def train(*args):
                idx = E._config_index(self_stub.all_configs)["save_lora_name"] + 5
                Path(self_dir, args[idx] + ".safetensors").write_bytes(b"x")
                return "Trained"
        self_stub, self_dir = self.stub, self.tmp
        prepared = Path(self.tmp) / "prepared"
        prepared.mkdir()
        msg, _ = E.start_training(FakeTrain, self.stub, str(prepared), "", "plain",
                                  "キャラクター", 512, 400, "m.safetensors", "None", "None")
        self.assertIn("トリガーワードなし", msg)
        self.assertNotIn("`` を入れて", msg)

    def test_empty_trigger_is_passed_through(self):
        vals = self.values(trigger="")
        self.assertEqual(self.get(vals, "lora_trigger_word"), "")

    def test_start_training_guards(self):
        msg, path = E.start_training(None, self.stub, "/nonexistent", "t", "n", "キャラクター", 512, 400, "m", "", "")
        self.assertIn("先に", msg)
        self.assertIsNone(path)
        msg, path = E.start_training(None, self.stub, self.tmp, "t", "n", "キャラクター", 512, 400, "", "", "")
        self.assertIn("モデル", msg)
        self.assertIsNone(path)

    def _train_returning(self, result, make_file=None):
        stub, tmp = self.stub, self.tmp

        class FakeTrain:
            @staticmethod
            def train(*args):
                if make_file:
                    Path(tmp, make_file).write_bytes(b"x")
                return result.replace("{dir}", tmp)
        prepared = Path(self.tmp) / "prepared"
        prepared.mkdir(exist_ok=True)
        return E.start_training(FakeTrain, stub, str(prepared), "", "n", "キャラクター",
                                512, 400, "m.safetensors", "None", "None")

    def test_stop_and_save_uses_the_renamed_file(self):
        # 途中で止めて保存すると、TrainTrainは「名前_12steps.safetensors」で保存する
        msg, path = self._train_returning(
            "Stopped. Successfully created to {dir}/n_12steps.safetensors", make_file="n_12steps.safetensors")
        self.assertTrue(path.endswith("n_12steps.safetensors"))
        self.assertIn("⏹", msg)
        self.assertIn("<lora:n_12steps:1>", msg)
        self.assertIn("途中で止めて", msg)
        self.assertNotIn("Stopped", msg)
        self.assertNotIn("Successfully", msg)

    def test_error_result_is_japanese_and_has_no_download(self):
        msg, path = self._train_returning("Error: CUDA out of memory. Tried to allocate 2.00 GiB")
        self.assertIsNone(path)
        self.assertIn("エラーが発生しました", msg)
        self.assertIn("学習解像度を下げて", msg)

    def test_no_data_message(self):
        msg, path = self._train_returning("No data!")
        self.assertIsNone(path)
        self.assertIn("学習に使える画像がありません", msg)
        self.assertNotIn("No data", msg)

    def test_guess_image_size(self):
        self.assertEqual(E.guess_image_size("animagineXL_v3.safetensors"), 1024)
        self.assertEqual(E.guess_image_size("ponyDiffusionV6.safetensors"), 1024)
        self.assertEqual(E.guess_image_size("anything-v5.safetensors [abc]"), 512)
        self.assertEqual(E.resolve_image_size("768", "x"), 768)
        self.assertEqual(E.resolve_image_size("自動", "sdxl_base"), 1024)


# ---------------------------------------------------------------------------
# メッセージの日本語化・保存ファイル探索
# ---------------------------------------------------------------------------

class TestTranslate(unittest.TestCase):
    CASES = {
        "File exist!": "同じ名前のLoRAファイルが既にあります",
        "No Model Selected.": "モデルが選択されていません",
        "Stopped": "学習を途中で停止しました（保存はしていません）",
        "Successfully created to /x/a.safetensors": "LoRAを保存しました: /x/a.safetensors",
        "Stopped. Successfully created to /x/a_9steps.safetensors":
            "学習を途中で止めて、ここまでの結果を保存しました: /x/a_9steps.safetensors",
        "anima can only be trained from inside Forge Neo": "anima はForge Neoの中でのみ学習できます",
        "Test mode": "テストモードのため、学習は行いませんでした",
    }

    def test_known_messages(self):
        for en, ja in self.CASES.items():
            self.assertEqual(E.translate_train_message(en), ja)

    def test_every_message_in_train_py_is_covered(self):
        """train.py が返す文言を実際に抜き出して、日本語化漏れがないことを確認する。"""
        import re
        src = (ROOT / "trainer" / "train.py").read_text(encoding="utf-8")
        found = set(re.findall(r'return "([^"]+)"', src)) | set(re.findall(r'result = "([^"]+)"', src))
        found = {x for x in found if x and not x.startswith(". ")}
        self.assertGreaterEqual(len(found), 6)
        untranslated = [m for m in found if E.translate_train_message(m) == m]
        self.assertEqual(untranslated, [], f"日本語化されていないメッセージ: {untranslated}")

    def test_unknown_messages_are_kept_and_multiline(self):
        self.assertEqual(E.translate_train_message("Something new"), "Something new")
        out = E.translate_train_message("File exist!\nNo data!")
        self.assertEqual(out.splitlines()[0], "同じ名前のLoRAファイルが既にあります")
        self.assertIn("学習に使える画像", out.splitlines()[1])

    def test_find_saved_lora(self):
        with tempfile.TemporaryDirectory() as d:
            a = Path(d, "x.safetensors"); a.write_bytes(b"1")
            b = Path(d, "x_30steps.safetensors"); b.write_bytes(b"1")
            t0 = __import__("time").time() - 1
            self.assertEqual(E.find_saved_lora(f"Successfully created to {b}", d, "x", t0), str(b))
            self.assertEqual(E.find_saved_lora(f"Stopped. Successfully created to {a}", d, "x", t0), str(a))
            self.assertIsNotNone(E.find_saved_lora("Done", d, "x", t0))     # メッセージが無くても探す
            self.assertIsNone(E.find_saved_lora("Done", d, "nope", t0))
            os.utime(a, (1, 1)); os.utime(b, (1, 1))                           # 古いファイルは対象外
            self.assertIsNone(E.find_saved_lora("Done", d, "x", t0))

    def test_prepare_download_copies_to_temp(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, "my.safetensors"); src.write_bytes(b"abc")
            out = E.prepare_download(str(src))
            self.assertEqual(Path(out).read_bytes(), b"abc")
            self.assertNotEqual(os.path.dirname(out), d)


# ---------------------------------------------------------------------------
# メモリ解放
# ---------------------------------------------------------------------------

class TestFreeMemory(unittest.TestCase):
    def setUp(self):
        self._saved = {k: sys.modules.get(k) for k in ("modules", "modules.sd_models", "modules.devices",
                                                       "backend", "backend.memory_management")}
        self.calls = []
        calls = self.calls
        sd = types.ModuleType("modules.sd_models")
        sd.unload_model_weights = lambda *a, **k: calls.append("unload_model_weights")
        sd.checkpoints_loaded = {"cached": object()}
        dev = types.ModuleType("modules.devices")
        dev.torch_gc = lambda: calls.append("torch_gc")
        mods = types.ModuleType("modules"); mods.sd_models = sd; mods.devices = dev
        be = types.ModuleType("backend")
        mm = types.ModuleType("backend.memory_management")
        mm.unload_all_models = lambda: calls.append("unload_all_models")
        mm.soft_empty_cache = lambda: calls.append("soft_empty_cache")
        be.memory_management = mm
        sys.modules.update({"modules": mods, "modules.sd_models": sd, "modules.devices": dev,
                            "backend": be, "backend.memory_management": mm})
        self.sd = sd

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_unloads_everything_when_enabled(self):
        r = E.free_memory(True)
        for must in ("unload_model_weights", "unload_all_models", "soft_empty_cache", "torch_gc"):
            self.assertIn(must, self.calls)
        self.assertEqual(self.sd.checkpoints_loaded, {}, "A1111のチェックポイントキャッシュも空にする")
        self.assertIn("メモリ使用量", r["note"])
        self.assertRegex(r["note"], r"(GB|MB) → ")

    def test_does_not_touch_webui_when_disabled(self):
        E.free_memory(False)
        self.assertEqual(self.calls, [])

    def test_survives_broken_webui_api(self):
        self.sd.unload_model_weights = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        r = E.free_memory(True)         # 例外を外に出さない
        self.assertIn("note", r)

    def test_actually_returns_memory(self):
        import gc
        big = np.ones((200, 1024, 1024), dtype=np.uint8)      # 約200MB
        before = E._memory_snapshot().get("rss")
        del big
        gc.collect()
        r = E.free_memory(False)
        if before is not None:
            self.assertLess(r["after"], before, "確保したメモリがOSに返っている")


# ---------------------------------------------------------------------------
# モデルのダウンロード
# ---------------------------------------------------------------------------

def make_safetensors(n_floats=1000) -> bytes:
    data = np.arange(n_floats, dtype=np.float32).tobytes()
    header = json.dumps({"w": {"dtype": "F32", "shape": [n_floats], "data_offsets": [0, len(data)]}}).encode()
    return len(header).to_bytes(8, "little") + header + data


class FileServer:
    """Range/リダイレクト/途中切断を再現できる、テスト用のHTTPサーバー。"""

    def __init__(self, files):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        outer = self
        self.files, self.requests, self.truncate_first, self.delay = files, [], False, 0

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.requests.append((self.path, self.headers.get("Range")))
                if self.path.startswith("/redirect/"):
                    self.send_response(302)
                    self.send_header("Location", "/files/" + self.path[len("/redirect/"):])
                    self.end_headers()
                    return
                name = self.path[len("/files/"):] if self.path.startswith("/files/") else None
                if name not in outer.files:
                    self.send_response(404); self.end_headers(); return
                body = outer.files[name]
                start, rng = 0, self.headers.get("Range")
                if rng:
                    start = int(rng.split("=")[1].split("-")[0])
                    if start >= len(body):
                        self.send_response(416); self.end_headers(); return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
                else:
                    self.send_response(200)
                chunk = body[start:]
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                if outer.truncate_first and not rng:      # 最初のリクエストだけ途中で切る
                    outer.truncate_first = False
                    self.wfile.write(chunk[: len(chunk) // 2])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                if outer.delay:                            # ゆっくり送る（進捗表示の確認用）
                    import time as _t
                    for i in range(0, len(chunk), 65536):
                        self.wfile.write(chunk[i:i + 65536])
                        self.wfile.flush()
                        _t.sleep(outer.delay)
                else:
                    self.wfile.write(chunk)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def close(self):
        self.server.shutdown()


class TestDownload(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="easylora_dl_"))
        self.blob = make_safetensors(50000)               # 約200KB
        self.srv = FileServer({"m.safetensors": self.blob, "bad.safetensors": self.blob[:-100]})

    def tearDown(self):
        self.srv.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_validate_safetensors(self):
        ok = self.tmp / "ok.safetensors"; ok.write_bytes(self.blob)
        self.assertTrue(E.is_valid_safetensors(ok))
        cut = self.tmp / "cut.safetensors"; cut.write_bytes(self.blob[:-1])
        self.assertFalse(E.is_valid_safetensors(cut), "1バイトでも足りなければ不正")
        junk = self.tmp / "junk.safetensors"; junk.write_bytes(b"<html>Not Found</html>" * 10)
        self.assertFalse(E.is_valid_safetensors(junk))
        self.assertFalse(E.is_valid_safetensors(self.tmp / "none.safetensors"))

    def test_download_with_redirect_and_progress(self):
        calls = []
        dest = E.download_file(self.srv.url("/redirect/m.safetensors"), self.tmp / "out" / "m.safetensors",
                               lambda d, t, s: calls.append((d, t)), chunk=16 * 1024)
        self.assertEqual(Path(dest).read_bytes(), self.blob)
        self.assertFalse((self.tmp / "out" / "m.safetensors.part").exists(), ".partは残さない")
        self.assertEqual(calls[-1], (len(self.blob), len(self.blob)))

    def test_resumes_from_part_file(self):
        part = self.tmp / "m.safetensors.part"
        part.write_bytes(self.blob[:70000])
        E.download_file(self.srv.url("/files/m.safetensors"), self.tmp / "m.safetensors")
        self.assertEqual((self.tmp / "m.safetensors").read_bytes(), self.blob)
        self.assertEqual(self.srv.requests[0][1], "bytes=70000-", "続きから取得している")

    def test_retries_when_connection_drops(self):
        self.srv.truncate_first = True
        E.download_file(self.srv.url("/files/m.safetensors"), self.tmp / "m.safetensors")
        self.assertEqual((self.tmp / "m.safetensors").read_bytes(), self.blob)
        self.assertGreaterEqual(len(self.srv.requests), 2)

    def test_404_is_a_readable_error(self):
        with self.assertRaises(RuntimeError) as cm:
            E.download_file(self.srv.url("/files/missing.safetensors"), self.tmp / "x.safetensors")
        self.assertIn("見つかりません", str(cm.exception))
        self.assertFalse((self.tmp / "x.safetensors").exists())

    def test_find_installed_model_ignores_naming_differences(self):
        spec = E.CatalogModel("l", "r", "split_files/diffusion_models/anima-base-v1.0.safetensors", 4.2)
        ckpt = self.tmp / "ckpt"; ckpt.mkdir()
        orig = E.checkpoint_dir
        E.checkpoint_dir = lambda: ckpt
        try:
            self.assertIsNone(E.find_installed_model(spec))
            (ckpt / "anima_baseV10.safetensors").write_bytes(b"truncated")     # 壊れたファイルは対象外
            self.assertIsNone(E.find_installed_model(spec))
            (ckpt / "anima_baseV10.safetensors").write_bytes(self.blob)
            self.assertEqual(E.find_installed_model(spec).name, "anima_baseV10.safetensors")
        finally:
            E.checkpoint_dir = orig

    def test_catalog_matches_requested_models(self):
        by_file = {m.filename: m for m in E.MODEL_CATALOG}
        self.assertEqual(by_file["Illustrious-XL-v2.0.safetensors"].url,
                         "https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0/resolve/main/Illustrious-XL-v2.0.safetensors")
        self.assertEqual(by_file["anima-base-v1.0.safetensors"].url,
                         "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-base-v1.0.safetensors")
        self.assertTrue(by_file["anima-base-v1.0.safetensors"].needs_modules)

    def _fake_catalog(self, companions=()):
        spec = E.CatalogModel("★ Fake（テスト）", "x/y", "dir/m.safetensors", 0.0, companions)
        spec.__class__ = type("S", (E.CatalogModel,), {"url": property(lambda s: self.srv.url("/redirect/m.safetensors"))})
        return spec

    def test_resolve_model_downloads_once_then_reuses(self):
        spec = self._fake_catalog()
        ckpt = self.tmp / "ckpt"; ckpt.mkdir()
        saved = (dict(E.CATALOG_BY_LABEL), E.checkpoint_dir)
        E.CATALOG_BY_LABEL[spec.label] = spec
        E.checkpoint_dir = lambda: ckpt
        try:
            progress = []
            name = E.resolve_model(spec.label, lambda d, t, s: progress.append(d))
            self.assertEqual(Path(name), ckpt / "m.safetensors")
            self.assertEqual((ckpt / "m.safetensors").read_bytes(), self.blob)
            self.assertTrue(progress)
            n = len(self.srv.requests)
            self.assertEqual(Path(E.resolve_model(spec.label)), ckpt / "m.safetensors")
            self.assertEqual(len(self.srv.requests), n, "2回目はダウンロードしない")
            self.assertEqual(E.resolve_model("plain.safetensors [abc]"), "plain.safetensors [abc]")
        finally:
            E.CATALOG_BY_LABEL.clear(); E.CATALOG_BY_LABEL.update(saved[0]); E.checkpoint_dir = saved[1]



# ---------------------------------------------------------------------------
# 進捗表示
# ---------------------------------------------------------------------------

class TestProgress(unittest.TestCase):
    def test_render(self):
        self.assertEqual(E.render_progress({}), "")
        html = E.render_progress({"title": "学習中", "frac": 0.426, "detail": "a<b"})
        self.assertIn("42%", html)
        self.assertIn("width:42%", html)
        self.assertIn("a&lt;b", html, "HTMLはエスケープする")
        self.assertIn("tt-prog-indeterminate", E.render_progress({"title": "x", "frac": None, "detail": ""}))
        self.assertIn("width:100%", E.render_progress({"title": "x", "frac": 7.0, "detail": ""}))

    def test_format_duration(self):
        self.assertEqual(E.format_duration(None), "計算中")
        self.assertEqual(E.format_duration(45), "45秒")
        self.assertEqual(E.format_duration(125), "2分05秒")
        self.assertEqual(E.format_duration(3700), "1時間01分")

    def test_tqdm_hook_reports_both_phases_and_restores(self):
        import time as _time
        from tqdm import tqdm as real_tqdm
        mod = types.SimpleNamespace(tqdm=real_tqdm)
        tracker = E.ProgressTracker()
        seen = []
        with E.hook_training_progress(mod, tracker, total_steps=40):
            self.assertIsNot(mod.tqdm, real_tqdm)
            # 1) 画像の前処理 (total=画像枚数)
            bar = mod.tqdm(total=5, file=io.StringIO())
            bar.update(2)
            seen.append(tracker.snapshot())
            # 2) 本番の学習ループ (total=step数)
            pbar = mod.tqdm(range(40), file=io.StringIO())
            pbar.set_description("Loss EMA * 1000: 12.3456, Current LR: 1.00e-04, Epoch: 3")
            for _ in range(10):
                _time.sleep(0.01)
                pbar.update(1)
            seen.append(tracker.snapshot())
        self.assertIs(mod.tqdm, real_tqdm, "終了後は元のtqdmに戻す")
        self.assertIn("前処理", seen[0]["title"])
        self.assertAlmostEqual(seen[0]["frac"], 0.4)
        self.assertEqual(seen[1]["title"], "学習中")
        self.assertAlmostEqual(seen[1]["frac"], 0.25)
        for must in ("10 / 40 step", "Epoch 3", "Loss 12.3456", "残り約"):
            self.assertIn(must, seen[1]["detail"])

    def test_hook_restores_even_on_error_and_tolerates_missing_tqdm(self):
        from tqdm import tqdm as real_tqdm
        mod = types.SimpleNamespace(tqdm=real_tqdm)
        with self.assertRaises(ValueError):
            with E.hook_training_progress(mod, E.ProgressTracker(), 10):
                raise ValueError("x")
        self.assertIs(mod.tqdm, real_tqdm)
        with E.hook_training_progress(types.SimpleNamespace(), E.ProgressTracker(), 10):
            pass  # tqdm属性が無くても落ちない


# ---------------------------------------------------------------------------
# Anima の VAE / Text Encoder（自動ダウンロードと一時的な選択）
# ---------------------------------------------------------------------------

def local_companion(kind, label, srv, name):
    comp = E.CompanionFile(kind, label, "x/y", f"split/{name}")
    comp.__class__ = type("LC", (E.CompanionFile,),
                          {"url": property(lambda s: srv.url(f"/redirect/{s.filename}"))})
    return comp


class TestAnimaCompanions(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="easylora_anima_"))
        self.blob = make_safetensors(20000)
        self.srv = FileServer({"qwen_image_vae.safetensors": self.blob,
                               "qwen_3_06b_base.safetensors": self.blob})
        self.vae_dir, self.te_dir = self.tmp / "VAE", self.tmp / "text_encoder"
        self._orig_dirs = E.module_dirs
        E.module_dirs = lambda kind: [self.vae_dir if kind == "vae" else self.te_dir]
        self.comps = (local_companion("vae", "VAE", self.srv, "qwen_image_vae.safetensors"),
                      local_companion("text_encoder", "Text Encoder", self.srv, "qwen_3_06b_base.safetensors"))

    def tearDown(self):
        E.module_dirs = self._orig_dirs
        self.srv.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_looks_like_anima(self):
        for yes in ("anima_baseV10.safetensors", "anima-base-v1.0.safetensors [abc123]",
                    "Anima_preview2.safetensors"):
            self.assertTrue(E.looks_like_anima(yes), yes)
        for no in ("animagineXL_v3.safetensors", "illustriousXL.safetensors", "ponyDiffusionV6.safetensors",
                   "myanima.safetensors", ""):
            self.assertFalse(E.looks_like_anima(no), no)

    def test_companions_only_for_anima(self):
        anima = [m for m in E.MODEL_CATALOG if "anima" in m.filename.lower()][0]
        illust = [m for m in E.MODEL_CATALOG if "illustrious" in m.filename.lower()][0]
        self.assertEqual(E.companions_for(anima.label), E.ANIMA_COMPANIONS)
        self.assertEqual(E.companions_for("anima_baseV10.safetensors"), E.ANIMA_COMPANIONS)
        self.assertEqual(E.companions_for(illust.label), ())
        self.assertEqual(E.companions_for("animagineXL_v3.safetensors"), ())
        self.assertEqual(E.companions_for("sd15.safetensors"), ())

    def test_anima_companion_urls_match_the_requested_ones(self):
        urls = {c.kind: c.url for c in E.ANIMA_COMPANIONS}
        self.assertEqual(urls["vae"],
                         "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors")
        self.assertEqual(urls["text_encoder"],
                         "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors")
        anima = [m for m in E.MODEL_CATALOG if "anima" in m.filename.lower()][0]
        self.assertEqual(anima.companions, E.ANIMA_COMPANIONS)
        self.assertTrue(anima.needs_modules)

    def test_downloads_missing_files_into_the_right_folders(self):
        calls = []
        paths = E.ensure_companions(self.comps, lambda c, d, t, s: calls.append(c.kind), refresh=False)
        self.assertEqual([Path(p) for p in paths],
                         [self.vae_dir / "qwen_image_vae.safetensors", self.te_dir / "qwen_3_06b_base.safetensors"])
        for p in paths:
            self.assertEqual(Path(p).read_bytes(), self.blob)
        self.assertEqual(set(calls), {"vae", "text_encoder"})

    def test_existing_files_are_reused_even_with_different_naming(self):
        self.vae_dir.mkdir(); self.te_dir.mkdir()
        (self.vae_dir / "Qwen-Image-VAE.safetensors").write_bytes(self.blob)
        paths = E.ensure_companions(self.comps, refresh=False)
        self.assertEqual(Path(paths[0]).name, "Qwen-Image-VAE.safetensors")
        self.assertEqual(Path(paths[1]).name, "qwen_3_06b_base.safetensors")
        self.assertEqual([r for r in self.srv.requests if "qwen_image_vae" in r[0]], [],
                         "VAEは既にあるのでダウンロードしない")

    def test_broken_existing_file_is_replaced(self):
        self.vae_dir.mkdir()
        (self.vae_dir / "qwen_image_vae.safetensors").write_bytes(b"truncated")
        paths = E.ensure_companions(self.comps, refresh=False)
        self.assertEqual(Path(paths[0]).read_bytes(), self.blob)


class TestUseWebuiModules(unittest.TestCase):
    def setUp(self):
        self._saved = {k: sys.modules.get(k) for k in ("modules", "modules.shared", "modules_forge",
                                                       "modules_forge.main_entry")}
        self.log = []

        class Opts:
            def __init__(s):
                s.forge_additional_modules = ["/x/user_vae.safetensors"]

            def set(s, key, value):
                setattr(s, key, value)
        log = self.log
        shared = types.ModuleType("modules.shared"); shared.opts = Opts()
        mods = types.ModuleType("modules"); mods.shared = shared
        me = types.ModuleType("modules_forge.main_entry")
        me.refresh_model_loading_parameters = lambda refresh=True: log.append("refresh_params")
        mf = types.ModuleType("modules_forge"); mf.main_entry = me
        sys.modules.update({"modules": mods, "modules.shared": shared,
                            "modules_forge": mf, "modules_forge.main_entry": me})
        self.shared = shared

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_switches_for_training_and_restores_afterwards(self):
        with E.use_webui_modules(["/m/vae.safetensors", "/m/te.safetensors"]) as changed:
            self.assertTrue(changed)
            self.assertEqual(self.shared.opts.forge_additional_modules,
                             ["/m/te.safetensors", "/m/vae.safetensors"])
        self.assertEqual(self.shared.opts.forge_additional_modules, ["/x/user_vae.safetensors"],
                         "ユーザーの選択は元に戻る")
        self.assertEqual(self.log, ["refresh_params"])

    def test_restores_even_if_training_fails(self):
        with self.assertRaises(ValueError):
            with E.use_webui_modules(["/m/a.safetensors"]):
                raise ValueError("boom")
        self.assertEqual(self.shared.opts.forge_additional_modules, ["/x/user_vae.safetensors"])

    def test_no_change_when_already_selected(self):
        with E.use_webui_modules(["/x/user_vae.safetensors"]) as changed:
            self.assertFalse(changed)
        self.assertEqual(self.log, [])

    def test_noop_outside_forge(self):
        del self.shared.opts.forge_additional_modules
        with E.use_webui_modules(["/m/a.safetensors"]) as changed:
            self.assertFalse(changed)


# ---------------------------------------------------------------------------
# 画面状態の組み立て（サーバー側の状態 → 画面）
# ---------------------------------------------------------------------------

def idle_snapshot(**kw):
    snap = {"running": False, "stage": "", "progress": {"title": "", "frac": None, "detail": ""},
            "prep_version": 0, "prep_out": None, "train_version": 0, "train_msg": "",
            "stop_visible": False, "download": {"visible": False}}
    snap.update(kw)
    return snap


class TestComposeUI(unittest.TestCase):
    def test_first_render_sends_everything_then_goes_idle(self):
        up, seen = E.compose_ui(idle_snapshot(), {})
        self.assertEqual(up["prep_progress"], "")
        self.assertEqual(up["train_result"], "")
        self.assertIs(up["stop_visible"], False)
        self.assertTrue(up["active"], "初回は送った直後なので、もう1回だけ問い合わせる")
        up2, _ = E.compose_ui(idle_snapshot(), seen)
        self.assertTrue(all(v is E.KEEP for k, v in up2.items() if k != "active"))
        self.assertFalse(up2["active"], "何も変わらず実行中でもなければ、問い合わせを止める")

    def test_running_sends_only_the_progress_bar_when_it_changes(self):
        snap = idle_snapshot(running=True, stage="train",
                             progress={"title": "学習中", "frac": 0.5, "detail": "5 / 10 step"})
        _, seen = E.compose_ui(idle_snapshot(), {})
        up, seen = E.compose_ui(snap, seen)
        self.assertIn("50%", up["train_progress"])
        self.assertTrue(up["active"])
        same, _ = E.compose_ui(snap, seen)
        self.assertIs(same["train_progress"], E.KEEP, "同じ表示は送り直さない")
        self.assertTrue(same["active"], "実行中は問い合わせを続ける")

    def test_results_are_sent_once_per_version(self):
        out = {"status": "ok", "checkup": "c", "gallery": [("a.jpg", "cap")], "tag_rows": [["t"]],
               "image_rows": [["i"]], "stats": "{}", "zip_file": "/z.zip", "prepared_dir": "/p", "steps": 600}
        snap = idle_snapshot(prep_version=3, prep_out=out)
        up, seen = E.compose_ui(snap, {})
        for k, v in out.items():
            self.assertEqual(up[k], v)
        up2, _ = E.compose_ui(snap, seen)
        self.assertIs(up2["status"], E.KEEP)
        self.assertIs(up2["gallery"], E.KEEP)

    def test_a_returning_browser_catches_up(self):
        """タブを離れて、その間に処理が終わった。戻った時の1回の問い合わせで、完了状態になる。"""
        running = idle_snapshot(running=True, stage="train",
                                progress={"title": "学習中", "frac": 0.4, "detail": "x"})
        _, seen = E.compose_ui(running, {})
        finished = idle_snapshot(train_version=2, train_msg="### ✅ 学習完了",
                                 download={"value": "/t/a.safetensors", "visible": True, "label": "x"})
        up, _ = E.compose_ui(finished, seen)
        self.assertEqual(up["train_progress"], "", "進捗バーは消える")
        self.assertEqual(up["train_result"], "### ✅ 学習完了")
        self.assertTrue(up["download"]["visible"])

    def test_reload_restores_everything(self):
        snap = idle_snapshot(train_version=2, train_msg="### ✅ 学習完了",
                             download={"value": "/t/a.safetensors", "visible": True, "label": "x"})
        up, _ = E.compose_ui(snap, {})
        self.assertEqual(up["train_result"], "### ✅ 学習完了")
        self.assertTrue(up["download"]["visible"])


# ---------------------------------------------------------------------------
# ジョブ実行（サーバー側に状態を持つ）
# ---------------------------------------------------------------------------

import time


def make_fake_train_module(stub, lora_dir, load_s=0.0, step_s=0.01, loop_hook=None):
    """train.py の挙動を真似た偽物。特に「停止フラグは学習ループの直前にリセットされる」点を再現する。"""
    from tqdm import tqdm
    idx = E._config_index(stub.all_configs)

    class FakeTrainModule:
        def __init__(self):
            self.tqdm = tqdm
            self.stopped = []
            self.flag = 0
            self.args = []

        def stop_time(self, save):
            self.stopped.append(save)
            self.flag = 2 if save else 1

        def train(self, *args):
            self.args.append(args)
            steps = int(args[5 + idx["train_iterations"]])
            name = args[5 + idx["save_lora_name"]]
            time.sleep(load_s)                                  # モデルの読み込み
            if loop_hook:
                loop_hook(self)
            self.flag = 0                                       # 本物の train_lora と同じ位置でリセット
            bar = self.tqdm(range(steps), file=io.StringIO())
            for i in range(steps):
                if self.flag > 0:
                    if self.flag > 1:
                        p = Path(lora_dir, f"{name}_{i}steps.safetensors")
                        p.write_bytes(b"LORA")
                        return f"Stopped. Successfully created to {p}"
                    return "Stopped"
                time.sleep(step_s)
                bar.update(1)
            p = Path(lora_dir, f"{name}.safetensors")
            p.write_bytes(b"LORA")
            return f"Successfully created to {p}"

    return FakeTrainModule()


class RunnerBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="easylora_runner_"))
        self.ext_root = self.tmp / "ext"; self.ext_root.mkdir()
        self._o = (E._extension_root, E.WD14Tagger, E.checkpoint_dir, E._RUNNER)
        E._extension_root = lambda: self.ext_root
        E.WD14Tagger = FakeTagger
        FakeTagger.script = staticmethod(character_script)
        self.ckpt = self.tmp / "ckpt"; self.ckpt.mkdir()
        E.checkpoint_dir = lambda: self.ckpt
        E._RUNNER = None
        self.src = self.tmp / "src"; self.src.mkdir()
        for i in range(6):
            structured_image(i).save(self.src / f"img{i}.png")
        self.stub = load_real_configs()
        self.lora_dir = self.tmp / "lora"; self.lora_dir.mkdir()
        self.stub.lora_dir = str(self.lora_dir)
        (self.ckpt / "plain.safetensors").write_bytes(make_safetensors(100))

    def tearDown(self):
        E._extension_root, E.WD14Tagger, E.checkpoint_dir, E._RUNNER = self._o
        shutil.rmtree(self.tmp, ignore_errors=True)

    def prep_kwargs(self, **kw):
        d = dict(files=None, folder_path=str(self.src), dataset_name="job", preset_name="キャラクター",
                 trigger="", general_threshold=0.35, character_threshold=0.85,
                 manual_keep_text="", manual_remove_text="", auto_threshold=True, free_mem=False)
        d.update(kw)
        return d

    def train_kwargs(self, **kw):
        d = dict(prepared_dir=None, trigger="", dataset_name="job", output_name="", preset_name="キャラクター",
                 size_choice="自動", steps=30, model="plain.safetensors", vae="None", te="None", free_mem=False)
        d.update(kw)
        return d

    def wait(self, runner, timeout=30):
        runner.join(timeout)
        self.assertFalse(runner.running, "処理が終わらなかった")

    def record_titles(self, runner):
        """進捗の更新をすべて記録する（ポーリングだと、短い表示を取り逃がすことがあるため）。"""
        titles = []
        original = runner.tracker.set

        def recording(title, frac=None, detail=""):
            if title and (not titles or titles[-1] != title):
                titles.append(title)
            original(title, frac, detail)
        runner.tracker.set = recording
        return titles

    def wait_for(self, condition, timeout=10, what="条件"):
        """条件が満たされるまで待つ。無限には待たず、満たされなければテストを失敗させる。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if condition():
                return
            time.sleep(0.01)
        self.fail(f"{timeout}秒待っても {what} になりませんでした")


class TestJobRunner(RunnerBase):
    def test_prep_only(self):
        tm = make_fake_train_module(self.stub, self.lora_dir)
        runner = E.JobRunner(tm, self.stub)
        self.assertIsNone(runner.start({"prep": self.prep_kwargs(), "train": None}))
        self.wait(runner)
        snap = runner.snapshot()
        self.assertIn("✅ 学習データの準備ができました", snap["prep_out"]["status"])
        self.assertIn("準備済みデータで学習だけ実行", snap["prep_out"]["status"])
        self.assertEqual(tm.args, [], "準備だけの時は学習しない")
        self.assertEqual(snap["train_msg"], "")

    def test_prep_failure_is_reported_not_raised(self):
        runner = E.JobRunner(make_fake_train_module(self.stub, self.lora_dir), self.stub)
        runner.start({"prep": self.prep_kwargs(folder_path=str(self.tmp / "nope")), "train": None})
        self.wait(runner)
        self.assertIn("❌ 準備に失敗しました", runner.snapshot()["prep_out"]["status"])
        self.assertFalse(runner.snapshot()["running"])

    def test_one_click_flow_and_phases(self):
        tm = make_fake_train_module(self.stub, self.lora_dir, load_s=0.6, step_s=0.02)
        runner = E.JobRunner(tm, self.stub)
        titles, details = [], []
        runner.start({"prep": self.prep_kwargs(), "train": self.train_kwargs()})
        deadline = time.time() + 40
        while runner.running and time.time() < deadline:
            p = runner.snapshot()["progress"]
            if p["title"] and (not titles or titles[-1] != p["title"]):
                titles.append(p["title"]); details.append(p["detail"])
            time.sleep(0.02)
        self.wait(runner)
        snap = runner.snapshot()
        self.assertIn("✅ 学習完了", snap["train_msg"])
        self.assertIn("LoRAを保存しました", snap["train_msg"])
        self.assertTrue(snap["download"]["visible"] and os.path.isfile(snap["download"]["value"]))
        self.assertFalse(snap["stop_visible"])
        joined = " ".join(titles)
        for phase in ("準備中", "メモリを空けています", "モデルを読み込み中", "学習中"):
            self.assertIn(phase, joined)
        # 要望: モデルの読み込み中は、メモリ使用量ではなく「読み込み中」のメッセージを出す
        i = titles.index("モデルを読み込み中")
        self.assertNotIn("メモリ使用量", details[i])
        self.assertIn("読み込んでいます", details[i])
        self.assertEqual(snap["prep_out"]["steps"], 400)
        idx = E._config_index(self.stub.all_configs)
        self.assertEqual(int(tm.args[0][5 + idx["train_iterations"]]), 400,
                         "おまかせ実行では、準備で決まった学習stepを使う")

    def test_second_start_is_refused_while_running(self):
        runner = E.JobRunner(make_fake_train_module(self.stub, self.lora_dir, load_s=0.5), self.stub)
        runner.start({"prep": None, "train": self.train_kwargs(prepared_dir=str(self.src))})
        refusal = runner.start({"prep": None, "train": self.train_kwargs(prepared_dir=str(self.src))})
        self.assertIn("すでに別の処理を実行中", refusal)
        self.wait(runner)

    def test_stop_during_training_saves(self):
        tm = make_fake_train_module(self.stub, self.lora_dir, step_s=0.05)
        runner = E.JobRunner(tm, self.stub)
        runner.start({"prep": None, "train": self.train_kwargs(prepared_dir=str(self.src), steps=200)})
        self.wait_for(lambda: runner.loop_started, what="学習ループの開始")
        time.sleep(0.2)
        self.assertIn("保存して止まります", runner.request_stop())
        self.wait(runner)
        self.assertEqual(tm.stopped, [True])
        snap = runner.snapshot()
        self.assertIn("⏹", snap["train_msg"])
        self.assertIn("steps.safetensors", snap["download"]["value"], "停止保存の別名ファイルがダウンロードできる")

    def test_stop_while_loading_does_not_leave_a_one_step_lora(self):
        tm = make_fake_train_module(self.stub, self.lora_dir, load_s=0.6, step_s=0.02)
        runner = E.JobRunner(tm, self.stub)
        runner.start({"prep": None, "train": self.train_kwargs(prepared_dir=str(self.src), steps=200)})
        self.wait_for(lambda: runner.snapshot()["progress"]["title"] == "モデルを読み込み中", what="モデル読み込み中")
        self.assertIn("読み込みが終わった時点", runner.request_stop())
        self.wait(runner)
        self.assertEqual(tm.stopped, [False], "保存しない停止で止める")
        self.assertEqual(list(self.lora_dir.glob("*.safetensors")), [])
        self.assertFalse(runner.snapshot()["download"]["visible"])

    def test_stop_when_idle(self):
        runner = E.JobRunner(make_fake_train_module(self.stub, self.lora_dir), self.stub)
        self.assertIn("実行中の処理はありません", runner.request_stop())

    def test_training_error_is_reported_and_runner_recovers(self):
        tm = make_fake_train_module(self.stub, self.lora_dir)
        tm.train = lambda *a: "Error: CUDA out of memory. Tried to allocate 2 GiB"
        runner = E.JobRunner(tm, self.stub)
        runner.start({"prep": None, "train": self.train_kwargs(prepared_dir=str(self.src))})
        self.wait(runner)
        msg = runner.snapshot()["train_msg"]
        self.assertIn("エラーが発生しました", msg)
        self.assertIn("学習解像度を下げて", msg)
        self.assertIsNone(runner.start({"prep": None, "train": self.train_kwargs(prepared_dir=str(self.src))}))
        self.wait(runner)

    def test_get_runner_survives_ui_reload(self):
        tm = make_fake_train_module(self.stub, self.lora_dir)
        self.assertIs(E.get_runner(tm, self.stub), E.get_runner(tm, self.stub))

    def test_work_finishes_without_anyone_watching(self):
        """画面(ブラウザ)が1度も問い合わせに来なくても、処理は最後まで進み、結果が保持される。"""
        tm = make_fake_train_module(self.stub, self.lora_dir, step_s=0.01)
        runner = E.JobRunner(tm, self.stub)
        runner.start({"prep": self.prep_kwargs(), "train": self.train_kwargs()})
        self.wait(runner)                                     # snapshot() は一度も呼ばない
        up, _ = E.compose_ui(runner.snapshot(), {})           # 後から戻ってきたブラウザ
        self.assertIn("✅ 学習完了", up["train_result"])
        self.assertTrue(up["download"]["visible"])
        self.assertIn("準備ができました", up["status"])


class TestJobRunnerDownloadsAndAnima(RunnerBase):
    def setUp(self):
        super().setUp()
        self.blob = make_safetensors(400000)            # 約1.6MB（ダウンロード中の状態を観測できる大きさ）
        self.srv = FileServer({"m.safetensors": self.blob, "vae.safetensors": self.blob, "te.safetensors": self.blob})
        self.vae_dir, self.te_dir = self.tmp / "VAE", self.tmp / "text_encoder"
        self._om = E.module_dirs
        E.module_dirs = lambda kind: [self.vae_dir if kind == "vae" else self.te_dir]
        self.companions = (local_companion("vae", "VAE", self.srv, "vae.safetensors"),
                           local_companion("text_encoder", "Text Encoder", self.srv, "te.safetensors"))
        self._ocat = dict(E.CATALOG_BY_LABEL)
        spec = E.CatalogModel("★ Anima（テスト）", "x/y", "d/m.safetensors", 0.0, self.companions)
        spec.__class__ = type("S", (E.CatalogModel,), {"url": property(lambda s: self.srv.url("/redirect/m.safetensors"))})
        E.CATALOG_BY_LABEL[spec.label] = spec
        self.spec = spec

        class Opts:
            forge_additional_modules = ["/user/own.safetensors"]

            def set(s, k, v):
                setattr(s, k, v)
        self._saved = {k: sys.modules.get(k) for k in ("modules", "modules.shared")}
        shared = types.ModuleType("modules.shared"); shared.opts = Opts()
        mods = types.ModuleType("modules"); mods.shared = shared
        sys.modules.update({"modules": mods, "modules.shared": shared})
        self.opts = shared.opts

    def tearDown(self):
        E.module_dirs = self._om
        E.CATALOG_BY_LABEL.clear(); E.CATALOG_BY_LABEL.update(self._ocat)
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        self.srv.close()
        super().tearDown()

    def test_anima_downloads_everything_and_uses_it_only_during_training(self):
        seen_during = {}
        tm = make_fake_train_module(self.stub, self.lora_dir,
                                    loop_hook=lambda t: seen_during.update(m=list(self.opts.forge_additional_modules)))
        runner = E.JobRunner(tm, self.stub)
        runner.start({"prep": None, "train": self.train_kwargs(prepared_dir=str(self.src), model=self.spec.label)})
        self.wait(runner)
        self.assertIn("✅ 学習完了", runner.snapshot()["train_msg"])
        self.assertEqual((self.ckpt / "m.safetensors").read_bytes(), self.blob)
        self.assertEqual((self.vae_dir / "vae.safetensors").read_bytes(), self.blob)
        self.assertEqual((self.te_dir / "te.safetensors").read_bytes(), self.blob)
        want = sorted(os.path.normpath(str(p)) for p in (self.vae_dir / "vae.safetensors", self.te_dir / "te.safetensors"))
        self.assertEqual(seen_during["m"], want, "学習中は、ダウンロードしたVAE/TEが使われる")
        self.assertEqual(self.opts.forge_additional_modules, ["/user/own.safetensors"], "学習後は元の選択に戻る")

    def test_second_run_does_not_download_again(self):
        tm = make_fake_train_module(self.stub, self.lora_dir)
        runner = E.JobRunner(tm, self.stub)
        for _ in range(2):
            runner.start({"prep": None, "train": self.train_kwargs(prepared_dir=str(self.src), model=self.spec.label)})
            self.wait(runner)
        files = [r[0] for r in self.srv.requests if r[0].startswith("/redirect/")]
        self.assertEqual(sorted(files), ["/redirect/m.safetensors", "/redirect/te.safetensors", "/redirect/vae.safetensors"])

    def test_non_anima_model_never_touches_modules_or_downloads_companions(self):
        seen_during = {}
        tm = make_fake_train_module(self.stub, self.lora_dir,
                                    loop_hook=lambda t: seen_during.update(m=list(self.opts.forge_additional_modules)))
        runner = E.JobRunner(tm, self.stub)
        runner.start({"prep": None, "train": self.train_kwargs(prepared_dir=str(self.src), model="plain.safetensors")})
        self.wait(runner)
        self.assertEqual(seen_during["m"], ["/user/own.safetensors"])
        self.assertFalse(self.vae_dir.exists() or self.te_dir.exists())

    def test_progress_names_each_download(self):
        tm = make_fake_train_module(self.stub, self.lora_dir)
        runner = E.JobRunner(tm, self.stub)
        titles = self.record_titles(runner)
        runner.start({"prep": None, "train": self.train_kwargs(prepared_dir=str(self.src), model=self.spec.label)})
        self.wait(runner)
        self.assertIn("モデルをダウンロード中", titles)
        self.assertIn("VAEをダウンロード中", titles)
        self.assertIn("Text Encoderをダウンロード中", titles)
        order = [titles.index(t) for t in ("モデルをダウンロード中", "VAEをダウンロード中",
                                           "Text Encoderをダウンロード中", "モデルを読み込み中", "学習中")]
        self.assertEqual(order, sorted(order), f"表示の順番がおかしい: {titles}")

    def test_stop_during_model_download_cancels_and_keeps_partial_file(self):
        self.srv.delay = 0.08
        tm = make_fake_train_module(self.stub, self.lora_dir)
        runner = E.JobRunner(tm, self.stub)
        runner.start({"prep": None, "train": self.train_kwargs(prepared_dir=str(self.src), model=self.spec.label)})
        self.wait_for(lambda: runner.snapshot()["progress"]["title"] == "モデルをダウンロード中",
                      what="モデルのダウンロード中")
        time.sleep(0.4)
        runner.request_stop()
        self.wait(runner)
        self.assertIn("中止しました", runner.snapshot()["train_msg"])
        self.assertEqual(tm.args, [], "学習は始まらない")
        self.assertFalse((self.ckpt / "m.safetensors").exists())
        self.assertTrue((self.ckpt / "m.safetensors.part").exists(), "次回は続きから再開できる")


class TestTabName(unittest.TestCase):
    def test_tab_is_named_lora_gakushu_but_keeps_internal_id(self):
        src = (ROOT / "scripts" / "traintrain.py").read_text(encoding="utf-8")
        self.assertIn('return (ui, "LoRA学習", "TrainTrain"),', src)
        self.assertNotIn('(ui, "TrainTrain", "TrainTrain")', src)


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

    def test_live_outputs_disable_gradio_pending_fade(self):
        """Timer polling must not dim the progress/status components between updates."""
        import gradio as gr
        with gr.Blocks() as demo:
            E.build_easy_tab(types.SimpleNamespace(), load_real_configs(), gradio_module=gr)

        live_outputs = [
            block for block in demo.blocks.values()
            if "tt-no-flicker" in (getattr(block, "elem_classes", None) or [])
        ]
        self.assertEqual(len(live_outputs), 5, "2 progress HTML + 3 dynamically polled Markdown outputs")

        style_blocks = [
            getattr(block, "value", "") for block in demo.blocks.values()
            if type(block).__name__ == "HTML"
        ]
        self.assertTrue(
            any(".tt-no-flicker .pending" in str(value) and "opacity: 1 !important" in str(value)
                for value in style_blocks),
            "Gradio's pending opacity animation must be disabled only for live outputs",
        )

    def test_build_tab_without_models(self):
        import gradio as gr
        with gr.Blocks():
            E.build_easy_tab(types.SimpleNamespace(), load_real_configs(), gradio_module=gr)

    def test_catalog_models_are_in_dropdown(self):
        import gradio as gr
        with gr.Blocks() as demo:
            E.build_easy_tab(types.SimpleNamespace(), load_real_configs(),
                             model_choices=["local.safetensors"], default_model="local.safetensors",
                             gradio_module=gr)
        dropdowns = [b for b in demo.blocks.values() if type(b).__name__ == "Dropdown"
                     and b.label == "モデル（チェックポイント）"]
        self.assertEqual(len(dropdowns), 1)
        values = [c[0] if isinstance(c, (tuple, list)) else c for c in dropdowns[0].choices]
        self.assertEqual(values[:2], [m.label for m in E.MODEL_CATALOG], "★付きの候補が先頭")
        self.assertIn("local.safetensors", values)
        self.assertEqual(dropdowns[0].value, "local.safetensors", "既定は今使っているモデル")

    def test_no_local_models_defaults_to_first_catalog_model(self):
        import gradio as gr
        with gr.Blocks() as demo:
            E.build_easy_tab(types.SimpleNamespace(), load_real_configs(), gradio_module=gr)
        dd = [b for b in demo.blocks.values() if type(b).__name__ == "Dropdown"
              and b.label == "モデル（チェックポイント）"][0]
        self.assertEqual(dd.value, E.MODEL_CATALOG[0].label)

    def test_image_size_guess_works_for_catalog_labels(self):
        for m in E.MODEL_CATALOG:
            self.assertEqual(E.guess_image_size(m.label), 1024)

    def test_preset_labels(self):
        for k, v in E.PRESETS.items():
            label = f"{k}｜{v['short']}"
            self.assertEqual(E.preset_key_from_label(label), k)
        self.assertEqual(E.preset_key_from_label("???"), E.DEFAULT_PRESET)


if __name__ == "__main__":
    unittest.main(verbosity=2)
