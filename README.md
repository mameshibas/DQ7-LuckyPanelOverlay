# DQ7R Lucky Panel Realtime — 画面認識 + 透過オーバーレイ

DQ7R（Steam）のミニゲーム「ラッキーパネル」盤面を **画面キャプチャして解析**し、推定したタイル画像を **Qt6 の透明オーバーレイ**で盤面上に重ねて表示するツールです。  
Steam版で動作確認を行っていますが、キャプチャーボード等を使用してPC画面上に表示可能であればSwitch/Switch2/Playstation版でも構造上は動作可能です。

- 盤面は台形（透視）なので、**ROI → 正対化(work) → 解析 → work → ROIへ逆ワープ**して重ねます
- **裏面（カード裏面）だけ**にオーバーレイを描画（表面には描かない/透明）
- オーバーレイがキャプチャに写り込んで点滅する問題を避けるため、Windowsの **Display Affinity** を使用します

> コード本体: `dq7_lucky_panel_realtime.py`

---
## DEMO

※DEMOは無理やり撮影した影響で一部ちらつきが発生しています。
---

## 主な機能

- **ウィンドウキャプチャ**: `mss`（ウィンドウ矩形は `DwmGetWindowAttribute(DWMWA_EXTENDED_FRAME_BOUNDS)` 優先）  
  キャプチャするウインドウに別のウインドウが重なっている場合、一緒にキャプチャされてしまい正しく機能しません。  
  ただし、当ツールで表示するOverlayはキャプチャされません。
- **ROI自動推定**: `auto_detect_board_roi()`（失敗時は `cv2.selectROI` で手動）
- **透視補正（正対化）**:
  - `Alt+p` で ROI画像上の4点をクリックして `perspective.json` に保存（ROI相対 0..1）
  - `warp_board()` で ROI → work (out_w,out_h) に正対化
- **グリッド推定**: `detect_grid_best()`
  - 第1候補: 溝（gap）法 + 透視
  - フォールバック: 裏面HSV / 表面矩形 / projection
- **タイル初期保存**: `Alt+i` で全セルのサムネイルを保存
- **swap追跡**: 青率 + 平均輝度で空きマス2つを検出し、交換履歴で `layout` 更新
- **オーバーレイ**:
  - `Overlay(QWidget)` + `QLabel` に BGRA画像を表示
  - 枠線は `paintEvent` で描画（画像に直接描かない）
  - `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)` を有効化


---

## 動作環境

- Windows 11  
  - `WDA_EXCLUDEFROMCAPTURE` は Windows 10 2004+ が前提です
- Python 3.10+ 推奨
- 必要ライブラリ（例）
  - `opencv-python`
  - `numpy`
  - `mss`
  - `PyQt6`
  - `pywin32`
  - `pynput`

インストール例:

```bash
pip install opencv-python numpy mss PyQt6 pywin32 pynput
```

---

## ファイル配置

同じフォルダに以下がある想定です。

- `dq7_lucky_panel_realtime.py`（本体）
- `perspective.json`（透視4点。`Alt+p`で自動生成）
- テンプレ画像（任意/推奨）
  - `back_template.png`（裏面検出用。推奨）
  - `start_template.png`（START検出用。任意）
  - `stop_template.png`（STOP検出用。任意）
  - `end_template.png`（未使用。定数として残っています）

> テンプレは「そのままのスクリーンショット」から切り出した画像を推奨します（解像度/スケールが合っているほど安定）。


---

## 使い方

### 1) 起動

```bash
python dq7_lucky_panel_realtime.py
```

- 既定では `DEFAULT_TARGET_WINDOW_TITLE = "DRAGON QUEST VII Reimagined"` を探します  
- 見つからない場合は、コンソール上でウィンドウ選択が表示されます

### 2) 自動セットアップ
- 環境にあったtemplate画像があると、自動でグリッド推定 → Trackingが始まります。  
  シャッフルが終わるとTrackingが終わりオーバーレイが表示されます。

- グリッド推定が開始されない、もしくは緑枠の範囲がずれている場合は**Alt + a**で手動実行ができます。


### 3) 必要なら透視補正を保存

- **Alt + p** : ROI画像上で盤面四隅を4点クリック → `perspective.json` 保存  
  - OpenCVのウィンドウにフォーカスが必要です
  - 保存後はROIサイズが変わっても（ROI相対保存なので）再利用できます

---

## ホットキー一覧（Alt + key）

- `r` : ROIを自動推定（失敗時は手動 `selectROI`）
- `p` : 盤面4隅クリック → 透視補正保存（ROI相対）
- `g` : グリッド推定（gap→裏→表→投影）
- `i` : 全オープン状態を保存（初期サムネ）
- `t` : 追跡ON/OFF
- `a` : （おすすめ）ROI更新→G→I→追跡ON
- `q` : 終了

---

## オーバーレイ表示について

- オーバーレイは「入力透過」になっています（クリックなどはゲーム側へ通ります）
- オーバーレイがキャプチャに写り込むと点滅するため、起動時に以下を有効化しています:
  - `SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)`

環境によっては `exclude-from-capture: False` と出ることがあります。その場合:
- Windowsバージョン（10 2004+）か確認
- 権限/互換性設定（管理者実行など）を見直してください

---

## 裏面だけ描画

裏面セルの判定は `back_template.png` を使い、各セルを中央クロップ→エッジ化→テンプレマッチでスコア化します。
青成分も併用したハイブリッド判定になっています。

- `BACK_MATCH_THRESH` を調整すると検出が安定します。（デフォルト `0.10`）  
  認識率が低いので閾値を低く設定していますが、動作上は安定しています。

デバッグ:
- `DEBUG_SHOW_BACK_DETECTION = True` にすると、裏面判定セルに枠とスコアを表示します

---

## よくあるトラブルシューティング

### ROIがうまく取れない
- Alt+r を何回か
- それでもダメなら `selectROI` で手動選択
- 窓の縁や影が入りすぎる場合は、ゲーム画面の表示倍率/フルスクリーン設定も影響します

### グリッドが崩れる/行列数が変になる
- まず Alt+p で透視4点を設定し、gap法（溝）で安定させるのが有効です
- 盤面のUI演出で溝が見えにくいタイミングだと失敗しやすいです（少し待ってから Alt+g）

### 裏面判定が不安定
- `back_template.png` を取り直す（同じ解像度・同じ表示状態から切り出す）
- `BACK_MATCH_THRESH` を少し上げ下げして様子を見る
- `DEBUG_SHOW_BACK_DETECTION = True` でスコア感を確認する
