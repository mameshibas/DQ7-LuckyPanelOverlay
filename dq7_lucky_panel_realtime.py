from __future__ import annotations

import sys
import time
import json
import threading
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import cv2
import mss
import numpy as np
from pynput import keyboard as pynput_keyboard
from ctypes import wintypes
from PyQt6 import QtCore, QtGui, QtWidgets
import win32gui
import win32con
import ctypes

# ==========================================
# DQ7R Lucky Panel Realtime (Steam)
#  - 自動ROI（毎回取り直し可）
#  - 透視補正Pは「ROI相対(0..1)」で保存 → ROIサイズが変わっても有効
#  - グリッド推定：gap(溝)法(推奨) → 裏/表/投影フォールバック
#  - Alt+a で「このラウンドの準備」を一括（ROI更新→G→I→追跡ON）
#
# 操作 (Alt + key):
#   r : ROIを自動推定（失敗時は手動selectROI）
#   p : 盤面4隅クリック → 透視補正保存（ROI相対）
#   g : グリッド推定（gap→裏→表→投影）
#   i : 全オープン状態を保存（初期サムネ）
#   t : 追跡ON/OFF
#   a : (おすすめ) ROI更新→G→I →追跡ON
#   q : 終了
# ==========================================

Pos = Tuple[int, int]  # (r, c)
DEBUG = False
DEBUG_SHOW_BACK_DETECTION = False

PERSPECTIVE_FILE = Path("perspective.json")

SWAP_LIMITS_BY_GRID = {
    # (4, 5): 5,
}
SWAP_LIMITS_BY_CELLS = {
    # 25: 7,
}

START_TEMPLATE_PATH = Path("start_template.png")
STOP_TEMPLATE_PATH = Path("stop_template.png")
UNVISIBLE_TEMPLATE_PATH = Path("end_template.png")
BACK_TEMPLATE_PATH = Path("back_template.png")

STOP_MATCH_THRESH = 0.95
START_MATCH_THRESH = 0.95
UNVISIBLE_MATCH_THRESH = 0.95
BACK_MATCH_THRESH = 0.10

TEMPLATE_CHECK_EVERY_N_FRAMES = 3
TEMPLATE_SEARCH_TARGET = "screen"  # "screen" | "roi" | "work"
STOP_COOLDOWN_SEC = 2.0
START_COOLDOWN_SEC = 2.0
UNVISIBLE_COOLDOWN_SEC = 2.0

DWMWA_EXTENDED_FRAME_BOUNDS = 9

PERSPECTIVE_INSET_RATIO = 0.00

# 既定のターゲットウィンドウタイトル（空なら選択UI）
DEFAULT_TARGET_WINDOW_TITLE = "DRAGON QUEST VII Reimagined"

WDA_NONE = 0x0
WDA_EXCLUDEFROMCAPTURE = 0x11  # Windows 10 2004+


# =========================
# 設定（環境で調整しやすい箇所）
# =========================
@dataclass
class Params:
    lower_blue: Tuple[int, int, int] = (90, 60, 60)
    upper_blue: Tuple[int, int, int] = (140, 255, 255)

    blue_frac_thresh: float = 0.12
    mean_bgr_thresh: float = 150.0

    empty_streak_frames: int = 2
    min_card_area: int = 20000

    row_tol: float = 40.0
    col_tol: float = 60.0

    mosaic_tile: int = 120

    auto_roi_min_area_ratio: float = 0.03

    # gap法（溝）
    gap_inv_thresh: int = 90
    gap_open_ratio: float = 0.40
    gap_smooth_ratio: float = 0.02


P = Params()


def set_exclude_from_capture(hwnd: int, enabled: bool) -> bool:
    try:
        val = WDA_EXCLUDEFROMCAPTURE if enabled else WDA_NONE
        ok = ctypes.windll.user32.SetWindowDisplayAffinity(wintypes.HWND(hwnd), val)
        return bool(ok)
    except Exception:
        return False


# =========================
# Qt Overlay
# =========================
class Overlay(QtWidgets.QWidget):
    def __init__(self, alpha=0.60):
        super().__init__()
        self._last_geo = None
        self.alpha = float(alpha)

        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        self.border_enabled = False
        self.border_color = QtGui.QColor(0, 255, 0)
        self.border_thickness = 3

        self.image_enabled = True
        self.show()

        hwnd = int(self.winId())  # Qtのウィンドウハンドル(HWND)
        ok = set_exclude_from_capture(hwnd, True)
        print("[Overlay] exclude-from-capture:", ok)

    def set_border(self, enabled: bool, color_bgr=(0, 255, 0), thickness: int = 3):
        self.border_enabled = bool(enabled)
        b, g, r = color_bgr
        self.border_color = QtGui.QColor(int(r), int(g), int(b))
        self.border_thickness = int(thickness)
        self.update()

    def set_image_enabled(self, enabled: bool):
        self.image_enabled = bool(enabled)
        self.label.setVisible(self.image_enabled)
        if not self.image_enabled:
            self.label.clear()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):
        super().paintEvent(event)
        if not self.border_enabled:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        pen = QtGui.QPen(self.border_color)
        pen.setWidth(self.border_thickness)
        p.setPen(pen)

        t = max(1, self.border_thickness)
        r = self.rect().adjusted(t // 2, t // 2, -(t // 2), -(t // 2))
        p.drawRect(r)
        p.end()

    def update_bgr(self, img: np.ndarray):
        if not self.image_enabled:
            return
        if img is None or img.size == 0:
            return
        if img.ndim != 3 or img.shape[2] not in (3, 4):
            return

        h, w = img.shape[:2]

        if img.shape[2] == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg)

            pm = QtGui.QPixmap(pix.size())
            pm.fill(QtCore.Qt.GlobalColor.transparent)
            painter = QtGui.QPainter(pm)
            painter.setOpacity(self.alpha)
            painter.drawPixmap(0, 0, pix)
            painter.end()
        else:
            rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            qimg = QtGui.QImage(rgba.data, w, h, 4 * w, QtGui.QImage.Format.Format_RGBA8888)
            pix = QtGui.QPixmap.fromImage(qimg)

            pm = QtGui.QPixmap(pix.size())
            pm.fill(QtCore.Qt.GlobalColor.transparent)
            painter = QtGui.QPainter(pm)
            painter.setOpacity(self.alpha)
            painter.drawPixmap(0, 0, pix)
            painter.end()

        self.label.setPixmap(pm)
        self.label.resize(w, h)
        self.resize(w, h)

    def place_at(self, x: int, y: int, w: int, h: int):
        geo = (int(x), int(y), int(w), int(h))
        if self._last_geo == geo:
            return
        self._last_geo = geo
        self.setGeometry(*geo)


# =========================
# DPI / Window geometry / Capture
# =========================
def get_window_rect(hwnd: int) -> tuple[int, int, int, int]:
    rect = wintypes.RECT()
    try:
        ctypes.windll.dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd),
            wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
            ctypes.byref(rect),
            ctypes.sizeof(rect),
        )
        return rect.left, rect.top, rect.right, rect.bottom
    except Exception:
        l, t, r, b = win32gui.GetWindowRect(hwnd)
        return l, t, r, b


def list_windows() -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []

    def enum_cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return
        if win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) & win32con.WS_EX_TOOLWINDOW:
            return
        out.append((hwnd, title))

    win32gui.EnumWindows(enum_cb, None)
    return out


def choose_window_interactive() -> int:
    wins = list_windows()
    print("\n=== Windows ===")
    for i, (hwnd, title) in enumerate(wins):
        print(f"[{i}] hwnd=0x{hwnd:08X}  {title}")

    while True:
        s = input("Select window index (or type part of title): ").strip()
        if s.isdigit():
            idx = int(s)
            if 0 <= idx < len(wins):
                hwnd = wins[idx][0]
                print(f"Using: {wins[idx][1]}")
                return hwnd

        if s:
            cand = [(h, t) for (h, t) in wins if s.lower() in t.lower()]
            if len(cand) == 1:
                print(f"Using: {cand[0][1]}")
                return cand[0][0]
            if len(cand) > 1:
                print("Multiple matches:")
                for h, t in cand[:20]:
                    print(f"  - {t}")

        print("Invalid selection. Try again.")


def find_window_by_title_prefer_default(default_title: str) -> int:
    default_title = (default_title or "").strip()
    if default_title:
        wins = list_windows()

        for hwnd, title in wins:
            if title == default_title:
                print(f"[WIN] Found exact title: {title}")
                return hwnd

        cand = [(h, t) for (h, t) in wins if default_title.lower() in t.lower()]
        if len(cand) == 1:
            print(f"[WIN] Found partial title: {cand[0][1]}")
            return cand[0][0]

        print(f"[WIN] Default title not found: '{default_title}' -> fallback to interactive selection")

    return choose_window_interactive()


def grab_window_bgr(sct: mss.mss, hwnd: int) -> Optional[np.ndarray]:
    if not win32gui.IsWindow(hwnd):
        return None
    if win32gui.IsIconic(hwnd):
        return None

    l, t, r, b = get_window_rect(hwnd)
    w = max(0, r - l)
    h = max(0, b - t)
    if w < 50 or h < 50:
        return None

    region = {"left": l, "top": t, "width": w, "height": h}
    img = np.array(sct.grab(region))[:, :, :3]  # BGRA -> BGR
    return img


# =========================
# Perspective save/load (ROI relative)
# =========================
def save_perspective_norm(src_pts_xy: np.ndarray, roi_w: int, roi_h: int, out_w: int, out_h: int):
    norm = []
    for x, y in src_pts_xy:
        norm.append([float(x) / float(roi_w), float(y) / float(roi_h)])
    d = {"src_pts_norm": norm, "out_w": int(out_w), "out_h": int(out_h)}
    PERSPECTIVE_FILE.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


def load_perspective_norm():
    if not PERSPECTIVE_FILE.exists():
        return None
    d = json.loads(PERSPECTIVE_FILE.read_text(encoding="utf-8"))
    if "src_pts_norm" not in d:
        return None
    src_norm = np.array(d["src_pts_norm"], dtype=np.float32)
    return src_norm, int(d.get("out_w", 900)), int(d.get("out_h", 700))


def denorm_pts(src_norm: np.ndarray, roi_w: int, roi_h: int) -> np.ndarray:
    pts = src_norm.copy()
    pts[:, 0] *= float(roi_w)
    pts[:, 1] *= float(roi_h)
    return pts.astype(np.float32)


def pick_four_points(img_bgr, win="pick 4 corners"):
    pts: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        nonlocal pts
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

    tmp = img_bgr.copy()
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = tmp.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(vis, (x, y), 6, (0, 255, 255), -1)
            cv2.putText(vis, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(
            vis,
            "Click 4 corners. Enter=OK  Esc=Cancel  Backspace=Undo",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(win, vis)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            cv2.destroyWindow(win)
            return None
        if k == 13:
            if len(pts) == 4:
                cv2.destroyWindow(win)
                return np.array(pts, dtype=np.float32)
        if k == 8 and pts:
            pts.pop()


def warp_board(roi_bgr: np.ndarray, src_pts: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    dst_pts = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(roi_bgr, M, (out_w, out_h))


def warp_work_to_roi_bgra_masked(
    work_bgr: np.ndarray,
    roi_w: int,
    roi_h: int,
    src_norm: Optional[np.ndarray],
    out_w: int,
    out_h: int,
    inset_ratio: float = 0.0,
    work_alpha_mask: Optional[np.ndarray] = None,  # 0..255 (work座標)
) -> Optional[np.ndarray]:
    if src_norm is None:
        return None

    # ROI四隅
    dst_roi_pts = denorm_pts(src_norm, roi_w, roi_h).astype(np.float32)
    if inset_ratio > 0:
        c = dst_roi_pts.mean(axis=0, keepdims=True)
        dst_roi_pts = (dst_roi_pts * (1.0 - inset_ratio) + c * inset_ratio).astype(np.float32)

    # work四隅
    src_work_pts = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(src_work_pts, dst_roi_pts)

    if work_bgr.shape[1] != out_w or work_bgr.shape[0] != out_h:
        work_fit = cv2.resize(work_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
    else:
        work_fit = work_bgr

    warped = cv2.warpPerspective(work_fit, M, (roi_w, roi_h), flags=cv2.INTER_LINEAR)

    # 盤面領域マスク
    base = np.full((out_h, out_w), 255, dtype=np.uint8)
    base_warp = cv2.warpPerspective(base, M, (roi_w, roi_h), flags=cv2.INTER_NEAREST)

    # 裏面だけマスク
    if work_alpha_mask is not None:
        if work_alpha_mask.shape[:2] != (out_h, out_w):
            m_fit = cv2.resize(work_alpha_mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        else:
            m_fit = work_alpha_mask
        m_warp = cv2.warpPerspective(m_fit, M, (roi_w, roi_h), flags=cv2.INTER_NEAREST)
        alpha = cv2.bitwise_and(base_warp, m_warp)
    else:
        alpha = base_warp

    bgra = cv2.cvtColor(warped, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    return bgra


# =========================
# Vision utilities
# =========================
def match_template_found(search_bgr: np.ndarray, tmpl_gray: np.ndarray, thresh: float) -> tuple[bool, float, tuple[int, int]]:
    if search_bgr is None or search_bgr.size == 0:
        return False, 0.0, (0, 0)

    search_gray = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2GRAY)

    th, tw = tmpl_gray.shape[:2]
    sh, sw = search_gray.shape[:2]
    if sh < th or sw < tw:
        return False, 0.0, (0, 0)

    res = cv2.matchTemplate(search_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return (float(max_val) >= thresh, float(max_val), (int(max_loc[0]), int(max_loc[1])))


def bgr_to_hsv(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def mask_hsv(hsv: np.ndarray, lower: Tuple[int, int, int], upper: Tuple[int, int, int]) -> np.ndarray:
    lo = np.array(lower, dtype=np.uint8)
    hi = np.array(upper, dtype=np.uint8)
    return cv2.inRange(hsv, lo, hi)


def cluster_1d(vals: List[float], tol: float) -> List[float]:
    vals = sorted(vals)
    if not vals:
        return []
    clusters = []
    cur = [vals[0]]
    for v in vals[1:]:
        if abs(v - statistics.mean(cur)) <= tol:
            cur.append(v)
        else:
            clusters.append(statistics.mean(cur))
            cur = [v]
    clusters.append(statistics.mean(cur))
    return clusters


def blue_fraction_and_mean(bgr_roi: np.ndarray, lower_blue, upper_blue) -> Tuple[float, float]:
    small = cv2.resize(bgr_roi, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = bgr_to_hsv(small)
    m = mask_hsv(hsv, lower_blue, upper_blue)
    blue_frac = float(m.mean() / 255.0)
    mean_bgr = float(small.mean())
    return blue_frac, mean_bgr


def build_cell_rois(frame_shape, row_centers: List[float], col_centers: List[float], cell_w: int, cell_h: int):
    H, W = frame_shape[:2]
    rois: Dict[Pos, Tuple[slice, slice]] = {}
    for r, cy in enumerate(row_centers):
        for c, cx in enumerate(col_centers):
            x1 = int(cx - cell_w / 2)
            y1 = int(cy - cell_h / 2)
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            y1 = max(0, y1)
            y2 = min(H, y2)
            x1 = max(0, x1)
            x2 = min(W, x2)
            rois[(r, c)] = (slice(y1, y2), slice(x1, x2))
    return rois


# ====== 裏面テンプレ判定（エッジ＋中央クロップ＋SQDIFF） ======
def _prep_edge(img_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    g = cv2.equalizeHist(g)
    e = cv2.Canny(g, 40, 120)
    return e


def _center_crop(img: np.ndarray, crop_ratio: float = 0.78) -> np.ndarray:
    h, w = img.shape[:2]
    cr = float(max(0.3, min(1.0, crop_ratio)))
    nw, nh = int(w * cr), int(h * cr)
    x1 = (w - nw) // 2
    y1 = (h - nh) // 2
    return img[y1 : y1 + nh, x1 : x1 + nw]


def is_back_card_by_template(cell_bgr: np.ndarray, tmpl_bgr: np.ndarray) -> float:
    """
    返り値: score (0..1) 大きいほど一致
    """
    if cell_bgr is None or cell_bgr.size == 0:
        return 0.0

    cell_c = _center_crop(cell_bgr, 0.78)
    tmpl_c = _center_crop(tmpl_bgr, 0.78)

    cell_e = _prep_edge(cell_c)
    tmpl_e = _prep_edge(tmpl_c)

    ch, cw = cell_e.shape[:2]
    if ch < 10 or cw < 10:
        return 0.0

    tmpl_fit = cv2.resize(tmpl_e, (cw, ch), interpolation=cv2.INTER_AREA)

    res = cv2.matchTemplate(cell_e, tmpl_fit, cv2.TM_SQDIFF_NORMED)
    v = float(res[0, 0])  # 0が完全一致
    return 1.0 - v


def blue_score(cell_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    lo = np.array([85, 40, 40], np.uint8)
    hi = np.array([140, 255, 255], np.uint8)
    m = cv2.inRange(hsv, lo, hi)
    return float(m.mean() / 255.0)  # 0..1


def is_back_card_hybrid(cell_bgr: np.ndarray, tmpl_bgr: np.ndarray) -> float:
    s_t = is_back_card_by_template(cell_bgr, tmpl_bgr)  # 0..1
    s_b = blue_score(_center_crop(cell_bgr, 0.85))      # 0..1
    return 0.75 * s_t + 0.25 * s_b

def detect_back_cells(
    work_bgr: np.ndarray,
    rois: Dict[Pos, Tuple[slice, slice]],
    tmpl_bgr: np.ndarray,
    thresh: float,
) -> tuple[np.ndarray, list[tuple[Pos, float]]]:
    """
    戻り値:
      - work座標mask(0/255)
      - 判定OKセルの一覧 [(pos, score), ...]
    """
    H, W = work_bgr.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    hits: list[tuple[Pos, float]] = []

    for pos, sl in rois.items():
        cell = work_bgr[sl]
        score = is_back_card_hybrid(cell, tmpl_bgr)
        if score >= thresh:
            mask[sl] = 255
            hits.append((pos, score))

    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask, hits


def draw_cell_boxes_on_work(
    work_bgr: np.ndarray,
    rois: Dict[Pos, Tuple[slice, slice]],
    hits: list[tuple[Pos, float]],
    color=(0, 255, 255),
    thickness: int = 2,
    put_score: bool = True,
) -> np.ndarray:
    out = work_bgr.copy()
    for pos, score in hits:
        sl = rois[pos]
        y1, y2 = sl[0].start, sl[0].stop
        x1, x2 = sl[1].start, sl[1].stop
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), color, thickness)
        if put_score:
            cv2.putText(
                out,
                f"{score:.2f}",
                (x1 + 4, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
    return out


def render_overlay_in_work(
    layout: List[List[Pos]],
    thumbs: Dict[Pos, np.ndarray],
    rois: Dict[Pos, Tuple[slice, slice]],
    work_shape: tuple[int, int, int],
) -> np.ndarray:
    """
    work座標(out_h,out_w)のキャンバスに、roisの位置へタイルを貼る。
    layout[r][c] = 元のPos（thumbsのキー）
    rois[(r,c)] = work座標での貼り先領域
    """
    H, W = work_shape[:2]
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    for dst_pos, (ys, xs) in rois.items():
        r, c = dst_pos
        src_pos = layout[r][c]
        src_img = thumbs.get(src_pos)
        if src_img is None:
            continue

        h = ys.stop - ys.start
        w = xs.stop - xs.start
        if h <= 0 or w <= 0:
            continue

        tile = cv2.resize(src_img, (w, h), interpolation=cv2.INTER_AREA)
        canvas[ys, xs] = tile

    return canvas


# =========================
# Auto ROI
# =========================
def auto_detect_board_roi(
    screen_bgr: np.ndarray,
    debug: bool = False,
    pad_ratio: float = 0.02,
    pad_lr_ratio: float = 0.02,
    pad_tb_ratio: float = 0.02,
    pad_px_lr: int = 0,
    pad_px_tb: int = 0,
):
    H, W = screen_bgr.shape[:2]
    gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (H * W) * P.auto_roi_min_area_ratio:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        x, y, w, h = cv2.boundingRect(approx)
        ar = w / max(h, 1)
        if not (0.6 <= ar <= 1.8):
            continue

        rect_area = w * h
        fill = area / max(rect_area, 1)
        score = area * fill
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        if debug:
            cv2.imshow("debug edges", edges)
        return None

    x, y, w, h = best

    base = int(min(w, h) * pad_ratio)
    pad_lr = base + int(min(w, h) * pad_lr_ratio) + int(pad_px_lr)
    pad_tb = base + int(min(w, h) * pad_tb_ratio) + int(pad_px_tb)

    x = max(0, x - pad_lr)
    y = max(0, y - pad_tb)
    w = min(W - x, w + 2 * pad_lr)
    h = min(H - y, h + 2 * pad_tb)

    if debug:
        vis = screen_bgr.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.imshow("debug auto roi", vis)
        cv2.imshow("debug edges", edges)

    return (x, y, w, h)


# =========================
# Grid detection (gap + fallbacks)
# =========================
def detect_grid_from_facedown(roi_bgr: np.ndarray, params: Params):
    hsv = bgr_to_hsv(roi_bgr)
    m = mask_hsv(hsv, params.lower_blue, params.upper_blue)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < params.min_card_area:
            continue
        ar = w / max(h, 1)
        if not (0.7 <= ar <= 1.4):
            continue
        rects.append((x, y, w, h))

    if len(rects) < 6:
        raise RuntimeError("カード裏面の検出が少なすぎます（ROI/HSV/閾値調整が必要）")

    centers = [(x + w / 2, y + h / 2, w, h) for (x, y, w, h) in rects]
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    cell_w = int(statistics.median([c[2] for c in centers]))
    cell_h = int(statistics.median([c[3] for c in centers]))

    row_centers = cluster_1d(ys, tol=params.row_tol)
    col_centers = cluster_1d(xs, tol=params.col_tol)

    R, C = len(row_centers), len(col_centers)
    if R * C < 8:
        raise RuntimeError(f"行列推定が不自然です: {R}x{C}")

    rois = build_cell_rois(roi_bgr.shape, row_centers, col_centers, cell_w, cell_h)
    return R, C, rois, cell_w, cell_h


def detect_grid_from_faces(roi_bgr: np.ndarray):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 180)

    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    H, W = gray.shape
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 15000:
            continue
        ar = w / max(h, 1)
        if not (0.7 <= ar <= 1.4):
            continue
        if x < 5 or y < 5 or x + w > W - 5 or y + h > H - 5:
            continue
        rects.append((x, y, w, h))

    if len(rects) < 6:
        raise RuntimeError("表面からカード矩形が十分に取れませんでした")

    centers = [(x + w / 2, y + h / 2, w, h) for x, y, w, h in rects]
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    cell_w = int(statistics.median([c[2] for c in centers]))
    cell_h = int(statistics.median([c[3] for c in centers]))

    row_centers = cluster_1d(ys, tol=P.row_tol)
    col_centers = cluster_1d(xs, tol=P.col_tol)

    R, C = len(row_centers), len(col_centers)
    if R * C < 8:
        raise RuntimeError(f"行列推定が不自然です（表面）: {R}x{C}")

    rois = build_cell_rois(roi_bgr.shape, row_centers, col_centers, cell_w, cell_h)
    return R, C, rois, cell_w, cell_h


def smooth_1d(a: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return a
    k = int(k)
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(a.astype(np.float32), kernel, mode="same")


def find_peaks_1d(a: np.ndarray, min_dist: int, rel_thresh: float) -> List[int]:
    if a.size == 0:
        return []
    mx = float(a.max())
    if mx <= 1e-6:
        return []
    thr = mx * rel_thresh
    cand = []
    for i in range(1, len(a) - 1):
        if a[i] > thr and a[i] >= a[i - 1] and a[i] >= a[i + 1]:
            cand.append(i)
    cand.sort(key=lambda i: a[i], reverse=True)
    chosen = []
    for i in cand:
        if all(abs(i - j) >= min_dist for j in chosen):
            chosen.append(i)
    return sorted(chosen)


def force_edge_peaks(px: np.ndarray, peaks: List[int], W: int, edge_ratio: float = 0.10, min_sep: int = 40) -> List[int]:
    if W <= 0:
        return peaks
    left_end = int(W * edge_ratio)
    right_start = int(W * (1.0 - edge_ratio))
    li = int(np.argmax(px[:max(1, left_end)]))
    ri = int(np.argmax(px[right_start:])) + right_start
    out = list(peaks)

    def far_enough(i):
        return all(abs(i - p) >= min_sep for p in out)

    if far_enough(li):
        out.append(li)
    if far_enough(ri):
        out.append(ri)
    return sorted(out)


def boundaries_to_centers(b: List[int]) -> List[float]:
    b = sorted(b)
    return [0.5 * (b[i] + b[i + 1]) for i in range(len(b) - 1)]


def detect_grid_by_projection(roi_bgr: np.ndarray, debug: bool = False):
    H, W = roi_bgr.shape[:2]

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    sobx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    soby = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    ax = cv2.convertScaleAbs(sobx)
    ay = cv2.convertScaleAbs(soby)

    _, bwx = cv2.threshold(ax, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bwy = cv2.threshold(ay, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bwx = cv2.morphologyEx(bwx, cv2.MORPH_CLOSE, np.ones((3, 7), np.uint8))
    bwy = cv2.morphologyEx(bwy, cv2.MORPH_CLOSE, np.ones((7, 3), np.uint8))

    proj_x = bwx.sum(axis=0)
    proj_y = bwy.sum(axis=1)

    kx = max(9, W // 70)
    ky = max(9, H // 70)
    px = smooth_1d(proj_x, kx)
    py = smooth_1d(proj_y, ky)

    min_dx = max(15, W // 12)
    min_dy = max(15, H // 12)

    peaks_x = find_peaks_1d(px, min_dist=min_dx, rel_thresh=0.50)
    peaks_y = find_peaks_1d(py, min_dist=min_dy, rel_thresh=0.40)

    bx = force_edge_peaks(px, peaks_x, W, edge_ratio=0.12, min_sep=max(12, W // 10))
    by = force_edge_peaks(py, peaks_y, H, edge_ratio=0.12, min_sep=max(12, H // 10))

    col_centers = boundaries_to_centers(bx)
    row_centers = boundaries_to_centers(by)

    if len(col_centers) < 3 or len(row_centers) < 3:
        raise RuntimeError(f"projection法で中心推定が不足: rows={len(row_centers)} cols={len(col_centers)}")

    C = len(col_centers)
    R = len(row_centers)

    col_d = [col_centers[i + 1] - col_centers[i] for i in range(C - 1)]
    row_d = [row_centers[i + 1] - row_centers[i] for i in range(R - 1)]
    cell_w = int(statistics.median(col_d)) if col_d else W // C
    cell_h = int(statistics.median(row_d)) if row_d else H // R

    rois = build_cell_rois(roi_bgr.shape, row_centers, col_centers, cell_w, cell_h)

    if debug:
        vis = roi_bgr.copy()
        for x in bx:
            cv2.line(vis, (int(x), 0), (int(x), H - 1), (0, 255, 0), 1)
        for y in by:
            cv2.line(vis, (0, int(y)), (W - 1, int(y)), (255, 0, 0), 1)
        cv2.imshow("debug projection bounds", vis)

    return R, C, rois, cell_w, cell_h


def detect_grid_by_gaps_with_perspective(
    roi_bgr: np.ndarray,
    debug: bool = False,
    gap_inv_thresh: int = 90,
    open_ratio: float = 0.08,
    smooth_ratio: float = 0.02,
    src_norm: Optional[np.ndarray] = None,
    out_w: int = 900,
    out_h: int = 700,
):
    work = roi_bgr
    if src_norm is not None:
        pts = denorm_pts(src_norm, roi_bgr.shape[1], roi_bgr.shape[0])
        work = warp_board(roi_bgr, pts, out_w, out_h)

    H, W = work.shape[:2]

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    inv = 255 - gray
    bw = cv2.threshold(inv, gap_inv_thresh, 255, cv2.THRESH_BINARY)[1]

    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    kx = max(15, int(W * open_ratio))
    ky = max(15, int(H * open_ratio))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky))

    bw_h = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_h)
    bw_v = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_v)

    proj_x = bw_v.sum(axis=0).astype(np.float32)
    proj_y = bw_h.sum(axis=1).astype(np.float32)

    k_sx = max(9, int(W * smooth_ratio))
    k_sy = max(9, int(H * smooth_ratio))
    px = smooth_1d(proj_x, k_sx)
    py = smooth_1d(proj_y, k_sy)

    min_dx = max(12, W // 12)
    min_dy = max(12, H // 12)

    bx = find_peaks_1d(px, min_dist=min_dx, rel_thresh=0.25)
    by = find_peaks_1d(py, min_dist=min_dy, rel_thresh=0.25)

    bx = force_edge_peaks(px, bx, W, edge_ratio=0.12, min_sep=max(12, W // 10))
    by = force_edge_peaks(py, by, H, edge_ratio=0.12, min_sep=max(12, H // 10))

    col_centers = boundaries_to_centers(bx)
    row_centers = boundaries_to_centers(by)

    if len(col_centers) < 3 or len(row_centers) < 3:
        raise RuntimeError(f"gap法で中心推定が不足: rows={len(row_centers)} cols={len(col_centers)}")

    C = len(col_centers)
    R = len(row_centers)

    col_d = [col_centers[i + 1] - col_centers[i] for i in range(C - 1)]
    row_d = [row_centers[i + 1] - row_centers[i] for i in range(R - 1)]
    cell_w = int(statistics.median(col_d)) if col_d else W // C
    cell_h = int(statistics.median(row_d)) if row_d else H // R

    rois = build_cell_rois(work.shape, row_centers, col_centers, cell_w, cell_h)

    if debug:
        cv2.imshow("debug gap work", work)
        cv2.imshow("debug gap_bw", bw)
        cv2.imshow("debug gap_bw_h", bw_h)
        cv2.imshow("debug gap_bw_v", bw_v)

    return R, C, rois, cell_w, cell_h, work


# =========================
# main
# =========================
def main():
    print("DQ7R ラッキーパネル 画面認識ツール（Steam想定）")
    print("操作: Alt+r/p/g/i/t/a/q")

    board_roi: Optional[tuple[int, int, int, int]] = None  # (x,y,w,h) in screen coords
    thumbs: Dict[Pos, np.ndarray] = {}
    rois: Dict[Pos, Tuple[slice, slice]] = {}
    R = C = 0
    layout: List[List[Pos]] = []
    tracking = False

    last_pair = None
    streak = 0
    last_print = 0.0
    swap_events: List[Tuple[Pos, Pos]] = []
    swap_limit = None
    frame_idx = 0
    last_stop_fire = 0.0
    last_start_fire = 0.0
    last_unvisible_fire = True

    # ---- template ----
    unvisible_tmpl_gray = None
    if UNVISIBLE_TEMPLATE_PATH.exists():
        b = cv2.imread(str(UNVISIBLE_TEMPLATE_PATH))
        if b is not None:
            unvisible_tmpl_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            print("[INFO] unvisible template loaded:", UNVISIBLE_TEMPLATE_PATH)
        else:
            print("[WARN] unvisible template exists but failed to load:", UNVISIBLE_TEMPLATE_PATH)
    else:
        print("[INFO] unvisible template not found:", UNVISIBLE_TEMPLATE_PATH)

    stop_tmpl_gray = None
    if STOP_TEMPLATE_PATH.exists():
        b = cv2.imread(str(STOP_TEMPLATE_PATH))
        if b is not None:
            stop_tmpl_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            print("[INFO] stop template loaded:", STOP_TEMPLATE_PATH)
        else:
            print("[WARN] stop template exists but failed to load:", STOP_TEMPLATE_PATH)
    else:
        print("[INFO] stop template not found:", STOP_TEMPLATE_PATH)

    start_tmpl_gray = None
    if START_TEMPLATE_PATH.exists():
        b = cv2.imread(str(START_TEMPLATE_PATH))
        if b is not None:
            start_tmpl_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            print("[INFO] start template loaded:", START_TEMPLATE_PATH)
        else:
            print("[WARN] start template exists but failed to load:", START_TEMPLATE_PATH)
    else:
        print("[INFO] start template not found:", START_TEMPLATE_PATH)

    back_tmpl_bgr: Optional[np.ndarray] = None
    if BACK_TEMPLATE_PATH.exists():
        back_tmpl_bgr = cv2.imread(str(BACK_TEMPLATE_PATH))
        if back_tmpl_bgr is None:
            print("[WARN] back template exists but failed to load:", BACK_TEMPLATE_PATH)
            back_tmpl_bgr = None
        else:
            print("[INFO] back template loaded:", BACK_TEMPLATE_PATH)
    else:
        print("[INFO] back template not found:", BACK_TEMPLATE_PATH)

    # ---- perspective ----
    src_norm = None
    out_w, out_h = 900, 700
    persp = load_perspective_norm()
    if persp:
        src_norm, out_w, out_h = persp
        print("[INFO] Loaded perspective(norm):", src_norm.tolist(), out_w, out_h)

    # ---- window + capture ----
    sct = mss.mss()
    hwnd = find_window_by_title_prefer_default(DEFAULT_TARGET_WINDOW_TITLE)

    # ---- hotkey ----
    hotkey_lock = threading.Lock()
    hotkey_queue: List[str] = []
    alt_down = False

    def push_key(k: str):
        with hotkey_lock:
            hotkey_queue.append(k)

    def on_press(key):
        nonlocal alt_down
        if key in (pynput_keyboard.Key.alt, pynput_keyboard.Key.alt_l, pynput_keyboard.Key.alt_r):
            alt_down = True
            return
        try:
            ch = key.char.lower()
        except Exception:
            return
        if not alt_down:
            return
        if ch in ("a", "r", "g", "i", "t", "p", "q"):
            push_key(ch)

    def on_release(key):
        nonlocal alt_down
        if key in (pynput_keyboard.Key.alt, pynput_keyboard.Key.alt_l, pynput_keyboard.Key.alt_r):
            alt_down = False

    listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()

    # ---- Qt overlay ----
    app = QtWidgets.QApplication(sys.argv)
    overlay = Overlay()

    screen: Optional[np.ndarray] = None
    roi_bgr: Optional[np.ndarray] = None
    work: Optional[np.ndarray] = None

    def get_roi_and_work(scr: np.ndarray):
        nonlocal board_roi, src_norm, out_w, out_h
        if board_roi is None:
            return None, None, None
        x, y, w, h = board_roi
        roi = scr[y : y + h, x : x + w]
        wk = roi
        if src_norm is not None:
            pts = denorm_pts(src_norm, roi.shape[1], roi.shape[0])
            wk = warp_board(roi, pts, out_w, out_h)
        return roi, wk, (x, y, w, h)

    def detect_grid_best(roi: np.ndarray):
        nonlocal src_norm, out_w, out_h
        try:
            R_, C_, rois_, _, _, wk_ = detect_grid_by_gaps_with_perspective(
                roi,
                debug=DEBUG,
                gap_inv_thresh=P.gap_inv_thresh,
                open_ratio=P.gap_open_ratio,
                smooth_ratio=P.gap_smooth_ratio,
                src_norm=src_norm,
                out_w=out_w,
                out_h=out_h,
            )
            print(f"[GRID] gap: {R_}x{C_}")
            return R_, C_, rois_, wk_
        except Exception as e_gap:
            if DEBUG:
                print("[gap] failed:", e_gap)

        wk = roi
        if src_norm is not None:
            pts = denorm_pts(src_norm, roi.shape[1], roi.shape[0])
            wk = warp_board(roi, pts, out_w, out_h)

        try:
            R_, C_, rois_, _, _ = detect_grid_from_facedown(wk, P)
            print(f"[GRID] facedown: {R_}x{C_}")
            return R_, C_, rois_, wk
        except Exception:
            try:
                R_, C_, rois_, _, _ = detect_grid_from_faces(wk)
                print(f"[GRID] faceup: {R_}x{C_}")
                return R_, C_, rois_, wk
            except Exception:
                R_, C_, rois_, _, _ = detect_grid_by_projection(wk, debug=DEBUG)
                print(f"[GRID] projection: {R_}x{C_}")
                return R_, C_, rois_, wk

    def run_setup_once(scr: np.ndarray) -> bool:
        nonlocal board_roi, rois, R, C, layout, thumbs, tracking
        nonlocal last_pair, streak, swap_events, swap_limit
        nonlocal work

        auto = auto_detect_board_roi(scr, debug=DEBUG)
        if auto is None:
            print("[AUTO] ROI auto failed. Press Alt+r for retry/manual.")
            return False

        board_roi = auto
        print("[AUTO] ROI updated:", board_roi)

        roi, _, _ = get_roi_and_work(scr)
        if roi is None:
            return False

        try:
            R, C, rois, work_for_grid = detect_grid_best(roi)
            work = work_for_grid
        except Exception as e:
            print("[AUTO] Grid failed:", e)
            rois = {}
            R = C = 0
            return False

        swap_limit = SWAP_LIMITS_BY_GRID.get((R, C))
        if swap_limit is None:
            swap_limit = SWAP_LIMITS_BY_CELLS.get(R * C)

        layout = [[(r, c) for c in range(C)] for r in range(R)]
        swap_events.clear()
        last_pair = None
        streak = 0

        thumbs = {pos: work[roi_].copy() for pos, roi_ in rois.items()}
        print(f"[AUTO] Saved thumbs: {len(thumbs)} tiles")

        tracking = True
        print("[AUTO] Tracking: ON")
        return True

    overlay_every_n = 2  # 2=30fps相当（負荷が高ければ3）

    def tick():
        nonlocal screen, roi_bgr, work
        nonlocal frame_idx, tracking, last_print, swap_limit
        nonlocal last_stop_fire, last_start_fire, last_unvisible_fire, last_pair, streak
        nonlocal board_roi, rois, R, C, layout, thumbs, swap_events
        nonlocal src_norm, out_w, out_h

        screen = grab_window_bgr(sct, hwnd)
        if screen is None:
            overlay.set_image_enabled(False)
            return

        frame_idx += 1

        # ---- キー取り出し ----
        with hotkey_lock:
            keys = hotkey_queue[:]
            hotkey_queue.clear()

        # ---- ROI/work 更新 ----
        if board_roi is not None:
            roi_bgr, work, _ = get_roi_and_work(screen)
        else:
            roi_bgr, work = None, None

        # ---- 定期ログ ----
        now = time.time()
        if now - last_print > 5.0:
            print(f"ROI={'set' if board_roi else 'none'} grid={R}x{C} tracking={tracking} limit={swap_limit} swaps={len(swap_events)}")
            last_print = now

        # ---- キー処理 ----
        for ch in keys:
            if ch == "q":
                overlay.close()
                QtWidgets.QApplication.quit()
                return

            if ch == "r":
                auto = auto_detect_board_roi(screen, debug=DEBUG)
                if auto is not None:
                    board_roi = auto
                    print("[ROI] auto:", board_roi)
                else:
                    print("[ROI] auto failed -> manual selectROI (needs focus on OpenCV window)")
                    tmp = screen.copy()
                    r = cv2.selectROI("Select Board ROI", tmp, fromCenter=False, showCrosshair=True)
                    cv2.destroyWindow("Select Board ROI")
                    if r and r[2] > 0 and r[3] > 0:
                        board_roi = (int(r[0]), int(r[1]), int(r[2]), int(r[3]))
                        print("[ROI] manual:", board_roi)
                continue

            if ch == "a":
                run_setup_once(screen)
                overlay.set_image_enabled(False)
                continue

            if board_roi is None or roi_bgr is None or work is None:
                continue

            if ch == "p":
                print("[P] Pick 4 corners on ROI image. (OpenCV window focus needed)")
                pts = pick_four_points(roi_bgr, win="pick 4 corners")
                if pts is not None:
                    out_w, out_h = 900, 700
                    save_perspective_norm(pts, roi_bgr.shape[1], roi_bgr.shape[0], out_w, out_h)
                    tmp = load_perspective_norm()
                    if tmp:
                        src_norm, out_w, out_h = tmp
                    print("[P] Saved perspective to perspective.json")
                continue

            if ch == "g":
                try:
                    R, C, rois, work_for_grid = detect_grid_best(roi_bgr)
                    work = work_for_grid
                    swap_limit = SWAP_LIMITS_BY_GRID.get((R, C)) or SWAP_LIMITS_BY_CELLS.get(R * C)
                    layout[:] = [[(r, c) for c in range(C)] for r in range(R)]
                    swap_events.clear()
                    last_pair = None
                    streak = 0
                    print("[G] Grid ready. Press Alt+i to save thumbs.")
                except Exception as e:
                    print("[G] Grid detect failed:", e)
                continue

            if ch == "i":
                if not rois or R == 0 or C == 0:
                    print("[I] Press Alt+g first.")
                    continue
                _, work_now, _ = get_roi_and_work(screen)
                if work_now is None:
                    continue
                thumbs = {pos: work_now[roi_].copy() for pos, roi_ in rois.items()}
                print(f"[I] Saved thumbs: {len(thumbs)} tiles")
                continue

            if ch == "t":
                tracking = not tracking
                last_pair = None
                streak = 0
                print("[T] Tracking:", tracking)
                continue

        # ---- テンプレで Tracking OFF/ON ----
        if frame_idx % TEMPLATE_CHECK_EVERY_N_FRAMES == 0:
            now = time.time()

            if TEMPLATE_SEARCH_TARGET == "screen":
                search_img = screen
            elif TEMPLATE_SEARCH_TARGET == "roi":
                search_img = roi_bgr if (roi_bgr is not None and roi_bgr.size) else screen
            else:
                search_img = work if (work is not None and work.size) else screen

            if (not tracking) and unvisible_tmpl_gray is not None and last_unvisible_fire:
                found, score, _ = match_template_found(search_img, unvisible_tmpl_gray, UNVISIBLE_MATCH_THRESH)
                if found:
                    last_unvisible_fire = False
                    print(f"[AUTO] UNVISIBLE template -> hide overlay (score={score:.3f})")
                    overlay.set_border(False)
                    overlay.set_image_enabled(False)

            if tracking and stop_tmpl_gray is not None and (now - last_stop_fire) > STOP_COOLDOWN_SEC:
                found, score, _ = match_template_found(search_img, stop_tmpl_gray, STOP_MATCH_THRESH)
                if found:
                    tracking = False
                    last_pair = None
                    streak = 0
                    last_stop_fire = now
                    last_unvisible_fire = True
                    overlay.set_border(False)
                    overlay.set_image_enabled(True)
                    print(f"[AUTO] STOP template -> Tracking OFF (score={score:.3f})")

            if (not tracking) and start_tmpl_gray is not None and (now - last_start_fire) > START_COOLDOWN_SEC:
                found, score, _ = match_template_found(search_img, start_tmpl_gray, START_MATCH_THRESH)
                if found:
                    last_start_fire = now
                    last_unvisible_fire = True
                    print(f"[AUTO] START template -> run_setup_once (score={score:.3f})")
                    run_setup_once(screen)
                    overlay.set_border(True)
                    overlay.set_image_enabled(False)

        # ---- 追跡処理（swap推定）----
        if tracking and rois and layout and work is not None:
            empties = []
            for pos, roi_ in rois.items():
                sub = work[roi_]
                blue_frac, mean_bgr = blue_fraction_and_mean(sub, P.lower_blue, P.upper_blue)
                if blue_frac < P.blue_frac_thresh and mean_bgr < P.mean_bgr_thresh:
                    empties.append(pos)

            if len(empties) == 2:
                pair = tuple(sorted(empties))
                if pair == last_pair:
                    streak += 1
                else:
                    last_pair = pair
                    streak = 1
            else:
                if last_pair is not None and streak >= P.empty_streak_frames:
                    if not swap_events or swap_events[-1] != last_pair:
                        swap_events.append(last_pair)
                        a, b = last_pair
                        ar, ac = a
                        br, bc = b
                        layout[ar][ac], layout[br][bc] = layout[br][bc], layout[ar][ac]
                        print("[SWAP]", last_pair, "total:", len(swap_events))
                        if swap_limit is not None and len(swap_events) >= swap_limit:
                            tracking = False
                            print(f"[AUTO] swap limit reached ({len(swap_events)}/{swap_limit}) -> Tracking OFF")
                last_pair = None
                streak = 0

        # ---- overlay描画 ----
        if thumbs and len(thumbs) == R * C and board_roi is not None and work is not None:
            if overlay_every_n <= 1 or (frame_idx % overlay_every_n == 0):
                wl, wt, wr, wb = get_window_rect(hwnd)
                rx, ry, rw, rh = board_roi
                overlay.place_at(wl + rx, wt + ry, rw, rh)

                overlay_work = render_overlay_in_work(layout, thumbs, rois, work.shape)

                work_mask = None
                hits: list[tuple[Pos, float]] = []
                if back_tmpl_bgr is not None and rois:
                    work_mask, hits = detect_back_cells(work, rois, back_tmpl_bgr, BACK_MATCH_THRESH)

                if DEBUG_SHOW_BACK_DETECTION and hits:
                    show_work = draw_cell_boxes_on_work(overlay_work, rois, hits, color=(0, 255, 255), thickness=2, put_score=True)
                else:
                    show_work = overlay_work

                if src_norm is not None:
                    bgra = warp_work_to_roi_bgra_masked(
                        show_work,
                        rw,
                        rh,
                        src_norm,
                        out_w,
                        out_h,
                        inset_ratio=PERSPECTIVE_INSET_RATIO,
                        work_alpha_mask=work_mask,
                    )
                    if bgra is not None:
                        overlay.update_bgr(bgra)
                else:
                    if show_work.shape[1] != rw or show_work.shape[0] != rh:
                        show_roi = cv2.resize(show_work, (rw, rh), interpolation=cv2.INTER_AREA)
                    else:
                        show_roi = show_work

                    if work_mask is not None:
                        if work_mask.shape[1] != rw or work_mask.shape[0] != rh:
                            m = cv2.resize(work_mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
                        else:
                            m = work_mask
                        bgra = cv2.cvtColor(show_roi, cv2.COLOR_BGR2BGRA)
                        bgra[:, :, 3] = m
                        overlay.update_bgr(bgra)
                    else:
                        overlay.update_bgr(show_roi)

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(16)  # 60fps相当,負荷が高ければ33(30fps相当)に
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
