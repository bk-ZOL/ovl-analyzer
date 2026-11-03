"""
通用 MESR 读取。

桌面界面和 Nikon Expert 的「OVL 分析」页共用这一份解析结果。
经典套刻（BASFIA 等）仍是 Shot / Point / Valid / dX / dY。
INC、DIS 在此基础上标明该画「按 Shot」还是「整片晶圆」；坐标单位沿用机台文件里的 μm。

不改 mesr_parser.parse_mesr_file 的列和坐标公式。10 参数拟合继续走那个解析器。
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


EVAL_MODE_DESC = {
    "BASFIA": "FIA 基线套刻",
    "BASLSA": "LSA 基线套刻",
    "FIA": "FIA 套刻",
    "LSA": "LSA 套刻",
    "OVL": "套刻",
    "REG": "套刻",
    "EGA": "EGA 对准",
    "FM": "焦点",
    "LEV": "调平",
    "INC": "增量畸变",
    "DIS": "畸变",
    "DIST": "畸变",
    "SDM": "Shot 畸变",
}

_OVERLAY_MODES = {"BASFIA", "BASLSA", "FIA", "LSA", "OVL", "REG", "EGA"}
_DIS_MODES = {"DIS", "DIST", "SDM", "DSTOR", "DISTORTION"}
_STRUCTURAL = {"SHOT", "MESP", "MEAS_DATA", "WAFER"}
_TOKEN = re.compile(r"([A-Za-z][A-Za-z0-9_]*)(?:\((\d+)\))?\s*:=")
def grid_center(shots: Dict[int, Tuple[int, int]]) -> Tuple[float, float]:
    """与 mesr_parser 相同：用全部 Shot 的 col/row 最小最大值取中心。"""
    if not shots:
        return 0.0, 0.0
    cols = [v[0] for v in shots.values()]
    rows = [v[1] for v in shots.values()]
    return (min(cols) + max(cols)) / 2.0, (min(rows) + max(rows)) / 2.0


def _tokens(text: str):
    matches = list(_TOKEN.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        yield m.group(1), (int(m.group(2)) if m.group(2) else None), text[m.end():end]


def _clean(value: str) -> str:
    value = value.strip().strip("\r")
    if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        return value[1:-1]
    return value


def _floats(value: str) -> List[float]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _has_token(text: str, word: str) -> bool:
    return re.search(rf"(^|[^A-Z0-9]){word}([^A-Z0-9]|$)", text.upper()) is not None


def mode_kind(eval_mode: str) -> Optional[str]:
    """从 EVAL_MODE 判断 INC / DIS。套刻模式返回 None。"""
    mode = (eval_mode or "").upper().strip()
    if not mode or mode in _OVERLAY_MODES:
        return None
    has_inc = _has_token(mode, "INC")
    has_dis = mode in _DIS_MODES or _has_token(mode, "DIS") or mode.startswith("DIST")
    if has_inc and has_dis:
        return "both"
    if has_inc:
        return "inc"
    if has_dis:
        return "dis"
    return None


def name_kind(filename: str = "", comment: str = "") -> Optional[str]:
    """文件名或备注里的 INC / DIS。不覆盖已经写明的套刻 EVAL_MODE。"""
    blob = f"{filename} {comment}"
    has_inc = _has_token(blob, "INC")
    has_dis = _has_token(blob, "DIS")
    if has_inc and has_dis:
        return "both"
    if has_inc:
        return "inc"
    if has_dis:
        return "dis"
    return None


def _is_int_col(col: np.ndarray) -> bool:
    if col.size == 0 or not np.all(np.isfinite(col)):
        return False
    return bool(np.all(np.abs(col - np.round(col)) < 1e-6))


def _looks_like_micrometer_coords(x: np.ndarray, y: np.ndarray, pitch_x: float, pitch_y: float) -> bool:
    """测量点设计坐标通常是几千 μm；套刻矢量一般远小于一个 Shot。"""
    if x.size == 0:
        return False
    limit = max(pitch_x, pitch_y, 1.0) * 0.25
    return float(np.nanmedian(np.abs(x))) > limit or float(np.nanmedian(np.abs(y))) > limit


@dataclass
class MesrFile:
    """一次 MESR 读取的结果。data_array() 最后一列是 Wafer，不在 column_names() 里。"""

    path: str = ""
    raw: str = ""
    header: Dict[str, str] = field(default_factory=dict)
    eval_mode: str = ""
    wafer_size: float = 300.0
    step_pitch: Tuple[float, float] = (0.0, 0.0)
    shots: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    mesps: Dict[int, Tuple[float, float, str, str]] = field(default_factory=dict)
    kind: str = "curve"
    columns: List[str] = field(default_factory=list)
    # 列下标指向 columns / data_array 的前 n_fields 列
    layout: Dict[str, object] = field(default_factory=dict)
    vector_cols: Dict[str, Optional[Tuple[int, int]]] = field(default_factory=dict)
    _array: np.ndarray = field(default_factory=lambda: np.zeros((0, 1)))

    def get(self, key: str, default: str = "") -> str:
        if key in self.header:
            return self.header[key]
        upper = key.upper()
        for name, value in self.header.items():
            if name.upper() == upper:
                return value
        return default

    def column_names(self) -> List[str]:
        return list(self.columns)

    def data_array(self) -> np.ndarray:
        if self._array.size == 0:
            return np.zeros((0, self.n_fields + 1))
        return self._array.copy()

    @property
    def n_fields(self) -> int:
        return len(self.columns)

    @property
    def is_vector_type(self) -> bool:
        return self.kind in ("ovl", "inc", "dis", "both")

    def wafer_ids(self) -> List[int]:
        arr = self.data_array()
        if arr.size == 0 or arr.shape[0] == 0:
            return []
        return sorted({int(w) for w in arr[:, -1]})

    def pretty_text(self) -> str:
        text = self.raw.replace("\r\n", "\n").replace("\r", "\n")
        if text.startswith("\ufeff"):
            text = text[1:]
        text = _TOKEN.sub(lambda m: ("\n" if m.start() else "") + m.group(0), text)
        return text.strip() + "\n"


def read_mesr(filepath: str) -> MesrFile:
    """读取 .MESR（含 `.MESR;5` 这类 VMS 文件名）。读不了时抛出异常，由调用方决定怎么标。"""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as handle:
        raw = handle.read()
    mf = MesrFile(path=filepath, raw=raw)
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    if text.startswith("\ufeff"):
        text = text[1:]

    current_wafer = 1
    meas_rows: List[Tuple[int, List[float]]] = []
    for key, index, value in _tokens(text):
        key_up = key.upper()
        if key_up == "WAFER":
            current_wafer = index or current_wafer
            continue
        if key_up == "SHOT":
            nums = _floats(value)
            if len(nums) >= 3:
                mf.shots[int(nums[0])] = (int(nums[1]), int(nums[2]))
            continue
        if key_up == "MESP":
            _store_mesp(mf, value)
            continue
        if key_up == "MEAS_DATA":
            try:
                nums = _floats(value)
            except ValueError:
                continue
            if nums:
                meas_rows.append((current_wafer, nums))
            continue
        if key_up in _STRUCTURAL:
            continue
        mf.header[key_up] = _clean(value)

    mf.eval_mode = mf.header.get("EVAL_MODE", "")
    mf.wafer_size = _first_float(mf.header.get("WAFER_SIZE", ""), 300.0)
    pitch = _floats(mf.header.get("STEP_PITCH", "") or "0,0")
    if len(pitch) >= 2:
        mf.step_pitch = (pitch[0], pitch[1])
    elif len(pitch) == 1:
        mf.step_pitch = (pitch[0], pitch[0])

    _build_array(mf, meas_rows)
    return mf


def _store_mesp(mf: MesrFile, value: str) -> None:
    parts = [p.strip() for p in value.split(",")]
    nums = []
    texts = []
    for part in parts:
        if not part:
            continue
        if part.startswith("'") and part.endswith("'") and len(part) >= 2:
            texts.append(part[1:-1])
            continue
        try:
            nums.append(float(part))
        except ValueError:
            texts.append(part.strip("'\""))
    if len(nums) >= 3:
        mesp_id = int(nums[0])
        mtype = texts[0] if texts else ""
        mname = texts[1] if len(texts) > 1 else ""
        mf.mesps[mesp_id] = (nums[1], nums[2], mtype, mname)


def _first_float(value: str, default: float) -> float:
    try:
        return float(str(value).split(",")[0].strip() or default)
    except ValueError:
        return default


def _build_array(mf: MesrFile, meas_rows: List[Tuple[int, List[float]]]) -> None:
    if not meas_rows:
        mf.kind = mode_kind(mf.eval_mode) or "curve"
        mf._array = np.zeros((0, 1))
        return

    lengths = [len(row) for _, row in meas_rows]
    width = max(set(lengths), key=lengths.count)
    kept = [(wafer, row) for wafer, row in meas_rows if len(row) == width]
    body = np.asarray([row for _, row in kept], dtype=float)
    wafers = np.asarray([wafer for wafer, _ in kept], dtype=float)
    pitch_x, pitch_y = mf.step_pitch

    kind_hint = mode_kind(mf.eval_mode)
    if kind_hint is None and mf.eval_mode.upper().strip() not in _OVERLAY_MODES:
        kind_hint = name_kind(os.path.basename(mf.path), mf.header.get("COMMENT", ""))

    layout_name, columns, layout, vector_cols, body = _classify_layout(
        body, pitch_x, pitch_y, mf.shots, kind_hint
    )
    if layout_name == "curve":
        mf.kind = "curve"
    elif kind_hint in ("inc", "dis", "both"):
        mf.kind = kind_hint
    elif vector_cols.get("inc") and vector_cols.get("dis"):
        mf.kind = "both"
    else:
        mf.kind = "ovl"

    mf.columns = columns
    mf.layout = layout
    mf.vector_cols = vector_cols
    mf._array = np.column_stack([body, wafers])


def _classify_layout(body, pitch_x, pitch_y, shots, kind_hint):
    """识别列。返回的 body 可能在绝对坐标文件前插入了 Shot 列。"""
    width = body.shape[1]
    empty_vectors = {"primary": None, "inc": None, "dis": None}

    def pack(name, columns, layout, vectors, data):
        return name, columns, layout, vectors, data

    int_prefix = width >= 3 and all(_is_int_col(body[:, i]) for i in range(3))
    if width == 5 and int_prefix:
        pair = (3, 4)
        vectors = {"primary": pair, "inc": None, "dis": None}
        if kind_hint == "inc":
            vectors["inc"] = pair
        elif kind_hint == "dis":
            vectors["dis"] = pair
        elif kind_hint == "both":
            vectors["inc"] = pair
            vectors["dis"] = pair
        return pack(
            "vector",
            ["Shot", "Point", "Valid", "dX", "dY"],
            {"shot": 0, "point": 1, "valid": 2, "x": None, "y": None, "absolute": False},
            vectors,
            body,
        )

    if width == 7 and int_prefix:
        coords = _looks_like_micrometer_coords(body[:, 3], body[:, 4], pitch_x, pitch_y)
        if coords:
            pair = (5, 6)
            vectors = {
                "primary": pair,
                "inc": pair if kind_hint == "inc" else None,
                "dis": pair if kind_hint in ("dis", "both") else None,
            }
            return pack(
                "vector",
                ["Shot", "Point", "Valid", "X", "Y", "dX", "dY"],
                {"shot": 0, "point": 1, "valid": 2, "x": 3, "y": 4, "absolute": False},
                vectors,
                body,
            )
        # 四个小量：INC 一对 + DIS 一对。网页自动倍率看的是最后两列，所以把要画的那一对放在最后。
        display = kind_hint or "both"
        if display == "inc":
            data = np.column_stack([body[:, 0:3], body[:, 5], body[:, 6], body[:, 3], body[:, 4]])
            vectors = {"primary": (5, 6), "inc": (5, 6), "dis": (3, 4)}
        else:
            data = body
            vectors = {"primary": (5, 6), "inc": (3, 4), "dis": (5, 6)}
        return pack(
            "vector",
            ["Shot", "Point", "Valid", "INC_X", "INC_Y", "DIS_X", "DIS_Y"]
            if display != "inc"
            else ["Shot", "Point", "Valid", "DIS_X", "DIS_Y", "INC_X", "INC_Y"],
            {"shot": 0, "point": 1, "valid": 2, "x": None, "y": None, "absolute": False},
            vectors,
            data,
        )

    if width == 4 and _is_int_col(body[:, 0]) and _is_int_col(body[:, 1]) and float(np.nanmax(np.abs(body[:, 0]))) < 5000:
        pair = (2, 3)
        vectors = {"primary": pair, "inc": pair if kind_hint == "inc" else None,
                   "dis": pair if kind_hint in ("dis", "both") else None}
        return pack(
            "vector",
            ["Shot", "Point", "dX", "dY"],
            {"shot": 0, "point": 1, "valid": None, "x": None, "y": None, "absolute": False},
            vectors,
            body,
        )

    if width >= 4 and (
        kind_hint in ("inc", "dis", "both")
        or _looks_like_micrometer_coords(body[:, 0], body[:, 1], pitch_x, pitch_y)
        or (width == 4 and not (_is_int_col(body[:, 0]) and float(np.nanmax(np.abs(body[:, 0]))) < 5000))
    ):
        # 绝对坐标：X Y dX dY，或更长时仍取前两列为位置、最后两列为矢量。
        x = body[:, 0]
        y = body[:, 1]
        dx = body[:, -2]
        dy = body[:, -1]
        shot_ids = _bin_shots(x, y, shots, pitch_x, pitch_y)
        data = np.column_stack([shot_ids, x, y, dx, dy])
        pair = (3, 4)
        vectors = {
            "primary": pair,
            "inc": pair if kind_hint == "inc" else None,
            "dis": pair if kind_hint in ("dis", "both", None) else None,
        }
        if kind_hint == "inc":
            vectors["inc"] = pair
            vectors["dis"] = None
        elif kind_hint == "dis":
            vectors["dis"] = pair
        return pack(
            "vector",
            ["Shot", "X", "Y", "dX", "dY"],
            {"shot": 0, "point": None, "valid": None, "x": 1, "y": 2, "absolute": True},
            vectors,
            data,
        )

    if width >= 2 and (kind_hint in ("inc", "dis", "both") or int_prefix):
        pair = (width - 2, width - 1)
        names = [f"C{i + 1}" for i in range(width)]
        names[-2], names[-1] = "dX", "dY"
        if int_prefix:
            names[0], names[1], names[2] = "Shot", "Point", "Valid"
        vectors = {"primary": pair, "inc": None, "dis": None}
        if kind_hint == "inc":
            vectors["inc"] = pair
        elif kind_hint in ("dis", "both"):
            vectors["dis"] = pair
        layout = {
            "shot": 0 if _is_int_col(body[:, 0]) else None,
            "point": 1 if width > 2 and _is_int_col(body[:, 1]) else None,
            "valid": 2 if int_prefix else None,
            "x": None,
            "y": None,
            "absolute": False,
        }
        return pack("vector", names, layout, vectors, body)

    names = [f"C{i + 1}" for i in range(width)]
    return pack(
        "curve",
        names,
        {"shot": None, "point": None, "valid": None, "x": None, "y": None, "absolute": False},
        empty_vectors,
        body,
    )


def _bin_shots(xs, ys, shots, pitch_x, pitch_y) -> np.ndarray:
    """把晶圆绝对坐标（μm，原点与 Shot 网格中心一致）归进 Shot。"""
    if not shots:
        return np.zeros(len(xs), dtype=float)
    col_center, row_center = grid_center(shots)
    centers = []
    for sid, (col, row) in shots.items():
        centers.append((
            sid,
            (col - col_center) * pitch_x,
            (row - row_center) * pitch_y,
        ))
    hx = pitch_x / 2.0 if pitch_x else 0.0
    hy = pitch_y / 2.0 if pitch_y else 0.0
    out = np.zeros(len(xs), dtype=float)
    for i, (x, y) in enumerate(zip(xs, ys)):
        inside = []
        for sid, cx, cy in centers:
            if hx and hy and abs(x - cx) <= hx + 1.0 and abs(y - cy) <= hy + 1.0:
                inside.append((sid, (x - cx) ** 2 + (y - cy) ** 2))
        if inside:
            out[i] = min(inside, key=lambda item: item[1])[0]
        else:
            out[i] = min(centers, key=lambda item: (x - item[1]) ** 2 + (y - item[2]) ** 2)[0]
    return out
