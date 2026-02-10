import argparse
import json
import re
from dataclasses import dataclass, asdict
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from paddleocr import PaddleOCR


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class OCRToken:
    text: str
    x: float
    y: float
    conf: float


# ---------------------------------------------------------------------------
# 坐标工具
# ---------------------------------------------------------------------------

def _center_of_box(box: List[List[float]]) -> Tuple[float, float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _as_points(box: Any) -> Optional[List[List[float]]]:
    """将 list / tuple / numpy.ndarray 等转为 [[x,y], ...] 点集"""
    if box is None:
        return None
    try:
        points = box.tolist() if hasattr(box, "tolist") else box
        if not isinstance(points, (list, tuple)) or len(points) < 3:
            return None
        out: List[List[float]] = []
        for p in points:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                return None
            out.append([float(p[0]), float(p[1])])
        return out
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 文本清洗 —— 仅用于数值解析，绝不用于名字
# ---------------------------------------------------------------------------

def _clean_numeric(text: str) -> str:
    """仅对数值上下文做 OCR 常见误识别修正"""
    t = text.strip()
    t = t.replace("，", ".").replace(",", ".")
    t = t.replace("：", ":").replace("／", "/")
    t = t.replace("|", "/").replace("\\", "/")
    # 只把独立 O/o 在纯数值串中替换
    t = re.sub(r"(?<![A-Za-z])[Oo](?![A-Za-z])", "0", t)
    return t


def _to_float(text: str) -> Optional[float]:
    t = _clean_numeric(text)
    m = re.search(r"[-+]?\d+(?:\.\d+)?", t)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _to_int(text: str) -> Optional[int]:
    t = _clean_numeric(text)
    m = re.search(r"[-+]?\d+", t)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# KDA / 爆头率 / Rating+ 解析
# ---------------------------------------------------------------------------

def _parse_kda(text: str) -> Optional[Dict[str, int]]:
    t = _clean_numeric(text)
    m = re.search(r"(\d+)\s*[/:\-]\s*(\d+)\s*[/:\-]\s*(\d+)", t)
    if not m:
        return None
    return {
        "kills": int(m.group(1)),
        "assists": int(m.group(2)),
        "deaths": int(m.group(3)),
    }


def _parse_headshot(text: str) -> float:
    t = _clean_numeric(text)
    has_percent = "%" in text
    v = _to_float(t)
    if v is None:
        return 0.0
    if has_percent:
        v = v / 100.0
    if v < 0:
        v = 0.0
    if v > 1:
        if v <= 100:
            v = v / 100.0
        else:
            v = 1.0
    return round(float(v), 6)


def _parse_rating_plus(text: str) -> float:
    """Rating+ 通常在 0.00 ~ 3.00 之间。OCR 偶尔丢失小数点，如 '149' 实为 '1.49'"""
    v = _to_float(_clean_numeric(text))
    if v is None:
        return 0.0
    # Rating+ 不可能 > 5，若超过说明小数点被 OCR 吃掉
    if v > 5.0:
        v = v / 100.0
    return round(float(v), 2)


# ---------------------------------------------------------------------------
# Token 分类
# ---------------------------------------------------------------------------

def _is_numeric_like(text: str) -> bool:
    t = _clean_numeric(text).strip()
    if not t:
        return False
    if _parse_kda(t) is not None:
        return True
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", t):
        return True
    return False


def _is_name_token(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if re.search(r"[A-Za-z]", t):
        return True
    if _is_numeric_like(t):
        return False
    # 符号型 token（. * - 等）允许作为名字的一部分
    return True


# ---------------------------------------------------------------------------
# PaddleOCR 3.x 结果解析 —— 支持 generator / list / dict / 对象
# ---------------------------------------------------------------------------

def _extract_tokens_from_predict(raw: Any) -> List[OCRToken]:
    """
    PaddleOCR 3.x 的 predict() 返回一个 **generator**，每次 yield 一个结果
    （对单张图只 yield 一次）。每个结果可能是 dict 或自定义对象，包含：
      dt_polys  : list[ndarray]  多边形坐标
      rec_texts : list[str]      识别文本
      rec_scores: list[float]    置信度
    """
    tokens: List[OCRToken] = []
    seen = set()

    def _add(text: str, box: Any, conf: float = 0.0) -> None:
        pts = _as_points(box)
        if pts is None:
            return
        text = str(text).strip()
        if not text:
            return
        x, y = _center_of_box(pts)
        key = (round(x, 1), round(y, 1), text)
        if key in seen:
            return
        seen.add(key)
        tokens.append(OCRToken(text=text, x=x, y=y, conf=float(conf)))

    def _process_result(res: Any) -> None:
        """处理单个结果对象（dict 或 attribute-based 对象）"""
        # 尝试转 dict
        d: Optional[dict] = None
        if isinstance(res, dict):
            d = res
        elif hasattr(res, "to_dict") and callable(res.to_dict):
            try:
                d = res.to_dict()
            except Exception:
                pass
        elif hasattr(res, "__dict__"):
            d = vars(res)

        if d is not None and "rec_texts" in d:
            polys = d.get("dt_polys") or []
            texts = d.get("rec_texts") or []
            scores = d.get("rec_scores") or []
            if hasattr(polys, "tolist"):
                polys = polys.tolist()
            if hasattr(texts, "tolist"):
                texts = texts.tolist()
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            for i, txt in enumerate(texts):
                poly = polys[i] if i < len(polys) else None
                score = float(scores[i]) if i < len(scores) else 0.0
                _add(txt, poly, score)
            return

        # 尝试通过属性直接访问
        if hasattr(res, "rec_texts") and hasattr(res, "dt_polys"):
            polys = getattr(res, "dt_polys", []) or []
            texts = getattr(res, "rec_texts", []) or []
            scores = getattr(res, "rec_scores", []) or []
            if hasattr(polys, "tolist"):
                polys = polys.tolist()
            if hasattr(texts, "tolist"):
                texts = texts.tolist()
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            for i, txt in enumerate(texts):
                poly = polys[i] if i < len(polys) else None
                score = float(scores[i]) if i < len(scores) else 0.0
                _add(txt, poly, score)
            return

        # 老格式兜底: [[box, (text, conf)], ...]
        if isinstance(res, (list, tuple)):
            for item in res:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    box_candidate = item[0]
                    info = item[1]
                    pts = _as_points(box_candidate)
                    if pts is not None:
                        if isinstance(info, (tuple, list)) and len(info) >= 2:
                            _add(str(info[0]), box_candidate, float(info[1]))
                        else:
                            _add(str(info), box_candidate, 0.0)
                    else:
                        _process_result(item)

    # ---- 主入口：处理 generator / list / 单个结果 ----
    try:
        iterator = iter(raw)
    except TypeError:
        _process_result(raw)
        return tokens

    for result_item in iterator:
        _process_result(result_item)

    return tokens


# ---------------------------------------------------------------------------
# 行聚类
# ---------------------------------------------------------------------------

def _cluster_rows_by_kda(tokens: List[OCRToken]) -> List[float]:
    """用 KDA token 的 y 坐标聚类得到玩家行中心"""
    kda_ys = sorted([t.y for t in tokens if _parse_kda(t.text) is not None])
    if not kda_ys:
        return []
    # 估算行间距
    if len(kda_ys) > 1:
        gaps = [kda_ys[i + 1] - kda_ys[i] for i in range(len(kda_ys) - 1)]
        typical_gap = median(gaps)
        merge_tol = max(8.0, min(30.0, typical_gap * 0.45))
    else:
        merge_tol = 15.0

    clusters: List[List[float]] = []
    for y in kda_ys:
        if not clusters or abs(y - clusters[-1][-1]) > merge_tol:
            clusters.append([y])
        else:
            clusters[-1].append(y)

    return [sum(c) / len(c) for c in clusters]


# ---------------------------------------------------------------------------
# KDA 锚点选择
# ---------------------------------------------------------------------------

def _pick_kda_index(row_tokens: List[OCRToken]) -> Optional[int]:
    candidates = [idx for idx, t in enumerate(row_tokens) if _parse_kda(t.text) is not None]
    if not candidates:
        return None

    best_idx: Optional[int] = None
    best_score = float("-inf")

    for idx in candidates:
        left = row_tokens[:idx]
        right = row_tokens[idx + 1:]

        left_name = sum(1 for t in left if _is_name_token(t.text))
        right_num = sum(1 for t in right if _is_numeric_like(t.text))

        score = right_num * 3.0 + left_name * 2.0
        if right_num < 2:
            score -= 10.0
        if left_name < 1:
            score -= 8.0
        score -= idx * 0.1

        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


# ---------------------------------------------------------------------------
# 列位置推断 & 基于 x 坐标的列分配
# ---------------------------------------------------------------------------

def _find_column_positions(
    tokens: List[OCRToken],
    row_centers: List[float],
    row_tol: float,
) -> Optional[List[float]]:
    """从数据列最完整的一行推断 KDA 右侧 8 列（爆头率~残局）的 x 中心坐标"""
    best_right: Optional[List[OCRToken]] = None
    best_count = 0

    for y in row_centers:
        row_tokens = [t for t in tokens if abs(t.y - y) <= row_tol]
        row_tokens.sort(key=lambda t: t.x)
        kda_idx = _pick_kda_index(row_tokens)
        if kda_idx is None:
            continue
        right = row_tokens[kda_idx + 1:]
        if len(right) > best_count:
            best_count = len(right)
            best_right = right

    if best_right is None or best_count < 8:
        return None

    return [t.x for t in best_right[:8]]


def _assign_to_columns(
    right_tokens: List[OCRToken],
    col_positions: List[float],
    col_tol: float = 60.0,
) -> List[str]:
    """
    将 KDA 右侧的 token 按 x 坐标匹配到 8 个固定列。
    若某列没有对应 token，则填 "0"。
    使用贪心分配：按距离从近到远逐对匹配，保证一一映射。
    """
    n_cols = len(col_positions)
    assigned = ["0"] * n_cols

    # 构建 (token_idx, col_idx, distance) 候选
    mappings: List[Tuple[int, int, float]] = []
    for ti, tok in enumerate(right_tokens):
        best_ci: Optional[int] = None
        best_dist = float("inf")
        for ci, cx in enumerate(col_positions):
            dist = abs(tok.x - cx)
            if dist < best_dist:
                best_dist = dist
                best_ci = ci
        if best_ci is not None and best_dist <= col_tol:
            mappings.append((ti, best_ci, best_dist))

    # 贪心：距离最近的先匹配
    mappings.sort(key=lambda m: m[2])
    used_tokens: set = set()
    used_cols: set = set()
    for ti, ci, _ in mappings:
        if ti in used_tokens or ci in used_cols:
            continue
        assigned[ci] = right_tokens[ti].text
        used_tokens.add(ti)
        used_cols.add(ci)

    return assigned


# ---------------------------------------------------------------------------
# 构建单个玩家
# ---------------------------------------------------------------------------

def _safe_float(text: str, default: float = 0.0) -> float:
    v = _to_float(text)
    return float(v) if v is not None else default


def _safe_int(text: str, default: int = 0) -> int:
    v = _to_int(text)
    return int(v) if v is not None else default


def _build_player(
    row_tokens: List[OCRToken],
    col_positions: Optional[List[float]] = None,
) -> Optional[Dict[str, Any]]:
    if not row_tokens:
        return None

    kda_idx = _pick_kda_index(row_tokens)
    if kda_idx is None:
        return None

    kda = _parse_kda(row_tokens[kda_idx].text)
    if kda is None:
        return None

    left_tokens = row_tokens[:kda_idx]
    right_tokens = row_tokens[kda_idx + 1:]

    # 名字：保留原始文本（不做数值清洗）
    name_parts = [t.text for t in left_tokens if _is_name_token(t.text)]
    player_name = " ".join(name_parts).strip()
    if not player_name:
        player_name = "unknown"

    # 右侧 8 列：爆头率 ADR RWS Rating+ 首杀 首死 多杀 残局
    if col_positions is not None and len(col_positions) == 8:
        vals = _assign_to_columns(right_tokens, col_positions)
    else:
        # 降级：按顺序取
        vals = [t.text for t in right_tokens]
        while len(vals) < 8:
            vals.append("0")
        vals = vals[:8]

    return {
        "player_name": player_name,
        "kda": kda,
        "headshot_percentage": _parse_headshot(vals[0]),
        "adr": _safe_float(vals[1], 0.0),
        "rws": _safe_float(vals[2], 0.0),
        "rating_plus": _parse_rating_plus(vals[3]),
        "first_kills": _safe_int(vals[4], 0),
        "first_deaths": _safe_int(vals[5], 0),
        "multi_kills": _safe_int(vals[6], 0),
        "clutches": _safe_int(vals[7], 0),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def extract_match_stats(
    image_path: str,
    debug_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    ocr = PaddleOCR(use_textline_orientation=True, lang="en")
    raw = ocr.predict(image_path)

    tokens = _extract_tokens_from_predict(raw)

    # debug：把所有 token 落盘，方便排查
    if debug_path:
        debug_data = [asdict(t) for t in tokens]
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug_data, f, ensure_ascii=False, indent=2)
        print(f"[DEBUG] 共提取 {len(tokens)} 个 token，已写入 {debug_path}")

    if not tokens:
        print("[WARN] 未提取到任何 OCR token，请检查图片或 OCR 模型")
        return []

    row_centers = _cluster_rows_by_kda(tokens)
    if not row_centers:
        print("[WARN] 未找到任何 KDA 行")
        return []

    # 行容差
    row_tol = 18.0
    if len(row_centers) > 1:
        gaps = [row_centers[i + 1] - row_centers[i] for i in range(len(row_centers) - 1)]
        row_tol = max(10.0, min(30.0, median(gaps) * 0.42))

    # 从最完整行推断列 x 坐标
    col_positions = _find_column_positions(tokens, row_centers, row_tol)
    if col_positions is not None:
        print(f"[INFO] 列坐标参考: {[round(x, 1) for x in col_positions]}")
    else:
        print("[WARN] 未能推断列坐标，将按顺序分配（可能不准确）")

    players: List[Dict[str, Any]] = []
    for y in row_centers:
        row_tokens = [t for t in tokens if abs(t.y - y) <= row_tol]
        row_tokens.sort(key=lambda t: t.x)
        player = _build_player(row_tokens, col_positions)
        if player is not None:
            players.append(player)

    # 去重
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for p in players:
        key = (
            p["player_name"],
            p["kda"]["kills"],
            p["kda"]["assists"],
            p["kda"]["deaths"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)

    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR CS 战绩截图并输出 test.json 格式")
    parser.add_argument("--image", type=str, default="img/test.png", help="输入图片路径")
    parser.add_argument("--output", type=str, default="test.json", help="输出 JSON 路径")
    parser.add_argument("--debug", type=str, default="debug_tokens.json",
                        help="调试用：将所有 OCR token 写入此文件（设为空字符串可关闭）")
    args = parser.parse_args()

    debug_path = args.debug if args.debug else None
    players = extract_match_stats(args.image, debug_path=debug_path)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(players, f, ensure_ascii=False, indent=2)

    print(f"已输出 {len(players)} 条数据到 {args.output}")


if __name__ == "__main__":
    main()
