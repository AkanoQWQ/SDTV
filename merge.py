import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 选手名字库 —— 从 players.json 读取
# ---------------------------------------------------------------------------

def _load_roster(path: str = "players.json") -> List[str]:
    if not os.path.exists(path):
        print(f"[WARN] 名字库文件 {path} 不存在，模糊匹配将不生效")
        return []
    with open(path, "r", encoding="utf-8") as f:
        roster = json.load(f)
    if not isinstance(roster, list):
        print(f"[WARN] {path} 格式不正确，应为字符串数组")
        return []
    return roster


# ---------------------------------------------------------------------------
# 模糊匹配
# ---------------------------------------------------------------------------

def _levenshtein(s1: str, s2: str) -> int:
    """Levenshtein 编辑距离"""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def _clean_name(name: str) -> str:
    """去除 OCR 常见尾部噪声（×、.、, 等），然后 lower + strip"""
    t = name.strip()
    # 去掉尾部常见 OCR 噪声字符
    t = re.sub(r"[\s×·•\-_.,;:!?]+$", "", t)
    return t.lower()


def fuzzy_match(ocr_name: str, roster: List[str], threshold: float = 0.4) -> str:
    """
    将 OCR 识别的名字模糊匹配到名字库。
    threshold: 允许的最大 编辑距离/名字长度 比例（越大越宽松）。
    匹配不上则返回原名。
    """
    ocr_clean = _clean_name(ocr_name)
    if not ocr_clean:
        return ocr_name

    best_match: Optional[str] = None
    best_ratio = float("inf")

    for canonical in roster:
        canon_clean = _clean_name(canonical)
        dist = _levenshtein(ocr_clean, canon_clean)
        max_len = max(len(ocr_clean), len(canon_clean))
        ratio = dist / max_len if max_len > 0 else 0.0

        if ratio < best_ratio:
            best_ratio = ratio
            best_match = canonical

    if best_match is not None and best_ratio <= threshold:
        return best_match

    return ocr_name


# ---------------------------------------------------------------------------
# 合并逻辑
# ---------------------------------------------------------------------------

def merge_players(
    all_records: List[Dict[str, Any]],
    roster: List[str],
) -> List[Dict[str, Any]]:
    """
    合并多场战绩：
      - KDA：累加
      - headshot_percentage / adr / rws / rating_plus：按 turns 加权平均
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in all_records:
        matched = fuzzy_match(record["player_name"], roster)
        groups.setdefault(matched, []).append(record)

    results: List[Dict[str, Any]] = []
    for name, records in groups.items():
        total_turns = sum(r["turns"] for r in records)

        def _weighted_avg(field: str) -> float:
            if total_turns == 0:
                return 0.0
            return sum(r[field] * r["turns"] for r in records) / total_turns

        merged = {
            "player_name": name,
            "kda": {
                "kills": sum(r["kda"]["kills"] for r in records),
                "assists": sum(r["kda"]["assists"] for r in records),
                "deaths": sum(r["kda"]["deaths"] for r in records),
            },
            "headshot_percentage": round(_weighted_avg("headshot_percentage"), 6),
            "adr": round(_weighted_avg("adr"), 2),
            "rws": round(_weighted_avg("rws"), 2),
            "rating_plus": round(_weighted_avg("rating_plus"), 2),
            "total_turns": total_turns,
        }
        results.append(merged)

    # 按 rating_plus 降序
    results.sort(key=lambda x: x["rating_plus"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="合并多场 CS 战绩数据")
    parser.add_argument("--input-dir", type=str, default="output",
                        help="输入 JSON 目录（默认 output/）")
    parser.add_argument("--output", type=str, default="merged.json",
                        help="合并后输出路径（默认 merged.json）")
    parser.add_argument("--roster", type=str, default="players.json",
                        help="选手名字库路径（默认 players.json）")
    args = parser.parse_args()

    roster = _load_roster(args.roster)
    print(f"已加载 {len(roster)} 名选手: {roster}")

    json_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not json_files:
        print(f"[WARN] 在 {args.input_dir}/ 下未找到任何 .json 文件")
        return

    all_records: List[Dict[str, Any]] = []
    for path in json_files:
        print(f"读取: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            all_records.extend(data)

    print(f"共读取 {len(all_records)} 条记录，来自 {len(json_files)} 个文件")

    merged = merge_players(all_records, roster)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"合并完成，已输出 {len(merged)} 名选手数据到 {args.output}")


if __name__ == "__main__":
    main()
