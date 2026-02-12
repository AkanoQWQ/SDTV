"""
为 merged.json 中每位玩家绘制雷达图，展示综合战力。
用法: python graph.py
      python graph.py --input result/merged.json --output result/radar.png
"""

import argparse
import json
import math
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


# ---------------------------------------------------------------------------
# 雷达图维度定义
# ---------------------------------------------------------------------------

# (字段名/计算方式, 显示标签)
DIMENSIONS = [
    ("rating_plus", "Rating+"),
    ("adr",         "ADR"),
    ("rws",         "RWS"),
    ("headshot_percentage", "爆头率"),
    ("kd_ratio",    "K/D"),
    ("dpr",         "DPR"),
]


def _compute_values(player: Dict[str, Any]) -> List[float]:
    """从玩家数据提取各维度原始值"""
    kda = player["kda"]
    kd = kda["kills"] / max(kda["deaths"], 1)
    dpr = kda["deaths"] / max(player.get("total_turns", 1), 1)
    return [
        player.get("rating_plus", 0.0),
        player.get("adr", 0.0),
        player.get("rws", 0.0),
        player.get("headshot_percentage", 0.0),
        round(kd, 2),
        round(dpr, 2),
    ]


def _normalize(
    values: List[float],
    maxvals: List[float],
    invert_indices: List[int] = [],
) -> List[float]:
    """归一化，允许超过 1.0（溢出效果）。invert_indices 中的维度做反转（值越小越好）"""
    result: List[float] = []
    for i, (v, m) in enumerate(zip(values, maxvals)):
        normed = v / m if m > 0 else 0.0
        if i in invert_indices:
            # 反转：最小值 → 最高分，最大值 → 最低分
            normed = max(2.0 - normed, 0.0)
        result.append(normed)
    return result


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def _setup_chinese_font() -> None:
    """尝试设置中文字体，找不到就用默认"""
    candidates = [
        "Microsoft YaHei", "SimHei", "PingFang SC",
        "WenQuanYi Micro Hei", "Noto Sans CJK SC",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return


def draw_radar(
    players: List[Dict[str, Any]],
    output_path: str,
) -> None:
    _setup_chinese_font()

    labels = [d[1] for d in DIMENSIONS]
    n = len(labels)

    # 自适应归一化上限：最大值 × 0.95，使最佳玩家略微溢出外圈
    # DPR 是反向指标（越低越好），需要特殊处理
    all_raw = [_compute_values(p) for p in players]
    dpr_idx = labels.index("DPR")
    max_vals = []
    for dim in range(n):
        col_max = max(raw[dim] for raw in all_raw)
        max_vals.append(col_max * 0.95 if col_max > 0 else 1.0)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    n_players = len(players)
    cols = min(n_players, 3)
    rows = math.ceil(n_players / cols)
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(6 * cols, 5.5 * rows),
        subplot_kw={"polar": True},
    )
    if n_players == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    # 颜色循环
    cmap = plt.cm.get_cmap("tab10")

    for i, player in enumerate(players):
        ax = axes[i]
        raw = _compute_values(player)
        normed = _normalize(raw, max_vals, invert_indices=[dpr_idx])
        normed += normed[:1]  # 闭合

        color = cmap(i % 10)
        ax.fill(angles, normed, alpha=0.25, color=color)
        ax.plot(angles, normed, linewidth=2, color=color)

        # 在顶点标注原始数值
        for j, (angle, val, norm) in enumerate(zip(angles[:-1], raw, normed[:-1])):
            if labels[j] == "爆头率":
                label_text = f"{val:.0%}"
            elif labels[j] == "DPR":
                label_text = f"{val:.2f}↓"
            else:
                label_text = f"{val:.2f}"
            ax.text(
                angle, norm + 0.08, label_text,
                ha="center", va="center", fontsize=9, color=color, fontweight="bold",
            )

        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="grey")
        ax.set_title(
            player["player_name"],
            fontsize=14, fontweight="bold", pad=18,
        )

    # 隐藏多余子图
    for j in range(n_players, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("SDTV 战绩雷达图", fontsize=18, fontweight="bold", y=1.01)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"雷达图已保存到 {output_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="为 CS 战绩数据绘制雷达图")
    parser.add_argument("--input", type=str, default="result/merged.json",
                        help="输入 JSON 路径（默认 result/merged.json）")
    parser.add_argument("--output", type=str, default="result/radar.png",
                        help="输出图片路径（默认 result/radar.png）")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 文件不存在: {args.input}")
        return

    with open(args.input, "r", encoding="utf-8") as f:
        players = json.load(f)

    if not isinstance(players, list) or not players:
        print("[ERROR] JSON 应为非空数组")
        return

    print(f"读取 {len(players)} 名选手数据")
    draw_radar(players, args.output)


if __name__ == "__main__":
    main()
