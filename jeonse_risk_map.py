"""
전세사기 위험 지역 시각화 프로그램
전세가율(전세가/매매가 비율)을 기반으로 지역별 위험도를 색상으로 표현합니다.

위험도 기준:
  위험   (빨강): 전세가율 80% 이상 - 깡통전세 고위험
  경계   (주황): 전세가율 70~80%
  주의   (노랑): 전세가율 60~70%
  안전   (초록): 전세가율 60% 미만
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import font_manager
import warnings
warnings.filterwarnings("ignore")

# ── 한글 폰트 설정 ─────────────────────────────────────────────────────────────
import subprocess, os

def setup_korean_font():
    # 나눔고딕 설치
    try:
        subprocess.run(
            ["apt-get", "install", "-y", "-q", "fonts-nanum"],
            capture_output=True, check=True
        )
    except Exception:
        pass
    font_manager.fontManager.__init__()
    nanum_fonts = [
        f.fname for f in font_manager.fontManager.ttflist
        if "Nanum" in f.name or "nanum" in f.fname.lower()
    ]
    if nanum_fonts:
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=nanum_fonts[0]).get_name()
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

setup_korean_font()

# ── 데이터 로드 & 전처리 ────────────────────────────────────────────────────────
DATA_FILE = "jeonse_data.csv"

df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
df.columns = ["주택유형", "광역", "권역", "시군구", "읍면동"] + list(df.columns[5:])
MONTH_COLS = [c for c in df.columns if c.startswith("202")]

# 숫자 변환 (하이픈 → NaN)
df[MONTH_COLS] = df[MONTH_COLS].replace("-", np.nan).astype(float)

# 최신월 기준 위험도 판정
LATEST = MONTH_COLS[-1]

def risk_level(ratio):
    if pd.isna(ratio):
        return ("데이터없음", "#cccccc", 0)
    if ratio >= 80:
        return ("위험 (80%↑)", "#d32f2f", 4)
    if ratio >= 70:
        return ("경계 (70~80%)", "#f57c00", 3)
    if ratio >= 60:
        return ("주의 (60~70%)", "#fbc02d", 2)
    return ("안전 (60%↓)", "#388e3c", 1)

df["위험등급"], df["색상"], df["위험점수"] = zip(*df[LATEST].map(risk_level))

# ── 분석 대상 필터링 ────────────────────────────────────────────────────────────
# 아파트 + 시군구 단위 소계(읍면동이 '소계')
apt = df[
    (df["주택유형"] == "아파트") &
    (df["읍면동"] == "소계") &
    (df["시군구"] != "소계") &
    (df["시군구"].notna())
].copy()

# 광역 단위 종합(모든 주택유형 합산, '소계'만)
metro = df[
    (df["주택유형"] == "아파트") &
    (df["시군구"] == "소계") &
    (df["권역"] == "소계") &
    (~df["광역"].isin(["전국", "수도권", "지방", "5대광역시", "6대광역시", "8개도", "9개도"]))
].copy()

# ── 색상 팔레트 ─────────────────────────────────────────────────────────────────
RISK_COLORS = {
    "위험 (80%↑)":   "#d32f2f",
    "경계 (70~80%)": "#f57c00",
    "주의 (60~70%)": "#fbc02d",
    "안전 (60%↓)":   "#388e3c",
    "데이터없음":     "#cccccc",
}
LEGEND_PATCHES = [
    mpatches.Patch(color=v, label=k) for k, v in RISK_COLORS.items()
]

# ══════════════════════════════════════════════════════════════════════════════
# 그림 1 : 광역시도별 전세가율 + 위험등급 (수평 막대)
# ══════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(14, 9))
fig1.patch.set_facecolor("#1a1a2e")
ax1.set_facecolor("#1a1a2e")

metro_sorted = metro.sort_values(LATEST, ascending=True).dropna(subset=[LATEST])

bars = ax1.barh(
    metro_sorted["광역"],
    metro_sorted[LATEST],
    color=metro_sorted["색상"],
    edgecolor="#333355",
    linewidth=0.6,
    height=0.7,
)

# 값 표시
for bar, val, grade in zip(bars, metro_sorted[LATEST], metro_sorted["위험등급"]):
    ax1.text(
        bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}%",
        va="center", ha="left", color="white", fontsize=9, fontweight="bold"
    )

# 80% 기준선
ax1.axvline(80, color="#ff4444", linestyle="--", linewidth=1.2, alpha=0.8, label="위험선 (80%)")
ax1.axvline(70, color="#ff9900", linestyle="--", linewidth=1.0, alpha=0.7, label="경계선 (70%)")
ax1.axvline(60, color="#ffee00", linestyle="--", linewidth=0.8, alpha=0.6, label="주의선 (60%)")

ax1.set_xlim(30, 95)
ax1.set_xlabel("전세가율 (%)", color="white", fontsize=11)
ax1.set_title(
    f"광역시도별 아파트 전세가율 위험도 현황  [{LATEST}]",
    color="white", fontsize=14, fontweight="bold", pad=15
)
ax1.tick_params(colors="white")
for spine in ax1.spines.values():
    spine.set_edgecolor("#444466")

legend1 = ax1.legend(
    handles=LEGEND_PATCHES,
    loc="lower right", framealpha=0.3,
    facecolor="#2a2a4e", labelcolor="white", fontsize=9
)
ax1.add_artist(legend1)
ax1.legend(loc="upper left", framealpha=0.3, facecolor="#2a2a4e", labelcolor="white", fontsize=9)

plt.tight_layout()
fig1.savefig("chart1_metro_risk.png", dpi=150, bbox_inches="tight",
             facecolor=fig1.get_facecolor())
print("✔  chart1_metro_risk.png 저장")

# ══════════════════════════════════════════════════════════════════════════════
# 그림 2 : 시군구 Top 30 위험 지역 (전세가율 높은 순)
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(14, 12))
fig2.patch.set_facecolor("#1a1a2e")
ax2.set_facecolor("#1a1a2e")

top30 = apt.nlargest(30, LATEST).sort_values(LATEST, ascending=True)
top30["지역명"] = top30["광역"] + " " + top30["시군구"]

bars2 = ax2.barh(
    top30["지역명"],
    top30[LATEST],
    color=top30["색상"],
    edgecolor="#333355",
    linewidth=0.6,
    height=0.75,
)

for bar, val in zip(bars2, top30[LATEST]):
    ax2.text(
        bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
        f"{val:.2f}%",
        va="center", ha="left", color="white", fontsize=8.5
    )

ax2.axvline(80, color="#ff4444", linestyle="--", linewidth=1.2, alpha=0.8)
ax2.axvline(70, color="#ff9900", linestyle="--", linewidth=1.0, alpha=0.7)
ax2.set_xlim(70, 92)
ax2.set_xlabel("전세가율 (%)", color="white", fontsize=11)
ax2.set_title(
    f"전세사기 고위험 지역 TOP 30 (아파트)  [{LATEST}]",
    color="white", fontsize=14, fontweight="bold", pad=15
)
ax2.tick_params(colors="white")
for spine in ax2.spines.values():
    spine.set_edgecolor("#444466")

ax2.legend(handles=LEGEND_PATCHES, loc="lower right",
           framealpha=0.3, facecolor="#2a2a4e", labelcolor="white", fontsize=9)

plt.tight_layout()
fig2.savefig("chart2_top30_risk.png", dpi=150, bbox_inches="tight",
             facecolor=fig2.get_facecolor())
print("✔  chart2_top30_risk.png 저장")

# ══════════════════════════════════════════════════════════════════════════════
# 그림 3 : 위험도 히트맵 (광역 × 월별 추이)
# ══════════════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(14, 9))
fig3.patch.set_facecolor("#1a1a2e")
ax3.set_facecolor("#1a1a2e")

heat_data = metro.set_index("광역")[MONTH_COLS].dropna(how="all")
# 최신월 기준으로 정렬
heat_data = heat_data.loc[heat_data[LATEST].sort_values(ascending=False).index]

# 커스텀 컬러맵 (60↓초록 → 70노랑 → 80빨강)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "jeonse_risk",
    [
        (0.0, "#1b5e20"),  # 진초록
        (0.4, "#388e3c"),  # 초록
        (0.6, "#fbc02d"),  # 노랑
        (0.75, "#f57c00"), # 주황
        (0.9, "#d32f2f"),  # 빨강
        (1.0, "#7f0000"),  # 진빨강
    ],
)

im = ax3.imshow(heat_data.values, cmap=cmap, vmin=45, vmax=90, aspect="auto")

ax3.set_xticks(range(len(MONTH_COLS)))
ax3.set_xticklabels(MONTH_COLS, color="white", fontsize=9)
ax3.set_yticks(range(len(heat_data)))
ax3.set_yticklabels(heat_data.index, color="white", fontsize=9)

# 셀 값 표시
for i in range(len(heat_data)):
    for j, col in enumerate(MONTH_COLS):
        val = heat_data.iloc[i, j]
        if not pd.isna(val):
            text_color = "white" if val > 75 else "#111111"
            ax3.text(j, i, f"{val:.1f}", ha="center", va="center",
                     fontsize=7.5, color=text_color, fontweight="bold")

cb = plt.colorbar(im, ax=ax3, fraction=0.03, pad=0.02)
cb.set_label("전세가율 (%)", color="white", fontsize=10)
cb.ax.yaxis.set_tick_params(color="white")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

ax3.set_title(
    "광역시도별 아파트 전세가율 월별 추이 히트맵",
    color="white", fontsize=14, fontweight="bold", pad=15
)

plt.tight_layout()
fig3.savefig("chart3_heatmap.png", dpi=150, bbox_inches="tight",
             facecolor=fig3.get_facecolor())
print("✔  chart3_heatmap.png 저장")

# ══════════════════════════════════════════════════════════════════════════════
# 그림 4 : 위험 분포 버블차트 (전세가율 vs 증감폭)
# ══════════════════════════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(14, 9))
fig4.patch.set_facecolor("#1a1a2e")
ax4.set_facecolor("#1a1a2e")
ax4.grid(color="#333355", linewidth=0.5, alpha=0.7)

# 최근 3개월 증감 계산
apt_bubble = apt.dropna(subset=[LATEST, MONTH_COLS[0]]).copy()
apt_bubble["증감"] = apt_bubble[LATEST] - apt_bubble[MONTH_COLS[0]]
apt_bubble["지역명"] = apt_bubble["광역"] + " " + apt_bubble["시군구"]

scatter = ax4.scatter(
    apt_bubble[LATEST],
    apt_bubble["증감"],
    c=apt_bubble["위험점수"],
    cmap=mcolors.LinearSegmentedColormap.from_list(
        "risk4", ["#388e3c", "#fbc02d", "#f57c00", "#d32f2f"]
    ),
    s=apt_bubble[LATEST].apply(lambda x: max(20, (x - 50) * 4)),
    alpha=0.75,
    edgecolors="#ffffff33",
    linewidth=0.4,
    vmin=1, vmax=4
)

# 고위험 레이블 (전세가율 82% 초과)
for _, row in apt_bubble[apt_bubble[LATEST] > 82].iterrows():
    ax4.annotate(
        row["지역명"],
        xy=(row[LATEST], row["증감"]),
        xytext=(5, 3), textcoords="offset points",
        fontsize=7, color="white", alpha=0.85,
    )

ax4.axvline(80, color="#ff4444", linestyle="--", linewidth=1.2, alpha=0.8, label="위험선 80%")
ax4.axvline(70, color="#ff9900", linestyle="--", linewidth=1.0, alpha=0.7, label="경계선 70%")
ax4.axhline(0, color="#aaaaaa", linestyle="-", linewidth=0.8, alpha=0.5)

ax4.set_xlabel(f"전세가율 (%) — {LATEST}", color="white", fontsize=11)
ax4.set_ylabel(f"전세가율 변동 ({MONTH_COLS[0]} → {LATEST})", color="white", fontsize=11)
ax4.set_title(
    "전세가율 수준 vs 변동폭 위험 산점도 (아파트 시군구)",
    color="white", fontsize=14, fontweight="bold", pad=15
)
ax4.tick_params(colors="white")
for spine in ax4.spines.values():
    spine.set_edgecolor("#444466")
ax4.legend(handles=LEGEND_PATCHES + [
    mpatches.Patch(color="none", label=""),
    mpatches.Patch(color="#ff4444", alpha=0.6, label=f"기준: {MONTH_COLS[0]}→{LATEST}")
], loc="upper left", framealpha=0.3, facecolor="#2a2a4e", labelcolor="white", fontsize=9)

plt.tight_layout()
fig4.savefig("chart4_scatter.png", dpi=150, bbox_inches="tight",
             facecolor=fig4.get_facecolor())
print("✔  chart4_scatter.png 저장")

# ══════════════════════════════════════════════════════════════════════════════
# 요약 출력
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print(f"  전세사기 위험 지역 분석 요약  ({LATEST} 기준)")
print("=" * 55)

for grade, color in [("위험 (80%↑)", "🔴"), ("경계 (70~80%)", "🟠"),
                     ("주의 (60~70%)", "🟡"), ("안전 (60%↓)", "🟢")]:
    subset = apt[apt["위험등급"] == grade]
    if subset.empty:
        continue
    count = len(subset)
    top = subset.nlargest(3, LATEST)[["광역", "시군구", LATEST]]
    names = ", ".join(f"{r['광역']} {r['시군구']}({r[LATEST]:.1f}%)" for _, r in top.iterrows())
    print(f"\n{color} {grade}: {count}개 지역")
    print(f"   대표: {names}")

print("\n생성된 파일:")
for f in ["chart1_metro_risk.png", "chart2_top30_risk.png",
          "chart3_heatmap.png", "chart4_scatter.png"]:
    print(f"  • {f}")
