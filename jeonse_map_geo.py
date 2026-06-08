"""
전세사기 위험 지역 인터랙티브 지도
folium + OpenStreetMap 기반으로 지역별 전세가율 위험도를 지도에 표시합니다.
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
import warnings
warnings.filterwarnings("ignore")

# ── 좌표 데이터베이스 ──────────────────────────────────────────────────────────
# {광역: {지역명: (위도, 경도)}}
COORDS = {
    # ── 광역시도 중심 ──────────────────────────────────────────────────────────
    "_province": {
        "서울": (37.5665, 126.9780),
        "부산": (35.1796, 129.0756),
        "대구": (35.8714, 128.6014),
        "인천": (37.4563, 126.7052),
        "광주": (35.1595, 126.8526),
        "대전": (36.3504, 127.3845),
        "울산": (35.5384, 129.3114),
        "세종": (36.4800, 127.2890),
        "경기": (37.4138, 127.5183),
        "강원": (37.8228, 128.1555),
        "충북": (36.6358, 127.4914),
        "충남": (36.6588, 126.6728),
        "전북": (35.7175, 127.1530),
        "전남": (34.8679, 126.9910),
        "경북": (36.4919, 128.8889),
        "경남": (35.4606, 128.2132),
        "제주": (33.4890, 126.4983),
    },
    # ── 서울 구 ───────────────────────────────────────────────────────────────
    "서울": {
        "종로": (37.5730, 126.9794), "중":    (37.5635, 126.9975),
        "용산": (37.5384, 126.9654), "성동":  (37.5635, 127.0369),
        "광진": (37.5384, 127.0824), "동대문":(37.5744, 127.0398),
        "중랑": (37.6065, 127.0926), "성북":  (37.5894, 127.0167),
        "강북": (37.6396, 127.0258), "도봉":  (37.6688, 127.0467),
        "노원": (37.6542, 127.0568), "은평":  (37.6176, 126.9227),
        "서대문":(37.5791,126.9368), "마포":  (37.5638, 126.9084),
        "양천": (37.5170, 126.8664), "강서":  (37.5509, 126.8496),
        "구로": (37.4954, 126.8874), "금천":  (37.4569, 126.8956),
        "영등포":(37.5264,126.8964), "동작":  (37.5122, 126.9394),
        "관악": (37.4784, 126.9516), "서초":  (37.4837, 127.0324),
        "강남": (37.5172, 127.0473), "송파":  (37.5145, 127.1059),
        "강동": (37.5301, 127.1238),
    },
    # ── 경기도 시 ─────────────────────────────────────────────────────────────
    "경기": {
        "수원":   (37.2636, 127.0286), "성남":   (37.4449, 127.1388),
        "용인":   (37.2411, 127.1776), "안양":   (37.3943, 126.9568),
        "안산":   (37.3219, 126.8309), "과천":   (37.4292, 126.9878),
        "광명":   (37.4784, 126.8647), "평택":   (36.9921, 127.1127),
        "오산":   (37.1497, 127.0776), "시흥":   (37.3798, 126.8028),
        "군포":   (37.3616, 126.9346), "의왕":   (37.3449, 126.9688),
        "하남":   (37.5398, 127.2149), "이천":   (37.2718, 127.4349),
        "안성":   (36.9997, 127.2798), "김포":   (37.6151, 126.7159),
        "화성":   (37.1998, 126.8311), "광주":   (37.4297, 127.2551),
        "여주":   (37.2984, 127.6374), "양주":   (37.7851, 127.0457),
        "포천":   (37.8944, 127.1999), "의정부": (37.7382, 127.0337),
        "구리":   (37.5943, 127.1297), "남양주": (37.6360, 127.2167),
        "파주":   (37.7601, 126.7799), "고양":   (37.6584, 126.8320),
        "동두천": (37.9035, 127.0607), "부천":   (37.5035, 126.7660),
    },
    # ── 부산 구 ───────────────────────────────────────────────────────────────
    "부산": {
        "중":    (35.1066, 129.0324), "서":     (35.0989, 129.0244),
        "동":    (35.1361, 129.0438), "영도":   (35.0910, 129.0681),
        "부산진":(35.1631, 129.0537), "동래":   (35.2047, 129.0831),
        "남":    (35.1363, 129.0836), "북":     (35.2095, 128.9895),
        "해운대":(35.1631, 129.1639), "사하":   (35.1021, 128.9741),
        "금정":  (35.2427, 129.0921), "강서":   (35.2095, 128.9814),
        "연제":  (35.1764, 129.0783), "수영":   (35.1459, 129.1138),
        "사상":  (35.1499, 128.9922), "기장":   (35.2447, 129.2220),
    },
    # ── 대구 구 ───────────────────────────────────────────────────────────────
    "대구": {
        "중":   (35.8703, 128.5911), "동":    (35.8869, 128.6352),
        "서":   (35.8716, 128.5603), "남":    (35.8461, 128.5980),
        "북":   (35.8849, 128.5827), "수성":  (35.8580, 128.6306),
        "달서": (35.8296, 128.5328), "달성":  (35.7749, 128.4316),
    },
    # ── 인천 구 ───────────────────────────────────────────────────────────────
    "인천": {
        "중":    (37.4738, 126.6164), "동":    (37.4739, 126.6434),
        "미추홀":(37.4637, 126.6502), "연수":  (37.4100, 126.6780),
        "남동":  (37.4469, 126.7310), "부평":  (37.5082, 126.7216),
        "계양":  (37.5376, 126.7378), "서":    (37.5452, 126.6759),
    },
    # ── 광주 구 ───────────────────────────────────────────────────────────────
    "광주": {
        "동":   (35.1461, 126.9231), "서":   (35.1519, 126.8854),
        "남":   (35.1338, 126.9024), "북":   (35.1746, 126.9116),
        "광산": (35.1396, 126.7933),
    },
    # ── 대전 구 ───────────────────────────────────────────────────────────────
    "대전": {
        "동":   (36.3496, 127.4546), "중":   (36.3253, 127.4210),
        "서":   (36.3551, 127.3836), "유성": (36.3624, 127.3564),
        "대덕": (36.3462, 127.4154),
    },
    # ── 울산 구 ───────────────────────────────────────────────────────────────
    "울산": {
        "중":   (35.5693, 129.3330), "남":   (35.5390, 129.3264),
        "동":   (35.5050, 129.4168), "북":   (35.5831, 129.3615),
        "울주": (35.5225, 129.1530),
    },
    # ── 강원 시군 ─────────────────────────────────────────────────────────────
    "강원": {
        "춘천": (37.8813, 127.7298), "원주": (37.3422, 127.9202),
        "강릉": (37.7519, 128.8761), "속초": (38.2069, 128.5913),
        "동해": (37.5247, 129.1138), "태백": (37.1651, 128.9855),
        "삼척": (37.4502, 129.1653),
    },
    # ── 충북 시군 ─────────────────────────────────────────────────────────────
    "충북": {
        "청주": (36.6424, 127.4890), "충주": (36.9910, 127.9260),
        "제천": (37.1329, 128.1909), "음성": (36.9396, 127.4506),
    },
    # ── 충남 시군 ─────────────────────────────────────────────────────────────
    "충남": {
        "천안": (36.8151, 127.1138), "공주": (36.4465, 127.1191),
        "보령": (36.3333, 126.6125), "아산": (36.7898, 127.0036),
        "서산": (36.7846, 126.4502), "당진": (36.8905, 126.6296),
        "논산": (36.1873, 127.0992), "계룡": (36.2745, 127.2499),
        "홍성": (36.6015, 126.6607), "예산": (36.6826, 126.8496),
    },
    # ── 전북 시군 ─────────────────────────────────────────────────────────────
    "전북": {
        "전주": (35.8242, 127.1480), "군산": (35.9678, 126.7368),
        "익산": (35.9482, 126.9575), "정읍": (35.5698, 126.8554),
        "남원": (35.4165, 127.3902), "김제": (35.8033, 126.8809),
    },
    # ── 전남 시군 ─────────────────────────────────────────────────────────────
    "전남": {
        "목포": (34.8118, 126.3922), "여수": (34.7604, 127.6622),
        "순천": (34.9506, 127.4872), "광양": (34.9432, 127.6959),
        "나주": (35.0160, 126.7108), "무안": (34.9904, 126.4820),
    },
    # ── 경북 시군 ─────────────────────────────────────────────────────────────
    "경북": {
        "포항": (36.0190, 129.3435), "경주": (35.8562, 129.2249),
        "김천": (36.1396, 128.1136), "안동": (36.5684, 128.7295),
        "구미": (36.1195, 128.3441), "영주": (36.8056, 128.6237),
        "영천": (35.9733, 128.9384), "경산": (35.8253, 128.7414),
        "칠곡": (35.9973, 128.4018), "상주": (36.4108, 128.1593),
        "문경": (36.5864, 128.1858),
    },
    # ── 경남 시군 ─────────────────────────────────────────────────────────────
    "경남": {
        "창원": (35.2279, 128.6811), "진주": (35.1799, 128.1075),
        "사천": (35.0037, 128.0640), "김해": (35.2281, 128.8888),
        "양산": (35.3350, 129.0378), "거제": (34.8800, 128.6211),
        "밀양": (35.5038, 128.7461), "통영": (34.8544, 128.4328),
    },
    # ── 제주 시군 ─────────────────────────────────────────────────────────────
    "제주": {
        "제주": (33.4996, 126.5312), "서귀포": (33.2541, 126.5600),
    },
    # ── 세종 ─────────────────────────────────────────────────────────────────
    "세종": {
        "세종": (36.4800, 127.2890),
    },
}

# 청주 하위 구(서원·상당·흥덕·청원) 좌표
COORDS["충북"]["서원"] = (36.6073, 127.4603)
COORDS["충북"]["상당"] = (36.6430, 127.5260)
COORDS["충북"]["흥덕"] = (36.6339, 127.4444)
COORDS["충북"]["청원"] = (36.7268, 127.4793)

# 창원 하위 구
COORDS["경남"]["마산회원"] = (35.2179, 128.5826)
COORDS["경남"]["마산합포"] = (35.2009, 128.5536)
COORDS["경남"]["진해"]    = (35.1539, 128.6941)
COORDS["경남"]["의창"]    = (35.2585, 128.6397)
COORDS["경남"]["성산"]    = (35.2249, 128.6996)

# 고양 하위 지구
COORDS["경기"]["일산서"] = (37.6776, 126.7621)
COORDS["경기"]["일산동"] = (37.6574, 126.7986)
COORDS["경기"]["덕양"]   = (37.6423, 126.8445)

# 수원 하위 구
COORDS["경기"]["권선"] = (37.2454, 126.9963)
COORDS["경기"]["장안"] = (37.3012, 127.0140)
COORDS["경기"]["영통"] = (37.2554, 127.0566)
COORDS["경기"]["팔달"] = (37.2818, 127.0148)

# 용인 하위 구
COORDS["경기"]["처인"] = (37.2336, 127.2022)
COORDS["경기"]["기흥"] = (37.2753, 127.1149)
COORDS["경기"]["수지"] = (37.3221, 127.0994)

# 안양 하위 구
COORDS["경기"]["만안"] = (37.3889, 126.9251)
COORDS["경기"]["동안"] = (37.3942, 126.9627)

# 성남 하위 구
COORDS["경기"]["수정"] = (37.4486, 127.1469)
COORDS["경기"]["중원"] = (37.4434, 127.1485)
COORDS["경기"]["분당"] = (37.3784, 127.1147)

# 안산 하위 구
COORDS["경기"]["상록"] = (37.3152, 126.8516)
COORDS["경기"]["단원"] = (37.3229, 126.8073)

# 부천 하위 구
COORDS["경기"]["오정"] = (37.5058, 126.7843)
COORDS["경기"]["원미"] = (37.5035, 126.7698)
COORDS["경기"]["소사"] = (37.4826, 126.7706)


# ── 위험등급 판정 ──────────────────────────────────────────────────────────────
def risk_info(ratio):
    if pd.isna(ratio):
        return "데이터없음", "#888888", 0
    if ratio >= 80:
        return "위험", "#d32f2f", 4
    if ratio >= 70:
        return "경계", "#f57c00", 3
    if ratio >= 60:
        return "주의", "#fbc02d", 2
    return "안전", "#388e3c", 1


# ── 데이터 로드 ────────────────────────────────────────────────────────────────
df = pd.read_csv("jeonse_data.csv", encoding="utf-8-sig")
df.columns = ["주택유형", "광역", "권역", "시군구", "읍면동"] + list(df.columns[5:])
MONTH_COLS = [c for c in df.columns if c.startswith("202")]
df[MONTH_COLS] = df[MONTH_COLS].replace("-", np.nan).astype(float)
LATEST = MONTH_COLS[-1]


# ── 시각화 데이터 수집 ─────────────────────────────────────────────────────────
def collect_markers(df):
    """지도 마커 데이터를 수집합니다."""
    markers = []

    for _, row in df[df["주택유형"] == "아파트"].iterrows():
        prov  = row["광역"]
        kwon  = row["권역"]
        sgg   = row["시군구"]
        emd   = row["읍면동"]
        val   = row[LATEST]
        if pd.isna(val):
            continue

        # 배제: 집계 광역(전국/수도권 등)
        if prov in ("전국", "수도권", "지방", "5대광역시", "6대광역시",
                    "8개도", "9개도"):
            continue

        # 좌표 찾기
        coord = None
        prov_dict = COORDS.get(prov, {})

        # ① 서울: 읍면동=구 이름
        if prov == "서울" and emd != "소계":
            coord = prov_dict.get(emd)
            label = f"서울 {emd}"
            level = "구"

        # ② 경기: 시군구=하위 명칭(읍면동=소계)
        elif prov == "경기" and sgg != "소계" and emd == "소계":
            coord = prov_dict.get(sgg)
            label = f"경기 {sgg}"
            level = "시"

        # ③ 부산: 시군구=구 이름
        elif prov == "부산" and sgg != "소계" and emd == "소계":
            coord = prov_dict.get(sgg)
            label = f"부산 {sgg}"
            level = "구"

        # ④ 대구·인천·광주·대전·울산: 권역=구 이름, 시군구=소계
        elif prov in ("대구", "인천", "광주", "대전", "울산") \
                and kwon != "소계" and sgg == "소계" and emd == "소계":
            coord = prov_dict.get(kwon)
            label = f"{prov} {kwon}"
            level = "구"

        # ⑤ 경남: 시군구에 하위구(마산회원 등) 혹은 도시(소계)
        elif prov == "경남" and sgg != "소계" and emd == "소계":
            coord = prov_dict.get(sgg)
            label = f"경남 {sgg}"
            level = "시구"

        # ⑥ 충북 청주 하위구
        elif prov == "충북" and kwon == "청주" and sgg != "소계" and emd == "소계":
            coord = prov_dict.get(sgg)
            label = f"청주 {sgg}"
            level = "구"

        # ⑦ 기타 도: 권역=도시, 시군구=소계
        elif prov not in ("서울", "경기", "부산", "대구", "인천", "광주",
                          "대전", "울산") \
                and kwon != "소계" and sgg == "소계" and emd == "소계":
            coord = prov_dict.get(kwon)
            label = f"{prov} {kwon}"
            level = "시군"

        if coord is None:
            continue

        grade, color, score = risk_info(val)
        # 월별 데이터
        trend = {m: row[m] for m in MONTH_COLS}

        markers.append({
            "label": label, "lat": coord[0], "lon": coord[1],
            "ratio": val, "grade": grade, "color": color,
            "score": score, "level": level,
            "trend": trend,
        })

    return markers


markers = collect_markers(df)
print(f"총 {len(markers)}개 마커 수집")


# ── 광역시도 레벨 데이터 ──────────────────────────────────────────────────────
metro_df = df[
    (df["주택유형"] == "아파트") &
    (df["권역"] == "소계") & (df["시군구"] == "소계") & (df["읍면동"] == "소계") &
    (~df["광역"].isin(["전국", "수도권", "지방", "5대광역시", "6대광역시", "8개도", "9개도"]))
].copy()
metro_df = metro_df.dropna(subset=[LATEST])


# ── Folium 지도 생성 ──────────────────────────────────────────────────────────
m = folium.Map(
    location=[36.5, 127.8],
    zoom_start=7,
    tiles=None,
)

# 타일 레이어
folium.TileLayer(
    "CartoDB positron",
    name="밝은 지도",
    control=True,
).add_to(m)
folium.TileLayer(
    "CartoDB dark_matter",
    name="어두운 지도",
    control=True,
).add_to(m)
folium.TileLayer(
    "OpenStreetMap",
    name="OpenStreetMap",
    control=True,
).add_to(m)

# ── 광역 레이어 ────────────────────────────────────────────────────────────────
province_layer = folium.FeatureGroup(name="광역시도 (아파트 전체)", show=True)

for _, row in metro_df.iterrows():
    prov = row["광역"]
    val  = row[LATEST]
    grade, color, _ = risk_info(val)
    coord = COORDS["_province"].get(prov)
    if coord is None:
        continue

    # 추이 텍스트
    trend_rows = "".join(
        f"<tr><td style='padding:2px 8px;color:#aaa'>{m}</td>"
        f"<td style='padding:2px 8px;text-align:right;font-weight:bold'>{row[m]:.2f}%</td></tr>"
        for m in MONTH_COLS if not pd.isna(row[m])
    )
    popup_html = f"""
    <div style='font-family:sans-serif;min-width:200px'>
      <div style='background:{color};color:white;padding:8px 12px;border-radius:6px 6px 0 0;
                  font-size:15px;font-weight:bold'>{prov} &nbsp;[{grade}]</div>
      <div style='padding:8px 12px;background:#1e1e2e;color:white;border-radius:0 0 6px 6px'>
        <span style='font-size:22px;font-weight:bold;color:{color}'>{val:.2f}%</span>
        <span style='font-size:11px;color:#aaa'> 전세가율 ({LATEST})</span>
        <hr style='border-color:#333;margin:8px 0'>
        <table style='width:100%;font-size:12px'>{trend_rows}</table>
      </div>
    </div>"""

    folium.CircleMarker(
        location=coord,
        radius=22,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.40,
        weight=2.5,
        tooltip=f"{prov}  전세가율 {val:.1f}% [{grade}]",
        popup=folium.Popup(popup_html, max_width=260),
    ).add_to(province_layer)

    # 이름 레이블
    folium.Marker(
        location=coord,
        icon=folium.DivIcon(
            html=f"<div style='font-size:11px;font-weight:bold;color:white;"
                 f"text-shadow:1px 1px 2px #000,0 0 4px #000;text-align:center;"
                 f"white-space:nowrap'>{prov}<br>{val:.1f}%</div>",
            icon_size=(60, 30),
            icon_anchor=(30, 15),
        ),
    ).add_to(province_layer)

province_layer.add_to(m)

# ── 시군구 레이어 ──────────────────────────────────────────────────────────────
city_layer = folium.FeatureGroup(name="시군구 상세 (아파트)", show=True)

for mk in markers:
    grade = mk["grade"]
    color = mk["color"]
    val   = mk["ratio"]
    label = mk["label"]
    radius = max(5, min(14, (val - 50) * 0.30))

    trend_rows = "".join(
        f"<tr><td style='padding:2px 8px;color:#aaa'>{m}</td>"
        f"<td style='padding:2px 8px;text-align:right;font-weight:bold'>"
        f"{mk['trend'][m]:.2f}%</td></tr>"
        for m in MONTH_COLS if not pd.isna(mk["trend"][m])
    )
    popup_html = f"""
    <div style='font-family:sans-serif;min-width:200px'>
      <div style='background:{color};color:white;padding:8px 12px;border-radius:6px 6px 0 0;
                  font-size:15px;font-weight:bold'>{label} &nbsp;[{grade}]</div>
      <div style='padding:8px 12px;background:#1e1e2e;color:white;border-radius:0 0 6px 6px'>
        <span style='font-size:22px;font-weight:bold;color:{color}'>{val:.2f}%</span>
        <span style='font-size:11px;color:#aaa'> 전세가율 ({LATEST})</span>
        <hr style='border-color:#333;margin:8px 0'>
        <table style='width:100%;font-size:12px'>{trend_rows}</table>
      </div>
    </div>"""

    folium.CircleMarker(
        location=(mk["lat"], mk["lon"]),
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.80,
        weight=1.5,
        tooltip=f"{label}  {val:.1f}% [{grade}]",
        popup=folium.Popup(popup_html, max_width=260),
    ).add_to(city_layer)

city_layer.add_to(m)

# ── 고위험 지역 마커 (80% 이상) ───────────────────────────────────────────────
danger_layer = folium.FeatureGroup(name="⚠ 위험 지역 (80% 이상)", show=True)

for mk in sorted(markers, key=lambda x: -x["ratio"]):
    if mk["ratio"] < 80:
        continue
    folium.Marker(
        location=(mk["lat"], mk["lon"]),
        icon=folium.Icon(color="red", icon="exclamation-sign", prefix="glyphicon"),
        tooltip=f"⚠ {mk['label']}  {mk['ratio']:.1f}% [위험]",
    ).add_to(danger_layer)

danger_layer.add_to(m)

# ── 범례 HTML ─────────────────────────────────────────────────────────────────
legend_html = """
<div style="
    position: fixed; bottom: 30px; left: 30px; z-index: 1000;
    background: rgba(20,20,40,0.92); color: white;
    padding: 16px 20px; border-radius: 10px;
    border: 1px solid #555; font-family: sans-serif; font-size: 13px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.5)">
  <b style="font-size:14px">전세사기 위험도</b>
  <div style="font-size:10px;color:#aaa;margin-bottom:8px">전세가율 기준 ({latest})</div>
  <div style="display:flex;align-items:center;margin:5px 0">
    <span style="background:#d32f2f;width:18px;height:18px;border-radius:50%;display:inline-block;margin-right:8px"></span>
    위험 &nbsp;<span style="color:#d32f2f;font-weight:bold">80% 이상</span>
  </div>
  <div style="display:flex;align-items:center;margin:5px 0">
    <span style="background:#f57c00;width:18px;height:18px;border-radius:50%;display:inline-block;margin-right:8px"></span>
    경계 &nbsp;<span style="color:#f57c00;font-weight:bold">70~80%</span>
  </div>
  <div style="display:flex;align-items:center;margin:5px 0">
    <span style="background:#fbc02d;width:18px;height:18px;border-radius:50%;display:inline-block;margin-right:8px"></span>
    주의 &nbsp;<span style="color:#fbc02d;font-weight:bold">60~70%</span>
  </div>
  <div style="display:flex;align-items:center;margin:5px 0">
    <span style="background:#388e3c;width:18px;height:18px;border-radius:50%;display:inline-block;margin-right:8px"></span>
    안전 &nbsp;<span style="color:#388e3c;font-weight:bold">60% 미만</span>
  </div>
  <hr style="border-color:#444;margin:10px 0">
  <div style="font-size:11px;color:#bbb">
    ○ 큰 원 = 광역시도 평균<br>
    ● 작은 원 = 시군구 상세<br>
    ⚠ 아이콘 = 위험 지역
  </div>
</div>
""".replace("{latest}", LATEST)

m.get_root().html.add_child(folium.Element(legend_html))

# ── 제목 ──────────────────────────────────────────────────────────────────────
title_html = f"""
<div style="
    position: fixed; top: 15px; left: 50%; transform: translateX(-50%);
    z-index: 1000; background: rgba(20,20,40,0.92); color: white;
    padding: 10px 24px; border-radius: 8px; border: 1px solid #555;
    font-family: sans-serif; font-size: 16px; font-weight: bold;
    box-shadow: 0 2px 10px rgba(0,0,0,0.5); text-align: center">
  🏠 전세사기 위험 지역 지도 &nbsp;|&nbsp;
  <span style="color:#fbc02d">아파트 전세가율</span> &nbsp;
  <span style="font-size:12px;font-weight:normal;color:#aaa">({LATEST} 기준)</span>
</div>"""
m.get_root().html.add_child(folium.Element(title_html))

# ── 레이어 컨트롤 ─────────────────────────────────────────────────────────────
folium.LayerControl(collapsed=False, position="topright").add_to(m)

# ── 저장 ──────────────────────────────────────────────────────────────────────
OUT = "jeonse_risk_map.html"
m.save(OUT)
print(f"✔  {OUT} 저장 완료")

# ── 요약 ──────────────────────────────────────────────────────────────────────
grade_counts = {}
for mk in markers:
    grade_counts[mk["grade"]] = grade_counts.get(mk["grade"], 0) + 1

print("\n지도 마커 통계:")
for g in ["위험", "경계", "주의", "안전"]:
    emoji = {"위험":"🔴","경계":"🟠","주의":"🟡","안전":"🟢"}[g]
    print(f"  {emoji} {g}: {grade_counts.get(g, 0)}개 지역")
