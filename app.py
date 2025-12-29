from __future__ import annotations

import hmac
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from io import BytesIO
from typing import Optional, List, Tuple

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from streamlit_local_storage import LocalStorage


# =========================
# 설정(색상/레이아웃)
# =========================
POS_COLOR = "#F4B000"
NEG_COLOR = "#E74C3C"
SELL_BG = "#F6C7C7"
HOLD_BG = "#EFEFEF"
BUY_BG  = "#F6DFC0"
TITLE_SUFFIX = "우하철도 액티브 포트 실시간 대응"

LS_KEY = "dashboard_state_v1"  # localStorage key


# =========================
# 데이터 모델
# =========================
@dataclass
class Row:
    name: str
    pct: float
    rule: str


@dataclass
class State:
    actions: List[str]  # length 4
    rows: List[Row]     # 2~4 (3,4 optional)


# =========================
# 유틸
# =========================
def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def parse_pct(s: str) -> Optional[float]:
    if s is None:
        return None
    s = s.strip().replace("%", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def pct_to_ratio(pct: float) -> float:
    return max(0.0, min(1.0, abs(pct) / 100.0))


def action_from_sign(pct: float) -> str:
    if pct > 0:
        return "매수"
    if pct < 0:
        return "매도"
    return "유지"


def action_bg_from_sign(pct: float) -> str:
    if pct > 0:
        return BUY_BG
    if pct < 0:
        return SELL_BG
    return HOLD_BG


def bar_color_from_sign(pct: float) -> str:
    return NEG_COLOR if pct < 0 else POS_COLOR


def format_pct_text(pct: float) -> str:
    # 숫자만 입력받는 조건이므로 표시는 +/-
    return f"- {abs(pct):g}%" if pct < 0 else f"+ {pct:g}%"


# =========================
# 로그인(단일 비밀번호)
# =========================
def require_login():
    if "authed" not in st.session_state:
        st.session_state.authed = False

    if st.session_state.authed:
        return

    st.title("대시보드 로그인")
    pw = st.text_input("비밀번호", type="password")
    if st.button("로그인"):
        if "APP_PASSWORD" not in st.secrets:
            st.error("서버에 APP_PASSWORD가 설정되지 않았습니다.")
            st.stop()
        if hmac.compare_digest(pw, st.secrets["APP_PASSWORD"]):
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("비밀번호가 올바르지 않습니다.")
    st.stop()


# =========================
# localStorage 로드/저장
# =========================
def state_to_json(state: State) -> str:
    payload = {
        "actions": state.actions,
        "rows": [asdict(r) for r in state.rows],
    }
    return json.dumps(payload, ensure_ascii=False)


def json_to_state(s: str) -> Optional[State]:
    try:
        obj = json.loads(s)
        actions = obj.get("actions", [])
        rows_obj = obj.get("rows", [])
        if not isinstance(actions, list) or not isinstance(rows_obj, list):
            return None
        actions = [(x if isinstance(x, str) else "") for x in actions][:4]
        while len(actions) < 4:
            actions.append("")
        rows: List[Row] = []
        for r in rows_obj:
            if not isinstance(r, dict):
                continue
            name = str(r.get("name", "") or "")
            pct = r.get("pct", 0.0)
            try:
                pct = float(pct)
            except Exception:
                pct = 0.0
            rule = str(r.get("rule", "") or "")
            rows.append(Row(name=name, pct=pct, rule=rule))
        if len(rows) < 2:
            return None
        rows = rows[:4]
        return State(actions=actions, rows=rows)
    except Exception:
        return None


def try_load_from_local_storage(localS: LocalStorage) -> Optional[State]:
    raw = None
    try:
        raw = localS.getItem(LS_KEY)
    except TypeError:
        try:
            localS.getItem(LS_KEY)
            for cand in ["__ls_get__", "getItem", "LocalStorage.getItem", LS_KEY]:
                v = st.session_state.get(cand)
                if isinstance(v, str) and v.strip():
                    raw = v
                    break
        except Exception:
            raw = None
    except Exception:
        raw = None

    if isinstance(raw, str) and raw.strip():
        return json_to_state(raw)
    return None


def save_to_local_storage(localS: LocalStorage, state: State):
    payload = state_to_json(state)
    try:
        localS.setItem(LS_KEY, payload)
    except TypeError:
        localS.setItem(LS_KEY, payload)


# =========================
# PNG 렌더링(PIL)
# =========================
def render_dashboard_png(state: State) -> Image.Image:
    import os

    W = 980
    pad = 24
    row_h = 54
    header_h = 56
    top_title_h = 54

    # --- 폰트 로드 먼저(레이아웃 계산 정확도를 위해) ---
    def load_korean_fonts():
        base = os.path.dirname(os.path.abspath(__file__))
        reg_path = os.path.join(base, "assets", "NanumGothic.ttf")
        bold_path = os.path.join(base, "assets", "NanumGothicBold.ttf")
        font_title = ImageFont.truetype(bold_path, 28)   # 타이틀
        font_h     = ImageFont.truetype(bold_path, 22)   # 헤더/강조
        font_rule  = ImageFont.truetype(bold_path, 18)
        font       = ImageFont.truetype(reg_path, 18)    # 일반 텍스트
        return font_title, font_h, font_rule, font

    def load_icon():
        base = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(base, "assets", "megaphone.png")
        try:
            icon = Image.open(icon_path).convert("RGBA")
            return icon
        except Exception:
            return None

    font_title, font_h, font_rule, font = load_korean_fonts()
    icon_img = load_icon()

    # --- ACTION 박스 높이: 입력 줄 수(0~4) 기반 ---
    action_lines = [a for a in state.actions if a.strip()]
    # 기본은 2줄, 3개면 3줄, 4개면 4줄
    n_lines = len(action_lines)
    n_lines_for_height = 2 if n_lines <= 2 else min(4, n_lines)

    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    spacing = 6
    text_h = n_lines_for_height * line_h + (n_lines_for_height - 1) * spacing
    action_box_h = max(60, text_h + 20)  # 위아래 패딩 포함

    # --- table_top/H 계산은 action_box_h 확정 후 ---
    n_rows = len(state.rows)
    table_top = pad + top_title_h + action_box_h + 24
    H = table_top + header_h + row_h * n_rows + pad

    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # --- 정수 좌표 기반 중앙정렬(미세 어긋남 방지) ---
    def draw_center_text(box, text, font_, fill=(0, 0, 0)):
        x1, y1, x2, y2 = box
        if text is None:
            text = ""
        text = str(text)
    
        # 가로폭은 bbox로 정확히
        bbox = draw.textbbox((0, 0), text, font=font_)
        tw = bbox[2] - bbox[0]
    
        # ✅ 세로는 bbox가 아니라 "폰트 라인 높이"로 통일 (광학적으로 더 안정적)
        ascent, descent = font_.getmetrics()
        line_h = ascent + descent
    
        x = x1 + (x2 - x1 - tw) / 2
        y = y1 + (y2 - y1 - line_h) / 2
    
        draw.text((int(round(x)), int(round(y))), text, fill=fill, font=font_)


    # --- 타이틀 ---
    now = datetime.now()
    title = f"{now.month}/{now.day} {TITLE_SUFFIX}"
    draw.text((pad, pad), title, fill=hex_to_rgb("#2F6DF6"), font=font_title)
    draw.rectangle([pad, pad + 34, W - pad, pad + 38], fill=hex_to_rgb("#2F6DF6"))

    # --- ACTION 박스 ---
    ax1, ay1 = pad, pad + 52
    ax2, ay2 = W - pad, ay1 + action_box_h
    draw.rectangle([ax1, ay1, ax2, ay2], outline=hex_to_rgb("#E66A6A"), width=3)

    # 아이콘 + "ACTION!" 텍스트(좌측)
    icon_x = ax1 + 14
    icon_size = 24
    # 박스 세로 중앙
    icon_y = ay1 + (action_box_h - icon_size) // 2

    text_x = icon_x
    text_y = ay1 + (action_box_h - (font_h.getmetrics()[0] + font_h.getmetrics()[1])) // 2

    if icon_img is not None:
        icon_resized = icon_img.resize((icon_size, icon_size), Image.LANCZOS)
        img.paste(icon_resized, (int(icon_x), int(icon_y)), icon_resized)
        text_x += icon_size + 8

    draw.text(
        (int(text_x), int(text_y)),
        "ACTION!",
        fill=hex_to_rgb("#E53935"),
        font=font_h
    )

    # ACTION 내용: 좌측 정렬 + 전체 블록 세로 중앙
    # 표시 줄 수 = n_lines_for_height (2/3/4)
    show_lines = action_lines[:n_lines_for_height]
    # 줄이 부족하면 빈줄로 채워서(2줄 기본 유지)
    while len(show_lines) < n_lines_for_height:
        show_lines.append("")

    # 텍스트 시작 x(좌측부터)
    tx1 = ax1 + 160

    # 전체 블록 높이 계산해서 박스 중앙에 오게 배치
    nonempty = [ln for ln in show_lines if ln.strip() != ""]
    # 빈줄도 높이는 차지해야 하므로 "줄 수"로 계산
    block_h = n_lines_for_height * line_h + (n_lines_for_height - 1) * spacing
    start_y = ay1 + (action_box_h - block_h) // 2

    for i, ln in enumerate(show_lines):
        y_line = start_y + i * (line_h + spacing)
        draw.text((int(tx1), int(y_line)), ln, fill=(0, 0, 0), font=font)

    # --- 테이블 헤더 ---
    col_w = [220, 180, 140, 200, W - pad * 2 - (220 + 180 + 140 + 200)]
    headers = ["종목", "시각화", "비중변경", "ACTION", "매수/매도 기준"]
    x = pad
    y = table_top
    for i, h in enumerate(headers):
        draw.rectangle([x, y, x + col_w[i], y + header_h], fill=hex_to_rgb("#E0E0E0"))
        draw_center_text((x, y, x + col_w[i], y + header_h), h, font_h, fill=(30, 30, 30))
        x += col_w[i]

    # --- 행 ---
    y += header_h
    for r in state.rows:
        ratio = pct_to_ratio(r.pct)
        bar_color = hex_to_rgb(bar_color_from_sign(r.pct))
        action = action_from_sign(r.pct)
        action_bg = hex_to_rgb(action_bg_from_sign(r.pct))
        pct_text = format_pct_text(r.pct)

        # 기본 배경(각 칸)
        x = pad
        for w in col_w:
            draw.rectangle([x, y, x + w, y + row_h], fill=hex_to_rgb("#F4F4F4"))
            x += w

        # 종목(중앙)
        draw_center_text((pad, y, pad + col_w[0], y + row_h), r.name, font_h, fill=(0, 0, 0))

        # 시각화 바(행 세로 중앙) - ✅ 칼럼 폭에 맞춰 자동 조정(겹침 방지)
        left_pad = 30
        right_pad = 20
        bx1 = pad + col_w[0] + left_pad
        bw = max(60, col_w[1] - (left_pad + right_pad))  # ✅ col_w[1]에 맞춰 폭 자동
        bh = 26
        by1 = y + (row_h - bh) // 2
        draw.rectangle([bx1, by1, bx1 + bw, by1 + bh], fill=hex_to_rgb("#DADADA"))
        draw.rectangle([bx1, by1, bx1 + int(bw * ratio), by1 + bh], fill=bar_color)


        # 비중변경(중앙)
        x1 = pad + col_w[0] + col_w[1]
        draw_center_text((x1, y, x1 + col_w[2], y + row_h), pct_text, font_h, fill=(0, 0, 0))

        # ACTION 셀 배경 + 텍스트 중앙
        ax = pad + col_w[0] + col_w[1] + col_w[2]
        draw.rectangle([ax, y, ax + col_w[3], y + row_h], fill=action_bg)
        draw_center_text((ax, y, ax + col_w[3], y + row_h), action, font_h, fill=(0, 0, 0))

        # 기준(중앙)
        x_rule = pad + col_w[0] + col_w[1] + col_w[2] + col_w[3]
        draw_center_text((x_rule, y, x_rule + col_w[4], y + row_h), (r.rule or ""), font_rule, fill=(0, 0, 0))

        y += row_h

    return img


# =========================
# UI
# =========================
st.set_page_config(page_title="대시보드", layout="centered")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');
html, body, [class*="css"]  {
  font-family: 'Nanum Gothic', sans-serif !important;
}
</style>
""",
    unsafe_allow_html=True,
)

require_login()

localS = LocalStorage()

# 1) 첫 진입 시 localStorage에서 상태 로드(있으면)
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    loaded = try_load_from_local_storage(localS)
    if loaded is not None:
        st.session_state.loaded_state = loaded
    else:
        st.session_state.loaded_state = None

# 기본값(원 코드 동일)
default_actions = ["오늘 대응 요약 1", "오늘 대응 요약 2", "", ""]
default_rows = [
    Row(name="셀트리온제약", pct=-50.0, rule="예: 시장가"),
    Row(name="에이비온", pct=50.0, rule="예: 2900 이내 분할"),
    Row(name="", pct=0.0, rule=""),
    Row(name="", pct=0.0, rule=""),
]

init_state = st.session_state.loaded_state if st.session_state.loaded_state is not None else State(
    actions=default_actions,
    rows=default_rows,
)

# 세션 상태에 폼 키 세팅(한 번만)
def ensure_session_keys():
    if "actions" not in st.session_state:
        st.session_state.actions = init_state.actions[:]
    if "rows" not in st.session_state:
        rows4 = init_state.rows[:]
        while len(rows4) < 4:
            rows4.append(Row(name="", pct=0.0, rule=""))
        rows4 = rows4[:4]
        st.session_state.rows = [asdict(r) for r in rows4]

ensure_session_keys()

# 저장 콜백: 입력이 바뀔 때마다 localStorage 갱신
def persist():
    rows = [Row(**r) for r in st.session_state.rows]
    final_rows: List[Row] = []
    final_rows.append(rows[0] if rows[0].name.strip() else Row("종목1", rows[0].pct, rows[0].rule))
    final_rows.append(rows[1] if rows[1].name.strip() else Row("종목2", rows[1].pct, rows[1].rule))

    for r in rows[2:]:
        has_any = (r.name.strip() != "") or (abs(r.pct) > 0.0) or (r.rule.strip() != "")
        if has_any:
            final_rows.append(Row(r.name.strip() or "종목", r.pct, r.rule))

    state = State(actions=st.session_state.actions[:4], rows=final_rows)
    save_to_local_storage(localS, state)

# ===== 입력 UI =====
st.title("대시보드")

st.subheader("Action(요약)")

# Action 1~2 (항상 표시)
for i in range(2):
    st.session_state.actions[i] = st.text_input(
        f"Action {i+1}",
        value=st.session_state.actions[i],
        key=f"action_{i}",
        on_change=persist,
    )

# Action 3~4 (접기)
with st.expander("Action 3~4 (펼치기)", expanded=False):
    for i in range(2, 4):
        st.session_state.actions[i] = st.text_input(
            f"Action {i+1}",
            value=st.session_state.actions[i],
            key=f"action_{i}",
            on_change=persist,
        )

st.subheader("종목 입력 (비중은 숫자만)")

def row_editor(i: int, optional: bool):
    c1, c2, c3 = st.columns([2, 1, 3])
    r = st.session_state.rows[i]

    name_label = f"종목{i+1}" + ("(선택)" if optional else "")
    pct_label  = f"비중변경{i+1}" + ("(선택)" if optional else "")
    rule_label = f"매수/매도 기준{i+1}" + ("(선택)" if optional else "")

    name = c1.text_input(name_label, value=r["name"], key=f"name_{i}", on_change=persist)
    pct_num = c2.number_input(
        pct_label,
        value=float(r["pct"]),
        step=float(1.0),
        format="%.0f",
        key=f"pct_{i}",
        on_change=persist,
    )
    rule = c3.text_input(rule_label, value=r["rule"], key=f"rule_{i}", on_change=persist)

    st.session_state.rows[i] = {"name": name, "pct": float(pct_num), "rule": rule}

row_editor(0, optional=False)
row_editor(1, optional=False)

with st.expander("종목 3~4 (선택) 입력 펼치기", expanded=False):
    row_editor(2, optional=True)
    row_editor(3, optional=True)

# 현재 상태 구성
rows_all = [Row(**r) for r in st.session_state.rows]
final_rows: List[Row] = []
final_rows.append(rows_all[0] if rows_all[0].name.strip() else Row("종목1", rows_all[0].pct, rows_all[0].rule))
final_rows.append(rows_all[1] if rows_all[1].name.strip() else Row("종목2", rows_all[1].pct, rows_all[1].rule))

for r in rows_all[2:]:
    has_any = (r.name.strip() != "") or (abs(r.pct) > 0.0) or (r.rule.strip() != "")
    if has_any:
        final_rows.append(Row(r.name.strip() or "종목", r.pct, r.rule))

current = State(actions=st.session_state.actions[:4], rows=final_rows)

# ===== 미리보기 =====
st.subheader("미리보기 (PNG 기반)")
img = render_dashboard_png(current)
st.image(img)

buf = BytesIO()
img.save(buf, format="PNG")
st.download_button(
    "PNG 다운로드",
    data=buf.getvalue(),
    file_name="dashboard.png",
    mime="image/png",
)

# ===== 복사용 텍스트 =====
st.subheader("복사용 텍스트")
lines = []
lines.append(f"{datetime.now().month}/{datetime.now().day} {TITLE_SUFFIX}")
lines.append("")
actions_txt = [a for a in current.actions if a.strip()]
if actions_txt:
    lines.append("[ACTION]")
    lines.extend([f"- {a}" for a in actions_txt])
    lines.append("")

lines.append("[종목]")
for r in current.rows:
    lines.append(f"- {r.name} | {format_pct_text(r.pct)} | {action_from_sign(r.pct)} | {r.rule}")

copy_text = "\n".join(lines)

st.code(copy_text, language="text")
st.text_area("길게 눌러 복사(모바일용)", value=copy_text, height=180)
