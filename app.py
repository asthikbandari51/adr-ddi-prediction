import sys
import os
import math
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_SRC_DIR = _APP_DIR / "src"
for _p in [str(_SRC_DIR), str(_APP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

st.set_page_config(
    page_title="ADR & DDI System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from predict import (
        predict_adr,
        predict_ddi,
        analyze_prescription,
        normalize_drug_name,
        extract_drugs_from_text,
        DRUG_SYNONYMS,
    )
except ModuleNotFoundError as exc:
    st.error(
        "**Cannot import predict.py** — make sure it exists in the project root or src/ folder.\n\n"
        f"Error: `{exc}`"
    )
    st.stop()

# ---------------------------------------------------------------------------
# SESSION STATE DEFAULTS
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "theme":       "dark",
    "rx_text":     "",
    "single_drug": "",
    "ddi_d1":      "",
    "ddi_d2":      "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# COLOUR TOKENS  (dark / light)
# ---------------------------------------------------------------------------
IS_DARK = st.session_state.theme == "dark"

if IS_DARK:
    BG          = "#0d1117"
    SURFACE     = "#161b22"
    SURFACE2    = "#21262d"
    BORDER      = "#30363d"
    TEXT        = "#e6edf3"
    TEXT_MUTED  = "#8b949e"
    TEXT_FAINT  = "#6e7681"
    ACCENT      = "#58a6ff"
    GREEN       = "#3fb950"
    AMBER       = "#d2993e"
    RED         = "#f85149"
    CARD_BG     = "#161b22"
    TOGGLE_ICO  = "☀️"
    TOGGLE_LBL  = "Switch to Light Mode"
    HOVER_ROW   = "rgba(255,255,255,0.03)"
    DDI_STRIPE  = "rgba(248,81,73,0.05)"
    DDI_BORDER  = "rgba(248,81,73,0.12)"
    DDI_HOVER   = "rgba(248,81,73,0.09)"
    HERO_GR     = "linear-gradient(135deg,#0d1117 0%,#161b22 50%,#0d1117 100%)"
    BADGE_SRC   = "#0d1117"
    SAFE_BG     = "rgba(63,185,80,0.06)"
    SAFE_BDR    = "rgba(63,185,80,0.2)"
else:
    BG          = "#ffffff"
    SURFACE     = "#f6f8fa"
    SURFACE2    = "#eaeef2"
    BORDER      = "#d0d7de"
    TEXT        = "#1f2328"
    TEXT_MUTED  = "#57606a"
    TEXT_FAINT  = "#8c959f"
    ACCENT      = "#0969da"
    GREEN       = "#1a7f37"
    AMBER       = "#9a6700"
    RED         = "#cf222e"
    CARD_BG     = "#ffffff"
    TOGGLE_ICO  = "🌙"
    TOGGLE_LBL  = "Switch to Dark Mode"
    HOVER_ROW   = "rgba(0,0,0,0.03)"
    DDI_STRIPE  = "rgba(207,34,46,0.04)"
    DDI_BORDER  = "rgba(207,34,46,0.15)"
    DDI_HOVER   = "rgba(207,34,46,0.08)"
    HERO_GR     = "linear-gradient(135deg,#f6f8fa 0%,#eaeef2 50%,#f6f8fa 100%)"
    BADGE_SRC   = "#eaeef2"
    SAFE_BG     = "rgba(26,127,55,0.06)"
    SAFE_BDR    = "rgba(26,127,55,0.2)"

# ---------------------------------------------------------------------------
# INJECT CSS  (pure ASCII inside the string)
# ---------------------------------------------------------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] .main,
[data-testid="stMain"] {{
    background: {BG} !important;
    font-family: 'Inter', sans-serif;
    color: {TEXT};
}}

#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stDeployButton"],
[data-testid="stToolbar"] {{ display: none; }}

::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {SURFACE}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {ACCENT}; }}

[data-testid="stSidebar"] {{
    background: {BG} !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
[data-testid="stSidebar"] hr {{ border-color: {BORDER} !important; }}
[data-testid="stSidebar"] .stExpander {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
}}

[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT} !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
}}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px {ACCENT}1a !important;
    outline: none !important;
}}
[data-testid="stTextArea"] label,
[data-testid="stTextInput"] label {{ display: none !important; }}

[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label {{
    color: {TEXT_MUTED} !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}}
[data-testid="stSelectbox"] > div > div {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT} !important;
}}
[data-testid="stSelectbox"] > div > div:hover {{ border-color: {ACCENT} !important; }}

[data-testid="stButton"] button {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT} !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    transition: all 0.16s ease !important;
}}
[data-testid="stButton"] button:hover {{
    background: {SURFACE2} !important;
    border-color: {ACCENT} !important;
    transform: translateY(-1px) !important;
}}
[data-testid="stButton"] button[kind="primary"] {{
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    box-shadow: 0 2px 8px rgba(46,160,67,0.3) !important;
}}
[data-testid="stButton"] button[kind="primary"]:hover {{
    background: linear-gradient(135deg, #2ea043, #3fb950) !important;
    transform: translateY(-1px) scale(1.01) !important;
    box-shadow: 0 4px 16px rgba(46,160,67,0.45) !important;
}}

[data-testid="stAlert"] {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
}}
[data-testid="column"] {{ padding: 0 0.4rem !important; }}

@keyframes fadeUp  {{ from {{ opacity:0; transform:translateY(14px); }} to {{ opacity:1; transform:translateY(0); }} }}
@keyframes rowIn   {{ from {{ opacity:0; transform:translateX(-8px); }} to {{ opacity:1; transform:translateX(0); }} }}
@keyframes chipIn  {{ from {{ opacity:0; transform:scale(0.82); }} to {{ opacity:1; transform:scale(1); }} }}
@keyframes cardIn  {{ from {{ opacity:0; transform:translateY(12px); }} to {{ opacity:1; transform:translateY(0); }} }}
@keyframes floatY  {{ 0%,100% {{ transform:translateY(0); }} 50% {{ transform:translateY(-7px); }} }}

.hero {{
    background: {HERO_GR};
    border-bottom: 1px solid {BORDER};
    padding: 2.2rem 1rem 1.8rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}}
.hero::before {{
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 40% at 20% 50%, {ACCENT}0f 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 80% 50%, {GREEN}0d 0%, transparent 70%);
    pointer-events: none;
}}
.hero-icon {{
    font-size: 2.8rem;
    display: block;
    filter: drop-shadow(0 0 18px {ACCENT}66);
    animation: floatY 3s ease-in-out infinite;
    margin-bottom: 0.4rem;
}}
.hero-title {{
    font-size: clamp(1.5rem, 3vw, 2.2rem);
    font-weight: 700;
    background: linear-gradient(135deg, {ACCENT} 0%, {GREEN} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    line-height: 1.2;
}}
.hero-sub {{
    color: {TEXT_MUTED};
    font-size: 0.875rem;
    margin-top: 0.35rem;
}}
.hero-badges {{
    display: flex;
    justify-content: center;
    gap: 0.4rem;
    flex-wrap: wrap;
    margin-top: 0.8rem;
}}
.hero-badge {{
    background: {SURFACE2};
    border: 1px solid {BORDER};
    color: {TEXT_MUTED};
    font-size: 0.7rem;
    font-weight: 500;
    padding: 0.18rem 0.6rem;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
}}

.sec-hdr {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 1.75rem 0 0.9rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid {BORDER};
}}
.sec-hdr-title {{ font-size: 1rem; font-weight: 600; color: {TEXT}; flex: 1; }}
.sec-count {{
    background: {SURFACE2};
    color: {TEXT_MUTED};
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.1rem 0.5rem;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
}}

.chips-row {{ display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.6rem 0; align-items: center; }}
.chip-lbl {{ font-size: 0.7rem; color: {TEXT_MUTED}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }}
.chip {{
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    background: {ACCENT}1a;
    border: 1px solid {ACCENT}4d;
    color: {ACCENT};
    padding: 0.18rem 0.6rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 500;
    animation: chipIn 0.25s ease both;
}}

.metrics-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.6rem;
    margin: 1rem 0;
}}
.metric-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
    animation: cardIn 0.3s ease both;
}}
.metric-val  {{ font-size: 1.6rem; font-weight: 700; line-height: 1; margin-bottom: 0.2rem; }}
.metric-lbl  {{ font-size: 0.68rem; color: {TEXT_MUTED}; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; }}

.drug-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.8rem;
    animation: cardIn 0.35s ease both;
}}
.drug-card-hdr {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-bottom: 0.75rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid {BORDER};
}}
.drug-name {{ font-size: 1rem; font-weight: 700; color: {TEXT}; }}
.badge-row  {{ display: flex; gap: 0.3rem; align-items: center; flex-wrap: wrap; }}

.badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.2rem;
    padding: 0.16rem 0.55rem;
    border-radius: 999px;
    font-size: 0.67rem;
    font-weight: 600;
    white-space: nowrap;
    letter-spacing: 0.02em;
}}
.badge-high   {{ background: #dcfce7; color: #166534; }}
.badge-medium {{ background: #fef9c3; color: #854d0e; }}
.badge-low    {{ background: #fee2e2; color: #991b1b; }}
.badge-blue   {{ background: {ACCENT}1a; color: {ACCENT}; border: 1px solid {ACCENT}33; }}
.badge-mono   {{ background: {BADGE_SRC}; color: {TEXT_FAINT}; border: 1px solid {BORDER}; font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; }}

.adr-list {{ display: flex; flex-direction: column; gap: 0.25rem; }}
.adr-row {{
    display: grid;
    grid-template-columns: 10px 1fr 80px 70px;
    align-items: center;
    gap: 0.55rem;
    padding: 0.45rem 0.5rem;
    border-radius: 7px;
    transition: background 0.12s;
    animation: rowIn 0.28s ease both;
}}
.adr-row:hover {{ background: {HOVER_ROW}; }}
.adr-dot  {{ width: 8px; height: 8px; border-radius: 50%; justify-self: center; flex-shrink: 0; }}
.d-vc  {{ background: {RED};       box-shadow: 0 0 6px {RED}88; }}
.d-c   {{ background: {AMBER};     box-shadow: 0 0 6px {AMBER}66; }}
.d-uc  {{ background: {ACCENT};    box-shadow: 0 0 6px {ACCENT}66; }}
.d-r   {{ background: {TEXT_MUTED}; }}
.d-uk  {{ background: {BORDER}; }}
.adr-name {{ font-size: 0.875rem; color: {TEXT}; font-weight: 450; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.freq-wrap {{ position: relative; height: 5px; background: {SURFACE2}; border-radius: 999px; overflow: hidden; }}
.freq-fill {{ position: absolute; top: 0; left: 0; height: 100%; border-radius: 999px; transition: width 0.6s ease; }}
.b-vc  {{ background: linear-gradient(90deg, {RED},   {RED}cc); }}
.b-c   {{ background: linear-gradient(90deg, {AMBER}, {AMBER}cc); }}
.b-uc  {{ background: linear-gradient(90deg, {ACCENT},{ACCENT}cc); }}
.b-r   {{ background: {TEXT_MUTED}; }}
.b-uk  {{ background: {BORDER}; }}
.adr-pct {{ font-size: 0.75rem; font-weight: 600; color: {TEXT_MUTED}; text-align: right; font-family: 'JetBrains Mono', monospace; white-space: nowrap; }}

.no-results {{ text-align: center; padding: 1.6rem 1rem; color: {TEXT_MUTED}; font-size: 0.875rem; }}
.no-results-icon {{ font-size: 2rem; display: block; margin-bottom: 0.4rem; }}

.ddi-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-left: 3px solid {AMBER};
    border-radius: 12px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.8rem;
    animation: cardIn 0.35s ease both;
}}
.ddi-pair-hdr {{
    display: flex;
    align-items: center;
    gap: 0.45rem;
    margin-bottom: 0.7rem;
    padding-bottom: 0.55rem;
    border-bottom: 1px solid {BORDER};
    flex-wrap: wrap;
}}
.ddi-drug  {{ font-size: 0.9rem; font-weight: 600; color: {TEXT}; }}
.ddi-plus  {{ color: {TEXT_MUTED}; font-size: 0.85rem; }}
.ddi-rows  {{ display: flex; flex-direction: column; gap: 0.3rem; }}
.ddi-row {{
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.45rem 0.7rem;
    background: {DDI_STRIPE};
    border: 1px solid {DDI_BORDER};
    border-radius: 7px;
    animation: rowIn 0.28s ease both;
    transition: background 0.12s;
}}
.ddi-row:hover {{ background: {DDI_HOVER}; }}
.ddi-name  {{ font-size: 0.875rem; color: {RED}; font-weight: 500; flex: 1; }}
.ddi-safe {{
    display: flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.5rem 0.7rem;
    background: {SAFE_BG};
    border: 1px solid {SAFE_BDR};
    border-radius: 7px;
    font-size: 0.85rem;
    color: {GREEN};
}}

.about-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 0.65rem; margin-top: 0.9rem; }}
.about-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 0.9rem 1rem;
    transition: transform 0.16s ease, border-color 0.16s;
}}
.about-card:hover {{ transform: translateY(-2px); border-color: {SURFACE2}; }}
.about-card-title {{ font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: {TEXT_MUTED}; margin-bottom: 0.3rem; }}
.about-card-body  {{ font-size: 0.84rem; color: {TEXT}; line-height: 1.55; }}
.pill {{
    display: inline-block;
    background: {SURFACE2};
    color: {TEXT_MUTED};
    font-size: 0.7rem;
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    margin: 0.1rem;
    font-family: 'JetBrains Mono', monospace;
}}

.input-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1.2rem 1.3rem;
    margin: 1rem 0;
    animation: cardIn 0.35s ease both;
}}
.input-card:focus-within {{ border-color: {ACCENT}; box-shadow: 0 0 0 3px {ACCENT}1a; }}
.input-lbl {{
    font-size: 0.7rem;
    font-weight: 600;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}}
.qt-lbl {{
    font-size: 0.7rem;
    font-weight: 600;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 0.6rem 0 0.3rem;
}}
.divider {{ height: 1px; background: {BORDER}; margin: 1.1rem 0; }}
.footer {{
    margin-top: 2rem;
    padding: 1rem 0;
    text-align: center;
    font-size: 0.78rem;
    color: {TEXT_FAINT};
    border-top: 1px solid {BORDER};
}}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def fmt_freq(freq):
    if freq <= 0:
        return "—"
    p = freq * 100
    if p < 0.001:  return "<0.001%"
    if p < 0.01:   return f"{p:.4f}%"
    if p < 0.1:    return f"{p:.3f}%"
    if p < 1.0:    return f"{p:.2f}%"
    if p < 10:     return f"{p:.1f}%"
    return f"{p:.0f}%"


def bar_pct(freq):
    if freq <= 0:
        return 0
    raw = (math.log10(max(freq, 1e-6)) + 6) / 6 * 100
    return min(max(int(raw), 3), 100)


def _cat_codes(cat):
    c = cat.lower()
    if "very common" in c: return "d-vc", "b-vc"
    if "common"      in c: return "d-c",  "b-c"
    if "uncommon"    in c: return "d-uc", "b-uc"
    if "rare"        in c: return "d-r",  "b-r"
    return "d-uk", "b-uk"


def conf_badge(conf):
    m = {
        "high":   ("badge-high",   "High Confidence"),
        "medium": ("badge-medium", "Model Prediction"),
    }
    cls, lbl = m.get(conf, ("badge-low", "Not Found"))
    ico = {"high": "🟢", "medium": "🟡"}.get(conf, "⚪")
    return f'<span class="badge {cls}">{ico} {lbl}</span>'


def src_badge(method):
    labels = {
        "database_lookup":  ("badge-blue", "SIDER DB"),
        "model_prediction": ("badge-medium", "ML Model"),
        "not_found":        ("badge-low", "Not Found"),
    }
    cls, lbl = labels.get(method, ("badge-mono", method.replace("_", " ").title()))
    return f'<span class="badge {cls}">{lbl}</span>'


def render_drug_card(drug_name, result):
    conf   = result.get("confidence", "none")
    method = result.get("method", "")
    se_all = result.get("predicted_side_effects", [])

    hdr = (
        f'<div class="drug-card-hdr">'
        f'<span class="drug-name">💊 {drug_name.title()}</span>'
        f'<span class="badge-row">{conf_badge(conf)} {src_badge(method)}</span>'
        f'</div>'
    )

    if conf == "none" or method == "not_found":
        msg = result.get("message", "Try the generic name (e.g. acetaminophen, ibuprofen).")
        body = (
            f'<div class="no-results">'
            f'<span class="no-results-icon">🔍</span>'
            f'<strong>Drug not found in database</strong><br>'
            f'<small>{msg}</small>'
            f'</div>'
        )
    elif not se_all:
        body = (
            f'<div class="no-results">'
            f'<span class="no-results-icon">📊</span>'
            f'No results at this threshold — lower the frequency filter.'
            f'</div>'
        )
    else:
        rows = []
        for i, s in enumerate(se_all):
            name  = s.get("name", "").title()
            freq  = s.get("freq", 0.0)
            cat   = s.get("category", "Unknown")
            dc, bc = _cat_codes(cat)
            wp    = bar_pct(freq)
            pct_s = fmt_freq(freq)
            rows.append(
                f'<div class="adr-row" style="animation-delay:{i*30}ms">'
                f'<span class="adr-dot {dc}"></span>'
                f'<span class="adr-name">{name}</span>'
                f'<div class="freq-wrap"><div class="freq-fill {bc}" style="width:{wp}%"></div></div>'
                f'<span class="adr-pct">{pct_s}</span>'
                f'</div>'
            )
        body = f'<div class="adr-list">{"".join(rows)}</div>'

    st.markdown(f'<div class="drug-card">{hdr}{body}</div>', unsafe_allow_html=True)


def render_ddi_card(ddi):
    # FIX: predict.py returns drug_1 / drug_2
    d1   = ddi.get("drug_1", ddi.get("drug1", "")).title()
    d2   = ddi.get("drug_2", ddi.get("drug2", "")).title()
    conf = ddi.get("confidence", "none")
    ixs  = ddi.get("interactions", [])

    hdr = (
        f'<div class="ddi-pair-hdr">'
        f'<span class="ddi-drug">💊 {d1}</span>'
        f'<span class="ddi-plus">+</span>'
        f'<span class="ddi-drug">💊 {d2}</span>'
        f'<span style="margin-left:auto">{conf_badge(conf)}</span>'
        f'</div>'
    )

    if not ixs:
        inner = '<div class="ddi-safe">✅ No significant interactions found between these drugs.</div>'
    else:
        items = "".join(
            f'<div class="ddi-row" style="animation-delay:{i*25}ms">'
            f'<span style="font-size:0.9rem">⚠️</span>'
            f'<span class="ddi-name">{ix.title()}</span>'
            f'<span class="badge badge-mono">TWOSIDES</span>'
            f'</div>'
            for i, ix in enumerate(ixs)
        )
        inner = f'<div class="ddi-rows">{items}</div>'

    st.markdown(f'<div class="ddi-card">{hdr}{inner}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 💊 ADR & DDI System")
    st.markdown("---")

    # Theme toggle
    c1, c2 = st.columns([3, 1])
    with c1:
        theme_label = "🌙 Dark Mode" if IS_DARK else "☀️ Light Mode"
        st.markdown(
            f'<div style="font-size:0.82rem;color:{TEXT_MUTED};padding-top:0.3rem">{theme_label}</div>',
            unsafe_allow_html=True,
        )
    with c2:
        if st.button(TOGGLE_ICO, key="theme_toggle", help=TOGGLE_LBL):
            st.session_state.theme = "light" if IS_DARK else "dark"
            st.rerun()

    st.markdown("---")

    mode = st.radio(
        "Navigation",
        [
            "📋 Prescription Analysis",
            "🔬 Single Drug ADR",
            "⚠️ Drug Pair DDI",
            "ℹ️ About",
        ],
        key="nav_mode",
    )

    st.markdown("---")
    st.markdown(
        f'<div style="font-size:0.7rem;font-weight:600;color:{TEXT_MUTED};text-transform:uppercase;'
        f'letter-spacing:0.06em;margin-bottom:0.4rem">Frequency Legend</div>'
        f'<div style="font-size:0.82rem;line-height:2.1">'
        f'<span style="color:{RED}">●</span> Very Common (&gt;10%)<br>'
        f'<span style="color:{AMBER}">●</span> Common (1–10%)<br>'
        f'<span style="color:{ACCENT}">●</span> Uncommon (0.1–1%)<br>'
        f'<span style="color:{TEXT_MUTED}">●</span> Rare (&lt;0.1%)'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    with st.expander("🔄 Synonym Map", expanded=False):
        for k, v in sorted(DRUG_SYNONYMS.items()):
            st.markdown(
                f'<span style="color:{ACCENT};font-size:0.8rem">{k.title()}</span>'
                f'<span style="color:{TEXT_FAINT};font-size:0.75rem"> → {v.title()}</span>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        f'<div style="font-size:0.72rem;color:{TEXT_FAINT};line-height:1.7">'
        f'🟢 High = Database match<br>🟡 Medium = ML Model<br>⚪ Low = Not found</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="hero">'
    '<span class="hero-icon">💊</span>'
    '<div class="hero-title">ADR &amp; DDI Prediction System</div>'
    '<div class="hero-sub">Clinical pharmacovigilance &nbsp;·&nbsp; SIDER &nbsp;·&nbsp; OFFSIDES &nbsp;·&nbsp; TWOSIDES &nbsp;·&nbsp; DrugBank</div>'
    '<div class="hero-badges">'
    '<span class="hero-badge">SIDER 4.1</span>'
    '<span class="hero-badge">OFFSIDES PRR</span>'
    '<span class="hero-badge">TWOSIDES 16M pairs</span>'
    '<span class="hero-badge">DrugBank 19K drugs</span>'
    '<span class="hero-badge">BC5CDR NLP</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# MODE 1  --  PRESCRIPTION ANALYSIS
# ---------------------------------------------------------------------------
if mode == "📋 Prescription Analysis":

    st.markdown(
        '<div class="sec-hdr"><span>📋</span>'
        '<span class="sec-hdr-title">Prescription / Clinical Text Analysis</span></div>',
        unsafe_allow_html=True,
    )

    # Quick example buttons  -- FIX: set widget key directly, then rerun
    st.markdown(f'<div class="qt-lbl">Quick-load example</div>', unsafe_allow_html=True)
    eq1, eq2, eq3 = st.columns(3)
    if eq1.button("🫀 Cardiac Patient", use_container_width=True, key="ex_cardiac"):
        st.session_state["rx_input_widget"] = (
            "Patient is prescribed Warfarin 5mg, Aspirin 75mg, Clopidogrel 75mg "
            "and Omeprazole 20mg following coronary artery bypass surgery."
        )
        st.rerun()
    if eq2.button("🩸 Diabetic Patient", use_container_width=True, key="ex_diabetic"):
        st.session_state["rx_input_widget"] = (
            "Patient takes Metformin 1000mg, Atorvastatin 20mg, Lisinopril 10mg "
            "and Aspirin 75mg for Type 2 diabetes with hypertension."
        )
        st.rerun()
    if eq3.button("💊 Fever & Pain", use_container_width=True, key="ex_fever"):
        st.session_state["rx_input_widget"] = (
            "Patient is prescribed Paracetamol 500mg and Ibuprofen 400mg "
            "for fever and body pain."
        )
        st.rerun()

    st.markdown('<div class="input-card"><div class="input-lbl">Clinical Prescription / Note</div>', unsafe_allow_html=True)
    prescription = st.text_area(
        label="prescription",
        placeholder="e.g. Patient is prescribed Metformin 500mg, Atorvastatin 20mg and Amlodipine 5mg...",
        height=95,
        key="rx_input_widget",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    cc1, cc2 = st.columns([2, 1])
    with cc1:
        freq_filter = st.selectbox(
            "Frequency Filter",
            ["All (including rare)", "Common & above (>=1%)", "Very common only (>=10%)"],
            index=1,
            key="rx_freq_filter",
        )
    with cc2:
        topn_rx = st.number_input("Top N per drug", min_value=5, max_value=25, value=10, key="rx_topn")

    analyse = st.button("🔍 Analyse Prescription", use_container_width=True, type="primary", key="btn_analyse")

    min_freq_rx = 0.10 if ">=10%" in freq_filter else (0.01 if ">=1%" in freq_filter else 0.0)

    if analyse:
        if not prescription.strip():
            st.warning("Please enter a prescription or clinical note.")
        else:
            with st.spinner("Extracting drugs and running predictions…"):
                # FIX: correct function + correct param names top_n / min_freq
                result = analyze_prescription(
                    prescription,
                    top_n=int(topn_rx),
                    min_freq=min_freq_rx,
                )

            # FIX: correct dict key = drugs_found
            drugs = result.get("drugs_found", [])

            if not drugs:
                st.error(
                    "No drug names detected. Use generic names: "
                    "Metformin, Warfarin, Aspirin, Ibuprofen, Paracetamol…"
                )
            else:
                chips = '<div class="chips-row"><span class="chip-lbl">Detected:</span>'
                for i, d in enumerate(drugs):
                    norm  = normalize_drug_name(d)
                    alias = f" → {norm.title()}" if norm != d.lower() else ""
                    chips += f'<span class="chip" style="animation-delay:{i*55}ms">💊 {d.title()}{alias}</span>'
                chips += "</div>"
                st.markdown(chips, unsafe_allow_html=True)

                # FIX: correct dict keys = adr_predictions, ddi_predictions
                adr_preds = result.get("adr_predictions", {})
                ddi_preds = result.get("ddi_predictions", [])
                adr_found = sum(1 for v in adr_preds.values() if v.get("confidence") != "none")
                ddi_found = sum(1 for d in ddi_preds if d.get("interactions"))

                st.markdown(
                    f'<div class="metrics-row">'
                    f'<div class="metric-card"><div class="metric-val" style="color:{ACCENT}">{len(drugs)}</div><div class="metric-lbl">Drugs Detected</div></div>'
                    f'<div class="metric-card"><div class="metric-val" style="color:{GREEN}">{adr_found}/{len(drugs)}</div><div class="metric-lbl">ADR Profiles Found</div></div>'
                    f'<div class="metric-card"><div class="metric-val" style="color:{AMBER}">{ddi_found}/{max(len(ddi_preds),1)}</div><div class="metric-lbl">DDI Pairs w/ Interactions</div></div>'
                    f'</div><div class="divider"></div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="sec-hdr"><span>🧪</span>'
                    f'<span class="sec-hdr-title">Adverse Drug Reactions (ADR)</span>'
                    f'<span class="sec-count">{len(drugs)} drugs</span></div>',
                    unsafe_allow_html=True,
                )
                ncols = min(len(drugs), 3)
                cols  = st.columns(ncols)
                for i, drug in enumerate(drugs):
                    with cols[i % ncols]:
                        render_drug_card(drug, adr_preds.get(drug, {}))

                if ddi_preds:
                    st.markdown(
                        f'<div class="sec-hdr"><span>⚠️</span>'
                        f'<span class="sec-hdr-title">Drug–Drug Interactions (DDI)</span>'
                        f'<span class="sec-count">{len(ddi_preds)} pairs</span></div>',
                        unsafe_allow_html=True,
                    )
                    for ddi in ddi_preds:
                        render_ddi_card(ddi)
                else:
                    st.info("Only one drug detected — DDI analysis requires at least two drugs.")

# ---------------------------------------------------------------------------
# MODE 2  --  SINGLE DRUG ADR
# ---------------------------------------------------------------------------
elif mode == "🔬 Single Drug ADR":

    st.markdown(
        '<div class="sec-hdr"><span>🔬</span>'
        '<span class="sec-hdr-title">Single Drug ADR Prediction</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:{TEXT_MUTED};font-size:0.875rem;margin-bottom:0.9rem">'
        "Enter any drug name — supports brand names (Crocin, Brufen, Lasix, Cordarone) "
        "and Indian/British spellings (Paracetamol, Frusemide, Salbutamol).</p>",
        unsafe_allow_html=True,
    )

    sc1, sc2 = st.columns([3, 1])
    with sc1:
        drug_input = st.text_input(
            label="drug_input",
            placeholder="e.g. paracetamol, warfarin, crocin, lasix…",
            key="drug_input_field",
            label_visibility="collapsed",
        )
    with sc2:
        topn_s = st.number_input("Top N", min_value=5, max_value=50, value=10, key="single_topn")

    # Quick test buttons  -- FIX: set widget key + rerun
    st.markdown(f'<div class="qt-lbl">Quick tests</div>', unsafe_allow_html=True)
    qt_drugs = ["paracetamol", "warfarin", "metformin", "amiodarone", "digoxin", "atorvastatin", "ibuprofen", "furosemide"]
    qt_cols  = st.columns(len(qt_drugs))
    for i, qd in enumerate(qt_drugs):
        if qt_cols[i].button(qd.title(), use_container_width=True, key=f"qt_{qd}"):
            st.session_state["drug_input_field"] = qd
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict ADR", use_container_width=True, type="primary", key="btn_predict")

    if predict_btn:
        current_drug = st.session_state.get("drug_input_field", "").strip()
        if not current_drug:
            st.warning("Please enter a drug name.")
        else:
            # FIX: correct function normalize_drug_name
            normalized = normalize_drug_name(current_drug)
            with st.spinner(f"Looking up {current_drug}…"):
                # FIX: correct function predict_adr + correct param top_n=
                result = predict_adr(current_drug, top_n=int(topn_s))

            if normalized != current_drug.lower():
                st.info(f"🔄 **{current_drug.title()}** recognized as **{normalized.title()}**")

            conf   = result.get("confidence", "none")
            method = result.get("method", "")
            se     = result.get("predicted_side_effects", [])

            st.markdown(
                f'<div class="sec-hdr" style="margin-top:1rem"><span>🧪</span>'
                f'<span class="sec-hdr-title">ADR Profile — {normalized.title()}</span>'
                f'<span style="margin-left:auto">{conf_badge(conf)} {src_badge(method)}</span></div>',
                unsafe_allow_html=True,
            )
            render_drug_card(normalized, result)

            if se:
                cats = {"Very Common": 0, "Common": 0, "Uncommon": 0, "Rare": 0, "Unknown": 0}
                for s in se:
                    cat = s.get("category", "Unknown")
                    for k in cats:
                        if k.lower() in cat.lower():
                            cats[k] += 1
                            break
                    else:
                        cats["Unknown"] += 1
                pills_html = " ".join(
                    f'<span class="pill">{k}: {v}</span>'
                    for k, v in cats.items() if v > 0
                )
                st.markdown(
                    f'<div style="margin-top:0.4rem;font-size:0.78rem;color:{TEXT_MUTED}">Category breakdown: {pills_html}</div>',
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------------------
# MODE 3  --  DRUG PAIR DDI
# ---------------------------------------------------------------------------
elif mode == "⚠️ Drug Pair DDI":

    st.markdown(
        '<div class="sec-hdr"><span>⚠️</span>'
        '<span class="sec-hdr-title">Drug–Drug Interaction Checker</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:{TEXT_MUTED};font-size:0.875rem;margin-bottom:0.9rem">'
        "Enter two drug names to check for known pharmacovigilance-confirmed interactions.</p>",
        unsafe_allow_html=True,
    )

    dc1, dc2 = st.columns(2)
    with dc1:
        drug1 = st.text_input(
            label="drug1",
            placeholder="First drug  e.g. warfarin",
            key="ddi_d1_field",
            label_visibility="collapsed",
        )
    with dc2:
        drug2 = st.text_input(
            label="drug2",
            placeholder="Second drug  e.g. aspirin",
            key="ddi_d2_field",
            label_visibility="collapsed",
        )

    # FIX: selectbox instead of slider
    topn_d_map = {"5 interactions": 5, "7 interactions": 7, "10 interactions": 10, "15 interactions": 15}
    topn_d_sel = st.selectbox("Max interactions to show", list(topn_d_map.keys()), index=1, key="ddi_topn")
    topn_d     = topn_d_map[topn_d_sel]

    # Quick pair buttons  -- FIX: set both widget keys + rerun
    st.markdown(f'<div class="qt-lbl">High-risk pair quick tests</div>', unsafe_allow_html=True)
    pairs = [
        ("warfarin",    "aspirin"),
        ("digoxin",     "amiodarone"),
        ("metformin",   "ibuprofen"),
        ("sertraline",  "tramadol"),
        ("clopidogrel", "omeprazole"),
        ("furosemide",  "metformin"),
    ]
    pc = st.columns(len(pairs))
    for i, (p1, p2) in enumerate(pairs):
        if pc[i].button(f"{p1[:5]}+{p2[:5]}", use_container_width=True, key=f"ddi_pair_{i}"):
            st.session_state["ddi_d1_field"] = p1
            st.session_state["ddi_d2_field"] = p2
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    check_btn = st.button("🔍 Check Interaction", use_container_width=True, type="primary", key="btn_ddi")

    if check_btn:
        d1_val = st.session_state.get("ddi_d1_field", "").strip()
        d2_val = st.session_state.get("ddi_d2_field", "").strip()

        if not d1_val or not d2_val:
            st.warning("Please enter both drug names.")
        else:
            # FIX: correct function normalize_drug_name
            n1 = normalize_drug_name(d1_val)
            n2 = normalize_drug_name(d2_val)

            if n1 != d1_val.lower():
                st.info(f"🔄 **{d1_val.title()}** → **{n1.title()}**")
            if n2 != d2_val.lower():
                st.info(f"🔄 **{d2_val.title()}** → **{n2.title()}**")

            with st.spinner("Checking TWOSIDES database…"):
                # FIX: correct function predict_ddi + correct param top_n=
                result = predict_ddi(d1_val, d2_val, top_n=topn_d)

            if "error" in result:
                st.error(result["error"])
            else:
                st.markdown(
                    f'<div class="sec-hdr" style="margin-top:1rem"><span>⚠️</span>'
                    f'<span class="sec-hdr-title">Interaction: {n1.title()} + {n2.title()}</span>'
                    f'<span style="margin-left:auto">{conf_badge(result.get("confidence","none"))}</span></div>',
                    unsafe_allow_html=True,
                )
                render_ddi_card(result)
                if result.get("confidence") == "high":
                    st.caption("Source: TWOSIDES database — real-world reports, PRR-filtered >= 3.0")

# ---------------------------------------------------------------------------
# MODE 4  --  ABOUT
# ---------------------------------------------------------------------------
elif mode == "ℹ️ About":

    st.markdown(
        '<div class="sec-hdr"><span>ℹ️</span>'
        '<span class="sec-hdr-title">About This System</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="about-grid">'

        '<div class="about-card">'
        '<div class="about-card-title">Data Sources</div>'
        '<div class="about-card-body">'
        '<span class="pill">SIDER 4.1</span> Official drug side effects (145K rows)<br>'
        '<span class="pill">OFFSIDES</span> Off-label ADR signals (PRR >= 2.0)<br>'
        '<span class="pill">TWOSIDES</span> DDI pairs (16M rows, PRR >= 3.0)<br>'
        '<span class="pill">DrugBank</span> Drug names + Lipinski (19K drugs)'
        '</div></div>'

        '<div class="about-card">'
        '<div class="about-card-title">ML Models</div>'
        '<div class="about-card-body">'
        '<span class="pill">NLP</span> SciSpacy BC5CDR drug NER<br>'
        '<span class="pill">ADR</span> Stacked RF + XGB + LightGBM<br>'
        '<span class="pill">DDI</span> Stacked RF + XGB + LR + LightGBM<br>'
        '<span class="pill">Features</span> Lipinski molecular descriptors'
        '</div></div>'

        '<div class="about-card">'
        '<div class="about-card-title">Pipeline</div>'
        '<div class="about-card-body">'
        '1. NLP free-text drug extraction<br>'
        '2. Synonym normalization (80+ mappings)<br>'
        '3. Database lookup → High Confidence<br>'
        '4. ML model fallback → Medium Confidence<br>'
        '5. Noise blacklist filter (204 entries)'
        '</div></div>'

        '<div class="about-card">'
        '<div class="about-card-title">Regional Names</div>'
        '<div class="about-card-body">'
        'Indian brands: Crocin, Brufen, Lasix, Cordarone<br>'
        'British spellings: Paracetamol, Frusemide, Salbutamol<br>'
        'US brands: Tylenol, Advil, Glucophage'
        '</div></div>'

        '<div class="about-card">'
        '<div class="about-card-title">Noise Filtering</div>'
        '<div class="about-card-body">'
        'ADR: PRR >= 2.0 threshold applied.<br>'
        'DDI: PRR >= 3.0 threshold applied.<br>'
        'Blacklist removes 204 non-pharmacological confounders: suicide attempts, injuries, social events.'
        '</div></div>'

        '<div class="about-card">'
        '<div class="about-card-title">Project</div>'
        '<div class="about-card-body">'
        'ADR &amp; DDI Prediction System<br>'
        'Python · scikit-learn · XGBoost<br>'
        'LightGBM · spaCy · Streamlit'
        '</div></div>'

        '</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown(
    f'<div class="footer">'
    f'<strong>ADR &amp; DDI Prediction System</strong>'
    f' &nbsp;·&nbsp; SIDER 4.1 · OFFSIDES · TWOSIDES · DrugBank'
    f' &nbsp;·&nbsp; For research use only — not a substitute for clinical judgement'
    f'</div>',
    unsafe_allow_html=True,
)
