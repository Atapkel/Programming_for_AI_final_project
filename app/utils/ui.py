"""Shared look and feel: colors, CSS, page header, rate tiles and the Plotly theme."""

from html import escape

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


NAVY = "#0B3C7D"
NAVY_DARK = "#072A59"
SKY = "#00A3D9"
GOLD = "#E0A800"
GREEN = "#1E9E5A"
RED = "#D64545"
MUTED = "#6B7A90"
BORDER = "#E1E7EF"
TEXT = "#1B2533"
COLORWAY = [NAVY, SKY, GOLD, "#7A5AF8", GREEN, "#E4572E", MUTED]

pio.templates["kz"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, sans-serif", size=13, color=TEXT),
        colorway=COLORWAY,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        title=dict(font=dict(size=15, color=NAVY_DARK), x=0, xanchor="left"),
        xaxis=dict(gridcolor="#EEF2F7", linecolor=BORDER, zeroline=False, ticks="outside",
                   tickcolor=BORDER),
        yaxis=dict(gridcolor="#EEF2F7", linecolor=BORDER, zeroline=False),
        legend=dict(orientation="h", y=-0.18, x=0, title_text=""),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=BORDER, font_color=TEXT),
        margin=dict(l=10, r=10, t=40, b=10),
    )
)
pio.templates.default = "kz"
TEMPLATE = "kz"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="st-"], .stMarkdown, button, input {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 1.6rem; max-width: 1320px; }}
h1, h2, h3 {{ color: {NAVY_DARK}; letter-spacing: -0.01em; }}

/* Sidebar as a navy navigation panel */
section[data-testid="stSidebar"] > div:first-child {{
  background: linear-gradient(180deg, {NAVY_DARK} 0%, {NAVY} 100%);
}}
section[data-testid="stSidebar"] * {{ color: #FFFFFF; }}
/* Wide-screen navigation; preserve native collapse behavior on smaller screens. */
@media (min-width: 1200px) {{
  section[data-testid="stSidebar"] {{
    width: 250px !important; min-width: 250px !important; max-width: 250px !important;
  }}
}}
@media (max-width: 767px) {{
  .block-container {{ padding: 1rem .65rem; }}
  .kz-hero {{ padding: 1rem; }}
  .kz-hero h1 {{ font-size: 1.4rem; }}
  .kz-tiles {{ grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
  .kz-table {{ display: block; overflow-x: auto; }}
}}
section[data-testid="stSidebar"] img {{
  border-radius: 8px; box-shadow: 0 6px 18px rgba(0,0,0,.35);
}}
section[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 2px; }}
section[data-testid="stSidebar"] [role="radiogroup"] label {{
  padding: .6rem .85rem; border-radius: 8px; width: 100%;
  border-left: 3px solid transparent; transition: background .15s;
}}
section[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{ display: none; }}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: rgba(255,255,255,.08); }}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
  background: rgba(255,255,255,.14); border-left-color: {GOLD}; font-weight: 600;
}}
.kz-side-brand {{ font-size: 1.05rem; font-weight: 700; margin: .6rem 0 0; }}
.kz-side-sub {{ font-size: .78rem; opacity: .7; letter-spacing: .08em; text-transform: uppercase; }}
.kz-side-label {{ font-size: .72rem; opacity: .6; letter-spacing: .1em; text-transform: uppercase;
  margin: 1.2rem 0 .3rem; }}

/* Page header band */
.kz-hero {{
  background: linear-gradient(120deg, {NAVY_DARK} 0%, {NAVY} 60%, #145AA8 100%);
  border-bottom: 4px solid {GOLD}; border-radius: 14px; padding: 1.6rem 1.9rem;
  color: #FFFFFF; margin-bottom: 1.3rem; display: flex; justify-content: space-between;
  align-items: flex-end; gap: 1rem; flex-wrap: wrap;
}}
.kz-hero .eyebrow {{ font-size: .75rem; letter-spacing: .14em; text-transform: uppercase;
  color: #9FD3F0; font-weight: 600; }}
.kz-hero h1 {{ color: #FFFFFF; margin: .25rem 0 .35rem; font-size: 1.9rem; padding: 0; }}
.kz-hero p {{ margin: 0; opacity: .85; max-width: 760px; }}
.kz-hero .stamp {{ font-size: .8rem; background: rgba(255,255,255,.12); padding: .4rem .75rem;
  border-radius: 999px; white-space: nowrap; }}

/* Rate tiles */
.kz-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 14px; margin-bottom: 1.2rem; }}
.kz-tile {{ background: #FFFFFF; border: 1px solid {BORDER}; border-radius: 12px;
  padding: 1rem 1.1rem; box-shadow: 0 1px 2px rgba(16,24,40,.04); border-top: 3px solid {NAVY}; }}
.kz-tile .label {{ font-size: .78rem; color: {MUTED}; text-transform: uppercase;
  letter-spacing: .06em; font-weight: 600; }}
.kz-tile .value {{ font-size: 1.65rem; font-weight: 700; color: {NAVY_DARK}; margin: .2rem 0; }}
.kz-tile .value small {{ font-size: .9rem; color: {MUTED}; font-weight: 500; }}
.kz-tile .delta {{ font-size: .85rem; font-weight: 600; }}
.kz-tile .note {{ font-size: .78rem; color: {MUTED}; margin-top: .15rem; }}
.kz-up {{ color: {GREEN}; }} .kz-down {{ color: {RED}; }} .kz-flat {{ color: {MUTED}; }}

/* Section titles, cards, callouts */
.kz-section {{ border-left: 4px solid {GOLD}; padding-left: .7rem; margin: 1.4rem 0 .7rem; }}
.kz-section h3 {{ margin: 0; padding: 0; font-size: 1.15rem; }}
.kz-section p {{ margin: .15rem 0 0; color: {MUTED}; font-size: .88rem; }}
.kz-card {{ background: #FFFFFF; border: 1px solid {BORDER}; border-radius: 12px;
  padding: 1.1rem 1.2rem; height: 100%; }}
.kz-card h4 {{ margin: 0 0 .35rem; color: {NAVY_DARK}; font-size: 1rem; }}
.kz-card p {{ margin: 0; color: #46546A; font-size: .9rem; }}
.kz-card .num {{ display: inline-block; width: 26px; height: 26px; border-radius: 50%;
  background: {NAVY}; color: #FFF; text-align: center; line-height: 26px; font-size: .8rem;
  font-weight: 700; margin-bottom: .5rem; }}
.kz-callout {{ background: #EEF6FC; border: 1px solid #CFE6F6; border-left: 4px solid {SKY};
  border-radius: 10px; padding: .9rem 1.1rem; margin: .3rem 0 1rem; color: {TEXT}; }}
.kz-callout b {{ color: {NAVY_DARK}; }}
.kz-callout.warn {{ background: #FFF8E6; border-color: #F5E2A8; border-left-color: {GOLD}; }}

/* Leaderboard table */
.kz-table {{ width: 100%; border-collapse: collapse; background: #FFF; border: 1px solid {BORDER};
  border-radius: 12px; overflow: hidden; font-size: .9rem; }}
.kz-table th {{ background: #F5F8FC; color: {MUTED}; font-weight: 600; text-transform: uppercase;
  font-size: .72rem; letter-spacing: .06em; text-align: left; padding: .65rem .8rem;
  border-bottom: 1px solid {BORDER}; }}
.kz-table td {{ padding: .6rem .8rem; border-bottom: 1px solid #F0F3F8; }}
.kz-table tr:last-child td {{ border-bottom: none; }}
.kz-table td.r, .kz-table th.r {{ text-align: right; font-variant-numeric: tabular-nums; }}
.kz-table tr.best td {{ background: #F1FAF5; }}
.kz-badge {{ display: inline-block; padding: .1rem .5rem; border-radius: 999px; font-size: .7rem;
  font-weight: 600; letter-spacing: .03em; }}
.kz-badge.base {{ background: #EEF1F5; color: {MUTED}; }}
.kz-badge.stat {{ background: #E6F0FB; color: {NAVY}; }}
.kz-badge.ml {{ background: #FFF4D6; color: #8A6A00; }}
.kz-badge.best {{ background: {GREEN}; color: #FFF; }}
.kz-bar {{ background: #EEF2F7; border-radius: 4px; height: 8px; width: 120px; display: inline-block;
  vertical-align: middle; margin-right: 8px; }}
.kz-bar > span {{ display: block; height: 100%; border-radius: 4px; background: {NAVY}; }}

/* Streamlit widgets */
[data-testid="stPlotlyChart"] {{ background: #FFF; border: 1px solid {BORDER}; border-radius: 12px;
  padding: 8px 8px 4px; }}
[data-testid="stMetric"] {{ background: #FFF; border: 1px solid {BORDER}; border-radius: 12px;
  padding: .8rem 1rem; }}
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; padding: .5rem 1rem; }}
.kz-footer {{ margin-top: 2.2rem; padding-top: .9rem; border-top: 1px solid {BORDER};
  color: {MUTED}; font-size: .78rem; }}
</style>
"""


def html(markup):
    """Render HTML; lines are stripped so Markdown never turns indentation into code blocks."""
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def apply_theme():
    html(CSS)


def page_header(title, subtitle, eyebrow="", stamp=""):
    stamp_html = '<div class="stamp">' + escape(stamp) + "</div>" if stamp else ""
    html(f"""
    <div class="kz-hero"><div>
      <div class="eyebrow">{escape(eyebrow)}</div>
      <h1>{escape(title)}</h1><p>{escape(subtitle)}</p>
    </div>{stamp_html}</div>
    """)


def section(title, subtitle=""):
    subtitle_html = "<p>" + escape(subtitle) + "</p>" if subtitle else ""
    html(f'<div class="kz-section"><h3>{escape(title)}</h3>{subtitle_html}</div>')


def delta_html(change, text, good_when_up=True):
    """Arrow + colored text; color shows whether the move is good, arrow shows direction."""
    if change is None or abs(change) < 1e-12:
        return f'<span class="kz-flat">● {escape(text)}</span>'
    arrow = "▲" if change > 0 else "▼"
    good = (change > 0) == good_when_up
    return f'<span class="{"kz-up" if good else "kz-down"}">{arrow} {escape(text)}</span>'


def tiles(items):
    """items: dicts with label, value, optional unit, delta (html) and note."""
    cells = []
    for item in items:
        unit = item.get("unit", "")
        unit_html = " <small>" + escape(unit) + "</small>" if unit else ""
        cells.append(
            '<div class="kz-tile">'
            '<div class="label">' + escape(item["label"]) + "</div>"
            '<div class="value">' + escape(item["value"]) + unit_html + "</div>"
            '<div class="delta">' + item.get("delta", "") + "</div>"
            '<div class="note">' + escape(item.get("note", "")) + "</div></div>"
        )
    html('<div class="kz-tiles">' + "".join(cells) + "</div>")


def callout(markup, warn=False):
    html(f'<div class="kz-callout{" warn" if warn else ""}">{markup}</div>')


def footer():
    html('<div class="kz-footer">Sources: National Bank of Kazakhstan · IMF · FRED · World Bank. '
         'Educational project, Programming for AI. Not an official publication.</div>')
