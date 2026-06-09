"""
World Cup 2026 stylistic utilities for the Streamlit app.

Features:
- Flag-emoji mapping for all FIFA national teams
- Custom CSS for World Cup trophy background and chat styling
- Countdown timer, bouncing ball animation, star ratings
- Player name → image URL resolution via BigQuery dim_player
"""
from __future__ import annotations

import os
import re
from datetime import date, datetime, timezone

# ---------------------------------------------------------------------------
# FIFA national team → flag emoji mapping
# Maps common team names / country names to their regional-indicator flag emoji.
# ---------------------------------------------------------------------------

_COUNTRY_TO_FLAG: dict[str, str] = {
    # AFC (Asia)
    "australia": "🇦🇺",
    "china": "🇨🇳",
    "iran": "🇮🇷",
    "iraq": "🇮🇶",
    "japan": "🇯🇵",
    "kuwait": "🇰🇼",
    "north korea": "🇰🇵",
    "south korea": "🇰🇷",
    "korea republic": "🇰🇷",
    "saudi arabia": "🇸🇦",
    "united arab emirates": "🇦🇪",
    "uae": "🇦🇪",
    "qatar": "🇶🇦",
    "uzbekistan": "🇺🇿",
    "syria": "🇸🇾",
    "jordan": "🇯🇴",
    "oman": "🇴🇲",
    "bahrain": "🇧🇭",
    "lebanon": "🇱🇧",
    "vietnam": "🇻🇳",
    "thailand": "🇹🇭",
    "indonesia": "🇮🇩",
    "india": "🇮🇳",
    "kyrgyzstan": "🇰🇬",
    "tajikistan": "🇹🇯",
    "turkmenistan": "🇹🇲",
    "palestine": "🇵🇸",
    "yemen": "🇾🇪",
    # CAF (Africa)
    "algeria": "🇩🇿",
    "angola": "🇦🇴",
    "benin": "🇧🇯",
    "botswana": "🇧🇼",
    "burkina faso": "🇧🇫",
    "burundi": "🇧🇮",
    "cameroon": "🇨🇲",
    "cape verde": "🇨🇻",
    "central african republic": "🇨🇫",
    "chad": "🇹🇩",
    "comoros": "🇰🇲",
    "congo": "🇨🇬",
    "congo dr": "🇨🇩",
    "dr congo": "🇨🇩",
    "ivory coast": "🇨🇮",
    "côte d'ivoire": "🇨🇮",
    "cote d'ivoire": "🇨🇮",
    "djibouti": "🇩🇯",
    "egypt": "🇪🇬",
    "equatorial guinea": "🇬🇶",
    "eritrea": "🇪🇷",
    "ethiopia": "🇪🇹",
    "gabon": "🇬🇦",
    "gambia": "🇬🇲",
    "ghana": "🇬🇭",
    "guinea": "🇬🇳",
    "guinea-bissau": "🇬🇼",
    "kenya": "🇰🇪",
    "lesotho": "🇱🇸",
    "liberia": "🇱🇷",
    "libya": "🇱🇾",
    "madagascar": "🇲🇬",
    "malawi": "🇲🇼",
    "mali": "🇲🇱",
    "mauritania": "🇲🇷",
    "mauritius": "🇲🇺",
    "morocco": "🇲🇦",
    "mozambique": "🇲🇿",
    "namibia": "🇳🇦",
    "niger": "🇳🇪",
    "nigeria": "🇳🇬",
    "rwanda": "🇷🇼",
    "senegal": "🇸🇳",
    "sierra leone": "🇸🇱",
    "somalia": "🇸🇴",
    "south africa": "🇿🇦",
    "south sudan": "🇸🇸",
    "sudan": "🇸🇩",
    "eswatini": "🇸🇿",
    "tanzania": "🇹🇿",
    "togo": "🇹🇬",
    "tunisia": "🇹🇳",
    "uganda": "🇺🇬",
    "zambia": "🇿🇲",
    "zimbabwe": "🇿🇼",
    # CONCACAF
    "canada": "🇨🇦",
    "mexico": "🇲🇽",
    "united states": "🇺🇸",
    "usa": "🇺🇸",
    "us": "🇺🇸",
    "costa rica": "🇨🇷",
    "el salvador": "🇸🇻",
    "guatemala": "🇬🇹",
    "honduras": "🇭🇳",
    "nicaragua": "🇳🇮",
    "panama": "🇵🇦",
    "belize": "🇧🇿",
    "jamaica": "🇯🇲",
    "trinidad and tobago": "🇹🇹",
    "trinidad": "🇹🇹",
    "haiti": "🇭🇹",
    "cuba": "🇨🇺",
    "dominican republic": "🇩🇴",
    "puerto rico": "🇵🇷",
    "suriname": "🇸🇷",
    "curaçao": "🇨🇼",
    "curacao": "🇨🇼",
    "guadeloupe": "🇬🇵",
    "martinique": "🇲🇶",
    "french guiana": "🇬🇫",
    "guyana": "🇬🇾",
    "bermuda": "🇧🇲",
    "aruba": "🇦🇼",
    "st kitts and nevis": "🇰🇳",
    "saint kitts and nevis": "🇰🇳",
    "antigua and barbuda": "🇦🇬",
    "dominica": "🇩🇲",
    "st lucia": "🇱🇨",
    "saint lucia": "🇱🇨",
    "st vincent and the grenadines": "🇻🇨",
    "grenada": "🇬🇩",
    "barbados": "🇧🇧",
    "bahamas": "🇧🇸",
    "cayman islands": "🇰🇾",
    # CONMEBOL
    "argentina": "🇦🇷",
    "bolivia": "🇧🇴",
    "brazil": "🇧🇷",
    "chile": "🇨🇱",
    "colombia": "🇨🇴",
    "ecuador": "🇪🇨",
    "paraguay": "🇵🇾",
    "peru": "🇵🇪",
    "uruguay": "🇺🇾",
    "venezuela": "🇻🇪",
    # OFC
    "new zealand": "🇳🇿",
    "fiji": "🇫🇯",
    "papua new guinea": "🇵🇬",
    "solomon islands": "🇸🇧",
    "vanuatu": "🇻🇺",
    "tahiti": "🇵🇫",
    "new caledonia": "🇳🇨",
    "samoa": "🇼🇸",
    "american samoa": "🇦🇸",
    "tonga": "🇹🇴",
    "cook islands": "🇨🇰",
    # UEFA (Europe)
    "albania": "🇦🇱",
    "andorra": "🇦🇩",
    "armenia": "🇦🇲",
    "austria": "🇦🇹",
    "azerbaijan": "🇦🇿",
    "belarus": "🇧🇾",
    "belgium": "🇧🇪",
    "bosnia and herzegovina": "🇧🇦",
    "bosnia": "🇧🇦",
    "bulgaria": "🇧🇬",
    "croatia": "🇭🇷",
    "cyprus": "🇨🇾",
    "czech republic": "🇨🇿",
    "czechia": "🇨🇿",
    "denmark": "🇩🇰",
    "england": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "estonia": "🇪🇪",
    "faroe islands": "🇫🇴",
    "finland": "🇫🇮",
    "france": "🇫🇷",
    "georgia": "🇬🇪",
    "germany": "🇩🇪",
    "gibraltar": "🇬🇮",
    "greece": "🇬🇷",
    "hungary": "🇭🇺",
    "iceland": "🇮🇸",
    "israel": "🇮🇱",
    "italy": "🇮🇹",
    "kazakhstan": "🇰🇿",
    "kosovo": "🇽🇰",
    "latvia": "🇱🇻",
    "liechtenstein": "🇱🇮",
    "lithuania": "🇱🇹",
    "luxembourg": "🇱🇺",
    "malta": "🇲🇹",
    "moldova": "🇲🇩",
    "montenegro": "🇲🇪",
    "netherlands": "🇳🇱",
    "holland": "🇳🇱",
    "north macedonia": "🇲🇰",
    "macedonia": "🇲🇰",
    "northern ireland": "🇬🇧",
    "norway": "🇳🇴",
    "poland": "🇵🇱",
    "portugal": "🇵🇹",
    "romania": "🇷🇴",
    "russia": "🇷🇺",
    "san marino": "🇸🇲",
    "scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "serbia": "🇷🇸",
    "slovakia": "🇸🇰",
    "slovenia": "🇸🇮",
    "spain": "🇪🇸",
    "sweden": "🇸🇪",
    "switzerland": "🇨🇭",
    "turkey": "🇹🇷",
    "türkiye": "🇹🇷",
    "ukraine": "🇺🇦",
    "wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿",
}

# Additional common variants / short names
_ALIASES: dict[str, str] = {
    "korea": "south korea",
    "united states of america": "united states",
    "united states men": "united states",
    "usmnt": "united states",
    "three lions": "england",
    "socceroos": "australia",
    "matildas": "australia",
    "les bleus": "france",
    "azzurri": "italy",
    "la roja": "spain",
    "seleção": "brazil",
    "selecao": "brazil",
    "die mannschaft": "germany",
    "oranje": "netherlands",
    "super eagles": "nigeria",
    "black stars": "ghana",
    "teranga lions": "senegal",
    "indomitable lions": "cameroon",
    "atlas lions": "morocco",
    "pharaohs": "egypt",
    "samba boys": "brazil",
    "albiceleste": "argentina",
    "la albiceleste": "argentina",
    "cafeteros": "colombia",
    "tricolores": "colombia",
    "vinotinto": "venezuela",
    "la vinotinto": "venezuela",
    "los cafeteros": "colombia",
    "team usa": "united states",
    "uswnt": "united states",
    "canmnt": "canada",
    "el tri": "mexico",
    "los ticos": "costa rica",
    "regragui": None,  # coach, not a team
    "特立尼达和多巴哥": "trinidad and tobago",
    "中国": "china",
    "日本": "japan",
    "韩国": "south korea",
    "巴西": "brazil",
    "阿根廷": "argentina",
    "德国": "germany",
    "法国": "france",
    "西班牙": "spain",
    "意大利": "italy",
    "英格兰": "england",
    "荷兰": "netherlands",
    "葡萄牙": "portugal",
    "墨西哥": "mexico",
    "乌拉圭": "uruguay",
    "哥伦比亚": "colombia",
    "智利": "chile",
    "秘鲁": "peru",
    "厄瓜多尔": "ecuador",
    "巴拉圭": "paraguay",
    "玻利维亚": "bolivia",
    "委内瑞拉": "venezuela",
}


def get_flag(country_name: str) -> str | None:
    """Return the flag emoji for a country/team name, or None if unknown."""
    if not country_name or not country_name.strip():
        return None
    key = country_name.strip().lower()
    # Check aliases first
    resolved = _ALIASES.get(key)
    if resolved is not None:
        key = resolved
    return _COUNTRY_TO_FLAG.get(key)


def inject_flag_emojis(text: str) -> str:
    """Scan text for known national team names and prepend their flag emoji.

    Only adds the flag *once* per team name per message.
    Uses case-insensitive matching with word boundaries to avoid
    partial matches (e.g. 'France' not matching 'frances').
    """
    if not text:
        return text

    # Build a regex that matches any known country name with word boundaries.
    # Sort by length descending so longer names match before shorter ones
    # (e.g., "south korea" before "korea").
    country_names = sorted(
        _COUNTRY_TO_FLAG.keys(),
        key=lambda n: (-len(n), n),
    )

    # Escape regex special chars in names
    escaped = [re.escape(name) for name in country_names if len(name) >= 2]
    pattern = re.compile(
        r"\b(" + "|".join(escaped) + r")\b",
        flags=re.IGNORECASE,
    )

    already_done: set[str] = set()

    def _replacement(match: re.Match) -> str:
        matched_text = match.group(1)
        key = matched_text.lower()
        if key in already_done:
            return matched_text  # already added flag
        flag = _COUNTRY_TO_FLAG.get(key)
        if flag is None:
            return matched_text
        already_done.add(key)
        return f"{flag} {matched_text}"

    return pattern.sub(_replacement, text)


# ---------------------------------------------------------------------------
# Agent → Emoji mapping
# ---------------------------------------------------------------------------

AGENT_EMOJIS: dict[str, str] = {
    "bigquery": "📊",
    "news": "📰",
    "sentiment": "💬",
    "prediction": "🔮",
    "match_facts": "⚽",
    "chat": "💬",
    "rules": "📋",
    "orchestrator": "🧠",
}


def agent_emoji(agent_name: str) -> str:
    """Return the representative emoji for an agent."""
    return AGENT_EMOJIS.get(agent_name.lower(), "🤖")


# ---------------------------------------------------------------------------
# Custom CSS for World Cup trophy background and chat styling
# ---------------------------------------------------------------------------

# Lazy-loaded base64 data URI for the World Cup trophy background image.
_bg_data_uri: str | None = None


def _get_bg_data_uri() -> str:
    """Load the trophy image and return a base64 data URI for CSS embedding."""
    global _bg_data_uri
    if _bg_data_uri is not None:
        return _bg_data_uri

    import base64

    # Resolve the image relative to this module's directory
    _img_path = os.path.join(os.path.dirname(__file__), "Images", "World-Cup-2026-Logo-500x281.png")
    try:
        with open(_img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        _bg_data_uri = f"data:image/png;base64,{b64}"
    except FileNotFoundError:
        # Fallback: use the trophy emoji approach
        _bg_data_uri = ""
    return _bg_data_uri


def _build_world_cup_css() -> str:
    """Build the full CSS with the trophy image injected as a title icon."""
    bg_uri = _get_bg_data_uri()

    trophy_title_css = ""
    if bg_uri:
        trophy_title_css = f"""
/* ── Trophy image next to the title (white bg removed via mix-blend-mode) ── */
.wc-title-icon {{
    display: inline-block;
    width: 10.5em;
    height: 10.5em;
    background: url("{bg_uri}");
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center center;
    vertical-align: middle;
    margin-right: 0.15em;
    mix-blend-mode: screen;
    filter: drop-shadow(0 0 10px rgba(255, 215, 0, 0.3));
}}

/* ── Trophy avatar in chat (overrides Streamlit's emoji avatar) ── */
[data-testid="stChatMessageAvatar"][aria-label="assistant"] {{
    background: url("{bg_uri}") !important;
    background-size: contain !important;
    background-position: center center !important;
    background-repeat: no-repeat !important;
    mix-blend-mode: screen;
}}
[data-testid="stChatMessageAvatar"][aria-label="assistant"] svg,
[data-testid="stChatMessageAvatar"][aria-label="assistant"] img,
[data-testid="stChatMessageAvatar"][aria-label="assistant"] span {{
    display: none !important;
}}
"""

    fallback_emoji_css = ""
    if not bg_uri:
        fallback_emoji_css = """
/* ── Trophy watermark (centered, large, very faint emoji fallback) ── */
[data-testid="stAppViewContainer"]::before {
    content: "🏆";
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: min(60vw, 400px);
    opacity: 0.04;
    pointer-events: none;
    z-index: 0;
    user-select: none;
}
"""

    return f"""<style>
/* ── Full-page background (dark navy gradient) ── */
[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(ellipse at 50% 0%, rgba(255, 215, 0, 0.08) 0%, transparent 60%),
        linear-gradient(180deg, #0a1f2e 0%, #0d2b3e 40%, #0f1a2e 100%);
}}
{trophy_title_css}{fallback_emoji_css}
/* ── Sidebar glass effect ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, rgba(10, 31, 46, 0.95) 0%, rgba(15, 26, 46, 0.95) 100%);
    border-right: 1px solid rgba(255, 215, 0, 0.15);
}}

/* ── Main content area ── */
[data-testid="stAppViewBlockContainer"] {{
    position: relative;
    z-index: 1;
}}

/* ── Chat message styling ── */
[data-testid="stChatMessage"] {{
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 215, 0, 0.1);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 8px;
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
}}

/* ── User chat bubble ── */
[data-testid="stChatMessage"][aria-label*="user"] {{
    background: rgba(255, 215, 0, 0.06);
    border-color: rgba(255, 215, 0, 0.25);
}}

/* ── Assistant chat bubble ── */
[data-testid="stChatMessage"][aria-label*="assistant"] {{
    background: rgba(30, 144, 255, 0.06);
    border-color: rgba(30, 144, 255, 0.2);
}}

/* ── Chat input field ── */
[data-testid="stChatInput"] textarea {{
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 215, 0, 0.2);
    border-radius: 10px;
    color: #e0e0e0;
}}

/* ── Title & captions ── */
h1, h2, h3 {{
    color: #f0c040 !important;
}}
h1 {{
    text-shadow: 0 0 20px rgba(255, 215, 0, 0.3);
}}

/* ── Agent badge in sidebar ── */
.agent-badge {{
    display: inline-block;
    background: rgba(255, 215, 0, 0.12);
    border: 1px solid rgba(255, 215, 0, 0.3);
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.85em;
    color: #f0c040;
}}

/* ── Countdown timer ── */
.wc-countdown {{
    text-align: center;
    padding: 12px 16px;
    margin: 8px 0 16px 0;
    background: linear-gradient(135deg, rgba(255, 215, 0, 0.1) 0%, rgba(255, 215, 0, 0.04) 100%);
    border: 1px solid rgba(255, 215, 0, 0.2);
    border-radius: 12px;
}}
.wc-countdown .days-left {{
    font-size: 2.4em;
    font-weight: 800;
    color: #f0c040;
    line-height: 1.1;
}}
.wc-countdown .countdown-label {{
    font-size: 0.85em;
    color: #aaa;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}

/* ── Next Match card (main page, horizontal layout) ── */
.next-match-card {{
    background: linear-gradient(135deg, rgba(255, 215, 0, 0.06) 0%, rgba(10, 31, 46, 0.5) 100%);
    border: 1px solid rgba(255, 215, 0, 0.15);
    border-radius: 14px;
    padding: 16px 20px;
    margin: 4px 0 16px 0;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 24px;
    flex-wrap: wrap;
    text-align: center;
}}
.next-match-countdown {{
    font-size: 1.6em;
    font-weight: 800;
    color: #f0c040;
    line-height: 1.2;
    white-space: nowrap;
}}
.next-match-countdown-sub {{
    font-size: 0.7em;
    color: #888;
}}
.next-match-teams {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
}}
.next-match-team {{
    text-align: center;
    min-width: 90px;
}}
.next-match-flag {{
    font-size: 1.5em;
    display: block;
}}
.next-match-name {{
    font-size: 0.8em;
    font-weight: 700;
    color: #e0e0e0;
    display: block;
    margin-top: 2px;
}}
.next-match-players {{
    margin-top: 3px;
    color: #ccc;
    font-size: 0.72em;
}}
.next-match-vs {{
    font-size: 0.65em;
    font-weight: 700;
    color: rgba(255, 215, 0, 0.45);
    align-self: center;
}}
.next-match-meta {{
    text-align: center;
    font-size: 0.72em;
    color: #aaa;
    line-height: 1.5;
    border-left: 1px solid rgba(255, 255, 255, 0.08);
    padding-left: 20px;
}}
.next-match-meta-row {{
    padding: 1px 0;
}}

/* ── Star confidence ratings ── */
.confidence-stars {{
    display: inline-flex;
    gap: 2px;
    font-size: 1.1em;
    margin-top: 6px;
}}
.confidence-stars .star-filled {{ color: #f0c040; }}
.confidence-stars .star-empty  {{ color: rgba(255, 255, 255, 0.15); }}

/* ── Player image card (inline in chat) ── */
.player-card {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 215, 0, 0.15);
    border-radius: 10px;
    padding: 6px 10px 6px 6px;
    margin: 3px 4px 3px 0;
    vertical-align: middle;
}}
.player-card img {{
    width: 36px;
    height: 36px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid rgba(255, 215, 0, 0.3);
}}

/* ── Bouncing football animation ── */
@keyframes bounceAcross {{
    0%   {{ left: -60px; bottom: 8%; }}
    15%  {{ left: 12%; bottom: 14%; }}
    30%  {{ left: 28%; bottom: 6%; }}
    45%  {{ left: 44%; bottom: 12%; }}
    60%  {{ left: 60%; bottom: 7%; }}
    75%  {{ left: 76%; bottom: 13%; }}
    90%  {{ left: 92%; bottom: 5%; }}
    100% {{ left: 105%; bottom: 10%; }}
}}

.wc-bouncing-ball {{
    position: fixed;
    bottom: 8%;
    left: -60px;
    font-size: 2em;
    z-index: 999;
    pointer-events: none;
    user-select: none;
    animation: bounceAcross 14s linear infinite;
    opacity: 0.18;
    filter: drop-shadow(0 4px 8px rgba(0,0,0,0.3));
}}

/* ── Subtle football-pattern dots overlay ── */
[data-testid="stAppViewContainer"]::after {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image:
        radial-gradient(circle at 20% 30%, rgba(255, 255, 255, 0.015) 1px, transparent 1px),
        radial-gradient(circle at 70% 60%, rgba(255, 255, 255, 0.015) 1px, transparent 1px),
        radial-gradient(circle at 40% 80%, rgba(255, 215, 0, 0.02) 1px, transparent 1px);
    background-size: 80px 80px, 100px 100px, 120px 120px;
    pointer-events: none;
    z-index: 0;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{
    width: 8px;
}}
::-webkit-scrollbar-track {{
    background: rgba(255, 255, 255, 0.02);
}}
::-webkit-scrollbar-thumb {{
    background: rgba(255, 215, 0, 0.15);
    border-radius: 4px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: rgba(255, 215, 0, 0.3);
}}
</style>
"""


# ── Module-level CSS accessor (lazy-generated, replaces old WORLD_CUP_CSS constant) ──

def get_world_cup_css() -> str:
    """Return the full World Cup CSS, with trophy background image embedded."""
    return _build_world_cup_css()


def world_cup_header_html() -> str:
    """Returns an inline HTML span with the trophy image for use next to the title."""
    bg_uri = _get_bg_data_uri()
    if not bg_uri:
        return '<span style="font-size:2.2em;">🏆</span>'
    return '<span class="wc-title-icon" aria-label="World Cup Trophy"></span>'


# ---------------------------------------------------------------------------
# Suggestion 1: Countdown timer
# ---------------------------------------------------------------------------

_WC2026_START = date(2026, 6, 11)  # tournament kick-off


def countdown_html() -> str:
    """Returns an HTML countdown widget showing days until World Cup 2026."""
    today = date.today()
    days_left = (_WC2026_START - today).days
    if days_left < 0:
        days_text = "🏁 The World Cup is LIVE!"
    elif days_left == 0:
        days_text = "⚽ Kick-off TODAY!"
    elif days_left == 1:
        days_text = "⏱️ Tomorrow!"
    else:
        days_text = str(days_left)

    return f"""
<div class="wc-countdown">
    <div class="days-left">{days_text}</div>
    <div class="countdown-label">Days until World Cup 2026</div>
    <div style="font-size:0.75em;color:#888;margin-top:4px;">
        📅 {_WC2026_START.strftime('%B %d, %Y')}
    </div>
</div>
"""


# ---------------------------------------------------------------------------
# Suggestion 2: Bouncing football animation element
# ---------------------------------------------------------------------------

BOUNCING_BALL_HTML = (
    '<div class="wc-bouncing-ball" aria-hidden="true">⚽</div>'
)


# ---------------------------------------------------------------------------
# Suggestion 3: Crowd roar audio on assistant reply
# ---------------------------------------------------------------------------

CROWD_ROAR_HTML = """
<script>
(function() {
    // Play a short crowd roar when a new assistant message appears.
    // Uses a tiny base64-encoded silent audio trick to avoid autoplay blocks.
    var lastAssistantCount = 0;
    var observer = new MutationObserver(function() {
        var messages = document.querySelectorAll('[data-testid="stChatMessage"]');
        var assistantMsgs = document.querySelectorAll('[data-testid="stChatMessage"][aria-label*="assistant"]');
        if (assistantMsgs.length > lastAssistantCount) {
            lastAssistantCount = assistantMsgs.length;
            // Use Web Audio API for a subtle crowd pop (no external file needed)
            try {
                var ctx = new (window.AudioContext || window.webkitAudioContext)();
                // Create a short burst of filtered noise → crowd-like "roar"
                var duration = 0.35;
                var bufferSize = ctx.sampleRate * duration;
                var buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
                var data = buffer.getChannelData(0);
                for (var i = 0; i < bufferSize; i++) {
                    var t = i / ctx.sampleRate;
                    // Pink-ish noise burst with quick decay
                    var noise = (Math.random() * 2 - 1) * 0.3;
                    var envelope = Math.exp(-t * 8) * (1 - Math.exp(-t * 30));
                    data[i] = noise * envelope;
                }
                var source = ctx.createBufferSource();
                source.buffer = buffer;
                var filter = ctx.createBiquadFilter();
                filter.type = 'bandpass';
                filter.frequency.value = 800;
                filter.Q.value = 0.8;
                var gain = ctx.createGain();
                gain.gain.value = 0.25;
                source.connect(filter);
                filter.connect(gain);
                gain.connect(ctx.destination);
                source.start();
            } catch(e) {
                // Silently fail if audio not available
            }
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
})();
</script>
"""


# ---------------------------------------------------------------------------
# Suggestion 4: Confidence → Gold Stars
# ---------------------------------------------------------------------------

def confidence_stars_html(score: float, label: str) -> str:
    """Convert a 0-1 confidence score to 1-5 gold stars HTML."""
    num_stars = max(1, min(5, round(score * 5)))
    filled = '<span class="star-filled">⭐</span>' * num_stars
    empty = '<span class="star-empty">⭐</span>' * (5 - num_stars)
    pct = round(score * 100)
    return (
        f'<div class="confidence-stars" title="Confidence: {label.upper()} ({pct}%)">'
        f'{filled}{empty}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Player images: known top players mapping (fallback when BigQuery unavailable)
# ---------------------------------------------------------------------------

# Mapping of common player names → image URLs (from reputable CDN sources).
# These are fallback images; the primary lookup queries dim_player in BigQuery.
_PLAYER_IMAGE_FALLBACK: dict[str, str] = {
    "lionel messi": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/95803.jpg",
    "cristiano ronaldo": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/63706.jpg",
    "kylian mbappé": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250076908.jpg",
    "kylian mbappe": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250076908.jpg",
    "neymar": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250039508.jpg",
    "harry kane": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250016834.jpg",
    "kevin de bruyne": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250009912.jpg",
    "robert lewandowski": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250002229.jpg",
    "mohamed salah": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250063625.jpg",
    "vinícius júnior": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250131924.jpg",
    "vinicius junior": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250131924.jpg",
    "jude bellingham": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250134977.jpg",
    "erling haaland": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250116362.jpg",
    "luka modrić": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/74699.jpg",
    "luka modric": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/74699.jpg",
    "antoine griezmann": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250020262.jpg",
    "rodri": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250081766.jpg",
    "phil foden": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250115352.jpg",
    "bukayo saka": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250139465.jpg",
    "jamal musiala": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250139578.jpg",
    "pedri": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250137778.jpg",
    "gavi": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250139560.jpg",
    "federico valverde": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250101405.jpg",
    "bruno fernandes": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250066439.jpg",
    "bernardo silva": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250056731.jpg",
    "julián álvarez": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250134247.jpg",
    "julian alvarez": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250134247.jpg",
    "lautaro martínez": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250081760.jpg",
    "lautaro martinez": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250081760.jpg",
    "virgil van dijk": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250024772.jpg",
    "son heung-min": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250024976.jpg",
    "heung-min son": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250024976.jpg",
    "rafael leão": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250115700.jpg",
    "rafael leao": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250115700.jpg",
    "khvicha kvaratskhelia": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250142290.jpg",
    "declan rice": "https://img.uefa.com/imgml/TP/players/1/2026/324x324/250091666.jpg",
}


def _get_player_image_fallback(player_name: str) -> str | None:
    """Return a fallback image URL for a known player name."""
    return _PLAYER_IMAGE_FALLBACK.get(player_name.strip().lower())


# Cache for BigQuery player lookups to avoid repeated queries
_player_image_cache: dict[str, str] = {}


def get_player_image_url(player_name: str) -> str | None:
    """Look up a player's image URL from dim_player in BigQuery, falling back to a known list.

    Returns None if no image is available for this player.
    """
    if not player_name or not player_name.strip():
        return None

    key = player_name.strip().lower()

    # Check in-memory cache first
    cached = _player_image_cache.get(key)
    if cached is not None:
        return cached if cached != "__NONE__" else None

    # Try BigQuery lookup
    try:
        from src.tools.bigquery_tools import run_query

        project = os.environ.get("BIGQUERY_PROJECT_ID")
        dataset = os.environ.get("BIGQUERY_DATASET_ID")
        if project and dataset:
            sql = f"""
                SELECT player_name, player_image_url
                FROM `{project}.{dataset}.dim_player`
                WHERE LOWER(player_name) = LOWER(@name)
                LIMIT 1
            """
            # Note: run_query doesn't support parameterized queries directly,
            # so we escape the name safely.
            safe_name = player_name.replace("'", "''")
            sql_final = f"""
                SELECT player_name, player_image_url
                FROM `{project}.{dataset}.dim_player`
                WHERE LOWER(player_name) = LOWER('{safe_name}')
                LIMIT 1
            """
            df = run_query(sql_final)
            if not df.empty:
                img_url = df.iloc[0].get("player_image_url") or None
                if img_url and str(img_url).startswith("http"):
                    _player_image_cache[key] = str(img_url)
                    return str(img_url)
    except Exception:
        pass  # BigQuery unavailable — use fallback

    # Fallback to static mapping
    fallback = _get_player_image_fallback(player_name)
    if fallback:
        _player_image_cache[key] = fallback
        return fallback

    # Mark as not found to avoid repeated lookups
    _player_image_cache[key] = "__NONE__"
    return None


def inject_player_images(text: str) -> str:
    """Scan text for known player names and embed inline player image cards.

    Uses word-boundary matching on known player names from fallback list
    and BigQuery dim_player. Only adds image once per player per message.
    """
    if not text:
        return text

    # Collect all known player names (fallback + any cached from BQ)
    known_players = set(_PLAYER_IMAGE_FALLBACK.keys())
    known_players.update(
        k for k, v in _player_image_cache.items() if v != "__NONE__"
    )

    if not known_players:
        return text

    # Sort by name length descending so "kylian mbappé" matches before "mbappé"
    sorted_players = sorted(known_players, key=lambda n: (-len(n), n))

    # Build regex with word boundaries
    escaped = [re.escape(name) for name in sorted_players if len(name) >= 3]
    if not escaped:
        return text

    pattern = re.compile(
        r"\b(" + "|".join(escaped) + r")\b",
        flags=re.IGNORECASE,
    )

    already_done: set[str] = set()

    def _replacement(match: re.Match) -> str:
        matched_text = match.group(1)
        key = matched_text.lower()
        if key in already_done:
            return matched_text
        img_url = get_player_image_url(matched_text)
        if img_url is None:
            return matched_text
        already_done.add(key)
        return (
            f'<span class="player-card">'
            f'<img src="{img_url}" alt="{matched_text}" loading="lazy" '
            f'onerror="this.style.display=\'none\'">'
            f'{matched_text}</span>'
        )

    return pattern.sub(_replacement, text)


# ---------------------------------------------------------------------------
# Next Match widget — queries BigQuery for the upcoming WC2026 match
# ---------------------------------------------------------------------------

_next_match_cache: dict | None = None
_next_match_cache_ts: float = 0.0


def get_next_match_html() -> str:
    """Query BigQuery for the next upcoming World Cup 2026 match and return
    a styled HTML card with countdown, teams, venue, referee, and top players.

    Cached for 60 seconds to avoid hitting BigQuery on every Streamlit rerun.
    """
    global _next_match_cache, _next_match_cache_ts
    import time as _time

    now = _time.time()
    if _next_match_cache is not None and (now - _next_match_cache_ts) < 60:
        data = _next_match_cache
    else:
        data = _fetch_next_match_from_bq()
        _next_match_cache = data
        _next_match_cache_ts = now

    if data is None:
        return _no_match_fallback_html()

    return _build_next_match_card(data)


def _fetch_next_match_from_bq() -> dict | None:
    """Fetch the next WC2026 match from BigQuery fact_match table."""
    try:
        from src.tools.bigquery_tools import run_query

        project = os.environ.get("BIGQUERY_PROJECT_ID")
        dataset = os.environ.get("BIGQUERY_DATASET_ID")
        if not project or not dataset:
            return None

        # 1. Get the next match
        match_sql = f"""
            SELECT
                match_id, match_date, kickoff_at,
                home_team_name, away_team_name,
                venue_name, venue_city, referee_name,
                match_status, competition_round
            FROM `{project}.{dataset}.fact_match`
            WHERE competition_id = 1
              AND season_year = 2026
              AND match_status IN ('SCHEDULED', 'LIVE')
            ORDER BY kickoff_at ASC
            LIMIT 1
        """
        df = run_query(match_sql)
        if df.empty:
            return None

        row = df.iloc[0]
        home_team = str(row.get("home_team_name", ""))
        away_team = str(row.get("away_team_name", ""))
        home_flag = get_flag(home_team) or ""
        away_flag = get_flag(away_team) or ""

        result: dict = {
            "match_id": int(row.get("match_id", 0)),
            "match_date": str(row.get("match_date", "")),
            "kickoff_at": str(row.get("kickoff_at", "")),
            "home_team": home_team,
            "away_team": away_team,
            "home_flag": home_flag,
            "away_flag": away_flag,
            "venue_name": str(row.get("venue_name", "")),
            "venue_city": str(row.get("venue_city", "")),
            "referee_name": str(row.get("referee_name", "")),
            "match_status": str(row.get("match_status", "SCHEDULED")),
            "competition_round": str(row.get("competition_round", "")),
            "home_players": [],
            "away_players": [],
        }

        # 2. Get top 2 players per team from fact_player_match_stat
        players_sql = f"""
            WITH ranked AS (
                SELECT
                    fps.team_id, fps.player_id,
                    dp.player_name, dt.team_name,
                    SUM(fps.goal_contributions) AS total_gc,
                    ROW_NUMBER() OVER (
                        PARTITION BY fps.team_id
                        ORDER BY SUM(fps.goal_contributions) DESC
                    ) AS rn
                FROM `{project}.{dataset}.fact_player_match_stat` fps
                JOIN `{project}.{dataset}.dim_player` dp USING (player_id)
                JOIN `{project}.{dataset}.dim_team` dt ON dt.team_id = fps.team_id
                WHERE fps.competition_id = 1
                  AND fps.season_year = 2026
                  AND (LOWER(dt.team_name) = LOWER('{home_team.replace("'", "''")}')
                       OR LOWER(dt.team_name) = LOWER('{away_team.replace("'", "''")}'))
                GROUP BY fps.team_id, fps.player_id, dp.player_name, dt.team_name
            )
            SELECT team_name, player_name, total_gc, rn
            FROM ranked
            WHERE rn <= 2
            ORDER BY team_name, rn
        """
        try:
            pdf = run_query(players_sql)
            if not pdf.empty:
                for _, prow in pdf.iterrows():
                    tname = str(prow.get("team_name", "")).lower()
                    pname = str(prow.get("player_name", ""))
                    if tname == home_team.lower():
                        result["home_players"].append(pname)
                    elif tname == away_team.lower():
                        result["away_players"].append(pname)
        except Exception:
            pass  # Best-effort; card still looks good without players

        return result
    except Exception:
        return None


def _build_next_match_card(data: dict) -> str:
    """Build the HTML card for the next World Cup match."""
    # Parse kickoff time for countdown
    kickoff_str = data.get("kickoff_at", "")
    match_date_str = data.get("match_date", "")
    try:
        from datetime import datetime as _dt, timezone as _tz
        kt = _dt.fromisoformat(kickoff_str.replace("Z", "+00:00"))
        now_utc = _dt.now(_tz.utc)
        delta = kt - now_utc
        if delta.total_seconds() < 0:
            countdown_text = "⚡ LIVE NOW!"
            countdown_sub = ""
        else:
            days = delta.days
            hours = delta.seconds // 3600
            minutes = (delta.seconds % 3600) // 60
            parts = []
            if days > 0:
                parts.append(f"{days}d")
            if hours > 0 or days > 0:
                parts.append(f"{hours}h")
            parts.append(f"{minutes}m")
            countdown_text = " ".join(parts)
            countdown_sub = "until kick-off"
    except Exception:
        countdown_text = match_date_str[:10] if match_date_str else "TBD"
        countdown_sub = ""

    home_flag = data.get("home_flag", "")
    away_flag = data.get("away_flag", "")
    home_team = data["home_team"]
    away_team = data["away_team"]
    venue = data.get("venue_name", "TBD")
    city = data.get("venue_city", "")
    venue_full = f"{venue}, {city}" if city else venue
    referee = data.get("referee_name", "TBD")
    round_label = data.get("competition_round", "")

    # Build player sections
    def _player_row(players: list[str]) -> str:
        if not players:
            return '<span style="color:#888;font-size:0.8em;">TBA</span>'
        return " &nbsp;·&nbsp; ".join(
            f'<span style="font-size:0.85em;">⭐ {p}</span>' for p in players
        )

    home_players_html = _player_row(data.get("home_players", []))
    away_players_html = _player_row(data.get("away_players", []))

    round_line = f'<div style="font-size:0.75em;color:#aaa;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">{round_label}</div>' if round_label else ""

    return f"""
<div class="next-match-card">
    {round_line}
    <div class="next-match-countdown">{countdown_text}</div>
    <div class="next-match-countdown-sub">{countdown_sub}</div>

    <div class="next-match-teams">
        <div class="next-match-team">
            <span class="next-match-flag">{home_flag}</span>
            <span class="next-match-name">{home_team}</span>
            <div class="next-match-players">{home_players_html}</div>
        </div>
        <div class="next-match-vs">VS</div>
        <div class="next-match-team">
            <span class="next-match-flag">{away_flag}</span>
            <span class="next-match-name">{away_team}</span>
            <div class="next-match-players">{away_players_html}</div>
        </div>
    </div>

    <div class="next-match-meta">
        <div class="next-match-meta-row">📅 {match_date_str[:10] if match_date_str else 'TBD'}</div>
        <div class="next-match-meta-row">🏟️ {venue_full}</div>
        <div class="next-match-meta-row">🦓 {referee}</div>
    </div>
</div>
"""


def _no_match_fallback_html() -> str:
    """Fallback when no upcoming match is found."""
    return """
<div class="next-match-card">
    <div style="text-align:center;color:#aaa;padding:16px;">
        🏟️ No upcoming match found.<br>
        <small>Check back soon for the full schedule.</small>
    </div>
</div>
"""

    return pattern.sub(_replacement, text)
