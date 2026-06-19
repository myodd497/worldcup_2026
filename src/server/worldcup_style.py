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
    padding-left: 10% !important;
    padding-right: 10% !important;
    padding-bottom: 8.5rem !important;
    padding-top: 0.2rem !important;
    max-width: 100% !important;
    display: flex;
    flex-direction: column;
    align-items: center;
}}

/* ── Center the chat input area (as a card) ── */
[data-testid="stChatInput"] {{
    position: fixed;
    left: 50%;
    bottom: max(12px, env(safe-area-inset-bottom));
    transform: translateX(-50%);
    width: min(700px, calc(100vw - 2rem));
    margin: 0;
    background: linear-gradient(180deg, rgba(10, 31, 46, 0.4) 0%, rgba(15, 26, 46, 0.3) 100%);
    border: 1px solid rgba(255, 215, 0, 0.1);
    border-radius: 16px;
    padding: 8px 12px;
    z-index: 999;
}}

/* ── Chat area card wrapper ── */
[data-testid="stChatMessageContainer"] {{
    max-width: 700px;
    margin: 0 auto;
    background: linear-gradient(180deg, rgba(10, 31, 46, 0.4) 0%, rgba(15, 26, 46, 0.3) 100%);
    border: 1px solid rgba(255, 215, 0, 0.08);
    border-radius: 16px;
    padding: 16px;
    margin: 8px auto 12px auto;
    max-height: 55vh;
    overflow-y: auto;
}}

/* ── Subtitle inside chat card ── */
[data-testid="stChatMessageContainer"]::before {{
    content: "⚽ Your AI-powered football assistant — ask about matches, predictions, standings, and more!";
    display: block;
    text-align: center;
    color: #888;
    font-size: 0.78em;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    margin-bottom: 8px;
}}

/* ── Widen the main block container to use more screen width ── */
[data-testid="stMainBlockContainer"] {{
    max-width: 100% !important;
    padding-left: 10% !important;
    padding-right: 10% !important;
    padding-bottom: 8.5rem !important;
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

/* ── Next Match card (centered, compact vertical layout) ── */
.next-match-card {{
    background: linear-gradient(135deg, rgba(255, 215, 0, 0.08) 0%, rgba(10, 31, 46, 0.55) 100%);
    border: 1px solid rgba(255, 215, 0, 0.2);
    border-radius: 14px;
    padding: 14px 16px;
    max-width: 100%;
    text-align: center;
    height: 294px;
    min-height: 294px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: center;
    overflow-y: auto;
}}
.next-match-round {{
    font-size: 0.65em;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 6px;
}}
.next-match-countdown {{
    font-size: 1.3em;
    font-weight: 800;
    color: #f0c040;
    line-height: 1.2;
}}
.next-match-countdown-sub {{
    font-size: 0.68em;
    color: #777;
    margin-bottom: 10px;
}}
.next-match-teams {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    margin: 8px 0;
}}
.next-match-team {{
    text-align: center;
    min-width: 80px;
}}
.next-match-flag {{
    font-size: 1.4em;
    display: block;
}}
.next-match-name {{
    font-size: 0.78em;
    font-weight: 700;
    color: #e0e0e0;
    display: block;
    margin-top: 2px;
}}
.next-match-players {{
    margin-top: 2px;
    color: #ccc;
    font-size: 0.68em;
}}
.next-match-vs {{
    font-size: 0.6em;
    font-weight: 700;
    color: rgba(255, 215, 0, 0.4);
    align-self: center;
}}
.next-match-meta {{
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    font-size: 0.68em;
    color: #999;
    line-height: 1.6;
}}
.next-match-meta-row {{
    padding: 1px 0;
}}

/* ── Standings card (side-by-side with next match, wider) ── */
.standings-card {{
    background: linear-gradient(135deg, rgba(255, 215, 0, 0.08) 0%, rgba(10, 31, 46, 0.55) 100%);
    border: 1px solid rgba(255, 215, 0, 0.2);
    border-radius: 14px;
    padding: 14px 16px;
    max-width: 100%;
    text-align: center;
    overflow-x: auto;
    overflow-y: auto;
    height: 294px;
    min-height: 294px;
    box-sizing: border-box;
}}
.standings-header {{
    font-size: 0.7em;
    color: #f0c040;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 10px;
    font-weight: 700;
}}
.standings-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.68em;
    color: #ccc;
}}
.standings-table th {{
    color: #888;
    font-weight: 600;
    padding: 4px 5px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    white-space: nowrap;
}}
.standings-table td {{
    padding: 3px 5px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    white-space: nowrap;
}}
.standings-rank {{
    font-weight: 700;
    color: #aaa;
    width: 18px;
}}
.standings-team {{
    text-align: left !important;
    font-weight: 600;
    color: #e0e0e0;
}}
.standings-stat {{
    text-align: center;
    color: #aaa;
    width: 22px;
}}
.standings-pts {{
    text-align: center;
    font-weight: 800;
    color: #f0c040;
    width: 24px;
}}
.standings-row-qualify {{
    background: rgba(255, 215, 0, 0.04);
}}
.standings-row-qualify .standings-rank {{
    color: #f0c040;
}}

/* ── Top Scorers card ── */
.topscorers-card {{
    background: linear-gradient(135deg, rgba(255, 215, 0, 0.08) 0%, rgba(10, 31, 46, 0.55) 100%);
    border: 1px solid rgba(255, 215, 0, 0.2);
    border-radius: 14px;
    padding: 14px 16px;
    max-width: 100%;
    text-align: center;
    overflow-x: auto;
    overflow-y: auto;
    height: 294px;
    min-height: 294px;
    box-sizing: border-box;
}}
.topscorers-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.68em;
    color: #ccc;
}}
.topscorers-table th {{
    color: #888;
    font-weight: 600;
    padding: 4px 5px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    white-space: nowrap;
}}
.topscorers-table td {{
    padding: 3px 5px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    white-space: nowrap;
}}
.ts-rank {{
    font-weight: 700;
    color: #aaa;
    width: 28px;
    white-space: nowrap;
}}
.ts-player {{
    text-align: left !important;
    font-weight: 600;
    color: #e0e0e0;
}}
.ts-team {{
    text-align: left !important;
    color: #bbb;
    font-size: 0.9em;
}}
.ts-goals {{
    text-align: center;
    font-weight: 800;
    color: #f0c040;
    width: 28px;
}}

/* ── Card wrapper for side-by-side layout ── */
.cards-wrapper {{
    display: flex;
    align-items: flex-start;
    justify-content: center;
    gap: 16px;
    flex-wrap: wrap;
    margin: 4px 0 16px 0;
}}

/* ── Group selector buttons ── */
.standings-group-selector {{
    display: flex;
    justify-content: center;
    gap: 4px;
    flex-wrap: wrap;
    margin-top: 8px;
}}
.standings-group-btn {{
    background: rgba(255, 215, 0, 0.08);
    border: 1px solid rgba(255, 215, 0, 0.2);
    color: #ccc;
    padding: 3px 10px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.7em;
    font-weight: 600;
    transition: all 0.15s;
}}
.standings-group-btn:hover {{
    background: rgba(255, 215, 0, 0.18);
    color: #fff;
}}
.standings-group-btn.active {{
    background: rgba(255, 215, 0, 0.2);
    border-color: rgba(255, 215, 0, 0.5);
    color: #f0c040;
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

/* ════════════════════════════════════════════════════════════════════
   MOBILE OVERRIDES (phones / narrow viewports)
   - Shrinks the trophy + title so "World Cup 2026" stays on ONE line.
   - Reduces side padding so the chat card uses the full width.
   - Leaves room on the right of the chat input so the send button is
     not hidden behind Streamlit Cloud's floating "Manage app" button.
   ════════════════════════════════════════════════════════════════════ */
@media (max-width: 768px) {{
    /* Tighter side padding on the main block so content breathes */
    [data-testid="stAppViewBlockContainer"],
    [data-testid="stMainBlockContainer"] {{
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 7.5rem !important;
    }}

    /* Smaller title so it fits one line next to the trophy */
    h1 {{
        font-size: 1.8rem !important;
        line-height: 1.1 !important;
        white-space: nowrap;
    }}
    h2 {{ font-size: 1.3rem !important; }}
    h3 {{ font-size: 1.1rem !important; }}

    /* Shrink the trophy logo dramatically on mobile (was 10.5em) */
    .wc-title-icon {{
        width: 2.4em !important;
        height: 2.4em !important;
        margin-right: 0.25em !important;
    }}

    /* Chat input: keep full width, but lift it ABOVE Streamlit Cloud's
       floating control buttons (Manage app / +) so the send arrow is
       not covered. ~70px clearance is enough on iOS Safari. */
    [data-testid="stChatInput"] {{
        left: 50% !important;
        right: auto !important;
        transform: translateX(-50%) !important;
        width: calc(100vw - 1.5rem) !important;
        margin: 0 !important;
        bottom: calc(70px + env(safe-area-inset-bottom)) !important;
    }}

    /* Chat card uses available width */
    [data-testid="stChatMessageContainer"] {{
        max-width: 100% !important;
        max-height: 50vh;
    }}

    /* ── Chat OUTPUT (assistant + user bubbles) ── */
    /* Tighter padding, smaller font, and proper word-wrap so long names,
       URLs, or table-like answers don't overflow horizontally. */
    [data-testid="stChatMessage"] {{
        padding: 10px 12px !important;
        margin-bottom: 6px !important;
        font-size: 0.92rem !important;
        line-height: 1.45 !important;
    }}
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span {{
        font-size: 0.92rem !important;
        line-height: 1.45 !important;
        overflow-wrap: anywhere;
        word-break: break-word;
    }}
    [data-testid="stChatMessage"] h1 {{ font-size: 1.2rem !important; }}
    [data-testid="stChatMessage"] h2 {{ font-size: 1.05rem !important; }}
    [data-testid="stChatMessage"] h3 {{ font-size: 0.95rem !important; }}

    /* Shrink assistant avatar (the trophy) so it doesn't dominate the bubble */
    [data-testid="stChatMessageAvatar"] {{
        width: 28px !important;
        height: 28px !important;
        min-width: 28px !important;
        flex-shrink: 0 !important;
    }}

    /* Tables / code blocks inside replies: scroll horizontally instead of
       breaking the layout. */
    [data-testid="stChatMessage"] table,
    [data-testid="stChatMessage"] pre {{
        display: block;
        max-width: 100%;
        overflow-x: auto;
        font-size: 0.8rem !important;
    }}
    [data-testid="stChatMessage"] pre code {{
        white-space: pre;
    }}

    /* Inline player image cards: scale down so 2-3 fit per line */
    .player-card {{
        font-size: 0.85em !important;
        padding: 4px 8px 4px 4px !important;
        gap: 6px !important;
    }}
    .player-card img {{
        width: 28px !important;
        height: 28px !important;
    }}

    /* Confidence stars: a touch smaller */
    .confidence-stars {{
        font-size: 0.95em !important;
    }}

    /* Dashboard cards stack vertically and get a sensible height */
    .next-match-card,
    .standings-card,
    .topscorers-card {{
        height: auto !important;
        min-height: 0 !important;
    }}
}}

/* Extra-small phones */
@media (max-width: 420px) {{
    h1 {{ font-size: 1.5rem !important; }}
    .wc-title-icon {{
        width: 2em !important;
        height: 2em !important;
    }}
    [data-testid="stChatInput"] {{
        margin: 0 !important;
        width: calc(100vw - 1rem) !important;
        bottom: calc(70px + env(safe-area-inset-bottom)) !important;
    }}
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
# Rotating "Thinking: <football fact>" spinner label
# ---------------------------------------------------------------------------
# A curated list of well-known, verified football facts. Heavily weighted
# toward the last decade (2014–2025). Rotated client-side every 3 seconds
# inside any Streamlit spinner whose label starts with "Thinking" — purely
# JS/CSS, no extra backend or network cost.

_FOOTBALL_FACTS: list[str] = [
    # ── World Cup (recent) ───────────────────────────────────────────────
    "Argentina won the 2022 World Cup, beating France 4-2 on penalties.",
    "Lionel Messi finally lifted the World Cup in 2022 at age 35.",
    "France won the 2018 World Cup in Russia, beating Croatia 4-2.",
    "Kylian Mbappé scored a hat-trick in the 2022 World Cup final and still lost.",
    "Germany won the 2014 World Cup in Brazil, beating Argentina 1-0.",
    "Mario Götze scored the winning goal in the 2014 World Cup final.",
    "Brazil lost 7-1 to Germany in the 2014 World Cup semi-final at home.",
    "Morocco became the first African team to reach a World Cup semi-final in 2022.",
    "Saudi Arabia stunned Argentina 2-1 in their 2022 World Cup opener.",
    "Japan beat both Germany and Spain in the 2022 World Cup group stage.",
    "Croatia reached the 2018 World Cup final, losing 4-2 to France.",
    "Luka Modrić won the 2018 World Cup Golden Ball.",
    "James Rodríguez was the 2014 World Cup top scorer with 6 goals.",
    "Harry Kane won the 2018 World Cup Golden Boot with 6 goals.",
    "Mbappé won the 2022 World Cup Golden Boot with 8 goals.",
    "Emiliano Martínez was named best goalkeeper at the 2022 World Cup.",
    "The 2026 World Cup will be the first hosted by three countries: USA, Canada, Mexico.",
    "The 2026 World Cup will be the first with 48 teams.",
    "The 2022 World Cup in Qatar was the first held in winter.",
    "VAR was first used at a World Cup in Russia 2018.",
    # ── Champions League (recent) ────────────────────────────────────────
    "Real Madrid won the Champions League in 2014, 2016, 2017, 2018, 2022, and 2024.",
    "Real Madrid have won 15 European Cups — more than any other club.",
    "Real Madrid won three Champions Leagues in a row (2016, 2017, 2018).",
    "Manchester City won their first Champions League in 2023.",
    "Liverpool won the Champions League in 2019, beating Tottenham 2-0.",
    "Liverpool came back from 3-0 down to beat Barcelona 4-0 in the 2019 semi-final.",
    "Bayern Munich won the 2020 Champions League with a 1-0 final vs PSG.",
    "Chelsea won the 2021 Champions League, beating Manchester City 1-0.",
    "Karim Benzema won the 2022 Ballon d'Or after carrying Real Madrid to the title.",
    "Rodrygo scored two stoppage-time goals to knock out Man City in 2022.",
    "Real Madrid eliminated PSG, Chelsea, City, and Liverpool to win the 2022 UCL.",
    "Vinícius Júnior scored the winner in the 2022 Champions League final.",
    "Karim Benzema scored a hat-trick at the Bernabéu vs PSG in 2022.",
    "Cristiano Ronaldo is the all-time top scorer in Champions League history.",
    "Lionel Messi is second on the all-time Champions League scoring list.",
    "Real Madrid's 2023/24 Champions League win was Carlo Ancelotti's 5th as a coach.",
    # ── Messi & Ronaldo era ──────────────────────────────────────────────
    "Lionel Messi has won the Ballon d'Or 8 times — more than anyone else.",
    "Cristiano Ronaldo has won the Ballon d'Or 5 times.",
    "Messi scored 91 goals in a calendar year (2012) — still a world record.",
    "Cristiano Ronaldo is the all-time top scorer in men's international football.",
    "Ronaldo scored his 900th career goal in 2024.",
    "Messi joined Inter Miami in 2023 after leaving PSG.",
    "Messi won the Leagues Cup with Inter Miami in his first tournament.",
    "Cristiano Ronaldo joined Al-Nassr in January 2023.",
    "Ronaldo left Manchester United in November 2022 after the Piers Morgan interview.",
    "Messi left Barcelona in 2021 due to La Liga's salary cap rules.",
    "Cristiano Ronaldo scored 5 goals in a single Champions League match for Real Madrid.",
    "Messi scored 5 goals in a Champions League match vs Bayer Leverkusen in 2012.",
    "Messi has the record for most goals at a single club: 672 for Barcelona.",
    "Cristiano Ronaldo is the only player to score in five different World Cups.",
    "Messi played 1000+ career club games.",
    # ── Euros & Copa América ────────────────────────────────────────────
    "Spain won Euro 2024, beating England 2-1 in the final.",
    "Italy won Euro 2020, beating England on penalties at Wembley.",
    "Portugal won Euro 2016 despite losing Ronaldo to injury in the final.",
    "Éder scored the Euro 2016 winning goal for Portugal vs France.",
    "Spain's Lamine Yamal became the youngest scorer in Euros history in 2024 at 16.",
    "Cody Gakpo and Dani Olmo shared the Euro 2024 Golden Boot with 3 goals.",
    "Argentina won the 2021 Copa América — Messi's first international trophy.",
    "Argentina won the 2024 Copa América, beating Colombia 1-0.",
    "Lautaro Martínez was the 2024 Copa América top scorer with 5 goals.",
    "Spain has now won Euro 2008, 2012, and 2024.",
    "Italy missed the 2018 AND 2022 World Cups despite winning Euro 2020.",
    "Greece's Euro 2004 win is still considered the biggest Euros shock.",
    # ── Premier League (recent) ─────────────────────────────────────────
    "Manchester City have won 6 of the last 7 Premier League titles.",
    "City won four Premier League titles in a row (2021–2024) — a league first.",
    "Leicester City won the Premier League in 2016 at 5000-1 pre-season odds.",
    "Liverpool ended their 30-year league wait by winning the 2019/20 Premier League.",
    "Manchester City completed the treble in 2022/23: PL, FA Cup, Champions League.",
    "Erling Haaland scored 36 league goals in his debut Premier League season.",
    "Mohamed Salah scored 32 league goals in 2017/18 — a 38-game PL record.",
    "Harry Kane joined Bayern Munich in 2023 for around €100m.",
    "Pep Guardiola has managed Man City since 2016.",
    "Jürgen Klopp left Liverpool in May 2024 after almost 9 years.",
    "Arne Slot replaced Klopp at Liverpool in 2024.",
    "Mikel Arteta took over Arsenal in December 2019.",
    "Erik ten Hag was sacked by Manchester United in October 2024.",
    "Newcastle were bought by a Saudi-led consortium in October 2021.",
    "Roman Abramovich sold Chelsea to Todd Boehly in 2022 due to UK sanctions.",
    "Chelsea spent over £1 billion on transfers under Boehly in 18 months.",
    "Declan Rice joined Arsenal from West Ham for £105m in 2023.",
    "Jack Grealish joined Man City for £100m in 2021 — first English £100m player.",
    "VAR was introduced to the Premier League in the 2019/20 season.",
    # ── La Liga & big transfers ─────────────────────────────────────────
    "Real Madrid won La Liga in 2024 with five games to spare.",
    "Jude Bellingham scored 19 La Liga goals in his debut Real Madrid season.",
    "Kylian Mbappé joined Real Madrid on a free transfer in summer 2024.",
    "Neymar's €222m move from Barcelona to PSG in 2017 is still a world record.",
    "Mbappé joined PSG from Monaco for €180m in 2018.",
    "Neymar joined Al-Hilal from PSG in 2023.",
    "Barcelona suffered a historic 8-2 defeat to Bayern Munich in 2020.",
    "Sergio Ramos left Real Madrid in 2021 after 16 years at the club.",
    "Carlo Ancelotti returned to Real Madrid in 2021 for a second spell.",
    "Real Madrid signed Eduardo Camavinga, Tchouameni, Bellingham, and Mbappé in 4 years.",
    # ── Other leagues & cups ─────────────────────────────────────────────
    "Bayer Leverkusen won their first-ever Bundesliga title in 2024 — unbeaten.",
    "Xabi Alonso led Leverkusen to that unbeaten Bundesliga title in his first full season.",
    "Bayern Munich's 11-year Bundesliga title streak ended in 2024.",
    "Napoli won the 2022/23 Serie A — their first title since Maradona in 1990.",
    "Inter Milan won Serie A in 2024 with five games to spare.",
    "Atalanta won the 2024 Europa League — their first major European trophy.",
    "PSG have dominated Ligue 1, winning 11 titles since 2013.",
    "Boca Juniors and River Plate met in the Copa Libertadores final in 2018.",
    "The 2018 Libertadores final second leg was played at the Bernabéu after fan violence.",
    "Flamengo won the Copa Libertadores in 2019 and 2022.",
    # ── Records & individual feats ──────────────────────────────────────
    "Erling Haaland scored 5 goals in a Champions League match vs RB Leipzig in 2023.",
    "Robert Lewandowski broke Gerd Müller's Bundesliga single-season record with 41 goals (2021).",
    "Karim Benzema scored 44 goals for Real Madrid in 2021/22.",
    "Cristiano Ronaldo became the first player to score 100 international goals (UEFA) in 2020.",
    "Iker Casillas won the World Cup, two Euros, and three Champions Leagues.",
    "Sergio Busquets won every major trophy at club and international level.",
    "Manuel Neuer redefined the sweeper-keeper role at Bayern and Germany.",
    "Thiago Silva is still playing top-level football well into his late 30s.",
    "Luka Modrić won the 2018 Ballon d'Or, ending the Messi-Ronaldo duopoly.",
    "Rodri won the 2024 Ballon d'Or after Spain's Euro win and City's PL title.",
    "Aitana Bonmatí won the 2023 and 2024 women's Ballon d'Or.",
    "Alexia Putellas won back-to-back Ballons d'Or in 2021 and 2022.",
    # ── Women's football ────────────────────────────────────────────────
    "Spain won the 2023 Women's World Cup, beating England 1-0.",
    "USA won the 2015 and 2019 Women's World Cups.",
    "Megan Rapinoe was the 2019 Women's World Cup Golden Ball winner.",
    "England won Euro 2022, beating Germany 2-1 at Wembley.",
    "Chloe Kelly scored the Euro 2022 extra-time winner for England.",
    "The 2023 Women's World Cup was co-hosted by Australia and New Zealand.",
    "Sam Kerr is Australia's all-time top scorer in international football.",
    "Marta has scored at five different Women's World Cups.",
    "Barcelona Femení won three of the last four Women's Champions Leagues.",
    "Aitana Bonmatí scored the only goal in Spain's 2023 World Cup final win.",
    # ── Historic / older facts ──────────────────────────────────────────
    "Brazil have won the World Cup five times — more than any other nation.",
    "Pelé is the only player to win three World Cups (1958, 1962, 1970).",
    "Diego Maradona's 'Hand of God' goal came vs England in the 1986 World Cup.",
    "Maradona then scored the 'Goal of the Century' four minutes later.",
    "The 1950 'Maracanazo' saw Uruguay beat Brazil 2-1 in the World Cup final.",
    "Germany's 1954 'Miracle of Bern' upset Hungary's mighty Magical Magyars.",
    "Italy won the World Cup four times (1934, 1938, 1982, 2006).",
    "France's Zinedine Zidane was sent off in his last-ever match — the 2006 final.",
    "Spain won their first World Cup in 2010 with Andrés Iniesta's extra-time goal.",
    "AC Milan went 58 league matches unbeaten between 1991 and 1993.",
    "Arsenal's 'Invincibles' went the entire 2003/04 Premier League season unbeaten.",
    "Sir Alex Ferguson won 13 Premier League titles with Manchester United.",
    "José Mourinho went 9 years unbeaten at home in league play.",
    "Johan Cruyff revolutionised football as a player and coach.",
    "The 'Cruyff Turn' was invented at the 1974 World Cup.",
    "Pep Guardiola's Barcelona won 14 of 19 possible trophies between 2008–2012.",
    "Liverpool's 'miracle of Istanbul' in 2005: came back from 3-0 down vs AC Milan to win on penalties.",
    "Manchester United won the Premier League, FA Cup, and Champions League treble in 1999.",
    "Barcelona did the sextuple in 2009 under Guardiola.",
    "Bayern Munich did the sextuple in 2020 under Hansi Flick.",
    # ── Tactical / quirky facts ─────────────────────────────────────────
    "A football match is 90 minutes long, plus stoppage time.",
    "The pitch can be 100–110m long and 64–75m wide for international games.",
    "Goalkeepers are the only players allowed to handle the ball in open play.",
    "A red card means immediate ejection — no substitute allowed.",
    "VAR can only review goals, penalties, red cards, and mistaken identity.",
    "The offside rule was modified in 2005 to be 'active vs passive.'",
    "Five substitutions per match became permanent post-COVID in most competitions.",
    "Concussion substitutes were introduced in 2020.",
    "The Premier League's record transfer is Enzo Fernández at £107m (2023).",
    "Real Madrid have never been relegated from La Liga.",
    "Athletic Bilbao only sign Basque players and have never been relegated.",
    # ── Notable goalkeepers ─────────────────────────────────────────────
    "Thibaut Courtois won the 2022 UCL final almost single-handedly vs Liverpool.",
    "Alisson Becker scored a header for Liverpool vs West Brom in 2021.",
    "Ederson set up multiple goals from his own box for Manchester City.",
    "Gianluigi Donnarumma was Euro 2020 Player of the Tournament at age 22.",
    "Emiliano Martínez saved a penalty in the last minute of extra time in the 2022 WC final.",
    "Yashin (1963) is the only goalkeeper to win the Ballon d'Or.",
    # ── Stadiums & atmosphere ───────────────────────────────────────────
    "Camp Nou's capacity is 99,354 — Europe's largest club stadium.",
    "Wembley Stadium has hosted three Champions League finals (2011, 2013, 2024).",
    "Anfield's 'You'll Never Walk Alone' is one of football's most iconic moments.",
    "El Clásico is Real Madrid vs Barcelona — football's biggest club rivalry.",
    "The Old Firm derby (Celtic vs Rangers) is one of football's fiercest rivalries.",
    "The Maracanã holds 78,838 — once held nearly 200,000 for the 1950 World Cup final.",
    # ── Coaches ─────────────────────────────────────────────────────────
    "Pep Guardiola has won league titles in Spain, Germany, and England.",
    "José Mourinho won the Champions League with Porto (2004) and Inter (2010).",
    "Carlo Ancelotti is the only coach to win the Champions League 5 times.",
    "Ancelotti is the only coach to win league titles in all top 5 European leagues.",
    "Klopp won the Champions League with Liverpool in 2019.",
    "Antonio Conte won three consecutive Serie A titles with Juventus (2012–2014).",
    "Diego Simeone has managed Atlético Madrid since December 2011.",
    "Zinedine Zidane won three consecutive Champions Leagues as Real Madrid coach.",
    # ── Recent transfers & moves ───────────────────────────────────────
    "Saudi Pro League signed Ronaldo, Neymar, Benzema, and Mané in 2023.",
    "Florian Wirtz, Jamal Musiala, and Lamine Yamal are leading the next-gen wave.",
    "Pedri and Gavi are Barcelona's midfield future, both Spain internationals.",
    "Cole Palmer scored 22 PL goals in his first Chelsea season (2023/24).",
    "Phil Foden was named Premier League Player of the Season in 2023/24.",
    "Bukayo Saka has been Arsenal's most consistent attacker post-Aubameyang.",
    "Jude Bellingham joined Real Madrid from Dortmund for €103m in 2023.",
    "Joshua Kimmich is one of the most versatile players in world football.",
    "Vinícius Júnior finished second in the 2024 Ballon d'Or, behind Rodri.",
    # ── Misc trivia ─────────────────────────────────────────────────────
    "Football is the most-watched sport in the world.",
    "The first official international match was Scotland 0-0 England in 1872.",
    "FIFA was founded in 1904.",
    "UEFA was founded in 1954.",
    "The Champions League was rebranded from the European Cup in 1992.",
    "Goal-line technology was first used at a World Cup in 2014.",
    "Penalty shootouts were introduced to the World Cup in 1978.",
    "Brazil's Ronaldo (R9) won the 2002 World Cup Golden Boot with 8 goals.",
    "Miroslav Klose is the all-time World Cup top scorer with 16 goals.",
    "Just Fontaine scored 13 goals at a single World Cup (1958) — still a record.",
    "Cristiano Ronaldo holds the record for most international caps for a European man.",
    "Bad Bunny performed at the 2026 World Cup-themed promo events.",
    "The Golden Boot is awarded to a World Cup's top scorer.",
    "The Golden Ball goes to a World Cup's best player.",
    "The Golden Glove goes to a World Cup's best goalkeeper.",
    "The 2030 World Cup will be co-hosted by Spain, Portugal, and Morocco.",
    "Three group-stage games of the 2030 World Cup will be played in South America.",
    "The 2034 World Cup will be hosted by Saudi Arabia.",
    "The Ballon d'Or moved its calendar in 2022 to follow the European season.",
    "The FIFA Best award is separate from the Ballon d'Or, voted by national coaches and captains.",
    # ── Refereeing & rules ──────────────────────────────────────────────
    "A drop ball restart was changed in 2019 — possession now goes to the team that last touched it.",
    "Handball rules have been clarified multiple times since 2019.",
    "Stoppage time at the 2022 World Cup was significantly longer due to a new FIFA directive.",
    "Semi-automated offside technology debuted at the 2022 World Cup.",
    "The 'Decisive Goal in Open Play' tiebreaker was used in 2022 group stages.",
    # ── Last decade highlights ──────────────────────────────────────────
    "Atlético Madrid won La Liga in 2014 and 2021 under Diego Simeone.",
    "Sevilla have won the Europa League 7 times — more than any other club.",
    "PSG reached their first Champions League final in 2020.",
    "Tottenham reached their first Champions League final in 2019.",
    "Ajax's run to the 2019 UCL semi-final reminded fans of their golden era.",
    "Frenkie de Jong, Matthijs de Ligt, and Donny van de Beek emerged from that Ajax side.",
    "Erling Haaland's 2019 Champions League hat-trick vs Atalanta announced his arrival.",
    "Kai Havertz scored Chelsea's 2021 Champions League final winner.",
    "Karim Benzema scored 15 goals in the 2021/22 Champions League knockouts.",
    "Real Madrid's 2022 UCL run is one of the most dramatic ever.",
]


THINKING_FACTS_HTML = (
    "<script>\n"
    "(function() {\n"
    "  var FACTS = " + __import__("json").dumps(_FOOTBALL_FACTS, ensure_ascii=False) + ";\n"
    "  // We are inside a Streamlit components iframe. Target the parent document.\n"
    "  var doc = window.parent && window.parent.document ? window.parent.document : document;\n"
    "  function pickFact(prev) {\n"
    "    var i = Math.floor(Math.random() * FACTS.length);\n"
    "    if (FACTS[i] === prev && FACTS.length > 1) i = (i + 1) % FACTS.length;\n"
    "    return FACTS[i];\n"
    "  }\n"
    "  var activeIntervals = new WeakMap();\n"
    "  function attach(spinnerEl) {\n"
    "    if (activeIntervals.has(spinnerEl)) return;\n"
    "    var txt = (spinnerEl.textContent || '').trim();\n"
    "    if (txt.indexOf('Thinking') === -1) return;\n"
    "    var label = spinnerEl.querySelector('div, span, p') || spinnerEl;\n"
    "    var current = pickFact(null);\n"
    "    label.textContent = '🏆 Thinking: ' + current;\n"
    "    var id = setInterval(function() {\n"
    "      if (!doc.body.contains(spinnerEl)) {\n"
    "        clearInterval(id);\n"
    "        return;\n"
    "      }\n"
    "      current = pickFact(current);\n"
    "      label.textContent = '🏆 Thinking: ' + current;\n"
    "    }, 3000);\n"
    "    activeIntervals.set(spinnerEl, id);\n"
    "  }\n"
    "  function scan() {\n"
    "    doc.querySelectorAll('[data-testid=\"stSpinner\"]').forEach(attach);\n"
    "  }\n"
    "  var observer = new MutationObserver(scan);\n"
    "  observer.observe(doc.body, { childList: true, subtree: true });\n"
    "  scan();\n"
    "})();\n"
    "</script>\n"
)


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
# Player images: lookup via dim_player in BigQuery
# ---------------------------------------------------------------------------

# Cache for BigQuery player lookups to avoid repeated queries
_player_image_cache: dict[str, str] = {}


def get_player_image_url(player_name: str) -> str | None:
    """Look up a player's image URL from dim_player in BigQuery.

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
        pass  # BigQuery unavailable

    # Mark as not found to avoid repeated lookups
    _player_image_cache[key] = "__NONE__"
    return None


def inject_player_images(text: str) -> str:
    """Scan text for known player names and embed inline player image cards.

    Only matches player names already cached from a prior BigQuery lookup.
    Adds the image at most once per player per message.
    """
    if not text:
        return text

    known_players = {
        k for k, v in _player_image_cache.items() if v != "__NONE__"
    }
    if not known_players:
        return text

    # Sort by name length descending so longer names match first
    sorted_players = sorted(known_players, key=lambda n: (-len(n), n))

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

    round_line = f'<div class="next-match-round">{round_label}</div>' if round_label else ""

    return (
f"""<div class="next-match-card">
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
</div>"""
    )


def _no_match_fallback_html() -> str:
    """Fallback when no upcoming match is found."""
    return (
        '<div class="next-match-card">'
        '<div style="text-align:center;color:#aaa;padding:16px;">'
        '🏟️ No upcoming match found.<br>'
        '<small>Check back soon for the full schedule.</small>'
        '</div>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Standings card — queries BigQuery for group standings with group selector
# ---------------------------------------------------------------------------

_standings_cache: dict[str, list[dict]] = {}
_standings_cache_ts: float = 0.0
_standings_groups_cache: list[str] = []


def get_standings_groups() -> list[str]:
    """Return available group names for WC2026."""
    global _standings_groups_cache, _standings_cache_ts
    import time as _time
    now = _time.time()
    if _standings_groups_cache and (now - _standings_cache_ts) < 300:
        return _standings_groups_cache

    try:
        from src.tools.bigquery_tools import run_query
        project = os.environ.get("BIGQUERY_PROJECT_ID")
        dataset = os.environ.get("BIGQUERY_DATASET_ID")
        if not project or not dataset:
            return ["A", "B", "C", "D", "E", "F", "G", "H"]

        sql = f"""
            SELECT DISTINCT group_name
            FROM `{project}.{dataset}.fact_standings_snapshot`
            WHERE competition_id = 1 AND season_year = 2026
            ORDER BY group_name
        """
        df = run_query(sql)
        if df.empty:
            return ["A", "B", "C", "D", "E", "F", "G", "H"]
        _standings_groups_cache = [str(g) for g in df["group_name"].tolist()]
        _standings_cache_ts = now
        return _standings_groups_cache
    except Exception:
        return ["A", "B", "C", "D", "E", "F", "G", "H"]


def get_standings_html(group_name: str) -> str:
    """Query BigQuery for standings of a given group and return styled HTML card."""
    try:
        from src.tools.bigquery_tools import run_query
        project = os.environ.get("BIGQUERY_PROJECT_ID")
        dataset = os.environ.get("BIGQUERY_DATASET_ID")
        if not project or not dataset:
            return _no_standings_fallback_html()

        sql = f"""
            SELECT
                standing_rank, team_name,
                points, played, wins, draws, losses, goals_for, goals_against, goal_diff
            FROM `{project}.{dataset}.mart_tournament_state`
            WHERE competition_id = 1
              AND season_year = 2026
              AND LOWER(group_name) = LOWER('{group_name.replace("'", "''")}')
            ORDER BY standing_rank ASC
        """
        df = run_query(sql)
        if df.empty:
            return _no_standings_fallback_html()

        rows = df.to_dict(orient="records")
        return _build_standings_card(group_name, rows)
    except Exception:
        return _no_standings_fallback_html()


def _build_standings_card(group_name: str, rows: list[dict]) -> str:
    """Build an HTML standings card for a group."""
    rows_html = ""
    for i, row in enumerate(rows):
        rank = int(row.get("standing_rank", i + 1))
        team = str(row.get("team_name", ""))
        flag = get_flag(team) or ""
        pts = int(row.get("points", 0) or 0)
        mp = int(row.get("played", 0) or 0)
        w = int(row.get("wins", 0) or 0)
        d = int(row.get("draws", 0) or 0)
        l = int(row.get("losses", 0) or 0)
        gf = int(row.get("goals_for", 0) or 0)
        ga = int(row.get("goals_against", 0) or 0)
        gd = int(row.get("goal_diff", 0) or 0)
        gd_str = f"+{gd}" if gd > 0 else str(gd)

        # Highlight top 2 (qualifying spots) with a subtle gold tint
        row_class = "standings-row-qualify" if rank <= 2 else ""
        rows_html += (
            f'<tr class="{row_class}">'
            f'<td class="standings-rank">{rank}</td>'
            f'<td class="standings-team">{flag} {team}</td>'
            f'<td class="standings-stat">{mp}</td>'
            f'<td class="standings-stat">{w}</td>'
            f'<td class="standings-stat">{d}</td>'
            f'<td class="standings-stat">{l}</td>'
            f'<td class="standings-stat">{gf}</td>'
            f'<td class="standings-stat">{ga}</td>'
            f'<td class="standings-stat">{gd_str}</td>'
            f'<td class="standings-pts">{pts}</td>'
            f'</tr>'
        )

    return f"""<div class="standings-card">
    <table class="standings-table">
        <thead>
            <tr>
                <th>#</th><th>Team</th><th>MP</th>
                <th>W</th><th>D</th><th>L</th>
                <th>GF</th><th>GA</th><th>GD</th><th>Pts</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
</div>"""


def _no_standings_fallback_html() -> str:
    return (
        '<div class="standings-card">'
        '<div style="text-align:center;color:#aaa;padding:16px;">'
        '📊 Standings not available yet.<br>'
        '<small>Check back once matches begin.</small>'
        '</div>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Top Scorers card — queries BigQuery for the tournament's top players by metric
# ---------------------------------------------------------------------------

_TOP_SCORER_METRICS = {
    "goals": ("Goals", "Gls", "goals"),
    "assists": ("Assists", "Ast", "assists"),
    "goal contributions": ("Goal Contrib.", "G+A", "goal_contributions"),
    "minutes played": ("Minutes", "Mins", "minutes_played"),
    "passes": ("Passes", "Pass", "passes_total"),
    "passes accurate": ("Pass Acc.", "PAcc", "passes_accurate"),
    "key passes": ("Key Passes", "KP", "key_passes"),
    "shots": ("Shots", "Sh", "shots_total"),
    "shots on target": ("Shots on Tgt", "SoT", "shots_on_target"),
    "interceptions": ("Intercept.", "Int", "interceptions"),
    "fouls committed": ("Fouls Cmt", "FC", "fouls_committed"),
    "fouls drawn": ("Fouls Drn", "FD", "fouls_drawn"),
    "yellow cards": ("Yellow C.", "YC", "yellow_cards"),
    "red cards": ("Red C.", "RC", "red_cards"),
    "saves": ("Saves", "Sv", "saves"),
    "goals conceded": ("Goals Conc.", "GC", "goals_conceded"),
    "rating": ("Rating", "Rat", "rating"),
    "dribbles total": ("Dribbles", "Drb", "dribbles_total"),
    "dribbles success": ("Drb. Succ", "DrbS", "dribbles_success"),
    "dribbles success %": ("Drb. Succ%", "Drb%", "SAFE_DIVIDE(dribbles_success, dribbles_total)"),
    "penalty scored": ("Pen. Scored", "PK+", "penalty_scored"),
    "penalty missed": ("Pen. Missed", "PK-", "penalty_missed"),
}

_top_scorers_cache: dict[str, tuple[str, list[dict]]] = {}  # metric -> (col_header, rows)
_top_scorers_cache_ts: float = 0.0


def get_top_scorer_metrics() -> list[str]:
    """Return list of available metric display names."""
    return list(_TOP_SCORER_METRICS.keys())


def get_top_scorers_html(metric: str = "goals") -> str:
    """Query BigQuery for the top 10 players by metric in WC2026 and return styled HTML card."""
    global _top_scorers_cache, _top_scorers_cache_ts
    import time as _time

    metric = metric.lower()
    if metric not in _TOP_SCORER_METRICS:
        metric = "goals"

    metric_label, col_header, db_column = _TOP_SCORER_METRICS[metric]

    now = _time.time()
    cache_key = f"{metric}"
    if cache_key in _top_scorers_cache and (now - _top_scorers_cache_ts) < 60:
        cached_label, rows = _top_scorers_cache[cache_key]
    else:
        rows = _fetch_top_by_metric_from_bq(db_column)
        _top_scorers_cache[cache_key] = (metric_label, rows)
        _top_scorers_cache_ts = now

    if not rows:
        return _no_top_scorers_fallback_html()

    return _build_top_scorers_card(rows, metric_label, col_header)


def _fetch_top_by_metric_from_bq(db_column: str) -> list[dict]:
    """Fetch top players by a given metric from fact_player_match_stat.
    
    Before June 11, 2026: uses last 2 friendly matches for all WC2026 teams.
    On/after June 11, 2026: queries all WC2026 matches.
    
    The db_column can be a raw column name or a calculated expression
    (e.g. 'SAFE_DIVIDE(dribbles_success, dribbles_total)' for dribble success %).
    """
    try:
        from src.tools.bigquery_tools import run_query
        project = os.environ.get("BIGQUERY_PROJECT_ID")
        dataset = os.environ.get("BIGQUERY_DATASET_ID")
        if not project or not dataset:
            return []

        today = date.today()
        wc_start = date(2026, 6, 11)

        if today < wc_start:
            # Last 2 friendly matches for all WC2026 teams
            where_clause = f"""AND fps.match_id IN (
                WITH wc_teams AS (
                    SELECT team_id, team_name
                    FROM `{project}.{dataset}.dim_team`
                    WHERE is_wc2026_participant = TRUE
                ),
                team_matches AS (
                    SELECT
                        fm.match_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY wc.team_id
                            ORDER BY fm.match_date DESC, fm.match_id DESC
                        ) AS rn
                    FROM `{project}.{dataset}.fact_match` fm
                    JOIN wc_teams wc
                        ON wc.team_id = fm.home_team_id OR wc.team_id = fm.away_team_id
                    WHERE fm.competition_id != 1
                        AND fm.match_status = 'FINISHED'
                        AND fm.match_date < '2026-06-11'
                        AND fm.home_goals IS NOT NULL
                )
                SELECT match_id FROM team_matches WHERE rn <= 2
            )"""
        else:
            where_clause = "AND fps.competition_id = 1 AND fps.season_year = 2026"

        # Determine whether db_column is a raw column or a calculated expression
        is_calculated = "(" in db_column  # e.g. SAFE_DIVIDE(x, y)

        if is_calculated:
            # For calculated metrics, we need a subquery first
            sql = f"""
                SELECT
                    player_name, team_name,
                    SUM(calc_value) AS total_value,
                    COUNT(*) AS matches
                FROM (
                    SELECT
                        dp.player_name,
                        dt.team_name,
                        {db_column} AS calc_value
                    FROM `{project}.{dataset}.fact_player_match_stat` fps
                    JOIN `{project}.{dataset}.dim_player` dp USING (player_id)
                    JOIN `{project}.{dataset}.dim_team` dt ON dt.team_id = fps.team_id
                    WHERE 1=1
                      {where_clause}
                      AND fps.dribbles_total > 0
                )
                GROUP BY player_name, team_name
                ORDER BY total_value DESC
                LIMIT 100
            """
        else:
            sql = f"""
                SELECT
                    dp.player_name,
                    dt.team_name,
                    SUM(fps.{db_column}) AS total_value,
                    COUNT(*) AS matches
                FROM `{project}.{dataset}.fact_player_match_stat` fps
                JOIN `{project}.{dataset}.dim_player` dp USING (player_id)
                JOIN `{project}.{dataset}.dim_team` dt ON dt.team_id = fps.team_id
                WHERE 1=1
                  {where_clause}
                  AND fps.{db_column} > 0
                GROUP BY dp.player_name, dt.team_name
                ORDER BY total_value DESC
                LIMIT 100
            """
        df = run_query(sql)
        if df.empty:
            return []
        return df.to_dict(orient="records")
    except Exception:
        return []


def _build_top_scorers_card(rows: list[dict], metric_label: str, col_header: str) -> str:
    """Build an HTML card for top players by metric."""
    rows_html = ""
    for i, row in enumerate(rows):
        rank = i + 1
        player = str(row.get("player_name", ""))
        team = str(row.get("team_name", ""))
        flag = get_flag(team) or ""
        value = row.get("total_value", 0)
        # Format value nicely
        is_pct_metric = "pct" in metric_label.lower() or "%" in metric_label.lower() or "dribbles success %" in metric_label.lower()
        if isinstance(value, float):
            if is_pct_metric:
                # For percentage metrics, multiply by 100 since SAFE_DIVIDE returns a decimal 0-1
                val_str = f"{value * 100:.1f}%"
            elif "rating" in metric_label.lower():
                val_str = f"{value:.2f}"
            else:
                val_str = f"{value:.1f}" if value != int(value) else str(int(value))
        else:
            val_str = str(int(value)) if value else "0"

        medal = ""
        if rank == 1: medal = "🥇 "
        elif rank == 2: medal = "🥈 "
        elif rank == 3: medal = "🥉 "

        rows_html += (
            f'<tr>'
            f'<td class="ts-rank">{medal}{rank}</td>'
            f'<td class="ts-player">⚽ {player}</td>'
            f'<td class="ts-team">{flag} {team}</td>'
            f'<td class="ts-goals">{val_str}</td>'
            f'</tr>'
        )

    return f"""<div class="topscorers-card">
    <table class="topscorers-table">
        <thead>
            <tr>
                <th>#</th><th>Player</th><th>Team</th><th>{col_header}</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
</div>"""


def _no_top_scorers_fallback_html() -> str:
    return (
        '<div class="topscorers-card">'
        '<div style="text-align:center;color:#aaa;padding:16px;">'
        '⚽ No goal data yet.<br>'
        '<small>Check back once matches begin.</small>'
        '</div>'
        '</div>'
    )
