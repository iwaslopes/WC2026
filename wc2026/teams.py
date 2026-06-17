"""Team-name harmonization between the competition fixtures, the editable
config files, and the martj42 results.csv (which uses its own naming).

Strategy: everything is matched on a normalized key (lowercase, accents and
punctuation stripped) plus a small alias table for the handful of nations whose
English names genuinely differ across sources. Resolution is always done
against the *actual* set of names present in results.csv, so we never guess a
name the data doesn't contain.
"""
import unicodedata
import pandas as pd
from . import config


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    for ch in ".'`-":
        s = s.replace(ch, "")
    return " ".join(s.split())


# groups of names that refer to the same nation (any spelling -> canonical group)
ALIAS_GROUPS = [
    {"united states", "usa", "us"},
    {"cape verde", "cabo verde"},
    {"ivory coast", "cote divoire", "cote d ivoire"},
    {"czechia", "czech republic"},
    {"turkey", "turkiye"},
    {"south korea", "korea republic", "korea south"},
    {"north korea", "korea dpr"},
    {"dr congo", "congo dr", "democratic republic of the congo", "congo kinshasa"},
    {"iran", "ir iran"},
    {"china", "china pr"},
    {"bosnia and herzegovina", "bosnia herzegovina"},
]
_ALIAS_LOOKUP = {}
for grp in ALIAS_GROUPS:
    for n in grp:
        _ALIAS_LOOKUP[n] = grp


def resolve_to_results(name: str, available: set) -> str | None:
    """Map any spelling of a team to the exact string used in results.csv."""
    avail_norm = {norm(a): a for a in available}
    key = norm(name)
    if key in avail_norm:
        return avail_norm[key]
    # try alias group members
    for alt in _ALIAS_LOOKUP.get(key, set()):
        if alt in avail_norm:
            return avail_norm[alt]
    return None


def load_placeholders() -> dict:
    df = pd.read_csv(config.PLACEHOLDERS)
    return dict(zip(df["slot"], df["real_team"]))


def resolve_fixture_team(name: str, placeholders: dict) -> str:
    """Turn a fixture cell (possibly a playoff placeholder) into a real team."""
    return placeholders.get(name, name)


def load_team_features() -> pd.DataFrame:
    """Editable per-team feature table (fifa_points, fm26_strength).

    Returned indexed by normalized name so it can be joined to anything.
    fm26_strength stays NaN until you supply Football Manager 26 data.
    """
    df = pd.read_csv(config.TEAM_FEATURES)
    df["key"] = df["team"].map(norm)
    return df.set_index("key")
