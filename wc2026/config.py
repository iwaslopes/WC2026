"""Central configuration: paths, constants, tunables.

Everything you might want to tweak lives here. Defaults are conservative;
refit/retune once the full results history (and later FM26) are in place.
"""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # the workspace folder
DATA = ROOT / "data"
CONFIG = ROOT / "config"
OUT = ROOT / "predictions"

GROUP_FIXTURES = DATA / "group_fixtures.csv"
KNOCKOUT_SLOTS = DATA / "knockout_slots.csv"
RESULTS_CSV = DATA / "results.csv"            # martj42 international results (drop this in)
TEAM_FEATURES = CONFIG / "team_features.csv"  # editable: fifa_points, fm26_strength
PLACEHOLDERS = CONFIG / "placeholder_teams.csv"  # editable: playoff slot -> real team

# --- Elo --------------------------------------------------------------------
ELO_START = 1500.0
ELO_HOME_ADV = 100.0       # added to home side (skipped if neutral)
ELO_K_BY_IMPORTANCE = {    # World Football Elo style base K
    "world_cup": 60,
    "continental_final": 50,
    "wc_qualifier": 40,
    "nations_league": 40,
    "confed_qualifier": 35,
    "friendly": 20,
    "other": 30,
}

# --- Poisson goals model ----------------------------------------------------
TRAIN_YEARS = 12
RECENCY_HALFLIFE_DAYS = 365 * 3
MAX_GOALS = 10             # score-matrix is (0..MAX_GOALS)^2
HOST_NATIONS = {"United States", "Canada", "Mexico"}  # canonical (martj42) names

# --- Corners / cards priors (rubric: corners off-by-2 ok, yellows off-by-1 ok)
CORNERS_GROUP, CORNERS_KO = 10, 11
YELLOWS_GROUP, YELLOWS_KO = 4, 5
REDS = 0

# --- Simulation -------------------------------------------------------------
N_SIMS = 20000
RANDOM_SEED = 42

# --- Strength blend weights (per-variable; calibratable via run.calibrate) ---
# effective_rating = (1-W_FIFA-W_FM)*Elo + W_FIFA*FIFA_in_elo + W_FM*FM26_in_elo
W_FIFA = 0.30
W_FM = 0.0        # raise once Football Manager 26 strengths are supplied
