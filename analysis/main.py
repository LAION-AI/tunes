"""
NeurIPS Analysis: Perceptual Evaluation of AI-Generated Music
=============================================================
Comprehensive statistical analysis of human annotations of AI-generated
vs. real music across multiple generative platforms.
"""

import json
import re
import warnings
from pathlib import Path
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.lines
import matplotlib.patches
import matplotlib.ticker
matplotlib.use("Agg")
from adjustText import adjust_text as _adjust_text
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import krippendorff
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

INPUT_DIR = Path("input/v2")
V1_INPUT_DIR = Path("input/v1")
MODEL_INPUT_DIR = Path("input/models")
OUTPUT_DIR = Path("output")
FIG_DIR = OUTPUT_DIR / "figures"

ANNOTATIONS_FILE = "annotations_export_2026-04-20.json"
PARTICIPANTS_FILE = "participants_export_2026-04-20.json"
FEEDBACK_FILE = "feedback_export_2026-04-20.json"
V1_ANNOTATIONS_FILE = "annotations_export_2026-04-07.json"

RATING_COLS = [
    "aesthetic_quality",
    "playlist_likelihood",
    "musical_creativity",
    "production_quality",
    "emotional_engagement",
]

RATING_LABELS = {
    "aesthetic_quality": "Aesthetic Quality",
    "playlist_likelihood": "Playlist Likelihood",
    "musical_creativity": "Musical Creativity",
    "production_quality": "Production Quality",
    "emotional_engagement": "Emotional Engagement",
}

MOOD_TAGS_CANONICAL = [
    "Wonder",
    "Transcendence",
    "Tenderness",
    "Nostalgia",
    "Peacefulness",
    "Power",
    "Joyful activation",
    "Tension",
    "Sadness",
]

# AI platforms only (kept for backwards-compat wherever AI-only comparisons are needed).
AI_SOURCES = ["suno", "udio", "sonauto", "mureka"]
# Full source list: human real songs + four AI platforms.
ALL_SOURCES = ["human"] + AI_SOURCES
SOURCE_ORDER = AI_SOURCES  # legacy alias used in AI-only analyses
AUTH_ORDER = ["real", "ai-generated", "uncertain"]
GROUND_TRUTH_ORDER = ["real", "ai-generated"]
SNIPPET_ORDER = ["original", "30s"]
THINKING_MODE_ORDER = ["off", "minimal", "low", "medium", "high", "on", "true", "false"]
THINKING_ALIAS_ORDER = ["non-thinking", "thinking"]

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
PALETTE_SOURCE = {
    "human": "#6A4C93",
    "suno": "#E63946",
    "udio": "#457B9D",
    "sonauto": "#2A9D8F",
    "mureka": "#E9C46A",
}
PALETTE_AUTH = {"real": "#2A9D8F", "ai-generated": "#E63946", "uncertain": "#A8DADC"}
PALETTE_TRUTH = {"real": "#6A4C93", "ai-generated": "#E63946"}
PALETTE_SNIPPET = {"original": "#264653", "30s": "#E76F51"}


# ---------------------------------------------------------------------------
# Data Loading & Cleaning
# ---------------------------------------------------------------------------

def load_data():
    """Load v2 annotations and participants.

    v2 is the authoritative dataset: it already contains every v1 annotation
    plus the new annotations collected with a 30-second song snippet and with
    real (human-made) songs interleaved with the AI platforms. We load v2 only
    to avoid double-counting, and we use the v1 export solely to label each
    annotation's listening condition ('original' snippet vs '30s' snippet).
    """
    with open(INPUT_DIR / ANNOTATIONS_FILE) as f:
        annotations = pd.DataFrame(json.load(f))
    with open(INPUT_DIR / PARTICIPANTS_FILE) as f:
        participants = pd.DataFrame(json.load(f))

    v1_annotation_ids = set()
    v1_path = V1_INPUT_DIR / V1_ANNOTATIONS_FILE
    if v1_path.exists():
        with open(v1_path) as f:
            v1_annotation_ids = {r["annotation_id"] for r in json.load(f)}

    annotations["created_at"] = pd.to_datetime(annotations["created_at"])
    participants["created_at"] = pd.to_datetime(participants["created_at"])

    def safe_parse_json_list(val):
        if pd.isna(val) or val is None:
            return []
        if isinstance(val, list):
            return val
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return []

    for col in ["ai_aspects", "mood_tags", "participant_musical_genres",
                 "song_genres", "song_moods", "song_tags"]:
        if col in annotations.columns:
            annotations[col] = annotations[col].apply(safe_parse_json_list)

    for col in ["musical_genres"]:
        if col in participants.columns:
            participants[col] = participants[col].apply(safe_parse_json_list)

    for col in RATING_COLS:
        annotations[col] = pd.to_numeric(annotations[col], errors="coerce")

    annotations["rating_mean"] = annotations[RATING_COLS].mean(axis=1)

    # Ground truth: 'human' songs are real; all other sources are AI platforms.
    annotations["is_ai_song"] = annotations["song_source"] != "human"
    annotations["ground_truth"] = np.where(
        annotations["is_ai_song"], "ai-generated", "real"
    )

    # Perceptual judgment encodings.
    annotations["detected_ai"] = (
        annotations["authenticity_assessment"] == "ai-generated"
    ).astype(int)
    annotations["said_real"] = (
        annotations["authenticity_assessment"] == "real"
    ).astype(int)
    annotations["said_uncertain"] = (
        annotations["authenticity_assessment"] == "uncertain"
    ).astype(int)

    # Correctness: only defined on non-uncertain trials (1 = correct, 0 = wrong,
    # NaN = uncertain response).
    correct_mask = (
        ((annotations["authenticity_assessment"] == "ai-generated") & annotations["is_ai_song"]) |
        ((annotations["authenticity_assessment"] == "real") & ~annotations["is_ai_song"])
    )
    annotations["is_correct"] = correct_mask.astype("float64")
    annotations.loc[
        annotations["authenticity_assessment"] == "uncertain", "is_correct"
    ] = np.nan

    # Listening condition: v1 annotations were made against full-length snippets;
    # annotations new in v2 used a 30-second snippet.
    annotations["snippet_condition"] = np.where(
        annotations["annotation_id"].isin(v1_annotation_ids), "original", "30s"
    )

    # Familiarity: per-trial self-report of whether the rater has heard
    # this song before. Levels in the export are {never, familiar,
    # uncertain, know}; "know" has a single observation, so we collapse
    # it into "familiar" to keep the cell estimable. The cleaned field
    # is a 3-level factor with "never" as the natural reference (the
    # uncontaminated perceptual condition).
    if "familiarity_level" in annotations.columns:
        annotations["familiarity_level"] = (
            annotations["familiarity_level"].astype(str).str.strip().str.lower()
            .replace({"know": "familiar", "nan": "never", "none": "never"})
        )
        annotations.loc[
            ~annotations["familiarity_level"].isin(["never", "familiar", "uncertain"]),
            "familiarity_level",
        ] = "never"
    else:
        annotations["familiarity_level"] = "never"

    # ---- Personalization helpers ---------------------------------------
    # Treat the rich participant-side fields as first-class predictors.
    # Age: keep as continuous, median-imputed with a missingness flag so
    # the ~30% of trials without age data are not dropped from models.
    age_raw = pd.to_numeric(annotations.get("participant_age"), errors="coerce")
    age_median = float(age_raw.median()) if age_raw.notna().any() else 30.0
    annotations["participant_age_imputed"] = age_raw.fillna(age_median)
    annotations["participant_age_missing"] = age_raw.isna().astype(int)

    # Taste breadth: how many genres the participant lists as preferred.
    fav_genres_col = (
        annotations["participant_musical_genres"]
        if "participant_musical_genres" in annotations.columns
        else pd.Series([[]] * len(annotations))
    )
    annotations["participant_taste_breadth"] = fav_genres_col.apply(
        lambda lst: len(lst) if isinstance(lst, list) else 0
    )

    # Favorite-genre / song-genre overlap. We compute two views:
    #   * exact_match  : any literal token shared between the song's
    #                    declared genres and the participant's favourites.
    #   * family_match : same comparison after collapsing each token to a
    #                    broad family (so e.g. "deep house" and "EDM"
    #                    are both Electronic).
    # Returns 1/0 when both sides have at least one tag, NaN otherwise so
    # that downstream models can decide whether to drop the trial.
    song_genres_col = (
        annotations["song_genres"]
        if "song_genres" in annotations.columns
        else pd.Series([[]] * len(annotations))
    )

    def _norm_set(lst):
        if not isinstance(lst, list) or not lst:
            return set()
        return {str(x).strip().lower() for x in lst if str(x).strip()}

    def _family_set(lst):
        if not isinstance(lst, list) or not lst:
            return set()
        out = set()
        for token in lst:
            fam = _genre_family([token])
            if fam and fam != "Other":
                out.add(fam)
        return out

    fav_norm = fav_genres_col.apply(_norm_set)
    song_norm = song_genres_col.apply(_norm_set)
    fav_fams = fav_genres_col.apply(_family_set)
    song_fams = song_genres_col.apply(_family_set)

    def _match(a, b):
        if not a or not b:
            return np.nan
        return 1.0 if a & b else 0.0

    annotations["fav_genre_exact_match"] = [
        _match(a, b) for a, b in zip(fav_norm, song_norm)
    ]
    annotations["fav_genre_family_match"] = [
        _match(a, b) for a, b in zip(fav_fams, song_fams)
    ]
    # A 3-level field that is always defined (match / no-match / unknown)
    # so that we do not have to drop trials with missing song tags.
    def _match_label(v):
        if pd.isna(v):
            return "unknown"
        return "match" if v >= 0.5 else "no-match"
    annotations["fav_genre_match_label"] = (
        annotations["fav_genre_family_match"].apply(_match_label)
    )

    return annotations, participants


# ---------------------------------------------------------------------------
# 1. Descriptive Statistics
# ---------------------------------------------------------------------------

def descriptive_statistics(ann, par, report):
    report.append("\n" + "=" * 80)
    report.append("1. DESCRIPTIVE STATISTICS")
    report.append("=" * 80)

    report.append(f"\n  Total annotations: {len(ann)}")
    report.append(f"  Unique participants (in annotations): {ann['participant_id'].nunique()}")
    report.append(f"  Unique songs: {ann['song_id'].nunique()}")
    report.append(f"  Unique sessions: {ann['session_id'].nunique()}")
    report.append(f"  Registered participants: {len(par)}")

    active = par[par["annotation_count"] > 0]
    report.append(f"  Active participants (>=1 annotation): {len(active)}")
    report.append(f"  Median annotations per active participant: {active['annotation_count'].median():.0f}")
    report.append(f"  Mean annotations per active participant: {active['annotation_count'].mean():.1f}")

    report.append("\n  --- Ground-Truth Balance (real vs AI) ---")
    truth_counts = ann["ground_truth"].value_counts()
    for gt, cnt in truth_counts.items():
        report.append(f"    {gt}: {cnt} ({cnt/len(ann)*100:.1f}%)")

    report.append("\n  --- Song Source Distribution ---")
    src_counts = ann["song_source"].value_counts()
    for src, cnt in src_counts.items():
        kind = "real" if src == "human" else "AI"
        report.append(f"    {src} ({kind}): {cnt} ({cnt/len(ann)*100:.1f}%)")

    report.append("\n  --- Snippet Condition Distribution ---")
    snip_counts = ann["snippet_condition"].value_counts()
    for cond, cnt in snip_counts.items():
        label = "full-length (v1)" if cond == "original" else "30-second (v2)"
        report.append(f"    {label}: {cnt} ({cnt/len(ann)*100:.1f}%)")

    report.append("\n  --- Authenticity Assessment Distribution ---")
    auth_counts = ann["authenticity_assessment"].value_counts()
    for auth, cnt in auth_counts.items():
        report.append(f"    {auth}: {cnt} ({cnt/len(ann)*100:.1f}%)")

    report.append("\n  --- Rating Summary Statistics ---")
    desc = ann[RATING_COLS].describe().T
    desc["median"] = ann[RATING_COLS].median()
    desc["skew"] = ann[RATING_COLS].skew()
    desc["kurtosis"] = ann[RATING_COLS].kurtosis()
    report.append(desc[["count", "mean", "std", "median", "min", "max", "skew", "kurtosis"]].to_string())

    report.append("\n  --- Annotation Duration (seconds) ---")
    dur = ann["annotation_duration_ms"].dropna() / 1000
    report.append(f"    Mean: {dur.mean():.1f}s, Median: {dur.median():.1f}s, "
                  f"Std: {dur.std():.1f}s, Min: {dur.min():.1f}s, Max: {dur.max():.1f}s")

    report.append("\n  --- Participant Demographics ---")
    engagement_counts = par["musical_engagement"].value_counts()
    report.append("  Musical Engagement:")
    for eng, cnt in engagement_counts.items():
        report.append(f"    {eng}: {cnt} ({cnt/len(par)*100:.1f}%)")

    ai_exp = par["ai_music_experience"].value_counts()
    report.append("\n  AI Music Experience:")
    for exp, cnt in ai_exp.items():
        report.append(f"    {exp}: {cnt} ({cnt/len(par)*100:.1f}%)")

    training = par["formal_training_years"].dropna()
    report.append(f"\n  Formal Training (years): Mean={training.mean():.1f}, "
                  f"Median={training.median():.0f}, Max={training.max():.0f}")


# ---------------------------------------------------------------------------
# 2. AI Detection Performance (Signal Detection Theory)
# ---------------------------------------------------------------------------

def _sdt_metrics(hits, misses, fas, crs):
    """Standard signal-detection theory metrics with log-linear correction."""
    # Log-linear correction (Hautus 1995) to avoid infinite d' at ceiling/floor.
    signal = hits + misses
    noise = fas + crs
    hr = (hits + 0.5) / (signal + 1.0)
    far = (fas + 0.5) / (noise + 1.0)
    d_prime = stats.norm.ppf(hr) - stats.norm.ppf(far)
    criterion = -0.5 * (stats.norm.ppf(hr) + stats.norm.ppf(far))
    accuracy = (hits + crs) / max(signal + noise, 1)
    precision_ai = hits / (hits + fas) if (hits + fas) else np.nan
    recall_ai = hits / signal if signal else np.nan
    f1_ai = (
        2 * precision_ai * recall_ai / (precision_ai + recall_ai)
        if pd.notna(precision_ai) and pd.notna(recall_ai) and (precision_ai + recall_ai) > 0
        else np.nan
    )
    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": fas,
        "correct_rejections": crs,
        "hit_rate": hits / signal if signal else np.nan,
        "false_alarm_rate": fas / noise if noise else np.nan,
        "precision_ai": precision_ai,
        "f1_ai": f1_ai,
        "d_prime": d_prime,
        "criterion": criterion,
        "accuracy": accuracy,
    }


def sdt_analysis(ann, report):
    report.append("\n" + "=" * 80)
    report.append("2. SIGNAL DETECTION THEORY — AI DETECTION ACCURACY")
    report.append("=" * 80)

    report.append("\n  Ground truth: 'human' songs are real; 'suno/udio/sonauto/mureka' are AI.")
    report.append("  Signal = AI trial, Noise = real (human) trial.")
    report.append("  Hit = 'ai-generated' on an AI trial. False Alarm = 'ai-generated' on a real trial.")
    report.append("  Uncertain responses are excluded from d' but reported separately.\n")

    total = len(ann)
    n_ai = int(ann["is_ai_song"].sum())
    n_real = int((~ann["is_ai_song"]).sum())
    report.append(f"  Total trials: {total}  (AI: {n_ai}, real: {n_real})")

    # Overall SDT (exclude uncertain)
    det = ann[ann["authenticity_assessment"] != "uncertain"]
    hits = int(((det["is_ai_song"]) & (det["detected_ai"] == 1)).sum())
    misses = int(((det["is_ai_song"]) & (det["detected_ai"] == 0)).sum())
    fas = int(((~det["is_ai_song"]) & (det["detected_ai"] == 1)).sum())
    crs = int(((~det["is_ai_song"]) & (det["detected_ai"] == 0)).sum())
    m = _sdt_metrics(hits, misses, fas, crs)
    uncertain_overall = (ann["authenticity_assessment"] == "uncertain").mean()

    report.append("\n  --- Overall Signal Detection ---")
    report.append(f"    Hits (AI -> 'ai'):          {m['hits']}  (HR={m['hit_rate']:.3f})")
    report.append(f"    Misses (AI -> 'real'):      {m['misses']}")
    report.append(f"    False alarms (real -> 'ai'): {m['false_alarms']}  (FAR={m['false_alarm_rate']:.3f})")
    report.append(f"    Correct rejections:         {m['correct_rejections']}")
    report.append(f"    Accuracy (non-uncertain):   {m['accuracy']*100:.1f}%")
    report.append(f"    d' = {m['d_prime']:.3f}     (0 = chance, higher = better discrimination)")
    report.append(f"    c (criterion) = {m['criterion']:.3f}  "
                  f"(negative = bias toward 'AI', positive = bias toward 'real')")
    report.append(f"    Uncertain rate overall:     {uncertain_overall*100:.1f}%")

    # Confusion matrix
    report.append("\n  --- Confusion Matrix (rows: ground truth, cols: response) ---")
    cm = pd.crosstab(ann["ground_truth"], ann["authenticity_assessment"],
                     dropna=False).reindex(index=GROUND_TRUTH_ORDER,
                                           columns=AUTH_ORDER, fill_value=0)
    report.append(cm.to_string())
    cm_row = cm.div(cm.sum(axis=1), axis=0)
    report.append("\n  Row-normalized (conditional probabilities):")
    report.append(cm_row.round(3).to_string())

    # Per-source breakdown (including human as CR/FA source)
    report.append("\n  --- Response Distribution by Source ---")
    by_source = pd.crosstab(ann["song_source"], ann["authenticity_assessment"]).reindex(
        index=ALL_SOURCES, columns=AUTH_ORDER, fill_value=0
    )
    by_source_prop = by_source.div(by_source.sum(axis=1), axis=0)
    for src in ALL_SOURCES:
        n = int(by_source.loc[src].sum())
        if src == "human":
            cr_rate = by_source_prop.loc[src, "real"]
            fa_rate = by_source_prop.loc[src, "ai-generated"]
            unc = by_source_prop.loc[src, "uncertain"]
            report.append(f"    {src} (real, n={n}): CR={cr_rate*100:.1f}% | "
                          f"FA={fa_rate*100:.1f}% | uncertain={unc*100:.1f}%")
        else:
            hit_rate = by_source_prop.loc[src, "ai-generated"]
            miss_rate = by_source_prop.loc[src, "real"]
            unc = by_source_prop.loc[src, "uncertain"]
            report.append(f"    {src} (AI,   n={n}): HR={hit_rate*100:.1f}% | "
                          f"Miss={miss_rate*100:.1f}% | uncertain={unc*100:.1f}%")

    chi2_data = pd.crosstab(ann["song_source"], ann["authenticity_assessment"])
    chi2, p, dof, expected = stats.chi2_contingency(chi2_data)
    cramers_v = np.sqrt(chi2 / (len(ann) * (min(chi2_data.shape) - 1)))
    report.append(f"\n  Chi-squared test (source x assessment): χ²={chi2:.2f}, df={dof}, "
                  f"p={p:.4e}, Cramér's V={cramers_v:.3f}")

    # Per-participant SDT
    report.append("\n  --- Per-Participant d' and Criterion ---")
    sdt_rows = []
    for pid, g in ann.groupby("participant_id"):
        g_det = g[g["authenticity_assessment"] != "uncertain"]
        if g_det["is_ai_song"].sum() == 0 or (~g_det["is_ai_song"]).sum() == 0:
            continue
        h = int(((g_det["is_ai_song"]) & (g_det["detected_ai"] == 1)).sum())
        mi = int(((g_det["is_ai_song"]) & (g_det["detected_ai"] == 0)).sum())
        fa = int(((~g_det["is_ai_song"]) & (g_det["detected_ai"] == 1)).sum())
        cr = int(((~g_det["is_ai_song"]) & (g_det["detected_ai"] == 0)).sum())
        mm = _sdt_metrics(h, mi, fa, cr)
        sdt_rows.append({"participant_id": pid, "n": len(g), **mm})
    sdt_df = pd.DataFrame(sdt_rows)
    if not sdt_df.empty:
        report.append(f"    Participants with both signal+noise trials: {len(sdt_df)}")
        report.append(f"    d':        mean={sdt_df['d_prime'].mean():.3f}, "
                      f"median={sdt_df['d_prime'].median():.3f}, "
                      f"std={sdt_df['d_prime'].std():.3f}")
        report.append(f"    criterion: mean={sdt_df['criterion'].mean():.3f}, "
                      f"median={sdt_df['criterion'].median():.3f}, "
                      f"std={sdt_df['criterion'].std():.3f}")
        report.append(f"    accuracy:  mean={sdt_df['accuracy'].mean()*100:.1f}%, "
                      f"median={sdt_df['accuracy'].median()*100:.1f}%")
        t_stat, t_p = stats.ttest_1samp(sdt_df["d_prime"].dropna(), 0.0)
        report.append(f"    One-sample t-test d' vs 0 (chance): t={t_stat:.3f}, p={t_p:.4e}")
        t_stat_c, t_p_c = stats.ttest_1samp(sdt_df["criterion"].dropna(), 0.0)
        report.append(f"    One-sample t-test criterion vs 0:    t={t_stat_c:.3f}, p={t_p_c:.4e}")
    ann.attrs["sdt_per_participant"] = sdt_df

    report.append("\n  --- Accuracy by Musical Engagement ---")
    by_eng = ann.groupby("participant_musical_engagement").apply(
        lambda g: pd.Series({
            "n": len(g),
            "accuracy": g["is_correct"].mean(skipna=True),
            "hit_rate_on_ai": g.loc[g["is_ai_song"], "detected_ai"].mean(),
            "fa_rate_on_real": g.loc[~g["is_ai_song"], "detected_ai"].mean(),
            "uncertain_rate": g["said_uncertain"].mean(),
        })
    )
    report.append(by_eng.round(3).to_string())

    report.append("\n  --- Accuracy by AI Experience ---")
    by_ai_exp = ann.groupby("participant_ai_music_experience").apply(
        lambda g: pd.Series({
            "n": len(g),
            "accuracy": g["is_correct"].mean(skipna=True),
            "hit_rate_on_ai": g.loc[g["is_ai_song"], "detected_ai"].mean(),
            "fa_rate_on_real": g.loc[~g["is_ai_song"], "detected_ai"].mean(),
        })
    )
    report.append(by_ai_exp.round(3).to_string())

    report.append("\n  --- Accuracy by Formal Training ---")
    ann["training_group"] = pd.cut(
        ann["participant_formal_training_years"].fillna(0),
        bins=[-1, 0, 3, 7, 100],
        labels=["None", "1-3 years", "4-7 years", "8+ years"]
    )
    by_train = ann.groupby("training_group", observed=True).apply(
        lambda g: pd.Series({
            "n": len(g),
            "accuracy": g["is_correct"].mean(skipna=True),
            "hit_rate_on_ai": g.loc[g["is_ai_song"], "detected_ai"].mean(),
            "fa_rate_on_real": g.loc[~g["is_ai_song"], "detected_ai"].mean(),
        })
    )
    report.append(by_train.round(3).to_string())

    # --- Bootstrap 95% CIs on aggregate SDT metrics ---
    # Resample annotations (with replacement) 2000 times to get CIs on d',
    # accuracy, HR, and FAR. Standard in NeurIPS perception papers where
    # only aggregate statistics are reported.
    report.append("\n  --- Bootstrap 95% CIs on Aggregate SDT Metrics (N=2000 resamples) ---")
    rng = np.random.default_rng(42)
    det_all = ann[ann["authenticity_assessment"] != "uncertain"].copy()
    boot_metrics = {"d_prime": [], "accuracy": [], "hit_rate": [], "false_alarm_rate": []}
    for _ in range(2000):
        samp = det_all.sample(len(det_all), replace=True, random_state=rng.integers(1e9))
        bh = int((samp["is_ai_song"] & (samp["detected_ai"] == 1)).sum())
        bm = int((samp["is_ai_song"] & (samp["detected_ai"] == 0)).sum())
        bf = int((~samp["is_ai_song"] & (samp["detected_ai"] == 1)).sum())
        bc = int((~samp["is_ai_song"] & (samp["detected_ai"] == 0)).sum())
        bmet = _sdt_metrics(bh, bm, bf, bc)
        for k in boot_metrics:
            boot_metrics[k].append(bmet[k])
    for k, vals in boot_metrics.items():
        arr = np.array(vals)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        report.append(f"    {k:20s}: mean={np.mean(arr):.3f}  95% CI [{lo:.3f}, {hi:.3f}]")


# ---------------------------------------------------------------------------
# 3. Rating Analysis Across Sources
# ---------------------------------------------------------------------------

def rating_analysis(ann, report):
    report.append("\n" + "=" * 80)
    report.append("3. RATING ANALYSIS ACROSS SOURCES (HUMAN + AI)")
    report.append("=" * 80)

    report.append("\n  --- Mean Ratings by Source (all five, human first) ---")
    means = ann.groupby("song_source")[RATING_COLS].mean().reindex(ALL_SOURCES)
    report.append(means.round(3).to_string())

    report.append("\n  --- Median Ratings by Source ---")
    medians = ann.groupby("song_source")[RATING_COLS].median().reindex(ALL_SOURCES)
    report.append(medians.round(3).to_string())

    report.append("\n  --- Real (human) vs AI-generated: Mann-Whitney U ---")
    for col in RATING_COLS:
        real_g = ann.loc[~ann["is_ai_song"], col].dropna()
        ai_g = ann.loc[ann["is_ai_song"], col].dropna()
        if len(real_g) < 2 or len(ai_g) < 2:
            continue
        u_stat, p_val = stats.mannwhitneyu(real_g, ai_g, alternative="two-sided")
        r_effect = 1 - (2 * u_stat) / (len(real_g) * len(ai_g))
        pooled_std = np.sqrt(((len(real_g) - 1) * real_g.std() ** 2 +
                              (len(ai_g) - 1) * ai_g.std() ** 2) /
                             max(len(real_g) + len(ai_g) - 2, 1))
        cohen_d = (real_g.mean() - ai_g.mean()) / pooled_std if pooled_std > 0 else 0.0
        report.append(f"    {RATING_LABELS[col]}: real={real_g.mean():.2f} (n={len(real_g)}) vs "
                      f"AI={ai_g.mean():.2f} (n={len(ai_g)}), U={u_stat:.0f}, p={p_val:.4e}, "
                      f"r={r_effect:.3f}, d={cohen_d:.3f}")

    report.append("\n  --- Kruskal-Wallis Across All Five Sources ---")
    for col in RATING_COLS:
        groups = [ann.loc[ann["song_source"] == s, col].dropna().values for s in ALL_SOURCES]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue
        h_stat, p_val = stats.kruskal(*groups)
        eta_sq = (h_stat - len(groups) + 1) / (len(ann) - len(groups))
        report.append(f"    {RATING_LABELS[col]}: H={h_stat:.3f}, p={p_val:.4e}, η²={eta_sq:.4f}")

    report.append("\n  --- Kruskal-Wallis Across AI Sources Only ---")
    ai_only = ann[ann["is_ai_song"]]
    for col in RATING_COLS:
        groups = [ai_only.loc[ai_only["song_source"] == s, col].dropna().values for s in AI_SOURCES]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue
        h_stat, p_val = stats.kruskal(*groups)
        eta_sq = (h_stat - len(groups) + 1) / (len(ai_only) - len(groups))
        report.append(f"    {RATING_LABELS[col]}: H={h_stat:.3f}, p={p_val:.4e}, η²={eta_sq:.4f}")

    report.append("\n  --- Post-hoc Pairwise Mann-Whitney U Across All Sources "
                  "(Bonferroni-corrected) ---")
    pairs_all = list(combinations(ALL_SOURCES, 2))
    n_comparisons = len(pairs_all)
    for col in RATING_COLS:
        report.append(f"\n    {RATING_LABELS[col]}:")
        for s1, s2 in pairs_all:
            g1 = ann.loc[ann["song_source"] == s1, col].dropna()
            g2 = ann.loc[ann["song_source"] == s2, col].dropna()
            if len(g1) < 2 or len(g2) < 2:
                continue
            u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            p_adj = min(p_val * n_comparisons, 1.0)
            r_effect = 1 - (2 * u_stat) / (len(g1) * len(g2))
            sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
            report.append(f"      {s1} vs {s2}: U={u_stat:.0f}, p_adj={p_adj:.4f}, "
                          f"r={r_effect:.3f} [{sig}]")

    report.append("\n  --- Ratings by Authenticity Assessment (perceived) ---")
    means_auth = ann.groupby("authenticity_assessment")[RATING_COLS].mean().reindex(AUTH_ORDER)
    report.append(means_auth.round(3).to_string())

    report.append("\n  --- Do songs rated higher get classified as 'real' more often? ---")
    for col in RATING_COLS:
        ai_rated = ann.loc[ann["authenticity_assessment"] == "ai-generated", col].dropna()
        real_rated = ann.loc[ann["authenticity_assessment"] == "real", col].dropna()
        u_stat, p_val = stats.mannwhitneyu(real_rated, ai_rated, alternative="two-sided")
        report.append(f"    {RATING_LABELS[col]}: 'real' mean={real_rated.mean():.2f} vs "
                      f"'ai' mean={ai_rated.mean():.2f}, U={u_stat:.0f}, p={p_val:.4e}")


# ---------------------------------------------------------------------------
# 4. Mixed-Effects Models
# ---------------------------------------------------------------------------

def mixed_effects_models(ann, report):
    report.append("\n" + "=" * 80)
    report.append("4. MIXED-EFFECTS MODELS")
    report.append("=" * 80)
    report.append("\n  Modeling ratings with fixed effects for song_source and random")
    report.append("  intercepts for participant_id (accounting for repeated measures).\n")

    ann_model = ann.dropna(subset=RATING_COLS + ["song_source", "participant_id"]).copy()
    ann_model["song_source"] = pd.Categorical(ann_model["song_source"], categories=ALL_SOURCES)

    for col in RATING_COLS:
        try:
            model = smf.mixedlm(
                f"{col} ~ C(song_source, Treatment(reference='human'))",
                data=ann_model,
                groups=ann_model["participant_id"],
            )
            result = model.fit(reml=True)
            report.append(f"\n  --- {RATING_LABELS[col]} ---")
            report.append(f"  Dependent variable: {col}")
            report.append(f"  N observations: {result.nobs:.0f}")
            report.append(f"  N groups: {result.k_fe + result.k_re}")
            report.append(f"  Log-Likelihood: {result.llf:.2f}")
            report.append(f"  Converged: {result.converged}")

            summary_df = pd.DataFrame({
                "Coef": result.fe_params,
                "Std.Err": result.bse_fe,
                "z": result.tvalues,
                "P>|z|": result.pvalues,
            })
            report.append(summary_df.round(4).to_string())
            report.append(f"  Random effect variance (participant): "
                          f"{result.cov_re.iloc[0, 0]:.4f}")
            icc = result.cov_re.iloc[0, 0] / (result.cov_re.iloc[0, 0] + result.scale)
            report.append(f"  ICC (Intraclass Correlation): {icc:.4f}")
        except Exception as e:
            report.append(f"\n  --- {RATING_LABELS[col]} --- FAILED: {e}")


# ---------------------------------------------------------------------------
# 5. Logistic Mixed-Effects Model for AI Detection
# ---------------------------------------------------------------------------

def detection_model(ann, report):
    report.append("\n" + "=" * 80)
    report.append("5. LOGISTIC REGRESSION — PREDICTORS OF 'SAID AI' AND CORRECT DETECTION")
    report.append("=" * 80)

    ann_det = ann.copy()
    ann_det["has_training"] = (ann_det["participant_formal_training_years"].fillna(0) > 0).astype(int)

    ai_exp_map = {
        "Heard about it but never tried": 0,
        "Tried once or twice": 1,
        "Use occasionally": 2,
        "Use regularly": 3,
        "Professional experience with AI music": 4,
    }
    ann_det["ai_exp_num"] = ann_det["participant_ai_music_experience"].map(ai_exp_map).fillna(0)

    eng_map = {"casual": 0, "enthusiast": 1, "musician": 2, "professional": 3}
    ann_det["engagement_num"] = ann_det["participant_musical_engagement"].map(eng_map).fillna(0)

    ann_det["log_duration"] = np.log1p(ann_det["annotation_duration_ms"].fillna(0))
    ann_det["is_ai_song_int"] = ann_det["is_ai_song"].astype(int)
    ann_det = ann_det.dropna(subset=["detected_ai", "song_source"])

    # Personalized covariates (see load_data for definitions): age (z-scored
    # with a missingness flag), taste breadth (# of self-reported favourite
    # genres, z-scored), favourite-genre vs.\ song-genre family match (3-level:
    # match / no-match / unknown), listening device, and listening
    # environment. Listening *context* is not included because it is highly
    # imbalanced (>95\% "alone").
    age_mean = ann_det["participant_age_imputed"].mean()
    age_std = ann_det["participant_age_imputed"].std()
    ann_det["age_z"] = (
        (ann_det["participant_age_imputed"] - age_mean) / age_std
        if age_std and age_std > 0 else 0.0
    )
    breadth_mean = ann_det["participant_taste_breadth"].mean()
    breadth_std = ann_det["participant_taste_breadth"].std()
    ann_det["taste_breadth_z"] = (
        (ann_det["participant_taste_breadth"] - breadth_mean) / breadth_std
        if breadth_std and breadth_std > 0 else 0.0
    )

    device_ref = _safe_ref_level(
        ann_det["participant_listening_device"].astype(str),
        preferred="Headphones (on-ear)",
        fallback_default="Laptop/phone speakers",
    )
    env_ref = _safe_ref_level(
        ann_det["participant_environment"].astype(str),
        preferred="Quiet room",
        fallback_default="Quiet room",
    )

    personal_terms = (
        " + age_z + C(participant_age_missing) "
        " + taste_breadth_z "
        f" + C(participant_listening_device, Treatment('{device_ref}')) "
        f" + C(participant_environment, Treatment('{env_ref}')) "
        " + C(fav_genre_match_label, Treatment('unknown')) "
    )

    # Model A: predict 'said ai-generated' across all trials, with ground truth
    # as a predictor plus participant covariates. Source enters via the
    # ground-truth flag and the per-source offsets for AI trials only.
    report.append("\n  --- Model A: P(response = 'ai-generated') on all non-uncertain trials ---")
    report.append(
        "  Personalized covariates added: age (z), age_missing, taste_breadth (z),\n"
        f"  listening device (ref={device_ref}), environment (ref={env_ref}),\n"
        "  favourite-genre match (ref=unknown)."
    )
    ann_a = ann_det[ann_det["authenticity_assessment"] != "uncertain"].copy()
    try:
        formula_a = (
            "detected_ai ~ is_ai_song_int + "
            "C(song_source, Treatment(reference='human')) + "
            "C(snippet_condition, Treatment(reference='original')) + "
            "ai_exp_num + engagement_num + has_training + log_duration"
            + personal_terms
        )
        model_a = smf.logit(formula_a, data=ann_a)
        result_a = model_a.fit(disp=False, maxiter=200)
        report.append(f"\n  N observations: {result_a.nobs:.0f}")
        report.append(f"  Pseudo R²: {result_a.prsquared:.4f}")
        report.append(f"  Log-Likelihood: {result_a.llf:.2f}")
        report.append(f"  AIC: {result_a.aic:.2f}")
        summary_a = pd.DataFrame({
            "Coef": result_a.params,
            "Std.Err": result_a.bse,
            "z": result_a.tvalues,
            "P>|z|": result_a.pvalues,
            "Odds Ratio": np.exp(result_a.params),
        })
        report.append("\n" + summary_a.round(4).to_string())
    except Exception as e:
        report.append(f"  Model A failed: {e}")

    # Model B: predict correctness (exclude uncertain), conditioning on ground truth.
    report.append("\n\n  --- Model B: P(correct) on non-uncertain trials ---")
    ann_b = ann_det[ann_det["authenticity_assessment"] != "uncertain"].copy()
    ann_b["correct_int"] = ann_b["is_correct"].astype(int)
    try:
        formula_b = (
            "correct_int ~ is_ai_song_int + "
            "C(snippet_condition, Treatment(reference='original')) + "
            "ai_exp_num + engagement_num + has_training + log_duration"
            + personal_terms
        )
        model_b = smf.logit(formula_b, data=ann_b)
        result_b = model_b.fit(disp=False, maxiter=200)
        report.append(f"\n  N observations: {result_b.nobs:.0f}")
        report.append(f"  Pseudo R²: {result_b.prsquared:.4f}")
        report.append(f"  Log-Likelihood: {result_b.llf:.2f}")
        report.append(f"  AIC: {result_b.aic:.2f}")
        summary_b = pd.DataFrame({
            "Coef": result_b.params,
            "Std.Err": result_b.bse,
            "z": result_b.tvalues,
            "P>|z|": result_b.pvalues,
            "Odds Ratio": np.exp(result_b.params),
        })
        report.append("\n" + summary_b.round(4).to_string())
    except Exception as e:
        report.append(f"  Model B failed: {e}")


# ---------------------------------------------------------------------------
# 6. Inter-Rater Reliability
# ---------------------------------------------------------------------------

def inter_rater_reliability(ann, report):
    report.append("\n" + "=" * 80)
    report.append("6. INTER-RATER RELIABILITY")
    report.append("=" * 80)

    songs_multi = ann.groupby("song_id").filter(lambda x: len(x) >= 2)
    n_songs_multi = songs_multi["song_id"].nunique()
    report.append(f"\n  Songs with >=2 annotations: {n_songs_multi}")

    if n_songs_multi == 0:
        report.append("  Insufficient overlap for inter-rater analysis.")
        return

    for col in RATING_COLS:
        pivot = songs_multi.pivot_table(index="song_id", columns="participant_id",
                                         values=col, aggfunc="first")
        reliability_data = pivot.values
        alpha = krippendorff.alpha(reliability_data.T, level_of_measurement="interval")
        report.append(f"  {RATING_LABELS[col]}: Krippendorff's α = {alpha:.4f}")

    pivot_auth = songs_multi.pivot_table(index="song_id", columns="participant_id",
                                          values="authenticity_assessment", aggfunc="first")
    auth_map = {"ai-generated": 0, "uncertain": 1, "real": 2}
    # Ensure numeric dtype for krippendorff.alpha; object arrays trigger errors.
    auth_numeric = pivot_auth.replace(auth_map).to_numpy(dtype=float)
    alpha_auth = krippendorff.alpha(auth_numeric.T, level_of_measurement="ordinal")
    report.append(f"  Authenticity Assessment: Krippendorff's α = {alpha_auth:.4f} (ordinal)")

    report.append("\n  --- Pairwise Agreement on Doubly-Rated Songs ---")
    report.append(f"  (Only {n_songs_multi} songs rated by 2 raters — ICC requires balanced design)")
    for col in RATING_COLS:
        pairs = songs_multi.groupby("song_id")[col].apply(list)
        pairs = pairs[pairs.apply(len) == 2]
        if len(pairs) > 0:
            r1 = [p[0] for p in pairs]
            r2 = [p[1] for p in pairs]
            abs_diff = np.mean(np.abs(np.array(r1) - np.array(r2)))
            r_corr, p_val = stats.pearsonr(r1, r2) if len(r1) >= 3 else (np.nan, np.nan)
            report.append(f"  {RATING_LABELS[col]}: mean |diff|={abs_diff:.2f}, "
                          f"r={r_corr:.3f} (p={p_val:.4f}, n={len(pairs)} pairs)")


# ---------------------------------------------------------------------------
# 7. PCA / Factor Analysis of Rating Dimensions
# ---------------------------------------------------------------------------

def pca_analysis(ann, report):
    report.append("\n" + "=" * 80)
    report.append("7. PCA OF RATING DIMENSIONS")
    report.append("=" * 80)

    ratings = ann[RATING_COLS].dropna()
    scaler = StandardScaler()
    ratings_scaled = scaler.fit_transform(ratings)

    pca = PCA()
    pca.fit(ratings_scaled)

    report.append(f"\n  N samples: {len(ratings)}")
    report.append("\n  --- Explained Variance ---")
    for i, (var, cum) in enumerate(zip(pca.explained_variance_ratio_,
                                        np.cumsum(pca.explained_variance_ratio_))):
        report.append(f"    PC{i+1}: {var*100:.1f}% (cumulative: {cum*100:.1f}%)")

    report.append("\n  --- Component Loadings ---")
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(len(RATING_COLS))],
        index=[RATING_LABELS[c] for c in RATING_COLS],
    )
    report.append(loadings.round(4).to_string())

    report.append("\n  --- Correlation Matrix of Ratings ---")
    corr = ann[RATING_COLS].corr()
    corr.index = [RATING_LABELS[c] for c in corr.index]
    corr.columns = [RATING_LABELS[c] for c in corr.columns]
    report.append(corr.round(3).to_string())

    return pca, scaler


# ---------------------------------------------------------------------------
# 8. Annotation Duration Analysis
# ---------------------------------------------------------------------------

def duration_analysis(ann, report):
    report.append("\n" + "=" * 80)
    report.append("8. ANNOTATION DURATION ANALYSIS")
    report.append("=" * 80)

    ann_dur = ann.dropna(subset=["annotation_duration_ms"]).copy()
    ann_dur["duration_sec"] = ann_dur["annotation_duration_ms"] / 1000

    report.append("\n  --- Duration by Source ---")
    dur_by_source = ann_dur.groupby("song_source")["duration_sec"].agg(["mean", "median", "std"])
    report.append(dur_by_source.round(2).to_string())

    report.append("\n  --- Duration by Authenticity Assessment ---")
    dur_by_auth = ann_dur.groupby("authenticity_assessment")["duration_sec"].agg(["mean", "median", "std"])
    report.append(dur_by_auth.round(2).to_string())

    h_stat, p_val = stats.kruskal(
        *[g["duration_sec"].values for _, g in ann_dur.groupby("authenticity_assessment")]
    )
    report.append(f"\n  Kruskal-Wallis (duration ~ assessment): H={h_stat:.3f}, p={p_val:.4e}")

    for col in RATING_COLS:
        r, p = stats.spearmanr(ann_dur["duration_sec"], ann_dur[col], nan_policy="omit")
        report.append(f"  Spearman correlation (duration vs {RATING_LABELS[col]}): "
                      f"ρ={r:.3f}, p={p:.4e}")


# ---------------------------------------------------------------------------
# 9. AI Aspects Analysis
# ---------------------------------------------------------------------------

def ai_aspects_analysis(ann, report):
    report.append("\n" + "=" * 80)
    report.append("9. AI ASPECTS IDENTIFIED BY PARTICIPANTS")
    report.append("=" * 80)

    ai_said = ann[ann["authenticity_assessment"] == "ai-generated"]

    def _collect(aspects_series):
        out = []
        for aspects in aspects_series:
            if isinstance(aspects, list):
                out.extend(aspects)
        return out

    # True positives: correctly flagged AI songs
    hits = ai_said[ai_said["is_ai_song"]]
    hit_aspects = _collect(hits["ai_aspects"])
    hit_counts = Counter(hit_aspects)
    report.append(f"\n  --- Aspects cited on TRUE POSITIVES (AI songs correctly flagged, n={len(hits)}) ---")
    report.append(f"  Total aspect mentions: {len(hit_aspects)}, unique: {len(hit_counts)}")
    for aspect, count in hit_counts.most_common(15):
        pct = count / max(len(hits), 1) * 100
        report.append(f"    {aspect}: {count} ({pct:.1f}%)")

    # False positives: real (human) songs misidentified as AI
    fas = ai_said[~ai_said["is_ai_song"]]
    fa_aspects = _collect(fas["ai_aspects"])
    fa_counts = Counter(fa_aspects)
    report.append(f"\n  --- Aspects cited on FALSE POSITIVES (real songs called AI, n={len(fas)}) ---")
    report.append(f"  Total aspect mentions: {len(fa_aspects)}, unique: {len(fa_counts)}")
    for aspect, count in fa_counts.most_common(15):
        pct = count / max(len(fas), 1) * 100
        report.append(f"    {aspect}: {count} ({pct:.1f}%)")

    report.append("\n  --- AI Aspects by Source (only true positives) ---")
    for source in AI_SOURCES:
        source_ai = hits[hits["song_source"] == source]
        source_aspects = _collect(source_ai["ai_aspects"])
        if source_aspects:
            sc = Counter(source_aspects)
            top = sc.most_common(5)
            report.append(f"    {source} (n={len(source_ai)}): " +
                          ", ".join(f"{a} ({c})" for a, c in top))


# ---------------------------------------------------------------------------
# 10. Effect Size Summary Table
# ---------------------------------------------------------------------------

def effect_size_summary(ann, report):
    report.append("\n" + "=" * 80)
    report.append("10. EFFECT SIZE SUMMARY (Cohen's d for source comparisons)")
    report.append("=" * 80)

    report.append("\n  Cohen's d for each rating dimension between sources "
                  "(all pairs including human):")
    for col in RATING_COLS:
        report.append(f"\n    {RATING_LABELS[col]}:")
        for s1, s2 in combinations(ALL_SOURCES, 2):
            g1 = ann.loc[ann["song_source"] == s1, col].dropna()
            g2 = ann.loc[ann["song_source"] == s2, col].dropna()
            if len(g1) < 2 or len(g2) < 2:
                continue
            pooled_std = np.sqrt(((len(g1) - 1) * g1.std()**2 + (len(g2) - 1) * g2.std()**2) /
                                 (len(g1) + len(g2) - 2))
            if pooled_std > 0:
                d = (g1.mean() - g2.mean()) / pooled_std
            else:
                d = 0.0
            magnitude = ("large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5
                         else "small" if abs(d) >= 0.2 else "negligible")
            report.append(f"      {s1} vs {s2}: d={d:.3f} ({magnitude})")


# ---------------------------------------------------------------------------
# 11. Per-Song Detection Rate Analysis
# ---------------------------------------------------------------------------

def per_song_analysis(ann, report):
    """Compute song-level fooling rates: fraction of annotators who said 'AI'."""
    report.append("\n" + "=" * 80)
    report.append("11. PER-SONG DETECTION RATE ANALYSIS")
    report.append("=" * 80)

    det = ann[ann["authenticity_assessment"] != "uncertain"].copy()

    # For AI songs, ai_response_rate = detection rate (correct label = AI).
    # For real songs, ai_response_rate = false-alarm rate at song level.
    song_stats = (
        det.groupby("song_id")
        .apply(lambda g: pd.Series({
            "is_ai_song": bool(g["is_ai_song"].iloc[0]),
            "source": g["source"].iloc[0] if "source" in g.columns else "unknown",
            "n_annotations": len(g),
            "ai_response_rate": g["detected_ai"].mean(),
        }))
        .reset_index()
    )

    ai_songs = song_stats[song_stats["is_ai_song"]].copy()
    real_songs = song_stats[~song_stats["is_ai_song"]].copy()

    report.append(
        f"\n  Songs evaluated: {len(song_stats)} "
        f"(AI: {len(ai_songs)}, real: {len(real_songs)})"
    )

    report.append("\n  --- AI Song Detection Rates (fraction of annotators labelling song 'AI') ---")
    report.append(f"    Mean detection rate: {ai_songs['ai_response_rate'].mean():.3f}")
    report.append(f"    Median:              {ai_songs['ai_response_rate'].median():.3f}")
    report.append(f"    Std dev:             {ai_songs['ai_response_rate'].std():.3f}")
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.001]
    bin_labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
    ai_songs["rate_bin"] = pd.cut(ai_songs["ai_response_rate"], bins=bins, labels=bin_labels, right=False)
    bin_counts = ai_songs["rate_bin"].value_counts().sort_index()
    report.append("    Distribution of per-song detection rates:")
    for lbl, cnt in bin_counts.items():
        pct = 100 * cnt / len(ai_songs)
        report.append(f"      {lbl}: {cnt} songs ({pct:.1f}%)")

    report.append("\n  --- Real Song False-Alarm Rates (fraction of annotators labelling song 'AI') ---")
    report.append(f"    Mean FA rate: {real_songs['ai_response_rate'].mean():.3f}")
    report.append(f"    Median:       {real_songs['ai_response_rate'].median():.3f}")
    report.append(f"    Std dev:      {real_songs['ai_response_rate'].std():.3f}")

    if "source" in ai_songs.columns:
        report.append("\n  --- Per-Platform Mean Detection Rate (AI songs only) ---")
        plat = ai_songs.groupby("source")["ai_response_rate"].agg(["mean", "std", "count"])
        plat.columns = ["mean_detection_rate", "std", "n_songs"]
        report.append(plat.round(3).to_string())

    # Use sort_values+head instead of nsmallest/nlargest to avoid a pandas
    # bug where attrs propagated from groupby.apply trigger an ambiguous truth
    # value error when attrs contains a DataFrame.
    ai_songs_plain = ai_songs[["song_id", "source", "n_annotations", "ai_response_rate"]].copy()
    ai_songs_plain.attrs = {}

    report.append("\n  --- Hardest AI Songs to Detect (lowest ai_response_rate) ---")
    hardest = ai_songs_plain.sort_values("ai_response_rate").head(5)
    report.append(hardest.round(3).to_string(index=False))

    report.append("\n  --- Easiest AI Songs to Detect (highest ai_response_rate) ---")
    easiest = ai_songs_plain.sort_values("ai_response_rate", ascending=False).head(5)
    report.append(easiest.round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# 12. Trial Order / Fatigue Effects
# ---------------------------------------------------------------------------

def trial_order_analysis(ann, report):
    """Check for learning or fatigue effects across trial order within a session."""
    report.append("\n" + "=" * 80)
    report.append("12. TRIAL ORDER / FATIGUE ANALYSIS")
    report.append("=" * 80)

    work = ann.copy()
    if "created_at" in work.columns:
        work = work.sort_values(["participant_id", "created_at"])
    work["trial_index"] = work.groupby("participant_id").cumcount() + 1

    det = work[work["authenticity_assessment"] != "uncertain"].copy()

    # Per-participant Spearman ρ between trial_index and accuracy
    rhos = []
    for pid, grp in det.groupby("participant_id"):
        if len(grp) < 4:
            continue
        r, _ = stats.spearmanr(grp["trial_index"], grp["is_correct"].astype(float), nan_policy="omit")
        if np.isfinite(r):
            rhos.append(r)

    report.append(f"\n  Participants with ≥4 non-uncertain trials: {len(rhos)}")
    if rhos:
        rhos_arr = np.array(rhos)
        z = np.arctanh(np.clip(rhos_arr, -0.9999, 0.9999))
        t_val, p_val = stats.ttest_1samp(z, 0)
        mean_rho = float(np.mean(rhos_arr))
        report.append(f"  Mean within-participant Spearman ρ (trial order vs. accuracy): {mean_rho:.3f}")
        report.append(f"  One-sample t-test (Fisher-z, H₀: ρ=0): t={t_val:.3f}, p={p_val:.4e}")
        if p_val < 0.05:
            direction = "learning (improvement)" if mean_rho > 0 else "fatigue (decline)"
            report.append(f"  Significant {direction} detected (p<0.05).")
        else:
            report.append("  No significant learning or fatigue effect (p≥0.05).")

        report.append("\n  --- Mean Accuracy by Trial-Index Quintile (all participants pooled) ---")
        det = det.copy()
        det["trial_quintile"] = pd.qcut(det["trial_index"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        by_q = det.groupby("trial_quintile", observed=True)["is_correct"].agg(["mean", "count"])
        by_q.columns = ["accuracy", "n_annotations"]
        report.append(by_q.round(3).to_string())


# ---------------------------------------------------------------------------
# 13. Snippet Condition (full-length v1 vs 30-second v2) Analysis
# ---------------------------------------------------------------------------

def snippet_condition_analysis(ann, report):
    report.append("\n" + "=" * 80)
    report.append("13. SNIPPET CONDITION: FULL-LENGTH (v1) vs 30-SECOND (v2)")
    report.append("=" * 80)
    report.append("\n  v1 annotations were collected against the full song snippet; all new")
    report.append("  annotations in v2 used a 30-second snippet. Every v1 annotation is reused")
    report.append("  in v2 (same annotation_id) and is labelled 'original' here; annotations new")
    report.append("  to v2 are labelled '30s'. Note: the v1 set contains AI platforms only; the")
    report.append("  30-second set adds real (human) songs, so human comparisons are omitted.\n")

    cond_counts = ann["snippet_condition"].value_counts().reindex(SNIPPET_ORDER, fill_value=0)
    for cond, cnt in cond_counts.items():
        report.append(f"    {cond}: {cnt} annotations")

    report.append("\n  --- Ratings by snippet condition (AI songs only, to compare like-for-like) ---")
    ai_ann = ann[ann["is_ai_song"]].copy()
    for col in RATING_COLS:
        g_orig = ai_ann.loc[ai_ann["snippet_condition"] == "original", col].dropna()
        g_new = ai_ann.loc[ai_ann["snippet_condition"] == "30s", col].dropna()
        if len(g_orig) < 2 or len(g_new) < 2:
            continue
        u_stat, p_val = stats.mannwhitneyu(g_orig, g_new, alternative="two-sided")
        pooled = np.sqrt(((len(g_orig) - 1) * g_orig.std() ** 2 +
                          (len(g_new) - 1) * g_new.std() ** 2) /
                         max(len(g_orig) + len(g_new) - 2, 1))
        d = (g_orig.mean() - g_new.mean()) / pooled if pooled > 0 else 0.0
        report.append(f"    {RATING_LABELS[col]}: original={g_orig.mean():.2f} (n={len(g_orig)}) vs "
                      f"30s={g_new.mean():.2f} (n={len(g_new)}), U={u_stat:.0f}, "
                      f"p={p_val:.4e}, d={d:.3f}")

    report.append("\n  --- AI Detection on AI songs by snippet condition ---")
    for cond in SNIPPET_ORDER:
        g = ai_ann[ai_ann["snippet_condition"] == cond]
        if len(g) == 0:
            continue
        hr = (g["authenticity_assessment"] == "ai-generated").mean()
        miss = (g["authenticity_assessment"] == "real").mean()
        unc = (g["authenticity_assessment"] == "uncertain").mean()
        report.append(f"    {cond}: n={len(g)}, HR={hr*100:.1f}%, miss={miss*100:.1f}%, "
                      f"uncertain={unc*100:.1f}%")

    # 2x2 chi-squared on AI trials only (response ai-generated vs not, excluding uncertain)
    ai_det = ai_ann[ai_ann["authenticity_assessment"] != "uncertain"]
    if not ai_det.empty:
        ct = pd.crosstab(ai_det["snippet_condition"], ai_det["detected_ai"])
        if ct.shape == (2, 2):
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            report.append(f"    Chi-squared (snippet x hit/miss on AI): χ²={chi2:.3f}, "
                          f"df={dof}, p={p:.4e}")

    report.append("\n  --- Annotation duration by snippet condition ---")
    for cond in SNIPPET_ORDER:
        d = ann.loc[ann["snippet_condition"] == cond, "annotation_duration_ms"].dropna() / 1000
        if len(d) == 0:
            continue
        report.append(f"    {cond}: n={len(d)}, median={d.median():.1f}s, "
                      f"mean={d.mean():.1f}s, P25={d.quantile(0.25):.1f}s, "
                      f"P75={d.quantile(0.75):.1f}s")


# ---------------------------------------------------------------------------
# 12. Model-vs-Human Benchmarking
# ---------------------------------------------------------------------------

def _compute_detection_metrics(df, assessment_col="authenticity_assessment"):
    """Compute detection metrics for rows with ground truth + authenticity labels."""
    work = df.copy()
    if assessment_col not in work.columns:
        work[assessment_col] = work.get("authenticity_assessment", np.nan)
    work["is_ai_song"] = work["song_source"] != "human"
    work["detected_ai"] = (work[assessment_col] == "ai-generated").astype(int)
    work["said_uncertain"] = (work[assessment_col] == "uncertain").astype(int)

    det = work[work[assessment_col] != "uncertain"]
    hits = int(((det["is_ai_song"]) & (det["detected_ai"] == 1)).sum())
    misses = int(((det["is_ai_song"]) & (det["detected_ai"] == 0)).sum())
    fas = int(((~det["is_ai_song"]) & (det["detected_ai"] == 1)).sum())
    crs = int(((~det["is_ai_song"]) & (det["detected_ai"] == 0)).sum())
    sdt = _sdt_metrics(hits, misses, fas, crs)
    sdt["uncertain_rate"] = work["said_uncertain"].mean()
    sdt["n_total"] = len(work)
    sdt["n_non_uncertain"] = len(det)
    return sdt


def _participant_detection_metrics(ann):
    """Per-participant detection metrics on non-uncertain responses."""
    rows = []
    for pid, g in ann.groupby("participant_id"):
        m = _compute_detection_metrics(g, assessment_col="authenticity_assessment")
        if m["n_non_uncertain"] == 0:
            continue
        rows.append({"participant_id": pid, **m})
    return pd.DataFrame(rows)


def model_vs_human_analysis(ann, report):
    report.append("\n" + "=" * 80)
    report.append("14. MODEL-VS-HUMAN BENCHMARK")
    report.append("=" * 80)

    report.append("\n  " + _THINKING_BINARY_EXPLANATION)
    report.append("  The per-run table below keeps the original thinking intensities for reference;")
    report.append("  figures 14-16 aggregate runs with the same (base_model, binary_thinking) label.")

    model_files = sorted(MODEL_INPUT_DIR.glob("*.json"))
    if not model_files:
        report.append("\n  No model outputs found in input/models/*.json; skipping.")
        return

    canonical_song_ids = set(ann["song_id"].unique())
    human_overall = _compute_detection_metrics(ann, assessment_col="authenticity_assessment")
    human_by_participant = _participant_detection_metrics(ann)

    model_rows = []
    rating_alignment_rows = []
    aesthetics_rows = []
    common_missing = None

    human_song_means = ann.groupby("song_id")[RATING_COLS].mean()
    human_aesthetic_song_mean = ann.groupby("song_id")["aesthetic_quality"].mean()
    human_aesthetic_global = ann["aesthetic_quality"].mean()

    for path in model_files:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, list) or not raw:
            continue

        df = pd.DataFrame(raw)
        if "song_id" not in df.columns or "song_source" not in df.columns:
            continue
        df = df.drop_duplicates(subset=["song_id"], keep="last").copy()
        df = df[df["song_id"].isin(canonical_song_ids)]
        if df.empty:
            continue

        run_name = str(df.get("model_run_name", pd.Series([path.stem])).iloc[0] or path.stem)
        model_label = model_alias(run_name)
        assess_col = (
            "authenticity_assessment_posthoc_uncertain"
            if "authenticity_assessment_posthoc_uncertain" in df.columns
            else "authenticity_assessment"
        )
        m = _compute_detection_metrics(df, assessment_col=assess_col)

        present_ids = set(df["song_id"].tolist())
        missing_ids = canonical_song_ids - present_ids
        common_missing = missing_ids if common_missing is None else (common_missing & missing_ids)

        model_rows.append({
            "model": model_label,
            "n_trials": m["n_total"],
            "coverage_pct": 100.0 * m["n_total"] / max(len(canonical_song_ids), 1),
            "accuracy_non_uncertain": m["accuracy"],
            "hit_rate_on_ai": m["hit_rate"],
            "false_alarm_rate_on_real": m["false_alarm_rate"],
            "uncertain_rate": m["uncertain_rate"],
            "d_prime": m["d_prime"],
            "criterion": m["criterion"],
            "missing_songs": len(missing_ids),
        })

        model_song_means = df.groupby("song_id")[RATING_COLS].mean()
        joined = model_song_means.join(human_song_means, how="inner", lsuffix="_model", rsuffix="_human")
        corr_vals = []
        mae_vals = []
        for col in RATING_COLS:
            x = joined[f"{col}_model"]
            y = joined[f"{col}_human"]
            valid = x.notna() & y.notna()
            if valid.sum() < 3:
                continue
            rho, _ = stats.spearmanr(x[valid], y[valid])
            corr_vals.append(rho)
            mae_vals.append(np.mean(np.abs(x[valid] - y[valid])))
        rating_alignment_rows.append({
            "model": model_label,
            "n_song_overlap": len(joined),
            "mean_spearman_rho": float(np.nanmean(corr_vals)) if corr_vals else np.nan,
            "mean_mae": float(np.nanmean(mae_vals)) if mae_vals else np.nan,
        })

        model_aesthetic_song_mean = df.groupby("song_id")["aesthetic_quality"].mean()
        joined_aes = model_aesthetic_song_mean.to_frame("model_aesthetic").join(
            human_aesthetic_song_mean.to_frame("human_aesthetic"),
            how="inner",
        )
        model_global_aesthetic = df["aesthetic_quality"].mean()
        aesthetic_delta_global = model_global_aesthetic - human_aesthetic_global
        if len(joined_aes) >= 3:
            rho_aes, _ = stats.spearmanr(joined_aes["model_aesthetic"], joined_aes["human_aesthetic"])
            mae_aes = np.mean(np.abs(joined_aes["model_aesthetic"] - joined_aes["human_aesthetic"]))
        else:
            rho_aes, mae_aes = np.nan, np.nan
        aesthetics_rows.append({
            "model": model_label,
            "model_aesthetic_mean": model_global_aesthetic,
            "human_aesthetic_mean": human_aesthetic_global,
            "delta_model_minus_human": aesthetic_delta_global,
            "n_song_overlap": len(joined_aes),
            "song_level_spearman_rho": rho_aes,
            "song_level_mae": mae_aes,
        })

    if not model_rows:
        report.append("\n  No valid model files could be parsed for benchmarking.")
        return

    model_df = pd.DataFrame(model_rows).sort_values(
        ["accuracy_non_uncertain", "d_prime"], ascending=False
    )
    align_df = pd.DataFrame(rating_alignment_rows).set_index("model") if rating_alignment_rows else pd.DataFrame()

    report.append(
        f"\n  Compared {len(model_df)} model runs against human annotations "
        f"on {len(canonical_song_ids)} unique songs."
    )
    if common_missing:
        report.append(
            f"  Songs missing in all model runs: {len(common_missing)} "
            f"(e.g., unavailable URL case)."
        )

    report.append("\n  --- Human Baseline (all human annotations) ---")
    report.append(
        f"    Accuracy (non-uncertain): {human_overall['accuracy']*100:.1f}% | "
        f"Hit rate on AI: {human_overall['hit_rate']*100:.1f}% | "
        f"FA rate on real: {human_overall['false_alarm_rate']*100:.1f}% | "
        f"Uncertain: {human_overall['uncertain_rate']*100:.1f}% | "
        f"d'={human_overall['d_prime']:.3f} | c={human_overall['criterion']:.3f}"
    )

    report.append("\n  --- Model Detection Performance ---")
    show_cols = [
        "model", "n_trials", "coverage_pct", "accuracy_non_uncertain",
        "hit_rate_on_ai", "false_alarm_rate_on_real", "uncertain_rate",
        "d_prime", "criterion", "missing_songs"
    ]
    pretty = model_df[show_cols].copy()
    for c in ["coverage_pct", "accuracy_non_uncertain", "hit_rate_on_ai", "false_alarm_rate_on_real", "uncertain_rate"]:
        pretty[c] = pretty[c].astype(float).round(3)
    pretty["d_prime"] = pretty["d_prime"].astype(float).round(3)
    pretty["criterion"] = pretty["criterion"].astype(float).round(3)
    report.append(pretty.to_string(index=False))

    if not human_by_participant.empty:
        report.append("\n  --- Model Position Relative to Human Participant Distribution ---")
        hp_acc = human_by_participant["accuracy"].dropna()
        hp_d = human_by_participant["d_prime"].dropna()
        for _, row in model_df.iterrows():
            better_acc = (hp_acc < row["accuracy_non_uncertain"]).mean() * 100 if len(hp_acc) else np.nan
            better_d = (hp_d < row["d_prime"]).mean() * 100 if len(hp_d) else np.nan
            report.append(
                f"    {row['model']}: accuracy > {better_acc:.1f}% of humans, "
                f"d' > {better_d:.1f}% of humans"
            )

    if not align_df.empty:
        report.append("\n  --- Model-Human Rating Alignment (song-level means) ---")
        align_show = align_df.copy()
        align_show["mean_spearman_rho"] = align_show["mean_spearman_rho"].round(3)
        align_show["mean_mae"] = align_show["mean_mae"].round(3)
        report.append(align_show.to_string())

    if aesthetics_rows:
        aes_df = pd.DataFrame(aesthetics_rows).sort_values(
            ["delta_model_minus_human"], ascending=False
        )
        aes_show = aes_df.copy()
        for c in [
            "model_aesthetic_mean",
            "human_aesthetic_mean",
            "delta_model_minus_human",
            "song_level_spearman_rho",
            "song_level_mae",
        ]:
            aes_show[c] = aes_show[c].astype(float).round(3)
        report.append("\n  --- Aesthetic Quality: Models vs Humans ---")
        report.append(
            "  (Model means are one judgment/song; human mean aggregates all annotators per song.)"
        )
        report.append(aes_show.to_string(index=False))

    # --- Statistical significance: each model vs. human aggregate ---
    # Proportion z-test comparing model accuracy to the human aggregate accuracy.
    # Human baseline: use all non-uncertain annotations.
    report.append("\n  --- Statistical Tests: Model Accuracy vs Human Baseline ---")
    report.append(
        "  Two-proportion z-test (model accuracy vs human aggregate; Bonferroni-corrected α=0.05/n_models)"
    )
    human_det = ann[ann["authenticity_assessment"] != "uncertain"]
    human_correct = int(human_det["is_correct"].sum(skipna=True))
    human_total = int(human_det["is_correct"].notna().sum())
    human_acc = human_correct / human_total if human_total > 0 else np.nan
    report.append(f"  Human baseline: {human_correct}/{human_total} = {human_acc:.3f}")

    if not model_df.empty:
        n_models = len(model_df)
        alpha_corrected = 0.05 / n_models
        for _, row in model_df.iterrows():
            m_acc = float(row["accuracy_non_uncertain"])
            m_n = int(row["n_trials"])
            if not np.isfinite(m_acc) or m_n == 0 or not np.isfinite(human_acc):
                continue
            # Two-proportion z-test (Newcombe-Wilson would be ideal, but the
            # large-sample normal approximation is standard for n>30)
            p_pool = (m_acc * m_n + human_acc * human_total) / (m_n + human_total)
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / m_n + 1 / human_total))
            if se == 0:
                z_stat, p_two = np.nan, np.nan
            else:
                z_stat = (m_acc - human_acc) / se
                p_two = float(2 * stats.norm.sf(abs(z_stat)))
            sig = "***" if p_two < alpha_corrected else ("ns" if p_two >= 0.05 else "*")
            report.append(
                f"    {row['model']:50s}: acc={m_acc:.3f} (n={m_n}), "
                f"z={z_stat:.2f}, p={p_two:.4e} {sig}"
            )
        report.append(f"  (Bonferroni threshold: α={alpha_corrected:.4f}; *** p<α, * p<0.05, ns p≥0.05)")


# ---------------------------------------------------------------------------
# Model figure helpers
# ---------------------------------------------------------------------------

# Binary thinking-mode grouping used in figures + model-vs-human section.
# Kept coarse on purpose: detailed per-run metrics are still reported in the
# numeric tables; the figures roll these up to the two categories below.
_THINKING_ALIAS = {
    "off": "non-thinking",
    "false": "non-thinking",
    "none": "non-thinking",
    "minimal": "non-thinking",
    "low": "non-thinking",
    "on": "thinking",
    "true": "thinking",
    "enabled": "thinking",
    "medium": "thinking",
    "high": "thinking",
}

_THINKING_BINARY_EXPLANATION = (
    "Binary thinking grouping: 'non-thinking' = {off, false, none, minimal, low}; "
    "'thinking' = {on, true, enabled, medium, high}."
)

_BASE_WORD_FIXES = {
    "gemma": "Gemma",
    "gemini": "Gemini",
    "flash": "Flash",
    "lite": "Lite",
    "pro": "Pro",
    "it": "IT",
    "google": "Google",
    "moss": "MOSS",
    "audio": "Audio",
}

# Trailing variant suffixes that should be folded into the thinking-mode axis
# rather than treated as separate base models. MOSS ships its non-thinking and
# thinking variants under different model names (MOSS-Audio-Nb-Instruct vs
# MOSS-Audio-Nb-Thinking); we strip these so both variants alias to the same
# base model (e.g. "MOSS Audio 4B"), letting the existing thinking-mode binary
# grouping in figures handle the contrast.
_BASE_VARIANT_SUFFIXES = ("-Instruct", "-Thinking")


def halo_effect_by_evaluator(ann, report):
    """Per-evaluator within-judge halo effect on aesthetic ratings.

    For each evaluator (humans + every AI model × thinking config), compute:
      - mean aesthetic rating when the evaluator judged the song 'ai-generated'
      - mean aesthetic rating when the evaluator judged the song 'real'
      - the difference (real - AI) and a 95% bootstrap CI on that difference
    The result is appended to the text report and also written to
    output/halo_table.tex as a LaTeX fragment ready to \\input{} into the paper.
    Positive deltas (halo: higher ratings for perceived-real) are colored
    green; negative deltas are colored red.
    """
    report.append("")
    report.append("=" * 80)
    report.append("15. HALO EFFECT BY EVALUATOR (AESTHETIC RATING)")
    report.append("=" * 80)

    def _boot_ci_mean(vals, n_boot=2000, seed=42):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return np.nan, np.nan, np.nan
        rng = np.random.default_rng(seed)
        boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
        return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    def _boot_ci_diff(a, b, n_boot=2000, seed=17):
        a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
        b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
        if len(a) == 0 or len(b) == 0:
            return np.nan, np.nan, np.nan
        rng = np.random.default_rng(seed)
        diffs = (
            rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
            - rng.choice(b, size=(n_boot, len(b)), replace=True).mean(axis=1)
        )
        return float(a.mean() - b.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    rows = []

    # --- Humans ---
    h_aes = pd.to_numeric(ann["aesthetic_quality"], errors="coerce")
    h_ai = h_aes[ann["authenticity_assessment"] == "ai-generated"]
    h_re = h_aes[ann["authenticity_assessment"] == "real"]
    m_ai, lo_ai, hi_ai = _boot_ci_mean(h_ai)
    m_re, lo_re, hi_re = _boot_ci_mean(h_re)
    d, d_lo, d_hi = _boot_ci_diff(h_re, h_ai)
    rows.append({
        "evaluator": "Humans",
        "kind": "human",
        "n_ai": int(h_ai.notna().sum()),
        "n_real": int(h_re.notna().sum()),
        "ai_mean": m_ai, "ai_lo": lo_ai, "ai_hi": hi_ai,
        "real_mean": m_re, "real_lo": lo_re, "real_hi": hi_re,
        "diff": d, "diff_lo": d_lo, "diff_hi": d_hi,
    })

    # --- Models ---
    model_long = _load_model_eval_long(ann)
    if not model_long.empty:
        # Group on (base_model_alias, thinking_binary). The existing
        # thinking_alias is already the binary grouping ("thinking" /
        # "non-thinking") produced in _load_model_eval_long.
        for (base, think), g in model_long.groupby(
            ["base_model_alias", "thinking_alias"]
        ):
            m_aes = pd.to_numeric(g["aesthetic_quality"], errors="coerce")
            m_ai_s = m_aes[g["assessment"] == "ai-generated"]
            m_re_s = m_aes[g["assessment"] == "real"]
            if m_ai_s.notna().sum() == 0 and m_re_s.notna().sum() == 0:
                continue
            m_ai, lo_ai, hi_ai = _boot_ci_mean(m_ai_s)
            m_re, lo_re, hi_re = _boot_ci_mean(m_re_s)
            d, d_lo, d_hi = _boot_ci_diff(m_re_s, m_ai_s)
            rows.append({
                "evaluator": f"{base} ({think})",
                "kind": "model",
                "n_ai": int(m_ai_s.notna().sum()),
                "n_real": int(m_re_s.notna().sum()),
                "ai_mean": m_ai, "ai_lo": lo_ai, "ai_hi": hi_ai,
                "real_mean": m_re, "real_lo": lo_re, "real_hi": hi_re,
                "diff": d, "diff_lo": d_lo, "diff_hi": d_hi,
            })

    # Keep humans first; sort models by descending halo delta
    model_rows = [r for r in rows if r["kind"] == "model"]
    model_rows.sort(key=lambda r: (r["diff"] if pd.notna(r["diff"]) else -np.inf), reverse=True)
    rows = [r for r in rows if r["kind"] == "human"] + model_rows

    # --- Text report ---
    hdr = (
        f"  {'Evaluator':<44s}  {'n_AI':>5s}  {'judged AI mean [95% CI]':>26s}  "
        f"{'n_real':>6s}  {'judged real mean [95% CI]':>28s}  "
        f"{'Δ real-AI [95% CI]':>22s}"
    )
    report.append(hdr)
    report.append("  " + "-" * (len(hdr) - 2))
    for r in rows:
        ai_str = (f"{r['ai_mean']:.2f} [{r['ai_lo']:.2f}, {r['ai_hi']:.2f}]"
                  if pd.notna(r["ai_mean"]) else "---")
        re_str = (f"{r['real_mean']:.2f} [{r['real_lo']:.2f}, {r['real_hi']:.2f}]"
                  if pd.notna(r["real_mean"]) else "---")
        d_str = (f"{r['diff']:+.2f} [{r['diff_lo']:+.2f}, {r['diff_hi']:+.2f}]"
                 if pd.notna(r["diff"]) else "---")
        report.append(
            f"  {r['evaluator']:<44s}  {r['n_ai']:>5d}  {ai_str:>26s}  "
            f"{r['n_real']:>6d}  {re_str:>28s}  {d_str:>22s}"
        )

    # --- LaTeX fragment ---
    tex_lines = []
    tex_lines.append(
        r"% Auto-generated by analysis/main.py -- halo_effect_by_evaluator"
    )
    tex_lines.append(r"\begin{tabular}{lrcrcc}")
    tex_lines.append(r"\toprule")
    tex_lines.append(
        r"\textbf{Evaluator} & $n_{\text{AI}}$ & "
        r"\textbf{Judged ``AI'' mean [95\% CI]} & "
        r"$n_{\text{real}}$ & "
        r"\textbf{Judged ``real'' mean [95\% CI]} & "
        r"$\boldsymbol{\Delta}_{\text{real-AI}}$ \textbf{[95\% CI]} \\")
    tex_lines.append(r"\midrule")

    def _fmt_mean_ci(m, lo, hi):
        if pd.isna(m):
            return "---"
        return f"${m:.2f}$ $[{lo:.2f}, {hi:.2f}]$"

    def _fmt_diff(d, lo, hi):
        if pd.isna(d):
            return "---"
        color = "ForestGreen" if d >= 0 else "BrickRed"
        sign = "+" if d >= 0 else ""
        return (rf"\textcolor{{{color}}}{{${sign}{d:.2f}$ "
                rf"$[{lo:+.2f}, {hi:+.2f}]$}}")

    prev_kind = None
    for r in rows:
        if prev_kind == "human" and r["kind"] == "model":
            tex_lines.append(r"\midrule")
        prev_kind = r["kind"]
        name = r["evaluator"].replace("_", r"\_").replace("&", r"\&")
        tex_lines.append(
            f"{name} & {r['n_ai']} & "
            f"{_fmt_mean_ci(r['ai_mean'], r['ai_lo'], r['ai_hi'])} & "
            f"{r['n_real']} & "
            f"{_fmt_mean_ci(r['real_mean'], r['real_lo'], r['real_hi'])} & "
            f"{_fmt_diff(r['diff'], r['diff_lo'], r['diff_hi'])} \\\\"
        )
    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")

    tex_path = OUTPUT_DIR / "halo_table.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    report.append("")
    report.append(f"  (LaTeX fragment written to {tex_path})")


def ground_truth_rating_by_evaluator(ann, report):
    """Per-evaluator aesthetic rating split by ground-truth provenance.

    Sibling of halo_effect_by_evaluator, but splits on ground-truth source
    (is the song actually AI vs actually human) rather than on the evaluator's
    own authenticity judgment. This exposes the true quality signal each
    evaluator assigns to real vs AI music, independent of their detection
    behaviour. Writes output/gt_rating_table.tex ready to \\input{}.
    """
    report.append("")
    report.append("=" * 80)
    report.append("16. GROUND-TRUTH AESTHETIC RATING BY EVALUATOR")
    report.append("=" * 80)

    def _boot_ci_mean(vals, n_boot=2000, seed=42):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return np.nan, np.nan, np.nan
        rng = np.random.default_rng(seed)
        boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
        return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    def _boot_ci_diff(a, b, n_boot=2000, seed=17):
        a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
        b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
        if len(a) == 0 or len(b) == 0:
            return np.nan, np.nan, np.nan
        rng = np.random.default_rng(seed)
        diffs = (
            rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
            - rng.choice(b, size=(n_boot, len(b)), replace=True).mean(axis=1)
        )
        return float(a.mean() - b.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    rows = []

    # --- Humans ---
    h_aes = pd.to_numeric(ann["aesthetic_quality"], errors="coerce")
    h_is_ai = ann["is_ai_song"].astype(bool)
    h_ai = h_aes[h_is_ai]
    h_re = h_aes[~h_is_ai]
    m_ai, lo_ai, hi_ai = _boot_ci_mean(h_ai)
    m_re, lo_re, hi_re = _boot_ci_mean(h_re)
    d, d_lo, d_hi = _boot_ci_diff(h_re, h_ai)
    rows.append({
        "evaluator": "Humans",
        "kind": "human",
        "n_ai": int(h_ai.notna().sum()),
        "n_real": int(h_re.notna().sum()),
        "ai_mean": m_ai, "ai_lo": lo_ai, "ai_hi": hi_ai,
        "real_mean": m_re, "real_lo": lo_re, "real_hi": hi_re,
        "diff": d, "diff_lo": d_lo, "diff_hi": d_hi,
    })

    # --- Models ---
    model_long = _load_model_eval_long(ann)
    if not model_long.empty:
        model_long = model_long.copy()
        model_long["is_ai_song"] = model_long["song_source"] != "human"
        for (base, think), g in model_long.groupby(
            ["base_model_alias", "thinking_alias"]
        ):
            m_aes = pd.to_numeric(g["aesthetic_quality"], errors="coerce")
            m_ai_s = m_aes[g["is_ai_song"].astype(bool)]
            m_re_s = m_aes[~g["is_ai_song"].astype(bool)]
            if m_ai_s.notna().sum() == 0 and m_re_s.notna().sum() == 0:
                continue
            m_ai, lo_ai, hi_ai = _boot_ci_mean(m_ai_s)
            m_re, lo_re, hi_re = _boot_ci_mean(m_re_s)
            d, d_lo, d_hi = _boot_ci_diff(m_re_s, m_ai_s)
            rows.append({
                "evaluator": f"{base} ({think})",
                "kind": "model",
                "n_ai": int(m_ai_s.notna().sum()),
                "n_real": int(m_re_s.notna().sum()),
                "ai_mean": m_ai, "ai_lo": lo_ai, "ai_hi": hi_ai,
                "real_mean": m_re, "real_lo": lo_re, "real_hi": hi_re,
                "diff": d, "diff_lo": d_lo, "diff_hi": d_hi,
            })

    # Keep humans first; sort models by descending delta
    model_rows = [r for r in rows if r["kind"] == "model"]
    model_rows.sort(key=lambda r: (r["diff"] if pd.notna(r["diff"]) else -np.inf), reverse=True)
    rows = [r for r in rows if r["kind"] == "human"] + model_rows

    # --- Text report ---
    hdr = (
        f"  {'Evaluator':<44s}  {'n_AI':>5s}  {'AI-song mean [95% CI]':>24s}  "
        f"{'n_real':>6s}  {'real-song mean [95% CI]':>26s}  "
        f"{'Δ real-AI [95% CI]':>22s}"
    )
    report.append(hdr)
    report.append("  " + "-" * (len(hdr) - 2))
    for r in rows:
        ai_str = (f"{r['ai_mean']:.2f} [{r['ai_lo']:.2f}, {r['ai_hi']:.2f}]"
                  if pd.notna(r["ai_mean"]) else "---")
        re_str = (f"{r['real_mean']:.2f} [{r['real_lo']:.2f}, {r['real_hi']:.2f}]"
                  if pd.notna(r["real_mean"]) else "---")
        d_str = (f"{r['diff']:+.2f} [{r['diff_lo']:+.2f}, {r['diff_hi']:+.2f}]"
                 if pd.notna(r["diff"]) else "---")
        report.append(
            f"  {r['evaluator']:<44s}  {r['n_ai']:>5d}  {ai_str:>24s}  "
            f"{r['n_real']:>6d}  {re_str:>26s}  {d_str:>22s}"
        )

    # --- LaTeX fragment ---
    tex_lines = []
    tex_lines.append(
        r"% Auto-generated by analysis/main.py -- ground_truth_rating_by_evaluator"
    )
    tex_lines.append(r"\begin{tabular}{lrcrcc}")
    tex_lines.append(r"\toprule")
    tex_lines.append(
        r"\textbf{Evaluator} & $n_{\text{AI}}$ & "
        r"\textbf{AI-song mean [95\% CI]} & "
        r"$n_{\text{real}}$ & "
        r"\textbf{Real-song mean [95\% CI]} & "
        r"$\boldsymbol{\Delta}_{\text{real-AI}}$ \textbf{[95\% CI]} \\")
    tex_lines.append(r"\midrule")

    def _fmt_mean_ci(m, lo, hi):
        if pd.isna(m):
            return "---"
        return f"${m:.2f}$ $[{lo:.2f}, {hi:.2f}]$"

    def _fmt_diff(d, lo, hi):
        if pd.isna(d):
            return "---"
        color = "ForestGreen" if d >= 0 else "BrickRed"
        sign = "+" if d >= 0 else ""
        return (rf"\textcolor{{{color}}}{{${sign}{d:.2f}$ "
                rf"$[{lo:+.2f}, {hi:+.2f}]$}}")

    prev_kind = None
    for r in rows:
        if prev_kind == "human" and r["kind"] == "model":
            tex_lines.append(r"\midrule")
        prev_kind = r["kind"]
        name = r["evaluator"].replace("_", r"\_").replace("&", r"\&")
        tex_lines.append(
            f"{name} & {r['n_ai']} & "
            f"{_fmt_mean_ci(r['ai_mean'], r['ai_lo'], r['ai_hi'])} & "
            f"{r['n_real']} & "
            f"{_fmt_mean_ci(r['real_mean'], r['real_lo'], r['real_hi'])} & "
            f"{_fmt_diff(r['diff'], r['diff_lo'], r['diff_hi'])} \\\\"
        )
    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")

    tex_path = OUTPUT_DIR / "gt_rating_table.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    report.append("")
    report.append(f"  (LaTeX fragment written to {tex_path})")


# Genre-family classification used by the aesthetic predictors LMM. Mapping is
# applied to the first matching token in song_genres (a JSON list field), with
# longer/more-specific keywords listed first so that "deep house" and "synth pop"
# resolve to Electronic and Pop respectively. Rare or untagged songs fall into
# "Other".
_GENRE_FAMILY_KEYWORDS = [
    ("metal", "Rock/Metal"), ("punk", "Rock/Metal"),
    ("grunge", "Rock/Metal"), ("rock", "Rock/Metal"),
    ("hip hop", "Hip-Hop/Rap"), ("hip-hop", "Hip-Hop/Rap"),
    ("rap", "Hip-Hop/Rap"), ("trap", "Hip-Hop/Rap"),
    ("r&b", "R&B/Soul"), ("rnb", "R&B/Soul"),
    ("soul", "R&B/Soul"), ("funk", "R&B/Soul"), ("disco", "R&B/Soul"),
    ("jazz", "Jazz"), ("blues", "Jazz"),
    ("country", "Country/Folk"), ("folk", "Country/Folk"),
    ("americana", "Country/Folk"),
    ("classical", "Classical"), ("orchestr", "Classical"),
    ("piano", "Classical"), ("cinematic", "Classical"),
    ("house", "Electronic"), ("techno", "Electronic"),
    ("edm", "Electronic"), ("trance", "Electronic"),
    ("dubstep", "Electronic"), ("synth", "Electronic"),
    ("ambient", "Electronic"), ("electro", "Electronic"),
    ("dance", "Electronic"), ("electronic", "Electronic"),
    ("reggae", "Reggae/Latin"), ("salsa", "Reggae/Latin"),
    ("bachata", "Reggae/Latin"), ("cumbia", "Reggae/Latin"),
    ("ska", "Reggae/Latin"), ("latin", "Reggae/Latin"),
    ("pop", "Pop"),
]


def _genre_family(genres):
    if not isinstance(genres, list) or not genres:
        return "Other"
    for g in genres:
        gl = str(g).lower()
        for kw, fam in _GENRE_FAMILY_KEYWORDS:
            if kw in gl:
                return fam
    return "Other"


def _safe_ref_level(series, preferred, fallback_default):
    """Pick a valid treatment reference level present in data."""
    counts = series.value_counts(dropna=True)
    if counts.empty:
        return fallback_default
    if preferred in counts.index:
        return preferred
    return str(counts.index[0])


def _collapse_rare_levels(series, min_count=20, keep_levels=None, other_label="Other"):
    """Collapse sparse categorical levels to stabilize mixed-model estimation."""
    keep_levels = set(keep_levels or [])
    counts = series.value_counts(dropna=True)
    rare = {k for k, v in counts.items() if v < min_count and k not in keep_levels}
    return series.apply(lambda x: other_label if x in rare else x)


def aesthetic_predictors_model(ann, report):
    """Crossed mixed-effects model: which factors predict higher aesthetic
    ratings?

    Fits a Gaussian linear mixed-effects model on aesthetic_quality with
    crossed random intercepts for participant_id and song_id (statsmodels
    REML, with song-level intercepts entered via vc_formula). Fixed effects
    cover ground-truth song source, the rater's authenticity judgment,
    genre family, snippet condition, and participant-level covariates
    (musical engagement, AI-music experience, log annotation duration).

    Reports each fixed effect with SE, 95% CI and Wald p, the participant /
    song / residual variance components, and Nakagawa & Schielzeth (2013)
    marginal/conditional pseudo-R^2. The same content is mirrored to
    output/aesthetic_lmm_table.tex for the paper.
    """
    report.append("\n" + "=" * 80)
    report.append("17. AESTHETIC PREDICTORS: CROSSED MIXED-EFFECTS MODEL")
    report.append("=" * 80)
    report.append(
        "\n  Linear mixed-effects model on aesthetic_quality (1-10) with"
        "\n  crossed random intercepts for participant and song (REML)."
        "\n  Genre is collapsed to broad families (first matching token in"
        "\n  song_genres). Continuous predictor (log annotation duration) is"
        "\n  z-scored for interpretable coefficients."
        "\n  Pseudo-R^2 follows Nakagawa & Schielzeth (2013)."
    )

    df = ann.copy()
    df["genre_family"] = df["song_genres"].apply(_genre_family)
    df["song_source"] = df["song_source"].astype(str).str.strip().str.lower()
    df["authenticity_assessment"] = (
        df["authenticity_assessment"].astype(str).str.strip().str.lower()
    )
    df["snippet_condition"] = df["snippet_condition"].astype(str).str.strip().str.lower()
    df["participant_musical_engagement"] = (
        df["participant_musical_engagement"].astype(str).str.strip().str.lower()
    )
    df["participant_ai_music_experience"] = (
        df["participant_ai_music_experience"].astype(str).str.strip().str.lower()
    )
    duration = pd.to_numeric(df["annotation_duration_ms"], errors="coerce")
    df["log_duration"] = np.log1p(duration.clip(lower=1.0))

    # Personalized covariates: age (median-imputed with missingness flag),
    # taste breadth, favourite-genre vs.\ song-genre family match,
    # listening device, environment, and per-trial song familiarity (a
    # potential confound: a rater who recognises the song is no longer
    # purely doing AI-vs-real perception). Listening context is excluded
    # because >95% of trials are in the "alone" cell.
    df["participant_listening_device"] = (
        df["participant_listening_device"].astype(str).str.strip()
    )
    df["participant_environment"] = (
        df["participant_environment"].astype(str).str.strip()
    )
    df["fav_genre_match_label"] = (
        df["fav_genre_match_label"].astype(str).str.strip()
    )
    df["familiarity_level"] = (
        df["familiarity_level"].astype(str).str.strip().str.lower()
    )

    keep_cols = [
        "aesthetic_quality", "song_source", "authenticity_assessment",
        "genre_family", "snippet_condition", "participant_musical_engagement",
        "participant_ai_music_experience", "log_duration",
        "participant_id", "song_id",
        "participant_age_imputed", "participant_age_missing",
        "participant_taste_breadth",
        "participant_listening_device", "participant_environment",
        "fav_genre_match_label", "familiarity_level",
    ]
    work = df.dropna(subset=keep_cols).copy()
    if len(work) < 50:
        report.append(f"\n  Insufficient data after dropna ({len(work)}); skipping.")
        return

    work["genre_family"] = _collapse_rare_levels(
        work["genre_family"], min_count=20, keep_levels={"Other"}, other_label="Other"
    )

    src_ref = _safe_ref_level(work["song_source"], preferred="human", fallback_default="human")
    auth_ref = _safe_ref_level(
        work["authenticity_assessment"], preferred="real", fallback_default="real"
    )
    snippet_ref = _safe_ref_level(
        work["snippet_condition"], preferred="original", fallback_default="original"
    )
    engage_ref = _safe_ref_level(
        work["participant_musical_engagement"],
        preferred="casual",
        fallback_default="casual",
    )

    duration_sd = work["log_duration"].std()
    if pd.isna(duration_sd) or duration_sd == 0:
        work["log_duration_z"] = 0.0
    else:
        work["log_duration_z"] = (
            (work["log_duration"] - work["log_duration"].mean())
            / duration_sd
        )

    age_sd = work["participant_age_imputed"].std()
    if pd.isna(age_sd) or age_sd == 0:
        work["age_z"] = 0.0
    else:
        work["age_z"] = (
            (work["participant_age_imputed"] - work["participant_age_imputed"].mean())
            / age_sd
        )
    breadth_sd = work["participant_taste_breadth"].std()
    if pd.isna(breadth_sd) or breadth_sd == 0:
        work["taste_breadth_z"] = 0.0
    else:
        work["taste_breadth_z"] = (
            (work["participant_taste_breadth"] - work["participant_taste_breadth"].mean())
            / breadth_sd
        )

    device_ref = _safe_ref_level(
        work["participant_listening_device"],
        preferred="Headphones (on-ear)",
        fallback_default="Laptop/phone speakers",
    )
    env_ref = _safe_ref_level(
        work["participant_environment"],
        preferred="Quiet room",
        fallback_default="Quiet room",
    )

    genre_counts = work["genre_family"].value_counts()
    genre_ref = _safe_ref_level(work["genre_family"], preferred="Pop", fallback_default="Other")

    fam_ref = _safe_ref_level(
        work["familiarity_level"], preferred="never", fallback_default="never"
    )

    formula = (
        "aesthetic_quality ~ "
        f"C(song_source, Treatment('{src_ref}')) "
        f"+ C(authenticity_assessment, Treatment('{auth_ref}')) "
        f"+ C(genre_family, Treatment('{genre_ref}')) "
        f"+ C(snippet_condition, Treatment('{snippet_ref}')) "
        f"+ C(participant_musical_engagement, Treatment('{engage_ref}')) "
        "+ C(participant_ai_music_experience) "
        "+ log_duration_z "
        "+ age_z + C(participant_age_missing) + taste_breadth_z "
        f"+ C(participant_listening_device, Treatment('{device_ref}')) "
        f"+ C(participant_environment, Treatment('{env_ref}')) "
        "+ C(fav_genre_match_label, Treatment('unknown')) "
        f"+ C(familiarity_level, Treatment('{fam_ref}'))"
    )

    report.append(
        f"\n  N = {len(work)} annotations | "
        f"{work['participant_id'].nunique()} participants | "
        f"{work['song_id'].nunique()} songs | "
        f"{work['genre_family'].nunique()} genre families "
        f"(genre reference: {genre_ref})"
    )
    report.append(
        "  Reference levels: "
        f"source={src_ref}, judged={auth_ref}, snippet={snippet_ref}, "
        f"engagement={engage_ref}, genre={genre_ref}, "
        f"device={device_ref}, environment={env_ref}, fav-genre=unknown, "
        f"familiarity={fam_ref}"
    )
    report.append(
        "  Genre distribution: "
        + ", ".join(f"{k}={v}" for k, v in genre_counts.items())
    )

    crossed = True
    try:
        md = smf.mixedlm(
            formula, data=work, groups=work["participant_id"],
            vc_formula={"song": "0 + C(song_id)"},
        )
        result = md.fit(reml=True, method="lbfgs", maxiter=1000)
    except Exception as e:
        report.append(f"\n  Crossed random-effects fit failed: {e}")
        report.append("  Falling back to participant-only random intercepts.")
        crossed = False
        try:
            md = smf.mixedlm(formula, data=work, groups=work["participant_id"])
            result = md.fit(reml=True, method="lbfgs", maxiter=1000)
        except Exception as e2:
            report.append(f"  Participant-only fit also failed: {e2}")
            return

    fe = result.fe_params
    bse_all = result.bse
    pv_all = result.pvalues
    ci_all = result.conf_int()
    ci_all.columns = ["lo", "hi"]

    se = bse_all.reindex(fe.index)
    pv = pv_all.reindex(fe.index)
    ci = ci_all.reindex(fe.index)

    def _pretty(name):
        s = name
        s = s.replace(f"C(song_source, Treatment('{src_ref}'))", "Source")
        s = s.replace(f"C(authenticity_assessment, Treatment('{auth_ref}'))", "Judged")
        s = s.replace(f"C(genre_family, Treatment('{genre_ref}'))", "Genre")
        s = s.replace(f"C(snippet_condition, Treatment('{snippet_ref}'))", "Snippet")
        s = s.replace(
            f"C(participant_musical_engagement, Treatment('{engage_ref}'))", "Engagement"
        )
        s = s.replace("C(participant_ai_music_experience)", "AI-exp.")
        s = s.replace(
            f"C(participant_listening_device, Treatment('{device_ref}'))", "Device"
        )
        s = s.replace(
            f"C(participant_environment, Treatment('{env_ref}'))", "Environment"
        )
        s = s.replace(
            "C(fav_genre_match_label, Treatment('unknown'))", "FavGenre"
        )
        s = s.replace(
            f"C(familiarity_level, Treatment('{fam_ref}'))", "Familiarity"
        )
        s = s.replace("C(participant_age_missing)", "AgeMissing")
        s = s.replace("age_z", "Age (z)")
        s = s.replace("taste_breadth_z", "TasteBreadth (z)")
        s = s.replace("[T.", " = ").replace("]", "")
        return s

    report.append("\n  --- Fixed Effects ---")
    hdr = (
        f"  {'Term':<60s}  {'beta':>8s}  {'SE':>6s}  "
        f"{'95% CI':>22s}  {'p':>10s}"
    )
    report.append(hdr)
    report.append("  " + "-" * (len(hdr) - 2))
    for name in fe.index:
        b = float(fe[name])
        s_ = float(se.loc[name]) if pd.notna(se.loc[name]) else np.nan
        p_ = float(pv.loc[name]) if pd.notna(pv.loc[name]) else np.nan
        lo = float(ci.loc[name, "lo"]) if pd.notna(ci.loc[name, "lo"]) else np.nan
        hi = float(ci.loc[name, "hi"]) if pd.notna(ci.loc[name, "hi"]) else np.nan
        ci_str = f"[{lo:+.2f}, {hi:+.2f}]"
        sig = (
            "***" if pd.notna(p_) and p_ < 0.001 else
            "**" if pd.notna(p_) and p_ < 0.01 else
            "*" if pd.notna(p_) and p_ < 0.05 else ""
        )
        report.append(
            f"  {_pretty(name):<60s}  {b:+8.3f}  {s_:6.3f}  "
            f"{ci_str:>22s}  {p_:10.4g} {sig}"
        )

    var_resid = float(result.scale)
    var_pid = np.nan
    var_pid_at_boundary = False
    if (
        hasattr(result, "cov_re") and result.cov_re is not None
        and getattr(result.cov_re, "size", 0)
    ):
        try:
            var_pid = float(np.asarray(result.cov_re)[0, 0])
        except Exception:
            var_pid = np.nan
    else:
        # Empty cov_re means REML pushed the participant variance to the
        # boundary (effectively zero) -- the fixed-effects covariates have
        # absorbed all between-participant variance.
        var_pid = 0.0
        var_pid_at_boundary = True
    var_song = np.nan
    if (
        crossed
        and hasattr(result, "vcomp")
        and result.vcomp is not None
        and len(result.vcomp)
    ):
        try:
            var_song = float(np.asarray(result.vcomp)[0])
        except Exception:
            var_song = np.nan

    Xb = np.asarray(md.exog) @ fe.values
    var_fixed = float(np.var(Xb, ddof=0))
    var_re_total = (
        (var_pid if pd.notna(var_pid) else 0.0)
        + (var_song if pd.notna(var_song) else 0.0)
    )
    denom = var_fixed + var_re_total + var_resid
    R2_m = var_fixed / denom if denom > 0 else np.nan
    R2_c = (var_fixed + var_re_total) / denom if denom > 0 else np.nan

    report.append("\n  --- Variance Components ---")
    report.append(f"  sigma^2 (residual)    = {var_resid:.3f}")
    if pd.notna(var_pid):
        suffix = " (boundary; absorbed by participant covariates)" if var_pid_at_boundary else ""
        report.append(f"  tau^2 (participant)   = {var_pid:.3f}{suffix}")
    else:
        report.append("  tau^2 (participant)   = ---")
    if crossed:
        if pd.notna(var_song):
            report.append(f"  tau^2 (song)          = {var_song:.3f}")
        else:
            report.append("  tau^2 (song)          = ---")
    report.append(f"  Marginal R^2 (fixed only)        = {R2_m:.3f}")
    report.append(f"  Conditional R^2 (fixed + random) = {R2_c:.3f}")
    report.append(f"  Log-likelihood (REML)            = {float(result.llf):.3f}")

    pos_sig = []
    neg_sig = []
    for name in fe.index:
        if name == "Intercept":
            continue
        p_ = float(pv.loc[name]) if pd.notna(pv.loc[name]) else np.nan
        b = float(fe[name])
        if pd.notna(p_) and p_ < 0.05:
            (pos_sig if b > 0 else neg_sig).append((_pretty(name), b, p_))
    pos_sig.sort(key=lambda x: x[1], reverse=True)
    neg_sig.sort(key=lambda x: x[1])
    if pos_sig:
        report.append("  Significant positive predictors (higher aesthetics):")
        for nm, b, p_ in pos_sig:
            report.append(f"    + {nm}: beta={b:+.3f}, p={p_:.4g}")
    if neg_sig:
        report.append("  Significant negative predictors (lower aesthetics):")
        for nm, b, p_ in neg_sig:
            report.append(f"    - {nm}: beta={b:+.3f}, p={p_:.4g}")
    if not pos_sig and not neg_sig:
        report.append("  No fixed-effect predictors reached p < 0.05.")

    tex = []
    tex.append(r"% Auto-generated by analysis/main.py -- aesthetic_predictors_model")
    tex.append(r"\begin{tabular}{lrrcc}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Predictor} & $\hat{\beta}$ & SE & 95\% CI & $p$ \\")
    tex.append(r"\midrule")
    for name in fe.index:
        b = float(fe[name])
        s_ = float(se.loc[name]) if pd.notna(se.loc[name]) else np.nan
        p_ = float(pv.loc[name]) if pd.notna(pv.loc[name]) else np.nan
        lo = float(ci.loc[name, "lo"]) if pd.notna(ci.loc[name, "lo"]) else np.nan
        hi = float(ci.loc[name, "hi"]) if pd.notna(ci.loc[name, "hi"]) else np.nan
        sig = (
            r"^{***}" if pd.notna(p_) and p_ < 0.001 else
            r"^{**}" if pd.notna(p_) and p_ < 0.01 else
            r"^{*}" if pd.notna(p_) and p_ < 0.05 else ""
        )
        nm = _pretty(name).replace("&", r"\&").replace("_", r"\_")
        ci_str = (
            rf"$[{lo:+.2f}, {hi:+.2f}]$" if pd.notna(lo) and pd.notna(hi) else "---"
        )
        p_str = f"${p_:.3g}$" if pd.notna(p_) else "---"
        se_str = f"${s_:.3f}$" if pd.notna(s_) else "---"
        tex.append(
            f"{nm} & ${b:+.3f}{sig}$ & {se_str} & {ci_str} & {p_str} \\\\"
        )
    tex.append(r"\midrule")
    tex.append(rf"$\sigma^2$ residual & {var_resid:.3f} & & & \\")
    if pd.notna(var_pid):
        suffix = r"$^{\dagger}$" if var_pid_at_boundary else ""
        tex.append(rf"$\tau^2$ participant{suffix} & {var_pid:.3f} & & & \\")
    else:
        tex.append(r"$\tau^2$ participant & --- & & & \\")
    if crossed:
        tex.append(
            (rf"$\tau^2$ song & {var_song:.3f} & & & \\")
            if pd.notna(var_song)
            else r"$\tau^2$ song & --- & & & \\"
        )
    tex.append(rf"Marginal $R^2$ & {R2_m:.3f} & & & \\")
    tex.append(rf"Conditional $R^2$ & {R2_c:.3f} & & & \\")
    tex.append(rf"LogLik (REML) & {float(result.llf):.3f} & & & \\")
    if pos_sig:
        tex.append(r"\midrule")
        tex.append(r"\multicolumn{5}{l}{\textit{Significant positive predictors (}p\textit{<0.05):}} \\")
        for nm, b, p_ in pos_sig:
            nm_tex = nm.replace("&", r"\&").replace("_", r"\_")
            tex.append(
                rf"\multicolumn{{5}}{{l}}{{\quad {nm_tex}: $\hat{{\beta}}={b:+.3f}$, $p={p_:.3g}$}} \\"
            )
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    tex_path = OUTPUT_DIR / "aesthetic_lmm_table.tex"
    tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    report.append("")
    report.append(f"  (LaTeX fragment written to {tex_path})")


# ---------------------------------------------------------------------------
# Personalization analysis (Section 18)
# ---------------------------------------------------------------------------

def personalization_analysis(ann, report):
    """Personalized perceptual benchmark.

    The dataset is not just a population-average judgement: every trial is
    tagged with the rater's age, taste profile (favourite genres),
    listening device and listening environment, and we can ask whether the
    *song's* declared genre matches one of the rater's favourites. This
    section reports descriptive accuracy and aesthetic-quality means by
    each of those axes, then fits two compact models that report only the
    personalized covariates after partialling out the previously reported
    baseline (source, snippet, and the standard listener traits).

    Output:
      * a textual breakdown for the report,
      * ``output/personalization_table.tex`` with the focused coefficient
        table referenced from the paper.
    """
    report.append("\n" + "=" * 80)
    report.append("18. PERSONALIZED PERCEPTION: AGE, TASTE & LISTENING CONDITIONS")
    report.append("=" * 80)
    report.append(
        "\n  Each annotation is tagged with the rater's age, favourite music"
        "\n  genres, listening device and ambient environment. The same"
        "\n  song can be heard by raters who do or do not list its genre as a"
        "\n  favourite, on different devices, in different rooms. We treat"
        "\n  these as first-class predictors of detection accuracy and"
        "\n  aesthetic ratings, turning the dataset into a *personalised*"
        "\n  perceptual benchmark."
    )

    # ---- Descriptives ----------------------------------------------------
    work = ann.copy()
    work["age_bucket"] = pd.cut(
        work["participant_age_imputed"],
        bins=[-1, 24, 34, 44, 200],
        labels=["<=24", "25-34", "35-44", "45+"],
    )
    work.loc[work["participant_age_missing"] == 1, "age_bucket"] = pd.NA

    def _row(g):
        n = len(g)
        acc = g["is_correct"].mean(skipna=True)
        aes = pd.to_numeric(g["aesthetic_quality"], errors="coerce").mean()
        return pd.Series({
            "n": n,
            "accuracy": acc,
            "aesthetic_mean": aes,
            "hit_rate_AI": g.loc[g["is_ai_song"], "detected_ai"].mean(),
            "fa_rate_real": g.loc[~g["is_ai_song"], "detected_ai"].mean(),
        })

    report.append("\n  --- Accuracy & aesthetics by age bucket ---")
    if work["age_bucket"].notna().any():
        report.append(
            work.dropna(subset=["age_bucket"]).groupby("age_bucket", observed=True)
                .apply(_row).round(3).to_string()
        )
    else:
        report.append("    (no age data available)")

    report.append("\n  --- By favourite-genre match (song genre vs rater favourites) ---")
    report.append(
        work.groupby("fav_genre_match_label").apply(_row).round(3).to_string()
    )

    report.append("\n  --- By listening device ---")
    report.append(
        work.groupby("participant_listening_device").apply(_row).round(3).to_string()
    )

    report.append("\n  --- By listening environment ---")
    report.append(
        work.groupby("participant_environment").apply(_row).round(3).to_string()
    )

    report.append("\n  --- By taste breadth (number of favourite genres) ---")
    work["breadth_q"] = pd.qcut(
        work["participant_taste_breadth"], q=4, duplicates="drop"
    )
    report.append(
        work.groupby("breadth_q", observed=True).apply(_row).round(3).to_string()
    )

    report.append("\n  --- By song familiarity (per-trial self-report) ---")
    report.append(
        work.groupby("familiarity_level").apply(_row).round(3).to_string()
    )

    # Robustness check: restrict to "never"-familiar trials so song
    # recognition cannot leak into the AI/real call.
    report.append(
        "\n  --- Robustness: detection metrics restricted to "
        "familiarity = 'never' (uncontaminated perceptual condition) ---"
    )
    fam_never = work[work["familiarity_level"] == "never"].copy()
    det_never = fam_never[fam_never["authenticity_assessment"] != "uncertain"]
    if len(det_never) >= 50:
        h = int(((det_never["is_ai_song"]) & (det_never["detected_ai"] == 1)).sum())
        m_ = int(((det_never["is_ai_song"]) & (det_never["detected_ai"] == 0)).sum())
        f_ = int(((~det_never["is_ai_song"]) & (det_never["detected_ai"] == 1)).sum())
        c_ = int(((~det_never["is_ai_song"]) & (det_never["detected_ai"] == 0)).sum())
        mm = _sdt_metrics(h, m_, f_, c_)
        unc_rate = (fam_never["authenticity_assessment"] == "uncertain").mean()
        report.append(
            f"    n_trials={len(fam_never)}, n_non_uncertain={len(det_never)}, "
            f"accuracy={mm['accuracy']*100:.1f}%, "
            f"hit_rate={mm['hit_rate']:.3f}, far={mm['false_alarm_rate']:.3f}, "
            f"d'={mm['d_prime']:.3f}, c={mm['criterion']:.3f}, "
            f"uncertain_rate={unc_rate*100:.1f}%"
        )
        # Save the headline numbers for the LaTeX prose to reference.
        ann.attrs["familiarity_never_metrics"] = mm
    else:
        report.append("    Insufficient 'never'-familiar trials for the check.")

    # ---- Compact LMM / GLM with only personalization coefficients ------
    rows = []  # tex rows: (label, beta, se, ci_lo, ci_hi, p, model_kind)

    # 1) Logistic on correctness (non-uncertain trials).
    det = work[work["authenticity_assessment"] != "uncertain"].copy()
    det["correct_int"] = det["is_correct"].astype(int)
    det["log_duration"] = np.log1p(
        pd.to_numeric(det["annotation_duration_ms"], errors="coerce").fillna(0)
    )
    det["log_duration_z"] = (
        (det["log_duration"] - det["log_duration"].mean())
        / (det["log_duration"].std() or 1.0)
    )
    age_sd = det["participant_age_imputed"].std() or 1.0
    det["age_z"] = (
        (det["participant_age_imputed"] - det["participant_age_imputed"].mean())
        / age_sd
    )
    breadth_sd = det["participant_taste_breadth"].std() or 1.0
    det["taste_breadth_z"] = (
        (det["participant_taste_breadth"] - det["participant_taste_breadth"].mean())
        / breadth_sd
    )
    eng_map = {"casual": 0, "enthusiast": 1, "musician": 2, "professional": 3}
    det["engagement_num"] = (
        det["participant_musical_engagement"].astype(str).str.strip().str.lower()
        .map(eng_map).fillna(0)
    )
    ai_exp_map = {
        "Heard about it but never tried": 0,
        "Tried once or twice": 1,
        "Use occasionally": 2,
        "Use regularly": 3,
        "Professional experience with AI music": 4,
    }
    det["ai_exp_num"] = det["participant_ai_music_experience"].map(ai_exp_map).fillna(0)
    det["is_ai_song_int"] = det["is_ai_song"].astype(int)

    device_ref = _safe_ref_level(
        det["participant_listening_device"].astype(str),
        preferred="Headphones (on-ear)",
        fallback_default="Laptop/phone speakers",
    )
    env_ref = _safe_ref_level(
        det["participant_environment"].astype(str),
        preferred="Quiet room",
        fallback_default="Quiet room",
    )

    formula_acc = (
        "correct_int ~ is_ai_song_int + "
        "C(snippet_condition, Treatment(reference='original')) + "
        "ai_exp_num + engagement_num + log_duration_z + "
        "age_z + C(participant_age_missing) + taste_breadth_z + "
        f"C(participant_listening_device, Treatment('{device_ref}')) + "
        f"C(participant_environment, Treatment('{env_ref}')) + "
        "C(fav_genre_match_label, Treatment('unknown'))"
    )
    report.append(
        "\n  --- Logistic regression on correctness (personalized terms only shown) ---"
    )
    try:
        res_acc = smf.logit(formula_acc, data=det).fit(disp=False, maxiter=200)
        report.append(f"  N={res_acc.nobs:.0f}, pseudo-R^2={res_acc.prsquared:.3f}, "
                      f"log-lik={res_acc.llf:.2f}, AIC={res_acc.aic:.2f}")
        keep = [
            n for n in res_acc.params.index
            if any(k in n for k in [
                "age_z", "participant_age_missing", "taste_breadth_z",
                "participant_listening_device", "participant_environment",
                "fav_genre_match_label",
            ])
        ]
        ci = res_acc.conf_int()
        ci.columns = ["lo", "hi"]
        for n in keep:
            b = float(res_acc.params[n])
            s_ = float(res_acc.bse[n])
            p_ = float(res_acc.pvalues[n])
            lo, hi = float(ci.loc[n, "lo"]), float(ci.loc[n, "hi"])
            rows.append(("Accuracy (logit)", n, b, s_, lo, hi, p_,
                         np.exp(b), np.exp(lo), np.exp(hi)))
            report.append(f"    {n}: beta={b:+.3f}, OR={np.exp(b):.3f}, "
                          f"95% CI [{np.exp(lo):.2f}, {np.exp(hi):.2f}], p={p_:.4g}")
    except Exception as e:
        report.append(f"  Accuracy model failed: {e}")

    # 2) Linear LMM on aesthetic quality with crossed random effects.
    aes = work.dropna(subset=[
        "aesthetic_quality", "song_source", "participant_id", "song_id",
        "participant_listening_device", "participant_environment",
        "participant_age_imputed",
    ]).copy()
    aes["log_duration"] = np.log1p(
        pd.to_numeric(aes["annotation_duration_ms"], errors="coerce").fillna(0)
    )
    aes["log_duration_z"] = (
        (aes["log_duration"] - aes["log_duration"].mean())
        / (aes["log_duration"].std() or 1.0)
    )
    aes["age_z"] = (
        (aes["participant_age_imputed"] - aes["participant_age_imputed"].mean())
        / (aes["participant_age_imputed"].std() or 1.0)
    )
    aes["taste_breadth_z"] = (
        (aes["participant_taste_breadth"] - aes["participant_taste_breadth"].mean())
        / (aes["participant_taste_breadth"].std() or 1.0)
    )
    device_ref_aes = _safe_ref_level(
        aes["participant_listening_device"].astype(str),
        preferred="Headphones (on-ear)",
        fallback_default="Laptop/phone speakers",
    )
    env_ref_aes = _safe_ref_level(
        aes["participant_environment"].astype(str),
        preferred="Quiet room",
        fallback_default="Quiet room",
    )

    formula_aes = (
        "aesthetic_quality ~ "
        "C(song_source, Treatment('human')) + "
        "C(authenticity_assessment, Treatment('real')) + "
        "C(snippet_condition, Treatment('original')) + log_duration_z + "
        "age_z + C(participant_age_missing) + taste_breadth_z + "
        f"C(participant_listening_device, Treatment('{device_ref_aes}')) + "
        f"C(participant_environment, Treatment('{env_ref_aes}')) + "
        "C(fav_genre_match_label, Treatment('unknown'))"
    )
    report.append(
        "\n  --- LMM on aesthetic quality (personalized terms only shown) ---"
    )
    try:
        md = smf.mixedlm(
            formula_aes, data=aes, groups=aes["participant_id"],
            vc_formula={"song": "0 + C(song_id)"},
        )
        res_aes = md.fit(reml=True, method="lbfgs", maxiter=1000)
        ci = res_aes.conf_int()
        ci.columns = ["lo", "hi"]
        report.append(f"  N={int(res_aes.nobs)}, "
                      f"participants={aes['participant_id'].nunique()}, "
                      f"songs={aes['song_id'].nunique()}, "
                      f"log-lik={float(res_aes.llf):.2f}")
        keep = [
            n for n in res_aes.fe_params.index
            if any(k in n for k in [
                "age_z", "participant_age_missing", "taste_breadth_z",
                "participant_listening_device", "participant_environment",
                "fav_genre_match_label",
            ])
        ]
        for n in keep:
            b = float(res_aes.fe_params[n])
            s_ = float(res_aes.bse[n]) if n in res_aes.bse.index else np.nan
            p_ = float(res_aes.pvalues[n]) if n in res_aes.pvalues.index else np.nan
            lo, hi = float(ci.loc[n, "lo"]), float(ci.loc[n, "hi"])
            rows.append(("Aesthetic (LMM)", n, b, s_, lo, hi, p_,
                         np.nan, np.nan, np.nan))
            report.append(f"    {n}: beta={b:+.3f}, 95% CI [{lo:+.2f}, {hi:+.2f}], "
                          f"p={p_:.4g}")
    except Exception as e:
        report.append(f"  Aesthetic LMM failed: {e}")

    # ---- LaTeX fragment -------------------------------------------------
    def _pretty_term(n):
        s = n
        s = s.replace(
            f"C(participant_listening_device, Treatment('{device_ref}'))",
            "Device",
        )
        s = s.replace(
            f"C(participant_listening_device, Treatment('{device_ref_aes}'))",
            "Device",
        )
        s = s.replace(
            f"C(participant_environment, Treatment('{env_ref}'))", "Environment"
        )
        s = s.replace(
            f"C(participant_environment, Treatment('{env_ref_aes}'))", "Environment"
        )
        s = s.replace("C(fav_genre_match_label, Treatment('unknown'))", "FavGenre")
        s = s.replace("C(participant_age_missing)", "AgeMissing")
        s = s.replace("age_z", "Age (z)")
        s = s.replace("taste_breadth_z", "TasteBreadth (z)")
        s = s.replace("[T.", " = ").replace("]", "")
        return s

    tex = []
    tex.append(r"% Auto-generated by analysis/main.py -- personalization_analysis")
    tex.append(r"\begin{tabular}{llrrcc}")
    tex.append(r"\toprule")
    tex.append(
        r"\textbf{Outcome} & \textbf{Predictor} & "
        r"$\hat{\beta}$ & SE & 95\% CI & $p$ \\"
    )
    tex.append(r"\midrule")
    last_outcome = None
    for r in rows:
        outcome, name, b, se, lo, hi, p, ORv, ORlo, ORhi = r
        sig = (
            r"^{***}" if pd.notna(p) and p < 0.001 else
            r"^{**}" if pd.notna(p) and p < 0.01 else
            r"^{*}" if pd.notna(p) and p < 0.05 else ""
        )
        nm = _pretty_term(name).replace("&", r"\&").replace("_", r"\_")
        if outcome == "Accuracy (logit)":
            ci_str = (
                rf"$[{ORlo:.2f}, {ORhi:.2f}]$ (OR)"
                if pd.notna(ORlo) and pd.notna(ORhi) else "---"
            )
        else:
            ci_str = (
                rf"$[{lo:+.2f}, {hi:+.2f}]$"
                if pd.notna(lo) and pd.notna(hi) else "---"
            )
        outcome_cell = outcome if outcome != last_outcome else ""
        last_outcome = outcome
        p_str = f"${p:.3g}$" if pd.notna(p) else "---"
        se_str = f"${se:.3f}$" if pd.notna(se) else "---"
        tex.append(
            f"{outcome_cell} & {nm} & ${b:+.3f}{sig}$ & {se_str} & {ci_str} & {p_str} \\\\"
        )
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    tex_path = OUTPUT_DIR / "personalization_table.tex"
    tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    report.append("")
    report.append(f"  (LaTeX fragment written to {tex_path})")

    # ---- Combined "accuracy by group" table (paper Table 8) ----------
    # The paper's Table 8 used to be a hand-typed breakdown by musical
    # engagement and formal training only. We auto-emit it here so that
    # the personalised covariates (age, fav-genre match, listening
    # device, environment, taste breadth) appear alongside the original
    # listener traits and stay in sync with the data.
    eng_order = ["casual", "enthusiast", "musician", "professional"]
    train_order = ["None", "1-3 years", "4-7 years", "8+ years"]
    age_order = ["<=24", "25-34", "35-44", "45+"]
    fav_order = ["match", "no-match", "unknown"]
    fam_order = ["never", "familiar", "uncertain"]
    device_order = [
        "Headphones (over-ear)", "Headphones (on-ear)",
        "Earbuds/In-ear", "External speakers", "Laptop/phone speakers",
    ]
    env_order = [
        "Quiet room", "Office/workplace",
        "On the street/outdoors", "Public transportation",
    ]

    work_eng = work.copy()
    work_eng["participant_musical_engagement"] = (
        work_eng["participant_musical_engagement"].astype(str).str.strip().str.lower()
    )
    work_eng["training_group"] = pd.cut(
        pd.to_numeric(work_eng["participant_formal_training_years"], errors="coerce").fillna(0),
        bins=[-1, 0, 3, 7, 100],
        labels=train_order,
    )
    breadth_levels = (
        work["breadth_q"].cat.categories
        if hasattr(work["breadth_q"], "cat") else
        list(work["breadth_q"].dropna().unique())
    )

    def _row_metrics(g):
        n = len(g)
        if n == 0:
            return None
        acc = g["is_correct"].mean(skipna=True)
        ai_g = g[g["is_ai_song"]]
        real_g = g[~g["is_ai_song"]]
        hr = ai_g["detected_ai"].mean() if len(ai_g) else np.nan
        fa = real_g["detected_ai"].mean() if len(real_g) else np.nan
        n_real = len(real_g)
        return n, acc, hr, fa, n_real

    def _fmt_pct(x):
        return "---" if pd.isna(x) else f"{x*100:.1f}\\%"

    def _section_rows(label, levels, series, work_df, indent=True):
        rows = [rf"\multicolumn{{5}}{{l}}{{\emph{{{label}}}}} \\"]
        for lvl in levels:
            sub = work_df[series == lvl]
            m = _row_metrics(sub)
            if m is None:
                continue
            n, acc, hr, fa, n_real = m
            # Label with insufficient-real-trials marker
            fa_cell = _fmt_pct(fa) if n_real >= 5 else "---"
            pretty = str(lvl).replace("&", r"\&").replace("_", r"\_")
            cap = pretty.capitalize() if label.startswith("Musical") else pretty
            prefix = r"\quad " if indent else ""
            rows.append(
                f"{prefix}{cap} & {n} & {_fmt_pct(acc)} & {_fmt_pct(hr)} & {fa_cell} \\\\"
            )
        return rows

    tex2 = []
    tex2.append(
        r"% Auto-generated by analysis/main.py -- personalization_analysis "
        r"(Table 8: accuracy by group)"
    )
    tex2.append(r"\begin{tabular}{lcccc}")
    tex2.append(r"\toprule")
    tex2.append(
        r" & $n$ \textbf{trials} & \textbf{Accuracy} & "
        r"\textbf{Hit rate (AI)} & \textbf{FA rate (real)} \\"
    )
    tex2.append(r"\midrule")
    tex2 += _section_rows(
        "Musical engagement",
        eng_order,
        work_eng["participant_musical_engagement"],
        work_eng,
    )
    tex2.append(r"\midrule")
    tex2 += _section_rows(
        "Formal training",
        train_order,
        work_eng["training_group"],
        work_eng,
    )
    tex2.append(r"\midrule")
    tex2 += _section_rows(
        "Age bucket",
        age_order,
        work_eng.get("age_bucket", pd.Series([pd.NA] * len(work_eng))),
        work_eng,
    )
    tex2.append(r"\midrule")
    tex2 += _section_rows(
        "Listening device",
        device_order,
        work_eng["participant_listening_device"],
        work_eng,
    )
    tex2.append(r"\midrule")
    tex2 += _section_rows(
        "Listening environment",
        env_order,
        work_eng["participant_environment"],
        work_eng,
    )
    tex2.append(r"\midrule")
    tex2 += _section_rows(
        "Favourite-genre vs.\\ song genre",
        fav_order,
        work_eng["fav_genre_match_label"],
        work_eng,
    )
    tex2.append(r"\midrule")
    tex2 += _section_rows(
        "Song familiarity (rater)",
        fam_order,
        work_eng["familiarity_level"],
        work_eng,
    )
    if len(breadth_levels) > 0:
        tex2.append(r"\midrule")
        # Render breadth quartiles as "Q1 (1-3)" style for readability.
        breadth_label_rows = [r"\multicolumn{5}{l}{\emph{Taste breadth (\# favourite genres, quartiles)}} \\"]
        for q_idx, lvl in enumerate(breadth_levels, start=1):
            sub = work_eng[work_eng["breadth_q"] == lvl]
            m = _row_metrics(sub)
            if m is None:
                continue
            n, acc, hr, fa, n_real = m
            fa_cell = _fmt_pct(fa) if n_real >= 5 else "---"
            try:
                rng = f"{int(np.ceil(lvl.left + 1e-9))}--{int(lvl.right)}"
            except Exception:
                rng = str(lvl)
            breadth_label_rows.append(
                rf"\quad Q{q_idx} ({rng}) & {n} & {_fmt_pct(acc)} & "
                rf"{_fmt_pct(hr)} & {fa_cell} \\"
            )
        tex2 += breadth_label_rows
    tex2.append(r"\bottomrule")
    tex2.append(r"\end{tabular}")

    tex2_path = OUTPUT_DIR / "accuracy_by_group_table.tex"
    tex2_path.write_text("\n".join(tex2) + "\n", encoding="utf-8")
    report.append(f"  (LaTeX fragment written to {tex2_path})")


# ---------------------------------------------------------------------------
# Detection-predictors LMM (parallel to aesthetic_predictors_model)
# ---------------------------------------------------------------------------

def detection_predictors_model(ann, report):
    """Mixed-effects logistic for AI-detection correctness.

    Mirrors :func:`aesthetic_predictors_model` predictor-for-predictor but
    on the binary outcome ``is_correct`` (1 = correct AI/real call,
    0 = incorrect; uncertain trials are dropped). To keep the table the
    same shape as the aesthetic LMM we fit a logistic regression with
    cluster-robust standard errors clustered by participant -- statsmodels
    does not provide a true binomial mixed model, and the cluster-robust
    sandwich gives valid inference under the same correlated-trials
    structure.

    Output:
      * ``output/detection_lmm_table.tex`` formatted exactly like
        ``aesthetic_lmm_table.tex`` so the paper can drop it in next to
        Table 9.
    """
    report.append("\n" + "=" * 80)
    report.append(
        "17b. DETECTION PREDICTORS: LOGISTIC REGRESSION (parallel to aesthetic LMM)"
    )
    report.append("=" * 80)
    report.append(
        "\n  Logistic regression on is_correct (1 = correct AI/real call) on"
        "\n  non-uncertain trials. Cluster-robust SE (clustered by participant)"
        "\n  to account for repeated measures. Predictors mirror the aesthetic"
        "\n  LMM exactly, swapping judged-authenticity for ground-truth (since"
        "\n  judged authenticity defines the outcome here)."
    )

    df = ann.copy()
    df = df[df["authenticity_assessment"] != "uncertain"].copy()
    df["correct_int"] = df["is_correct"].astype(int)
    df["genre_family"] = df["song_genres"].apply(_genre_family)
    df["song_source"] = df["song_source"].astype(str).str.strip().str.lower()
    df["snippet_condition"] = df["snippet_condition"].astype(str).str.strip().str.lower()
    df["participant_musical_engagement"] = (
        df["participant_musical_engagement"].astype(str).str.strip().str.lower()
    )
    df["participant_ai_music_experience"] = (
        df["participant_ai_music_experience"].astype(str).str.strip().str.lower()
    )
    df["participant_listening_device"] = (
        df["participant_listening_device"].astype(str).str.strip()
    )
    df["participant_environment"] = (
        df["participant_environment"].astype(str).str.strip()
    )
    df["fav_genre_match_label"] = (
        df["fav_genre_match_label"].astype(str).str.strip()
    )
    df["familiarity_level"] = (
        df["familiarity_level"].astype(str).str.strip().str.lower()
    )
    df["is_ai_song_int"] = df["is_ai_song"].astype(int)
    duration = pd.to_numeric(df["annotation_duration_ms"], errors="coerce")
    df["log_duration"] = np.log1p(duration.clip(lower=1.0))

    keep = [
        "correct_int", "is_ai_song_int", "song_source", "genre_family",
        "snippet_condition", "participant_musical_engagement",
        "participant_ai_music_experience", "participant_listening_device",
        "participant_environment", "fav_genre_match_label", "familiarity_level",
        "participant_age_imputed", "participant_age_missing",
        "participant_taste_breadth", "log_duration",
        "participant_id", "song_id",
    ]
    work = df.dropna(subset=keep).copy()
    if len(work) < 50:
        report.append(f"\n  Insufficient data after dropna ({len(work)}); skipping.")
        return

    work["genre_family"] = _collapse_rare_levels(
        work["genre_family"], min_count=20, keep_levels={"Other"}, other_label="Other"
    )
    for col in ("log_duration", "participant_age_imputed", "participant_taste_breadth"):
        sd = work[col].std()
        if pd.isna(sd) or sd == 0:
            work[col + "_z"] = 0.0
        else:
            work[col + "_z"] = (work[col] - work[col].mean()) / sd
    work.rename(columns={
        "log_duration_z": "log_duration_z",
        "participant_age_imputed_z": "age_z",
        "participant_taste_breadth_z": "taste_breadth_z",
    }, inplace=True)

    src_ref = _safe_ref_level(work["song_source"], preferred="human", fallback_default="human")
    snippet_ref = _safe_ref_level(
        work["snippet_condition"], preferred="original", fallback_default="original"
    )
    engage_ref = _safe_ref_level(
        work["participant_musical_engagement"], preferred="casual", fallback_default="casual"
    )
    genre_ref = _safe_ref_level(work["genre_family"], preferred="Pop", fallback_default="Other")
    device_ref = _safe_ref_level(
        work["participant_listening_device"],
        preferred="Headphones (on-ear)",
        fallback_default="Laptop/phone speakers",
    )
    env_ref = _safe_ref_level(
        work["participant_environment"],
        preferred="Quiet room",
        fallback_default="Quiet room",
    )
    fam_ref = _safe_ref_level(
        work["familiarity_level"], preferred="never", fallback_default="never"
    )

    # Note: ground truth (AI vs.\ real) is collinear with the source
    # dummies (human is the only "real" source), so we let source carry
    # the ground-truth information rather than including both.
    formula = (
        "correct_int ~ "
        f"C(song_source, Treatment('{src_ref}')) "
        f"+ C(genre_family, Treatment('{genre_ref}')) "
        f"+ C(snippet_condition, Treatment('{snippet_ref}')) "
        f"+ C(participant_musical_engagement, Treatment('{engage_ref}')) "
        "+ C(participant_ai_music_experience) "
        "+ log_duration_z "
        "+ age_z + C(participant_age_missing) + taste_breadth_z "
        f"+ C(participant_listening_device, Treatment('{device_ref}')) "
        f"+ C(participant_environment, Treatment('{env_ref}')) "
        "+ C(fav_genre_match_label, Treatment('unknown')) "
        f"+ C(familiarity_level, Treatment('{fam_ref}'))"
    )

    report.append(
        f"\n  N = {len(work)} non-uncertain annotations | "
        f"{work['participant_id'].nunique()} participants | "
        f"{work['song_id'].nunique()} songs"
    )
    report.append(
        "  Reference levels: "
        f"source={src_ref}, snippet={snippet_ref}, engagement={engage_ref}, "
        f"genre={genre_ref}, device={device_ref}, environment={env_ref}, "
        f"fav-genre=unknown, familiarity={fam_ref}"
    )

    try:
        result = smf.logit(formula, data=work).fit(
            disp=False,
            maxiter=200,
            cov_type="cluster",
            cov_kwds={"groups": work["participant_id"]},
        )
    except Exception as e:
        report.append(f"\n  Logit fit failed: {e}")
        return

    fe = result.params
    bse = result.bse
    pv = result.pvalues
    ci = result.conf_int()
    ci.columns = ["lo", "hi"]

    def _pretty(name):
        s = name
        s = s.replace(f"C(song_source, Treatment('{src_ref}'))", "Source")
        s = s.replace(f"C(genre_family, Treatment('{genre_ref}'))", "Genre")
        s = s.replace(f"C(snippet_condition, Treatment('{snippet_ref}'))", "Snippet")
        s = s.replace(
            f"C(participant_musical_engagement, Treatment('{engage_ref}'))",
            "Engagement",
        )
        s = s.replace("C(participant_ai_music_experience)", "AI-exp.")
        s = s.replace(
            f"C(participant_listening_device, Treatment('{device_ref}'))", "Device"
        )
        s = s.replace(
            f"C(participant_environment, Treatment('{env_ref}'))", "Environment"
        )
        s = s.replace("C(fav_genre_match_label, Treatment('unknown'))", "FavGenre")
        s = s.replace(
            f"C(familiarity_level, Treatment('{fam_ref}'))", "Familiarity"
        )
        s = s.replace("C(participant_age_missing)", "AgeMissing")
        s = s.replace("is_ai_song_int", "GroundTruth=AI")
        s = s.replace("age_z", "Age (z)")
        s = s.replace("taste_breadth_z", "TasteBreadth (z)")
        s = s.replace("log_duration_z", "log\\_duration\\_z")
        s = s.replace("[T.", " = ").replace("]", "")
        return s

    report.append("\n  --- Fixed effects (logistic; cluster-robust SE) ---")
    report.append(
        f"  N obs = {int(result.nobs)}, log-lik = {float(result.llf):.2f}, "
        f"pseudo-R^2 = {float(result.prsquared):.3f}, "
        f"AIC = {float(result.aic):.2f}"
    )
    hdr = (
        f"  {'Term':<60s}  {'beta':>8s}  {'OR':>6s}  "
        f"{'95% OR CI':>22s}  {'p':>10s}"
    )
    report.append(hdr)
    report.append("  " + "-" * (len(hdr) - 2))
    for name in fe.index:
        b = float(fe[name])
        s_ = float(bse.loc[name]) if pd.notna(bse.loc[name]) else np.nan
        p_ = float(pv.loc[name]) if pd.notna(pv.loc[name]) else np.nan
        lo = float(ci.loc[name, "lo"]) if pd.notna(ci.loc[name, "lo"]) else np.nan
        hi = float(ci.loc[name, "hi"]) if pd.notna(ci.loc[name, "hi"]) else np.nan
        ci_str = f"[{np.exp(lo):.2f}, {np.exp(hi):.2f}]"
        sig = (
            "***" if pd.notna(p_) and p_ < 0.001 else
            "**" if pd.notna(p_) and p_ < 0.01 else
            "*" if pd.notna(p_) and p_ < 0.05 else ""
        )
        report.append(
            f"  {_pretty(name):<60s}  {b:+8.3f}  {np.exp(b):6.2f}  "
            f"{ci_str:>22s}  {p_:10.4g} {sig}"
        )

    # Tjur's coefficient of discrimination as a goodness-of-fit summary.
    fitted = result.predict(work)
    y = work["correct_int"].astype(float)
    if y.sum() > 0 and (1 - y).sum() > 0:
        tjur_d = float(fitted[y == 1].mean() - fitted[y == 0].mean())
    else:
        tjur_d = np.nan
    report.append(f"\n  Tjur's coefficient of discrimination = {tjur_d:.3f}")

    # ----- LaTeX fragment ----------------------------------------------
    tex = []
    tex.append(r"% Auto-generated by analysis/main.py -- detection_predictors_model")
    tex.append(r"\begin{tabular}{lrrcc}")
    tex.append(r"\toprule")
    tex.append(
        r"\textbf{Predictor} & $\hat{\beta}$ & OR & 95\% OR CI & $p$ \\"
    )
    tex.append(r"\midrule")
    for name in fe.index:
        b = float(fe[name])
        s_ = float(bse.loc[name]) if pd.notna(bse.loc[name]) else np.nan
        p_ = float(pv.loc[name]) if pd.notna(pv.loc[name]) else np.nan
        lo = float(ci.loc[name, "lo"]) if pd.notna(ci.loc[name, "lo"]) else np.nan
        hi = float(ci.loc[name, "hi"]) if pd.notna(ci.loc[name, "hi"]) else np.nan
        sig = (
            r"^{***}" if pd.notna(p_) and p_ < 0.001 else
            r"^{**}" if pd.notna(p_) and p_ < 0.01 else
            r"^{*}" if pd.notna(p_) and p_ < 0.05 else ""
        )
        nm = _pretty(name).replace("&", r"\&")
        ci_str = (
            rf"$[{np.exp(lo):.2f}, {np.exp(hi):.2f}]$"
            if pd.notna(lo) and pd.notna(hi) else "---"
        )
        OR_str = f"${np.exp(b):.2f}$" if pd.notna(b) else "---"
        p_str = f"${p_:.3g}$" if pd.notna(p_) else "---"
        tex.append(
            f"{nm} & ${b:+.3f}{sig}$ & {OR_str} & {ci_str} & {p_str} \\\\"
        )
    tex.append(r"\midrule")
    tex.append(rf"$N$ observations & {int(result.nobs)} & & & \\")
    tex.append(
        rf"Pseudo-$R^2$ (McFadden) & {float(result.prsquared):.3f} & & & \\"
    )
    tex.append(rf"Tjur's $D$ & {tjur_d:.3f} & & & \\")
    tex.append(rf"Log-likelihood & {float(result.llf):.2f} & & & \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    tex_path = OUTPUT_DIR / "detection_lmm_table.tex"
    tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    report.append("")
    report.append(f"  (LaTeX fragment written to {tex_path})")


# ---------------------------------------------------------------------------
# Audiobox-Aesthetics comparison (Section 19)
# ---------------------------------------------------------------------------

# Mapping from audiobox-aesthetics output axes to our perceptual rating
# columns. Both scales run 1--10, so no rescaling is needed. Mappings are
# the conceptually closest pairs:
#   * PQ (Production Quality)    <-> production_quality   (direct match)
#   * CE (Content Enjoyment)     <-> aesthetic_quality    (subjective enjoyment)
#   * CU (Content Usefulness)    <-> playlist_likelihood  (would-listen-again)
#   * PC (Production Complexity) <-> musical_creativity   (closest available;
#       complexity != creativity but both index "non-trivial musical content")
# Audiobox does not provide an analogue for emotional_engagement.
AUDIOBOX_AXES = ["CE", "CU", "PC", "PQ"]
AUDIOBOX_AXIS_LABELS = {
    "CE": "Content Enjoyment",
    "CU": "Content Usefulness",
    "PC": "Production Complexity",
    "PQ": "Production Quality",
}
AUDIOBOX_TO_HUMAN = {
    "PQ": "production_quality",
    "CE": "aesthetic_quality",
    "CU": "playlist_likelihood",
    "PC": "musical_creativity",
}

AUDIOBOX_DIR = Path("aesthetics/audiobox")
AUDIOBOX_INPUT = AUDIOBOX_DIR / "input.jsonl"
AUDIOBOX_OUTPUT = AUDIOBOX_DIR / "output.jsonl"

SONGEVAL_DIR = Path("aesthetics/SongEval")
SONGEVAL_INPUT = SONGEVAL_DIR / "input.jsonl"
SONGEVAL_OUTPUT = SONGEVAL_DIR / "output" / "result.json"

OUR_MUSIC_AESTHETICS_DIR = Path("aesthetics/our-music-aesthetics-model")
OUR_MUSIC_AESTHETICS_INPUT = OUR_MUSIC_AESTHETICS_DIR / "input.jsonl"
OUR_MUSIC_AESTHETICS_OUTPUT = OUR_MUSIC_AESTHETICS_DIR / "output" / "result.json"

OUR_MUSIC_POPULARITY_DIR = Path("aesthetics/our-music-popularity-model")
OUR_MUSIC_POPULARITY_INPUT = OUR_MUSIC_POPULARITY_DIR / "input.jsonl"
OUR_MUSIC_POPULARITY_OUTPUT = OUR_MUSIC_POPULARITY_DIR / "output" / "result.json"

FEATURE_DIR = Path("features")
FEATURE_SUMMARY_CSV = FEATURE_DIR / "results" / "summary.csv"
FEATURE_BASELINE_MD = FEATURE_DIR / "results" / "baselines" / "comparison.md"
FEATURE_INVARIANT_MD = FEATURE_DIR / "results" / "invariants" / "ranking.md"
FEATURE_AUGMENTATION_MD = FEATURE_DIR / "results" / "augmentations" / "regimes.md"

SONGEVAL_AXES = ["Musicality", "Coherence", "Memorability", "Clarity", "Naturalness"]
SONGEVAL_AXIS_LABELS = {
    "Musicality": "Overall Musicality",
    "Coherence": "Overall Coherence",
    "Memorability": "Memorability",
    "Clarity": "Clarity of Song Structure",
    "Naturalness": "Naturalness of Vocal Breathing/Phrasing",
}
# SongEval's dimensions are not a one-to-one match to our questionnaire.
# We benchmark against the closest available human dimensions, mirroring the
# pragmatic mapping approach used for audiobox above.
SONGEVAL_TO_HUMAN = {
    "Musicality": "aesthetic_quality",
    "Coherence": "production_quality",
    "Memorability": "playlist_likelihood",
    "Clarity": "musical_creativity",
    "Naturalness": "emotional_engagement",
}

AUDIOBOX_BY_HUMAN = {human_col: ax for ax, human_col in AUDIOBOX_TO_HUMAN.items()}
SONGEVAL_AUDIOBOX_PAIRS = [
    (se_axis, AUDIOBOX_BY_HUMAN[human_col], human_col)
    for se_axis, human_col in SONGEVAL_TO_HUMAN.items()
    if human_col in AUDIOBOX_BY_HUMAN
]


def _rescale_1_5_to_1_10(value):
    """Map a 1--5 score onto the human 1--10 scale, preserving endpoints."""
    return 1.0 + (float(value) - 1.0) * (9.0 / 4.0)


def _load_audiobox_scores():
    """Load audiobox per-song scores joined with the input metadata.

    Returns a DataFrame with columns ``song_id``, ``song_source``,
    ``snippet_kind`` and the four audiobox axes (``CE``, ``CU``, ``PC``,
    ``PQ``). Returns an empty DataFrame if the artefacts are missing.
    """
    if not (AUDIOBOX_INPUT.exists() and AUDIOBOX_OUTPUT.exists()):
        return pd.DataFrame()
    in_rows = []
    with AUDIOBOX_INPUT.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                in_rows.append(json.loads(line))
    out_rows = []
    with AUDIOBOX_OUTPUT.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out_rows.append(json.loads(line))
    if len(in_rows) != len(out_rows):
        # Partial run: keep the leading rows that match.
        n = min(len(in_rows), len(out_rows))
        in_rows = in_rows[:n]
        out_rows = out_rows[:n]
    if not in_rows:
        return pd.DataFrame()
    df = pd.DataFrame({
        "song_id": [r["song_id"] for r in in_rows],
        "song_source": [r.get("song_source") for r in in_rows],
        "snippet_kind": [r.get("snippet_kind") for r in in_rows],
        **{ax: [float(o.get(ax, np.nan)) for o in out_rows] for ax in AUDIOBOX_AXES},
    })
    return df


def audiobox_aesthetics_analysis(ann, report):
    """Compare Meta's audiobox-aesthetics scores to human + LLM ratings.

    Audiobox-aesthetics (Tjandra et al., 2024) is a recent reference-free
    music-quality model that emits four axes on the same 1-10 scale we
    use. Conceptual mapping to our annotation dimensions is documented in
    ``AUDIOBOX_TO_HUMAN``. We compute (a) global means by source, (b)
    song-level Spearman/Pearson correlations and MAE against human means,
    and (c) the equivalent alignment for every LLM judge in
    ``input/models/`` so the audiobox front-end can be benchmarked
    alongside the LLM judges. Output is mirrored to
    ``output/audiobox_table.tex``.
    """
    report.append("\n" + "=" * 80)
    report.append("19. AUDIOBOX-AESTHETICS COMPARISON")
    report.append("=" * 80)

    aes_df = _load_audiobox_scores()
    if aes_df.empty:
        report.append(
            "\n  No audiobox scores found. Run "
            "`uv run python aesthetics/audiobox/build_input.py` then "
            "`uv run audio-aes --batch-size 1 aesthetics/audiobox/input.jsonl > "
            "aesthetics/audiobox/output.jsonl` first."
        )
        return

    n_full = int((aes_df["snippet_kind"] == "full").sum())
    n_crop = int((aes_df["snippet_kind"] == "cropped").sum())
    report.append(
        f"\n  Loaded audiobox scores for {len(aes_df)} songs "
        f"(v1 full snippets: {n_full}; v2-v1 cropped 29s: {n_crop})."
    )
    report.append(
        "  Mapping audiobox -> our dimensions: "
        + ", ".join(f"{ax}->{AUDIOBOX_TO_HUMAN[ax]}" for ax in AUDIOBOX_AXES)
        + ". Both scales are 1-10."
    )

    # ----- Global means by source ---------------------------------------
    report.append("\n  --- Global means by source (audiobox vs humans) ---")
    src_rows = []
    for src in ALL_SOURCES:
        sub_aes = aes_df[aes_df["song_source"] == src]
        sub_human = ann[ann["song_source"] == src]
        row = {"source": src, "n_audiobox": len(sub_aes), "n_human": len(sub_human)}
        for ax in AUDIOBOX_AXES:
            human_col = AUDIOBOX_TO_HUMAN[ax]
            row[f"{ax}_aes"] = float(sub_aes[ax].mean()) if len(sub_aes) else np.nan
            row[f"{ax}_human"] = float(
                pd.to_numeric(sub_human[human_col], errors="coerce").mean()
            ) if len(sub_human) else np.nan
        src_rows.append(row)
    src_df = pd.DataFrame(src_rows).set_index("source")
    show = src_df.copy()
    for c in show.columns:
        if c not in ("n_audiobox", "n_human"):
            show[c] = show[c].astype(float).round(2)
    report.append(show.to_string())

    # ----- Song-level alignment (audiobox vs human song means) ----------
    report.append("\n  --- Song-level alignment with human song means ---")
    human_song = ann.groupby("song_id")[RATING_COLS].mean()
    aes_song = aes_df.set_index("song_id")
    align_rows = []
    for ax in AUDIOBOX_AXES:
        human_col = AUDIOBOX_TO_HUMAN[ax]
        joined = aes_song[[ax]].join(human_song[[human_col]], how="inner").dropna()
        if len(joined) >= 3:
            rho, p_rho = stats.spearmanr(joined[ax], joined[human_col])
            r, p_r = stats.pearsonr(joined[ax], joined[human_col])
            mae = float(np.mean(np.abs(joined[ax] - joined[human_col])))
            mean_diff = float((joined[ax] - joined[human_col]).mean())
        else:
            rho = p_rho = r = p_r = mae = mean_diff = np.nan
        align_rows.append({
            "audiobox_axis": ax,
            "axis_label": AUDIOBOX_AXIS_LABELS[ax],
            "human_dim": human_col,
            "n_songs": int(len(joined)),
            "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
            "pearson_r": float(r) if pd.notna(r) else np.nan,
            "mae": mae,
            "mean_diff_aes_minus_human": mean_diff,
            "p_spearman": float(p_rho) if pd.notna(p_rho) else np.nan,
        })
    align_df = pd.DataFrame(align_rows)
    show = align_df.copy()
    for c in ["spearman_rho", "pearson_r", "mae", "mean_diff_aes_minus_human", "p_spearman"]:
        show[c] = show[c].astype(float).round(3)
    report.append(show.to_string(index=False))

    # ----- Per-source song-level Spearman ------------------------------
    report.append("\n  --- Per-source song-level Spearman rho ---")
    per_src_rows = []
    for ax in AUDIOBOX_AXES:
        human_col = AUDIOBOX_TO_HUMAN[ax]
        row = {"axis": f"{ax} ({AUDIOBOX_TO_HUMAN[ax]})"}
        for src in ALL_SOURCES:
            sub_aes = aes_df[aes_df["song_source"] == src].set_index("song_id")[[ax]]
            sub_human = (ann[ann["song_source"] == src]
                         .groupby("song_id")[[human_col]].mean())
            joined = sub_aes.join(sub_human, how="inner").dropna()
            if len(joined) >= 3:
                rho, _ = stats.spearmanr(joined[ax], joined[human_col])
            else:
                rho = np.nan
            row[src] = round(float(rho), 3) if pd.notna(rho) else np.nan
            row[f"n_{src}"] = int(len(joined))
        per_src_rows.append(row)
    per_src_df = pd.DataFrame(per_src_rows)
    report.append(per_src_df.to_string(index=False))

    # ----- LLM judges' alignment on the same dimension pairs ------------
    model_long = _load_model_eval_long(ann)
    rho_table_rows = []
    # Audiobox row first
    audiobox_row = {"evaluator": "Audiobox-Aesthetics", "kind": "audiobox"}
    for ax in AUDIOBOX_AXES:
        a_row = align_df[align_df["audiobox_axis"] == ax].iloc[0]
        audiobox_row[f"rho_{ax}"] = round(float(a_row["spearman_rho"]), 3) if pd.notna(a_row["spearman_rho"]) else np.nan
        audiobox_row[f"mae_{ax}"] = round(float(a_row["mae"]), 3) if pd.notna(a_row["mae"]) else np.nan
        audiobox_row[f"mean_{ax}"] = round(float(aes_df[ax].mean()), 3) if not aes_df.empty else np.nan
    rho_table_rows.append(audiobox_row)

    # Human aggregate self-reference
    human_row = {"evaluator": "Humans", "kind": "human"}
    for ax in AUDIOBOX_AXES:
        human_col = AUDIOBOX_TO_HUMAN[ax]
        m = pd.to_numeric(ann[human_col], errors="coerce").mean()
        human_row[f"rho_{ax}"] = 1.0
        human_row[f"mae_{ax}"] = 0.0
        human_row[f"mean_{ax}"] = round(float(m), 3) if pd.notna(m) else np.nan
    rho_table_rows.append(human_row)

    if not model_long.empty:
        for (base, think), g in model_long.groupby(["base_model_alias", "thinking_alias"]):
            row = {"evaluator": f"{base} ({think})", "kind": "model"}
            # Models only emit aesthetic_quality in our schema; we project
            # that column onto each audiobox-mapped human dim, fitting the
            # same alignment metric so the magnitudes are comparable.
            # If the model output had richer dimensions per song they would
            # be added here.
            df_m = g.copy()
            df_m["aesthetic_quality"] = pd.to_numeric(df_m["aesthetic_quality"], errors="coerce")
            agg = df_m.groupby("song_id")["aesthetic_quality"].mean()
            for ax in AUDIOBOX_AXES:
                human_col = AUDIOBOX_TO_HUMAN[ax]
                human_song_axis = ann.groupby("song_id")[human_col].mean()
                joined = agg.to_frame("model_pred").join(human_song_axis, how="inner").dropna()
                if len(joined) >= 3:
                    rho, _ = stats.spearmanr(joined["model_pred"], joined[human_col])
                    mae = float(np.mean(np.abs(joined["model_pred"] - joined[human_col])))
                else:
                    rho = np.nan
                    mae = np.nan
                row[f"rho_{ax}"] = round(float(rho), 3) if pd.notna(rho) else np.nan
                row[f"mae_{ax}"] = round(float(mae), 3) if pd.notna(mae) else np.nan
                row[f"mean_{ax}"] = round(float(agg.mean()), 3) if not agg.empty else np.nan
            rho_table_rows.append(row)

    rho_df = pd.DataFrame(rho_table_rows)

    report.append("\n  --- Cross-evaluator alignment with humans (per audiobox axis) ---")
    report.append(
        "  rho_X = song-level Spearman between evaluator's "
        "estimate of X and human song means on the mapped dimension."
    )
    show_cols = ["evaluator", "kind"] + [f"rho_{ax}" for ax in AUDIOBOX_AXES] + [f"mae_{ax}" for ax in AUDIOBOX_AXES]
    report.append(rho_df[show_cols].to_string(index=False))

    # ----- LaTeX fragment ----------------------------------------------
    tex = []
    tex.append(r"% Auto-generated by analysis/main.py -- audiobox_aesthetics_analysis")
    tex.append(r"\begin{tabular}{lcccc cccc}")
    tex.append(r"\toprule")
    tex.append(
        r" & \multicolumn{4}{c}{\textbf{Spearman} $\rho$ \textbf{vs.\ humans}} & "
        r"\multicolumn{4}{c}{\textbf{Mean abs.\ error (1--10)}} \\"
    )
    tex.append(r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}")
    head_axes = " & ".join(rf"\textbf{{{ax}}}" for ax in AUDIOBOX_AXES)
    tex.append(rf"\textbf{{Evaluator}} & {head_axes} & {head_axes} \\")
    tex.append(r"\midrule")

    def _fmt(v):
        if pd.isna(v):
            return "---"
        return f"${v:.2f}$"

    # Audiobox row first, then humans (skip the trivial 1.0/0.0 row to save space if desired,
    # but keep it for completeness as it anchors the scale)
    order = ["audiobox", "human", "model"]
    for kind in order:
        sub = rho_df[rho_df["kind"] == kind]
        if sub.empty:
            continue
        if kind == "model":
            tex.append(r"\midrule")
        for _, r in sub.iterrows():
            name = str(r["evaluator"]).replace("&", r"\&").replace("_", r"\_")
            rhos = " & ".join(_fmt(r[f"rho_{ax}"]) for ax in AUDIOBOX_AXES)
            maes = " & ".join(_fmt(r[f"mae_{ax}"]) for ax in AUDIOBOX_AXES)
            tex.append(f"{name} & {rhos} & {maes} \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    tex_path = OUTPUT_DIR / "audiobox_table.tex"
    tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    report.append("")
    report.append(f"  (LaTeX fragment written to {tex_path})")

    # ----- Take-aways ---------------------------------------------------
    best_axis = align_df.loc[align_df["spearman_rho"].idxmax()] if align_df["spearman_rho"].notna().any() else None
    worst_axis = align_df.loc[align_df["spearman_rho"].idxmin()] if align_df["spearman_rho"].notna().any() else None
    report.append("\n  --- Take-aways ---")
    if best_axis is not None:
        report.append(
            f"  Best-aligned axis: {best_axis['audiobox_axis']} ({best_axis['axis_label']}) "
            f"<-> {best_axis['human_dim']}: rho={best_axis['spearman_rho']:.3f}, "
            f"MAE={best_axis['mae']:.2f}, Δ(aes-human)={best_axis['mean_diff_aes_minus_human']:+.2f}"
        )
    if worst_axis is not None and worst_axis["audiobox_axis"] != (best_axis["audiobox_axis"] if best_axis is not None else None):
        report.append(
            f"  Weakest-aligned axis: {worst_axis['audiobox_axis']} ({worst_axis['axis_label']}) "
            f"<-> {worst_axis['human_dim']}: rho={worst_axis['spearman_rho']:.3f}, "
            f"MAE={worst_axis['mae']:.2f}, Δ(aes-human)={worst_axis['mean_diff_aes_minus_human']:+.2f}"
        )

    # Persist a CSV for the figure-generation step.
    out_csv = OUTPUT_DIR / "audiobox_alignment.csv"
    align_df.to_csv(out_csv, index=False)
    rho_df.to_csv(OUTPUT_DIR / "audiobox_evaluator_alignment.csv", index=False)
    aes_df.to_csv(OUTPUT_DIR / "audiobox_song_scores.csv", index=False)
    report.append(f"  (Per-song scores: {OUTPUT_DIR / 'audiobox_song_scores.csv'})")


def _load_songeval_scores():
    """Load SongEval per-song scores joined with staged input metadata."""
    if not (SONGEVAL_INPUT.exists() and SONGEVAL_OUTPUT.exists()):
        return pd.DataFrame()
    try:
        input_rows = [
            json.loads(line)
            for line in SONGEVAL_INPUT.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        score_obj = json.loads(SONGEVAL_OUTPUT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return pd.DataFrame()

    rows = []
    for r in input_rows:
        sid = str(r.get("song_id") or "")
        if not sid:
            continue
        song_scores = score_obj.get(sid, {})
        row = {
            "song_id": sid,
            "song_source": r.get("song_source"),
            "snippet_kind": r.get("snippet_kind"),
        }
        for ax in SONGEVAL_AXES:
            row[ax] = float(song_scores.get(ax, np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def _load_our_music_aesthetics_scores():
    """Load our REDACTED music-aesthetics scores joined with staged metadata.

    The model emits 1--5 scores. Returned metric columns are endpoint-rescaled
    to 1--10 for comparability with human ratings and Audiobox; raw values are
    retained as ``raw_<metric>``.
    """
    if not (OUR_MUSIC_AESTHETICS_INPUT.exists() and OUR_MUSIC_AESTHETICS_OUTPUT.exists()):
        return pd.DataFrame()
    try:
        input_rows = [
            json.loads(line)
            for line in OUR_MUSIC_AESTHETICS_INPUT.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        score_obj = json.loads(OUR_MUSIC_AESTHETICS_OUTPUT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return pd.DataFrame()

    rows = []
    for r in input_rows:
        sid = str(r.get("song_id") or "")
        if not sid:
            continue
        song_scores = score_obj.get(sid, {})
        row = {
            "song_id": sid,
            "song_source": r.get("song_source"),
            "snippet_kind": r.get("snippet_kind"),
        }
        for ax in SONGEVAL_AXES:
            raw = float(song_scores.get(ax, np.nan))
            row[f"raw_{ax}"] = raw
            row[ax] = _rescale_1_5_to_1_10(raw) if pd.notna(raw) else np.nan
        raw_overall = float(song_scores.get("Overall_Aesthetics", np.nan))
        row["raw_Overall_Aesthetics"] = raw_overall
        row["Overall_Aesthetics"] = (
            _rescale_1_5_to_1_10(raw_overall)
            if pd.notna(raw_overall)
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _load_our_music_popularity_scores():
    """Load our REDACTED music-popularity predictions joined with staged metadata."""
    if not (OUR_MUSIC_POPULARITY_INPUT.exists() and OUR_MUSIC_POPULARITY_OUTPUT.exists()):
        return pd.DataFrame()
    try:
        input_rows = [
            json.loads(line)
            for line in OUR_MUSIC_POPULARITY_INPUT.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        score_obj = json.loads(OUR_MUSIC_POPULARITY_OUTPUT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return pd.DataFrame()

    rows = []
    for r in input_rows:
        sid = str(r.get("song_id") or "")
        if not sid:
            continue
        pred = score_obj.get(sid, {})
        rows.append({
            "song_id": sid,
            "song_source": r.get("song_source"),
            "snippet_kind": r.get("snippet_kind"),
            "log1p_play_count": float(pred.get("log1p_play_count", np.nan)),
            "log1p_upvote_count": float(pred.get("log1p_upvote_count", np.nan)),
            "estimated_play_count": float(pred.get("estimated_play_count", np.nan)),
            "estimated_upvote_count": float(pred.get("estimated_upvote_count", np.nan)),
        })
    return pd.DataFrame(rows)


def popularity_model_analysis(ann, report):
    """Compare our popularity predictions against human and audio aesthetics signals."""
    report.append("\n" + "=" * 80)
    report.append("21. OUR MUSIC-POPULARITY MODEL COMPARISON")
    report.append("=" * 80)

    pop_df = _load_our_music_popularity_scores()
    if pop_df.empty:
        report.append(
            "\n  No Our Music-Popularity scores found. Run "
            "`uv run python aesthetics/our-music-popularity-model/build_input.py` then "
            "`uv run python aesthetics/our-music-popularity-model/eval.py "
            "-i aesthetics/our-music-popularity-model/input.jsonl "
            "-o aesthetics/our-music-popularity-model/output --use_cpu True` first."
        )
        return

    n_full = int((pop_df["snippet_kind"] == "full").sum())
    n_crop = int((pop_df["snippet_kind"] == "cropped").sum())
    report.append(
        f"\n  Loaded popularity predictions for {len(pop_df)} songs "
        f"(v1 full snippets: {n_full}; v2-v1 cropped 29s: {n_crop})."
    )
    report.append(
        "  Predictions are log1p(count) targets; estimated counts are expm1(log1p). "
        "The human benchmark has no real platform play/upvote labels, so this section "
        "tests whether predicted popularity behaves like human/aesthetic preference signals."
    )

    pop_song = pop_df.set_index("song_id")
    human_song = ann.groupby("song_id")[RATING_COLS].mean()

    report.append("\n  --- Predicted popularity by source ---")
    source_rows = []
    for src in ALL_SOURCES:
        sub = pop_df[pop_df["song_source"] == src]
        source_rows.append({
            "source": src,
            "n": int(len(sub)),
            "mean_log1p_plays": float(sub["log1p_play_count"].mean()) if len(sub) else np.nan,
            "median_est_plays": float(sub["estimated_play_count"].median()) if len(sub) else np.nan,
            "mean_log1p_upvotes": float(sub["log1p_upvote_count"].mean()) if len(sub) else np.nan,
            "median_est_upvotes": float(sub["estimated_upvote_count"].median()) if len(sub) else np.nan,
        })
    source_df = pd.DataFrame(source_rows)
    show_source = source_df.copy()
    for c in ["mean_log1p_plays", "mean_log1p_upvotes"]:
        show_source[c] = show_source[c].astype(float).round(3)
    for c in ["median_est_plays", "median_est_upvotes"]:
        show_source[c] = show_source[c].astype(float).round(1)
    report.append(show_source.to_string(index=False))

    report.append("\n  --- Popularity predictions vs human rating means ---")
    human_rows = []
    for target in ["log1p_play_count", "log1p_upvote_count"]:
        for col in RATING_COLS:
            joined = pop_song[[target]].join(human_song[[col]], how="inner").dropna()
            if len(joined) >= 3:
                rho, p_rho = stats.spearmanr(joined[target], joined[col])
                r, p_r = stats.pearsonr(joined[target], joined[col])
            else:
                rho = p_rho = r = p_r = np.nan
            human_rows.append({
                "popularity_target": target,
                "human_dim": col,
                "n_songs": int(len(joined)),
                "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                "pearson_r": float(r) if pd.notna(r) else np.nan,
                "p_spearman": float(p_rho) if pd.notna(p_rho) else np.nan,
                "p_pearson": float(p_r) if pd.notna(p_r) else np.nan,
            })
    human_align_df = pd.DataFrame(human_rows)
    show_human = human_align_df.copy()
    for c in ["spearman_rho", "pearson_r", "p_spearman", "p_pearson"]:
        show_human[c] = show_human[c].astype(float).round(3)
    report.append(show_human.to_string(index=False))

    report.append("\n  --- Popularity predictions vs audio aesthetics front-ends ---")
    frontend_rows = []

    def _add_frontend(frontend_name, df, axes):
        if df.empty:
            return
        f_song = df.set_index("song_id")
        for target in ["log1p_play_count", "log1p_upvote_count"]:
            for ax in axes:
                if ax not in f_song.columns:
                    continue
                joined = pop_song[[target]].join(f_song[[ax]], how="inner").dropna()
                if len(joined) >= 3:
                    rho, p_rho = stats.spearmanr(joined[target], joined[ax])
                    r, p_r = stats.pearsonr(joined[target], joined[ax])
                else:
                    rho = p_rho = r = p_r = np.nan
                frontend_rows.append({
                    "frontend": frontend_name,
                    "popularity_target": target,
                    "axis": ax,
                    "n_songs": int(len(joined)),
                    "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                    "pearson_r": float(r) if pd.notna(r) else np.nan,
                    "p_spearman": float(p_rho) if pd.notna(p_rho) else np.nan,
                    "p_pearson": float(p_r) if pd.notna(p_r) else np.nan,
                })

    _add_frontend("Audiobox-Aesthetics", _load_audiobox_scores(), AUDIOBOX_AXES)
    _add_frontend("SongEval", _load_songeval_scores(), SONGEVAL_AXES)
    _add_frontend("Our Music-Aesthetics", _load_our_music_aesthetics_scores(), SONGEVAL_AXES)

    frontend_df = pd.DataFrame(frontend_rows)
    if frontend_df.empty:
        report.append("  No audio aesthetics front-end scores available for comparison.")
    else:
        show_front = frontend_df.copy()
        for c in ["spearman_rho", "pearson_r", "p_spearman", "p_pearson"]:
            show_front[c] = show_front[c].astype(float).round(3)
        report.append(show_front.to_string(index=False))

        summary = (
            frontend_df.groupby(["frontend", "popularity_target"], as_index=False)
            .agg(mean_abs_spearman=("spearman_rho", lambda x: float(np.nanmean(np.abs(x)))))
            .sort_values(["popularity_target", "mean_abs_spearman"], ascending=[True, False])
        )
        summary["mean_abs_spearman"] = summary["mean_abs_spearman"].round(3)
        report.append("\n  Mean absolute Spearman rho by front-end:")
        report.append(summary.to_string(index=False))

    pop_df.to_csv(OUTPUT_DIR / "our_music_popularity_song_scores.csv", index=False)
    human_align_df.to_csv(OUTPUT_DIR / "our_music_popularity_human_alignment.csv", index=False)
    if not frontend_df.empty:
        frontend_df.to_csv(OUTPUT_DIR / "our_music_popularity_frontend_alignment.csv", index=False)

    best_human = human_align_df.loc[
        human_align_df["spearman_rho"].abs().idxmax()
    ] if human_align_df["spearman_rho"].notna().any() else None
    report.append("\n  --- Take-aways ---")
    if best_human is not None:
        report.append(
            f"  Strongest human-rating association: {best_human['popularity_target']} "
            f"vs {best_human['human_dim']}, rho={best_human['spearman_rho']:.3f} "
            f"(p={best_human['p_spearman']:.3g})."
        )
    report.append(
        f"  (Per-song popularity predictions: {OUTPUT_DIR / 'our_music_popularity_song_scores.csv'})"
    )


def _parse_markdown_tables(path):
    """Return all simple pipe tables from a markdown file as DataFrames."""
    if not path.exists():
        return []

    def _split_row(line):
        return [c.strip() for c in line.strip().strip("|").split("|")]

    tables = []
    current = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("|") and line.endswith("|"):
            current.append(line)
            continue
        if current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)

    parsed = []
    for rows in tables:
        if len(rows) < 2:
            continue
        header = _split_row(rows[0])
        body = []
        for row in rows[1:]:
            cells = _split_row(row)
            if all(set(c) <= set("-: ") for c in cells):
                continue
            if len(cells) == len(header):
                body.append(cells)
        if body:
            parsed.append(pd.DataFrame(body, columns=header))
    return parsed


def _clean_md_cell(value):
    text = str(value).strip()
    text = text.replace("`", "").replace("**", "")
    return text


def _parse_numeric(value):
    text = _clean_md_cell(value).replace(",", "")
    match = re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", text, flags=re.I)
    return float(match.group(0)) if match else np.nan


def _parse_ci_value(value):
    text = _clean_md_cell(value).replace(",", "")
    match = re.search(
        r"([-+]?\d*\.?\d+)\s*\[\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s*\]",
        text,
    )
    if match:
        return tuple(float(match.group(i)) for i in range(1, 4))
    return _parse_numeric(text), np.nan, np.nan


def _feature_display_name(run_id):
    run_id = _clean_md_cell(run_id)
    mapping = {
        "multiview_aug-combined_safe_seed": "Multiview ensemble",
        "lcnn_aug-none_seed0": "LCNN",
        "mert_head_aug-none_seed0": "MERT head",
        "muq_head_aug-none_seed0": "MuQ head",
        "moss_nano_head_aug-none_seed0": "MOSS Nano head",
        "clap_head_aug-none_seed0": "CLAP head",
        "convnext_aug-none_seed0": "ConvNeXt-Tiny",
        "vit_aug-none_seed0": "ViT-S/16",
        "efficientvit_aug-none_seed0": "EfficientViT-B1",
        "specttra_alpha_aug-none_seed0": "SpecTTTra alpha",
        "specttra_beta_aug-none_seed0": "SpecTTTra beta",
        "specttra_gamma_aug-none_seed0": "SpecTTTra gamma",
    }
    return mapping.get(run_id, run_id.replace("_", " "))


def _feature_model_family(model_name):
    name = model_name.lower()
    if "multiview" in name:
        return "proposed"
    if any(k in name for k in ["mert", "muq", "moss", "clap"]):
        return "foundation"
    if any(k in name for k in ["convnext", "vit", "efficientvit"]):
        return "vision"
    if "specttra" in name:
        return "sonics"
    return "cnn"


def _feature_verdict(value):
    text = _clean_md_cell(value)
    if "✓" in text:
        return "better"
    if "✗" in text:
        return "not better"
    return "tied"


def _load_feature_summary_rows():
    if not FEATURE_SUMMARY_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(FEATURE_SUMMARY_CSV)
    for col in ["auc", "eer", "tpr_at_1pct_fpr", "tpr_at_01pct_fpr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_feature_baseline_comparison():
    tables = _parse_markdown_tables(FEATURE_BASELINE_MD)
    if len(tables) < 2:
        return pd.DataFrame()

    proposed_tbl = tables[0].copy()
    baseline_tbl = tables[1].copy()
    summary_df = _load_feature_summary_rows()

    metric_lookup = {}
    for _, row in proposed_tbl.iterrows():
        value, low, high = _parse_ci_value(row["Value [95% CI]"])
        metric_lookup[_clean_md_cell(row["Metric"]).lower()] = (value, low, high)

    rows = [{
        "run_id": "multiview_aug-combined_safe_seed",
        "model": "Multiview ensemble",
        "family": "proposed",
        "auc": metric_lookup.get("auc", (np.nan, np.nan, np.nan))[0],
        "auc_ci_low": metric_lookup.get("auc", (np.nan, np.nan, np.nan))[1],
        "auc_ci_high": metric_lookup.get("auc", (np.nan, np.nan, np.nan))[2],
        "eer": metric_lookup.get("eer", (np.nan, np.nan, np.nan))[0],
        "tpr_at_1pct_fpr": metric_lookup.get("tpr@1%fpr", (np.nan, np.nan, np.nan))[0],
        "tpr_at_01pct_fpr": metric_lookup.get("tpr@0.1%fpr", (np.nan, np.nan, np.nan))[0],
        "delta_auc_vs_proposed": 0.0,
        "delong_p": np.nan,
        "wilcoxon_p": np.nan,
        "verdict": "proposed",
    }]

    summary_ood = pd.DataFrame()
    if not summary_df.empty:
        summary_ood = (
            summary_df[summary_df["split"] == "ood"]
            .sort_values(["run_id", "auc"])
            .drop_duplicates(subset=["run_id"], keep="last")
            .set_index("run_id")
        )

    for _, row in baseline_tbl.iterrows():
        run_id = _clean_md_cell(row["Baseline"])
        auc, ci_low, ci_high = _parse_ci_value(row["AUC [95% CI]"])
        metrics = summary_ood.loc[run_id] if run_id in summary_ood.index else {}
        rows.append({
            "run_id": run_id,
            "model": _feature_display_name(run_id),
            "family": _feature_model_family(run_id),
            "auc": auc,
            "auc_ci_low": ci_low,
            "auc_ci_high": ci_high,
            "eer": float(metrics.get("eer", np.nan)) if hasattr(metrics, "get") else np.nan,
            "tpr_at_1pct_fpr": (
                float(metrics.get("tpr_at_1pct_fpr", np.nan))
                if hasattr(metrics, "get") else np.nan
            ),
            "tpr_at_01pct_fpr": (
                float(metrics.get("tpr_at_01pct_fpr", np.nan))
                if hasattr(metrics, "get") else np.nan
            ),
            "delta_auc_vs_proposed": _parse_numeric(row["ΔAUC"]),
            "delong_p": _parse_numeric(row["DeLong p"]),
            "wilcoxon_p": _parse_numeric(row["Wilcoxon p"]),
            "verdict": _feature_verdict(row["Significant"]),
        })
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def _load_feature_invariants():
    tables = _parse_markdown_tables(FEATURE_INVARIANT_MD)
    if not tables:
        return pd.DataFrame()
    df = tables[0].copy()
    df = df.rename(columns={
        "Rank": "rank",
        "Probe": "probe",
        "OOD AUC": "ood_auc",
        "OOD TPR@1%FPR": "ood_tpr_at_1pct_fpr",
        "Val AUC": "val_auc",
    })
    for c in ["rank", "ood_auc", "ood_tpr_at_1pct_fpr", "val_auc"]:
        df[c] = df[c].map(_parse_numeric)
    df["probe"] = df["probe"].map(_clean_md_cell)
    return df.sort_values("rank")


def _load_feature_augmentations():
    tables = _parse_markdown_tables(FEATURE_AUGMENTATION_MD)
    if not tables:
        return pd.DataFrame()
    df = tables[0].copy()
    df = df.rename(columns={
        "Regime": "regime",
        "Val AUC": "val_auc",
        "OOD AUC": "ood_auc",
        "OOD TPR@1%FPR": "ood_tpr_at_1pct_fpr",
        "Gap (val−ood)": "gap_val_minus_ood",
        "Verdict": "verdict",
    })
    df["regime"] = df["regime"].map(_clean_md_cell)
    df["verdict"] = df["verdict"].map(_clean_md_cell)
    for c in ["val_auc", "ood_auc", "ood_tpr_at_1pct_fpr", "gap_val_minus_ood"]:
        df[c] = df[c].map(_parse_numeric)
    return df


def _latex_escape(value):
    return str(value).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def _format_feature_metric(value, digits=3):
    return "---" if pd.isna(value) else f"{float(value):.{digits}f}"


def _write_feature_latex_table(baseline_df):
    if baseline_df.empty:
        return
    rows = []
    rows.append(r"% Auto-generated by analysis/main.py -- feature analysis")
    rows.append(r"\begin{tabular}{llccccc}")
    rows.append(r"\toprule")
    rows.append(
        r"\textbf{Model} & \textbf{Type} & \textbf{OOD AUC [95\% CI]} & "
        r"\textbf{OOD EER} & \textbf{TPR@1\%FPR} & \textbf{$\Delta$AUC} & "
        r"\textbf{Verdict} \\"
    )
    rows.append(r"\midrule")
    best_auc = baseline_df["auc"].max()
    for _, row in baseline_df.iterrows():
        name = _latex_escape(row["model"])
        if row["family"] == "proposed":
            name = r"\textbf{" + name + "}"
        auc_text = (
            f"{_format_feature_metric(row['auc'])} "
            f"[{_format_feature_metric(row['auc_ci_low'])}, "
            f"{_format_feature_metric(row['auc_ci_high'])}]"
        )
        if np.isclose(float(row["auc"]), float(best_auc), equal_nan=False):
            auc_text = r"\textbf{" + auc_text + "}"
        delta = "---" if row["family"] == "proposed" else _format_feature_metric(row["delta_auc_vs_proposed"])
        verdict = {
            "proposed": "reference",
            "better": "multiview better",
            "tied": "tied",
            "not better": "baseline stronger",
        }.get(row["verdict"], row["verdict"])
        rows.append(
            f"{name} & {_latex_escape(row['family'])} & {auc_text} & "
            f"{_format_feature_metric(row['eer'])} & "
            f"{_format_feature_metric(row['tpr_at_1pct_fpr'])} & "
            f"{delta} & {_latex_escape(verdict)} \\\\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    (OUTPUT_DIR / "feature_detection_table.tex").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def feature_detection_analysis(report):
    """Integrate copied feature-detection experiment outputs."""
    report.append("\n" + "=" * 80)
    report.append("22. FEATURE-BASED AI-MUSIC DETECTION ANALYSIS")
    report.append("=" * 80)

    baseline_df = _load_feature_baseline_comparison()
    invariant_df = _load_feature_invariants()
    augmentation_df = _load_feature_augmentations()

    if baseline_df.empty and invariant_df.empty and augmentation_df.empty:
        report.append(f"\n  No feature-analysis outputs found under {FEATURE_DIR}.")
        return

    if not baseline_df.empty:
        baseline_df.to_csv(OUTPUT_DIR / "feature_detection_baselines.csv", index=False)
        _write_feature_latex_table(baseline_df)
        show = baseline_df[[
            "model", "family", "auc", "auc_ci_low", "auc_ci_high",
            "eer", "tpr_at_1pct_fpr", "delta_auc_vs_proposed", "verdict",
        ]].copy()
        for c in ["auc", "auc_ci_low", "auc_ci_high", "eer", "tpr_at_1pct_fpr", "delta_auc_vs_proposed"]:
            show[c] = show[c].astype(float).round(3)
        report.append("\n  --- OOD detector head-to-head ---")
        report.append(show.to_string(index=False))

    if not invariant_df.empty:
        invariant_df.to_csv(OUTPUT_DIR / "feature_invariant_ranking.csv", index=False)
        show_inv = invariant_df.copy()
        for c in ["ood_auc", "ood_tpr_at_1pct_fpr", "val_auc"]:
            show_inv[c] = show_inv[c].astype(float).round(3)
        report.append("\n  --- Invariant probe ranking ---")
        report.append(show_inv.to_string(index=False))

    if not augmentation_df.empty:
        augmentation_df.to_csv(OUTPUT_DIR / "feature_augmentation_ablation.csv", index=False)
        show_aug = augmentation_df.copy()
        for c in ["val_auc", "ood_auc", "ood_tpr_at_1pct_fpr", "gap_val_minus_ood"]:
            show_aug[c] = show_aug[c].astype(float).round(3)
        report.append("\n  --- Augmentation ablation ---")
        report.append(show_aug.to_string(index=False))

    report.append("\n  --- Take-aways ---")
    if not invariant_df.empty:
        best_probe = invariant_df.sort_values("ood_auc", ascending=False).iloc[0]
        report.append(
            f"  Best standalone invariant: {best_probe['probe']} "
            f"(OOD AUC={best_probe['ood_auc']:.3f}, "
            f"TPR@1%FPR={best_probe['ood_tpr_at_1pct_fpr']:.3f})."
        )
    if not baseline_df.empty:
        best_model = baseline_df.sort_values("auc", ascending=False).iloc[0]
        proposed = baseline_df[baseline_df["family"] == "proposed"].iloc[0]
        report.append(
            f"  Best OOD detector: {best_model['model']} "
            f"(AUC={best_model['auc']:.3f}); multiview ensemble reaches "
            f"AUC={proposed['auc']:.3f} with TPR@1%FPR={proposed['tpr_at_1pct_fpr']:.3f}."
        )
    if not augmentation_df.empty:
        loud = augmentation_df[augmentation_df["regime"] == "loudness"]
        if not loud.empty:
            loud = loud.iloc[0]
            report.append(
                f"  Loudness augmentation is the clearest failure mode: "
                f"OOD TPR@1%FPR={loud['ood_tpr_at_1pct_fpr']:.3f} "
                f"despite OOD AUC={loud['ood_auc']:.3f}."
            )
    report.append(
        f"  (Feature-analysis outputs: {OUTPUT_DIR / 'feature_detection_table.tex'}, "
        f"{OUTPUT_DIR / 'feature_detection_baselines.csv'}, "
        f"{OUTPUT_DIR / 'feature_invariant_ranking.csv'}, "
        f"{OUTPUT_DIR / 'feature_augmentation_ablation.csv'})"
    )


def _mood_tag_frame(ann):
    """Return annotation-level mood indicators for the canonical mood tags."""
    rows = ann.copy()

    def _tags(value):
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if pd.isna(value) or value is None:
            return []
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            parsed = []
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
        if str(parsed).strip():
            return [str(parsed).strip()]
        return []

    rows["mood_tags_clean"] = rows["mood_tags"].apply(_tags)
    rows["mood_tag_count"] = rows["mood_tags_clean"].apply(len)
    rows["has_mood_tag"] = (rows["mood_tag_count"] > 0).astype(int)
    for tag in MOOD_TAGS_CANONICAL:
        rows[f"mood::{tag}"] = rows["mood_tags_clean"].apply(lambda tags, t=tag: int(t in set(tags)))
    return rows


def _mood_contrast_stats(df, contrast, group_a_label, group_b_label, mask_a, mask_b):
    """Per-tag prevalence difference with approximate CI and Fisher exact p-values."""
    rows = []
    a = df.loc[mask_a].copy()
    b = df.loc[mask_b].copy()
    n_a = int(len(a))
    n_b = int(len(b))
    for tag in MOOD_TAGS_CANONICAL:
        col = f"mood::{tag}"
        k_a = int(a[col].sum())
        k_b = int(b[col].sum())
        p_a = k_a / n_a if n_a else np.nan
        p_b = k_b / n_b if n_b else np.nan
        diff = p_a - p_b
        se = np.sqrt((p_a * (1 - p_a) / n_a) + (p_b * (1 - p_b) / n_b)) if n_a and n_b else np.nan
        ci_low = diff - 1.96 * se if pd.notna(se) else np.nan
        ci_high = diff + 1.96 * se if pd.notna(se) else np.nan
        try:
            _, p = stats.fisher_exact([[k_a, n_a - k_a], [k_b, n_b - k_b]], alternative="two-sided")
        except ValueError:
            p = np.nan
        rows.append({
            "contrast": contrast,
            "group_a": group_a_label,
            "group_b": group_b_label,
            "mood_tag": tag,
            "n_a": n_a,
            "n_b": n_b,
            "count_a": k_a,
            "count_b": k_b,
            "prevalence_a": p_a,
            "prevalence_b": p_b,
            "diff": diff,
            "diff_ci_low": ci_low,
            "diff_ci_high": ci_high,
            "p_fisher": p,
        })
    out = pd.DataFrame(rows)
    valid = out["p_fisher"].notna()
    out["q_fdr"] = np.nan
    if valid.any():
        out.loc[valid, "q_fdr"] = multipletests(out.loc[valid, "p_fisher"], method="fdr_bh")[1]
    return out


def _compute_mood_tag_analysis(ann):
    df = _mood_tag_frame(ann)
    contrasts = [
        _mood_contrast_stats(
            df,
            "Ground truth: AI songs - real songs",
            "AI songs",
            "real songs",
            df["is_ai_song"],
            ~df["is_ai_song"],
        ),
        _mood_contrast_stats(
            df,
            "Perceived: judged AI - judged real",
            "judged AI",
            "judged real",
            df["authenticity_assessment"] == "ai-generated",
            df["authenticity_assessment"] == "real",
        ),
    ]
    stats_df = pd.concat(contrasts, ignore_index=True)

    summary_rows = []
    group_specs = [
        ("AI songs", df["is_ai_song"]),
        ("real songs", ~df["is_ai_song"]),
        ("judged AI", df["authenticity_assessment"] == "ai-generated"),
        ("judged real", df["authenticity_assessment"] == "real"),
        ("judged uncertain", df["authenticity_assessment"] == "uncertain"),
    ]
    for label, mask in group_specs:
        sub = df.loc[mask]
        summary_rows.append({
            "group": label,
            "n": int(len(sub)),
            "nonempty_mood_annotations": int(sub["has_mood_tag"].sum()),
            "nonempty_share": float(sub["has_mood_tag"].mean()) if len(sub) else np.nan,
            "mean_mood_tags": float(sub["mood_tag_count"].mean()) if len(sub) else np.nan,
            "median_mood_tags": float(sub["mood_tag_count"].median()) if len(sub) else np.nan,
        })

    for contrast, group_a, group_b, mask_a, mask_b in [
        ("Ground truth: AI songs vs real songs", "AI songs", "real songs", df["is_ai_song"], ~df["is_ai_song"]),
        (
            "Perceived: judged AI vs judged real",
            "judged AI",
            "judged real",
            df["authenticity_assessment"] == "ai-generated",
            df["authenticity_assessment"] == "real",
        ),
    ]:
        a_count = df.loc[mask_a, "mood_tag_count"]
        b_count = df.loc[mask_b, "mood_tag_count"]
        try:
            u_stat, p = stats.mannwhitneyu(a_count, b_count, alternative="two-sided")
        except ValueError:
            u_stat, p = np.nan, np.nan
        summary_rows.append({
            "group": contrast,
            "n": int(mask_a.sum() + mask_b.sum()),
            "nonempty_mood_annotations": np.nan,
            "nonempty_share": np.nan,
            "mean_mood_tags": float(a_count.mean() - b_count.mean()),
            "median_mood_tags": float(a_count.median() - b_count.median()),
            "mannwhitney_u": u_stat,
            "mannwhitney_p": p,
            "comparison": f"{group_a} minus {group_b}",
        })
    summary_df = pd.DataFrame(summary_rows)

    tag_counts = Counter(tag for tags in df["mood_tags_clean"] for tag in tags)
    canonical_total = sum(tag_counts.get(tag, 0) for tag in MOOD_TAGS_CANONICAL)
    all_total = sum(tag_counts.values())
    meta = {
        "total_annotations": int(len(df)),
        "canonical_tag_count": int(canonical_total),
        "all_tag_count": int(all_total),
        "canonical_tag_share": float(canonical_total / all_total) if all_total else np.nan,
        "free_text_tag_count": int(all_total - canonical_total),
    }
    return df, stats_df, summary_df, meta


def _write_mood_tag_latex_table(stats_df):
    if stats_df.empty:
        return
    gt = stats_df[stats_df["contrast"] == "Ground truth: AI songs - real songs"].set_index("mood_tag")
    perc = stats_df[stats_df["contrast"] == "Perceived: judged AI - judged real"].set_index("mood_tag")

    def _pct(v):
        return "---" if pd.isna(v) else f"{100 * float(v):.1f}"

    def _diff(v, q):
        if pd.isna(v):
            return "---"
        star = r"$^{*}$" if pd.notna(q) and q < 0.05 else ""
        sign = "+" if v >= 0 else ""
        return f"{sign}{100 * float(v):.1f}{star}"

    rows = [
        r"% Auto-generated by analysis/main.py -- mood tag analysis",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        (
            r"\textbf{Mood tag} & \textbf{AI \%} & \textbf{Real \%} & "
            r"\textbf{$\Delta$ pp} & \textbf{$q$} & "
            r"\textbf{Judged AI \%} & \textbf{Judged real \%} & "
            r"\textbf{$\Delta$ pp} & \textbf{$q$} \\"
        ),
        r"\midrule",
    ]
    for tag in MOOD_TAGS_CANONICAL:
        g = gt.loc[tag]
        p = perc.loc[tag]
        rows.append(
            f"{_latex_escape(tag)} & {_pct(g['prevalence_a'])} & {_pct(g['prevalence_b'])} & "
            f"{_diff(g['diff'], g['q_fdr'])} & {_format_feature_metric(g['q_fdr'], 3)} & "
            f"{_pct(p['prevalence_a'])} & {_pct(p['prevalence_b'])} & "
            f"{_diff(p['diff'], p['q_fdr'])} & {_format_feature_metric(p['q_fdr'], 3)} \\\\"
        )
    rows += [
        r"\bottomrule",
        r"\end{tabular}",
        r"% Positive deltas indicate higher prevalence in AI songs / judged-AI trials.",
        r"% Asterisks mark FDR q < 0.05 within each contrast.",
    ]
    (OUTPUT_DIR / "mood_tag_table.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def mood_tag_analysis(ann, report):
    """Compare human mood-tag selections by provenance and perceived authenticity."""
    report.append("\n" + "=" * 80)
    report.append("23. MOOD-TAG ANALYSIS")
    report.append("=" * 80)

    _, stats_df, summary_df, meta = _compute_mood_tag_analysis(ann)
    stats_df.to_csv(OUTPUT_DIR / "mood_tag_prevalence_comparison.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "mood_tag_summary.csv", index=False)
    _write_mood_tag_latex_table(stats_df)

    report.append(
        "\n  Mood tags are treated as multi-label binary annotations at the trial level. "
        f"The main analysis uses the {len(MOOD_TAGS_CANONICAL)} canonical prompt tags; "
        f"they account for {100 * meta['canonical_tag_share']:.1f}% of all mood-tag selections "
        f"({meta['canonical_tag_count']}/{meta['all_tag_count']}). Sparse free-text tags are "
        "excluded from Figure 22 to keep the comparison estimable and readable."
    )

    report.append("\n  --- Mood-tag completeness and tag-count intensity ---")
    show_summary = summary_df.copy()
    for c in ["nonempty_share", "mean_mood_tags", "median_mood_tags", "mannwhitney_p"]:
        if c in show_summary.columns:
            show_summary[c] = show_summary[c].astype(float).round(4)
    report.append(show_summary.to_string(index=False))

    report.append("\n  --- Canonical mood prevalence differences ---")
    show = stats_df.copy()
    for c in ["prevalence_a", "prevalence_b", "diff", "diff_ci_low", "diff_ci_high", "p_fisher", "q_fdr"]:
        show[c] = show[c].astype(float).round(4)
    report.append(show[[
        "contrast", "mood_tag", "n_a", "n_b", "prevalence_a", "prevalence_b",
        "diff", "diff_ci_low", "diff_ci_high", "p_fisher", "q_fdr",
    ]].to_string(index=False))

    report.append("\n  --- Take-aways ---")
    for contrast in stats_df["contrast"].unique():
        sub = stats_df[stats_df["contrast"] == contrast].copy()
        largest = sub.iloc[sub["diff"].abs().argmax()]
        sig = sub[sub["q_fdr"] < 0.05].sort_values("diff", key=lambda s: s.abs(), ascending=False)
        report.append(
            f"  {contrast}: largest canonical shift is {largest['mood_tag']} "
            f"({100 * largest['diff']:+.1f} pp; 95% CI "
            f"[{100 * largest['diff_ci_low']:+.1f}, {100 * largest['diff_ci_high']:+.1f}] pp; "
            f"FDR q={largest['q_fdr']:.3f})."
        )
        if sig.empty:
            report.append("    No canonical mood tag survives FDR correction at q<0.05.")
        else:
            sig_txt = ", ".join(f"{r.mood_tag} ({100 * r.diff:+.1f} pp)" for r in sig.itertuples())
            report.append(f"    FDR-significant mood shifts: {sig_txt}.")
    report.append(
        f"  (Mood-tag outputs: {OUTPUT_DIR / 'mood_tag_table.tex'}, "
        f"{OUTPUT_DIR / 'mood_tag_prevalence_comparison.csv'}, "
        f"{OUTPUT_DIR / 'mood_tag_summary.csv'})"
    )


def _load_model_mood_long(ann):
    """Load model mood tags as one row per model-run song judgment."""
    model_files = sorted(MODEL_INPUT_DIR.glob("*.json"))
    if not model_files:
        return pd.DataFrame()

    canonical_song_ids = set(ann["song_id"].unique())
    rows = []
    for path in model_files:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, list) or not raw:
            continue

        df = pd.DataFrame(raw)
        if "song_id" not in df.columns or "song_source" not in df.columns or "mood_tags" not in df.columns:
            continue
        df = df.drop_duplicates(subset=["song_id"], keep="last").copy()
        df = df[df["song_id"].isin(canonical_song_ids)]
        if df.empty:
            continue

        run_name = str(df.get("model_run_name", pd.Series([path.stem])).iloc[0] or path.stem)
        if "__think-" in run_name:
            _, thinking_mode = run_name.rsplit("__think-", 1)
        else:
            thinking_mode = "unknown"
        out = pd.DataFrame({
            "song_id": df["song_id"],
            "song_source": df["song_source"],
            "mood_tags": df["mood_tags"],
            "model": run_name,
            "model_alias": model_alias(run_name),
            "base_model_alias": base_model_alias(run_name),
            "thinking_mode": str(thinking_mode).lower(),
        })
        out["thinking_alias"] = _THINKING_ALIAS.get(
            str(thinking_mode).lower(), str(thinking_mode).lower()
        )
        rows.append(out)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _compute_model_human_mood_tag_analysis(ann):
    """Compare each model-run's canonical mood distribution against humans."""
    human_df = _mood_tag_frame(ann)
    model_df = _load_model_mood_long(ann)
    if model_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    model_df = _mood_tag_frame(model_df)

    human_n = int(len(human_df))
    human_prev = {
        tag: float(human_df[f"mood::{tag}"].mean())
        for tag in MOOD_TAGS_CANONICAL
    }
    human_counts = {
        tag: int(human_df[f"mood::{tag}"].sum())
        for tag in MOOD_TAGS_CANONICAL
    }
    mood_cols = [f"mood::{tag}" for tag in MOOD_TAGS_CANONICAL]
    human_song = (
        human_df[["song_id"] + mood_cols]
        .groupby("song_id", as_index=True)
        .max()
    )

    rows = []
    summary_rows = []
    for model_alias_name, sub in model_df.groupby("model_alias", sort=False):
        sub = sub.copy()
        n_model = int(len(sub))
        thinking_alias = str(sub["thinking_alias"].iloc[0])
        base_model = str(sub["base_model_alias"].iloc[0])
        diffs = []
        pvals = []
        for tag in MOOD_TAGS_CANONICAL:
            col = f"mood::{tag}"
            k_model = int(sub[col].sum())
            k_human = human_counts[tag]
            p_model = k_model / n_model if n_model else np.nan
            p_human = human_prev[tag]
            diff = p_model - p_human
            diffs.append(diff)
            try:
                _, p = stats.fisher_exact(
                    [[k_model, n_model - k_model], [k_human, human_n - k_human]],
                    alternative="two-sided",
                )
            except ValueError:
                p = np.nan
            pvals.append(p)
            se = (
                np.sqrt((p_model * (1 - p_model) / n_model) + (p_human * (1 - p_human) / human_n))
                if n_model and human_n else np.nan
            )
            rows.append({
                "model": model_alias_name,
                "base_model": base_model,
                "thinking_alias": thinking_alias,
                "mood_tag": tag,
                "n_model": n_model,
                "n_human": human_n,
                "count_model": k_model,
                "count_human": k_human,
                "prevalence_model": p_model,
                "prevalence_human": p_human,
                "diff_model_minus_human": diff,
                "diff_ci_low": diff - 1.96 * se if pd.notna(se) else np.nan,
                "diff_ci_high": diff + 1.96 * se if pd.notna(se) else np.nan,
                "p_fisher": p,
            })

        model_song = (
            sub[["song_id"] + mood_cols]
            .groupby("song_id", as_index=True)
            .max()
        )
        joined_song = model_song.join(human_song, how="inner", lsuffix="_model", rsuffix="_human")
        model_arr = joined_song[[f"{c}_model" for c in mood_cols]].to_numpy(dtype=int)
        human_arr = joined_song[[f"{c}_human" for c in mood_cols]].to_numpy(dtype=int)
        intersections = (model_arr & human_arr).sum(axis=1)
        model_sizes = model_arr.sum(axis=1)
        human_sizes = human_arr.sum(axis=1)
        denom = model_sizes + human_sizes
        nonempty = denom > 0
        dice = np.full(len(denom), np.nan, dtype=float)
        np.divide(2.0 * intersections, denom, out=dice, where=nonempty)
        jaccard_denom = model_sizes + human_sizes - intersections
        jaccard = np.full(len(jaccard_denom), np.nan, dtype=float)
        np.divide(intersections, jaccard_denom, out=jaccard, where=jaccard_denom > 0)
        tp = int((model_arr & human_arr).sum())
        fp = int((model_arr & (1 - human_arr)).sum())
        fn = int(((1 - model_arr) & human_arr).sum())
        micro_f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else np.nan

        model_dist = np.array([max(float(sub[f"mood::{tag}"].mean()), 0.0) for tag in MOOD_TAGS_CANONICAL])
        human_dist = np.array([max(human_prev[tag], 0.0) for tag in MOOD_TAGS_CANONICAL])
        if model_dist.sum() > 0 and human_dist.sum() > 0:
            model_dist = model_dist / model_dist.sum()
            human_dist = human_dist / human_dist.sum()
            js_distance = float(np.sqrt(0.5 * (
                stats.entropy(model_dist, 0.5 * (model_dist + human_dist)) +
                stats.entropy(human_dist, 0.5 * (model_dist + human_dist))
            )))
        else:
            js_distance = np.nan

        summary_rows.append({
            "model": model_alias_name,
            "base_model": base_model,
            "thinking_alias": thinking_alias,
            "n_model": n_model,
            "n_human": human_n,
            "n_song_overlap": int(len(joined_song)),
            "n_song_overlap_nonempty": int(nonempty.sum()),
            "mean_song_dice": float(np.nanmean(dice)) if np.isfinite(dice).any() else np.nan,
            "mean_song_jaccard": float(np.nanmean(jaccard)) if np.isfinite(jaccard).any() else np.nan,
            "micro_f1_song_tag": float(micro_f1) if pd.notna(micro_f1) else np.nan,
            "mean_abs_diff_pp": float(np.nanmean(np.abs(diffs)) * 100.0),
            "median_abs_diff_pp": float(np.nanmedian(np.abs(diffs)) * 100.0),
            "max_abs_diff_pp": float(np.nanmax(np.abs(diffs)) * 100.0),
            "mean_model_tags_per_trial": float(sub["mood_tag_count"].mean()),
            "mean_human_tags_per_trial": float(human_df["mood_tag_count"].mean()),
            "tag_count_delta": float(sub["mood_tag_count"].mean() - human_df["mood_tag_count"].mean()),
            "nonempty_share_model": float(sub["has_mood_tag"].mean()),
            "nonempty_share_human": float(human_df["has_mood_tag"].mean()),
            "js_distance_canonical_distribution": js_distance,
        })

    comp_df = pd.DataFrame(rows)
    valid = comp_df["p_fisher"].notna()
    comp_df["q_fdr_global"] = np.nan
    if valid.any():
        comp_df.loc[valid, "q_fdr_global"] = multipletests(
            comp_df.loc[valid, "p_fisher"], method="fdr_bh"
        )[1]

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["mean_song_dice", "median_abs_diff_pp"],
        ascending=[False, True],
    )
    human_profile = pd.DataFrame({
        "mood_tag": MOOD_TAGS_CANONICAL,
        "human_prevalence": [human_prev[tag] for tag in MOOD_TAGS_CANONICAL],
        "human_count": [human_counts[tag] for tag in MOOD_TAGS_CANONICAL],
        "human_n": human_n,
    })
    meta = {
        "n_models": int(summary_df["model"].nunique()) if not summary_df.empty else 0,
        "human_n": human_n,
        "human_mean_tags_per_trial": float(human_df["mood_tag_count"].mean()),
        "human_nonempty_share": float(human_df["has_mood_tag"].mean()),
    }
    return comp_df, summary_df, human_profile, meta


def _write_model_mood_latex_table(summary_df):
    if summary_df.empty:
        return
    rows = [
        r"% Auto-generated by analysis/main.py -- model-human mood tag analysis",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        (
            r"\textbf{Evaluator} & \textbf{Thinking} & \textbf{Song Dice} & "
            r"\textbf{Median $|\Delta|$ pp} & \textbf{Mean $|\Delta|$ pp} & "
            r"\textbf{Tags/trial} & "
            r"\textbf{$\Delta$ tags/trial} \\"
        ),
        r"\midrule",
    ]
    for _, row in summary_df.iterrows():
        rows.append(
            f"{_latex_escape(row['model'])} & {_latex_escape(row['thinking_alias'])} & "
            f"{float(row['mean_song_dice']):.3f} & "
            f"{float(row['median_abs_diff_pp']):.1f} & "
            f"{float(row['mean_abs_diff_pp']):.1f} & "
            f"{float(row['mean_model_tags_per_trial']):.2f} & "
            f"{float(row['tag_count_delta']):+.2f} \\\\"
        )
    rows += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (OUTPUT_DIR / "model_human_mood_tag_table.tex").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def model_human_mood_tag_analysis(ann, report):
    """Compare model mood-tag distributions against human annotators."""
    report.append("\n" + "=" * 80)
    report.append("24. MODEL-HUMAN MOOD-TAG ANALYSIS")
    report.append("=" * 80)

    comp_df, summary_df, human_profile, meta = _compute_model_human_mood_tag_analysis(ann)
    if comp_df.empty:
        report.append("\n  No model mood-tag outputs found under input/models/*.json.")
        return

    comp_df.to_csv(OUTPUT_DIR / "model_human_mood_tag_comparison.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "model_human_mood_tag_summary.csv", index=False)
    human_profile.to_csv(OUTPUT_DIR / "human_mood_tag_profile.csv", index=False)
    _write_model_mood_latex_table(summary_df)

    report.append(
        "\n  Model mood tags are compared against the human trial-level canonical mood-tag "
        f"profile (n={meta['human_n']}; mean tags/trial={meta['human_mean_tags_per_trial']:.2f}; "
        f"non-empty share={100 * meta['human_nonempty_share']:.1f}%). "
        "For each model run, the table reports both per-song overlap with human mood tags "
        "(Dice coefficient on shared songs) and marginal prevalence distance across the "
        "nine canonical mood tags. The Dice score is the primary model-ranking metric "
        "because it detects song-level mismatches that a marginal prevalence comparison "
        "would miss."
    )

    show = summary_df.copy()
    for c in [
        "mean_song_dice", "mean_song_jaccard", "micro_f1_song_tag",
        "mean_abs_diff_pp", "median_abs_diff_pp", "max_abs_diff_pp",
        "mean_model_tags_per_trial", "tag_count_delta", "nonempty_share_model",
        "js_distance_canonical_distribution",
    ]:
        show[c] = show[c].astype(float).round(3)
    report.append("\n  --- Model mood-distribution distance from humans ---")
    report.append(show.to_string(index=False))

    sig = comp_df[comp_df["q_fdr_global"] < 0.05].copy()
    report.append("\n  --- Take-aways ---")
    best = summary_df.sort_values("mean_song_dice", ascending=False).iloc[0]
    worst = summary_df.sort_values("mean_song_dice", ascending=True).iloc[0]
    closest_prevalence = summary_df.sort_values("median_abs_diff_pp", ascending=True).iloc[0]
    report.append(
        f"  Best per-song mood overlap: {best['model']} "
        f"(mean Dice={best['mean_song_dice']:.3f}; "
        f"median |prevalence delta|={best['median_abs_diff_pp']:.1f} pp; "
        f"mean |prevalence delta|={best['mean_abs_diff_pp']:.1f} pp)."
    )
    report.append(
        f"  Weakest per-song mood overlap: {worst['model']} "
        f"(mean Dice={worst['mean_song_dice']:.3f}; "
        f"median |prevalence delta|={worst['median_abs_diff_pp']:.1f} pp; "
        f"mean |prevalence delta|={worst['mean_abs_diff_pp']:.1f} pp)."
    )
    report.append(
        f"  Closest robust marginal prevalence profile: {closest_prevalence['model']} "
        f"(median |delta|={closest_prevalence['median_abs_diff_pp']:.1f} pp; "
        f"mean |delta|={closest_prevalence['mean_abs_diff_pp']:.1f} pp)."
    )
    if sig.empty:
        report.append("  No individual model-tag difference survives global FDR correction.")
    else:
        top_sig = (
            sig.assign(abs_diff=sig["diff_model_minus_human"].abs())
            .sort_values("abs_diff", ascending=False)
            .head(8)
        )
        sig_txt = "; ".join(
            f"{r.model} / {r.mood_tag}: {100 * r.diff_model_minus_human:+.1f} pp "
            f"(q={r.q_fdr_global:.3g})"
            for r in top_sig.itertuples()
        )
        report.append(f"  Largest FDR-significant model-tag shifts: {sig_txt}.")
    report.append(
        f"  (Model-human mood outputs: {OUTPUT_DIR / 'model_human_mood_tag_table.tex'}, "
        f"{OUTPUT_DIR / 'model_human_mood_tag_comparison.csv'}, "
        f"{OUTPUT_DIR / 'model_human_mood_tag_summary.csv'})"
    )


def _spearman_against_humans(score_df, score_col, human_song, human_col):
    """Return (rho, p, n) for song-level score vs. a human rating dimension."""
    if score_df.empty or score_col not in score_df.columns:
        return np.nan, np.nan, 0
    joined = (
        score_df.set_index("song_id")[[score_col]]
        .join(human_song[[human_col]], how="inner")
        .dropna()
    )
    if (
        len(joined) < 3
        or joined[score_col].nunique(dropna=True) < 2
        or joined[human_col].nunique(dropna=True) < 2
    ):
        return np.nan, np.nan, int(len(joined))
    rho, p_val = stats.spearmanr(joined[score_col], joined[human_col])
    return (
        float(rho) if pd.notna(rho) else np.nan,
        float(p_val) if pd.notna(p_val) else np.nan,
        int(len(joined)),
    )


def _our_music_aesthetics_axis_for_human(se_axis, human_col):
    """Use the documented overall score for human aesthetic quality."""
    if human_col == "aesthetic_quality":
        return "Overall_Aesthetics"
    return se_axis


def _spearman_fisher_ci(rho, n, alpha=0.05):
    """Approximate CI for a correlation using Fisher's z transform."""
    if pd.isna(rho) or n < 4:
        return np.nan, np.nan
    rho = float(np.clip(rho, -0.999999, 0.999999))
    z = np.arctanh(rho)
    z_delta = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n - 3)
    return float(np.tanh(z - z_delta)), float(np.tanh(z + z_delta))


def custom_model_human_alignment(ann):
    """Build Figure-18 data: custom audio models plus compact LLM reference."""
    human_song = ann.groupby("song_id")[RATING_COLS].mean()
    audiobox_df = _load_audiobox_scores()
    songeval_df = _load_songeval_scores()
    our_aes_df = _load_our_music_aesthetics_scores()
    popularity_df = _load_our_music_popularity_scores()
    rows = []

    def add_row(human_col, evaluator, family, score_label, score_df, score_col, note=""):
        rho, p_val, n = _spearman_against_humans(score_df, score_col, human_song, human_col)
        ci_low, ci_high = _spearman_fisher_ci(rho, n)
        rows.append({
            "human_dim": human_col,
            "human_label": RATING_LABELS[human_col],
            "evaluator": evaluator,
            "family": family,
            "score_label": score_label,
            "n_songs": n,
            "spearman_rho": rho,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "p_spearman": p_val,
            "note": note,
        })

    for human_col in RATING_COLS:
        ab_axis = AUDIOBOX_BY_HUMAN.get(human_col)
        if ab_axis is not None:
            add_row(
                human_col,
                "Audiobox-Aesthetics",
                "custom_aesthetics",
                f"{ab_axis}: {AUDIOBOX_AXIS_LABELS[ab_axis]}",
                audiobox_df,
                ab_axis,
            )

        se_axes = [ax for ax, dim in SONGEVAL_TO_HUMAN.items() if dim == human_col]
        for se_axis in se_axes:
            add_row(
                human_col,
                "SongEval",
                "custom_aesthetics",
                f"{se_axis}: {SONGEVAL_AXIS_LABELS[se_axis]}",
                songeval_df,
                se_axis,
            )
            if human_col == "aesthetic_quality":
                add_row(
                    human_col,
                    "Our Music-Aesthetics",
                    "our_aesthetics",
                    "Overall_Aesthetics: Overall Aesthetics Score",
                    our_aes_df,
                    "Overall_Aesthetics",
                )
            else:
                add_row(
                    human_col,
                    "Our Music-Aesthetics",
                    "our_aesthetics",
                    f"{se_axis}: {SONGEVAL_AXIS_LABELS[se_axis]}",
                    our_aes_df,
                    se_axis,
                )

        add_row(
            human_col,
            "Our Music-Popularity",
            "our_popularity",
            "log1p play count",
            popularity_df,
            "log1p_play_count",
            "preference proxy, not an aesthetic-rating head",
        )
        add_row(
            human_col,
            "Our Music-Popularity",
            "our_popularity",
            "log1p upvote count",
            popularity_df,
            "log1p_upvote_count",
            "preference proxy, not an aesthetic-rating head",
        )

    model_long = _load_model_eval_long(ann)
    if not model_long.empty:
        for human_col in RATING_COLS:
            if human_col not in model_long.columns:
                continue
            human_song_axis = human_song[[human_col]]
            for (base, think), g in model_long.groupby(["base_model_alias", "thinking_alias"]):
                pred_col = f"llm_{human_col}"
                agg = (
                    g.assign(**{human_col: pd.to_numeric(g[human_col], errors="coerce")})
                    .groupby("song_id")[human_col]
                    .mean()
                    .to_frame(pred_col)
                )
                joined = agg.join(human_song_axis, how="inner").dropna()
                if len(joined) >= 3 and joined[pred_col].nunique() >= 2:
                    rho, p_val = stats.spearmanr(joined[pred_col], joined[human_col])
                else:
                    rho = p_val = np.nan
                ci_low, ci_high = _spearman_fisher_ci(rho, len(joined))
                rows.append({
                    "human_dim": human_col,
                    "human_label": RATING_LABELS[human_col],
                    "evaluator": str(base),
                    "family": "llm_thinking" if think == "thinking" else "llm_non_thinking",
                    "score_label": str(think),
                    "n_songs": int(len(joined)),
                    "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "p_spearman": float(p_val) if pd.notna(p_val) else np.nan,
                    "note": f"LLM {human_col} rating compared to human {human_col} song mean",
                })

    return pd.DataFrame(rows)


def _write_aesthetics_llm_custom_summary_table(custom_align):
    """Write a compact paper table summarising Figure 18's per-dimension rhos."""
    if custom_align.empty:
        return

    panel_order = [
        "aesthetic_quality",
        "production_quality",
        "playlist_likelihood",
        "musical_creativity",
        "emotional_engagement",
    ]

    # Compact paper-style display names for LLM evaluators.
    base_model_display = {
        "Gemini 3.1 Pro": "Gemini 3.1 Pro",
        "Gemini 3.1 Flash Lite": "Gemini 3.1 Flash-Lite",
        "Gemma 4 E2B IT": "Gemma 4 E2B",
        "Gemma 4 E4B IT": "Gemma 4 E4B",
        "MOSS Audio 4B": "MOSS-Audio 4B",
        "MOSS Audio 8B": "MOSS-Audio 8B",
    }

    def _best_rho(frame, mask):
        vals = frame.loc[mask, "spearman_rho"].dropna()
        if vals.empty:
            return np.nan
        return float(vals.max())

    def _best_llm_run(frame, mask):
        """Return (rho, evaluator_name, score_label) for the argmax-rho LLM run."""
        sub = frame.loc[mask].dropna(subset=["spearman_rho"])
        if sub.empty:
            return np.nan, None, None
        idx = sub["spearman_rho"].idxmax()
        return (
            float(sub.loc[idx, "spearman_rho"]),
            str(sub.loc[idx, "evaluator"]),
            str(sub.loc[idx, "score_label"]) if "score_label" in sub.columns else None,
        )

    def _fmt(value, row_best):
        if pd.isna(value):
            return "---"
        txt = f"{float(value):+.2f}"
        if pd.notna(row_best) and np.isclose(float(value), float(row_best), atol=5e-4):
            txt = r"\textbf{" + txt + "}"
        return txt

    def _fmt_llm_label(evaluator, score_label):
        if evaluator is None:
            return "---"
        name = base_model_display.get(evaluator, evaluator)
        if score_label is None:
            return _latex_escape(name)
        sl = score_label.lower()
        if "non" in sl:
            suffix = "non-think"
        elif "think" in sl:
            suffix = "think"
        else:
            suffix = sl
        return _latex_escape(f"{name} ({suffix})")

    rows = [
        r"% Auto-generated by analysis/main.py -- aesthetics/custom-model summary",
        r"\begin{tabular}{lrrrrrl}",
        r"\toprule",
        (
            r"\textbf{Human dimension} & \textbf{Audiobox} & \textbf{SongEval} & "
            r"\textbf{Our aesth.} & \textbf{Our pop.} & \textbf{Best LLM} & "
            r"\textbf{Best LLM run} \\"
        ),
        r"\midrule",
    ]
    for human_col in panel_order:
        sub = custom_align[custom_align["human_dim"] == human_col]
        llm_mask = sub["family"].isin(["llm_non_thinking", "llm_thinking"])
        best_llm_rho, best_llm_evaluator, best_llm_label = _best_llm_run(sub, llm_mask)
        vals = {
            "audiobox": _best_rho(sub, sub["evaluator"] == "Audiobox-Aesthetics"),
            "songeval": _best_rho(sub, sub["evaluator"] == "SongEval"),
            "our_aesthetics": _best_rho(sub, sub["evaluator"] == "Our Music-Aesthetics"),
            "our_popularity": _best_rho(sub, sub["evaluator"] == "Our Music-Popularity"),
            "best_llm": best_llm_rho,
        }
        row_best = max([v for v in vals.values() if pd.notna(v)], default=np.nan)
        rows.append(
            f"{_latex_escape(RATING_LABELS[human_col])} & "
            f"{_fmt(vals['audiobox'], row_best)} & "
            f"{_fmt(vals['songeval'], row_best)} & "
            f"{_fmt(vals['our_aesthetics'], row_best)} & "
            f"{_fmt(vals['our_popularity'], row_best)} & "
            f"{_fmt(vals['best_llm'], row_best)} & "
            f"{_fmt_llm_label(best_llm_evaluator, best_llm_label)} \\\\"
        )
    rows += [
        r"\bottomrule",
        r"\end{tabular}",
        r"% Entries are song-level Spearman rho against human song means.",
        r"% Mapping: Audiobox CE/CU/PC/PQ -> aesthetic/playlist/creativity/production.",
        r"% Mapping: SongEval Musicality/Coherence/Memorability/Clarity/Naturalness ->",
        r"% aesthetic/production/playlist/creativity/emotional; our aesthetics uses Overall_Aesthetics for aesthetic.",
        r"% Our pop. is the better of log1p play-count and log1p upvote-count proxies.",
        r"% Best LLM is the best of the twelve LLM judge configurations on that dimension;",
        r"% Best LLM run names the configuration achieving that value (model + thinking mode).",
    ]
    (OUTPUT_DIR / "aesthetics_llm_custom_summary_table.tex").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def songeval_aesthetics_analysis(ann, report):
    """Compare SongEval scores to human, Audiobox-Aesthetics, and LLM ratings."""
    report.append("\n" + "=" * 80)
    report.append("20. SONGEVAL + AUDIO AESTHETICS COMPARISON")
    report.append("=" * 80)

    aes_df = _load_songeval_scores()
    if aes_df.empty:
        report.append(
            "\n  No SongEval scores found. Run "
            "`uv run python aesthetics/SongEval/build_input.py` then "
            "`uv run python aesthetics/SongEval/eval.py -i aesthetics/SongEval/input.txt "
            "-o aesthetics/SongEval/output --use_cpu True` first."
        )
        return

    n_full = int((aes_df["snippet_kind"] == "full").sum())
    n_crop = int((aes_df["snippet_kind"] == "cropped").sum())
    report.append(
        f"\n  Loaded SongEval scores for {len(aes_df)} songs "
        f"(v1 full snippets: {n_full}; v2-v1 cropped 29s: {n_crop})."
    )
    report.append(
        "  Mapping SongEval -> our dimensions: "
        + ", ".join(f"{ax}->{SONGEVAL_TO_HUMAN[ax]}" for ax in SONGEVAL_AXES)
        + ". Both scales are 1-10."
    )

    report.append("\n  --- Global means by source (SongEval vs humans) ---")
    src_rows = []
    for src in ALL_SOURCES:
        sub_aes = aes_df[aes_df["song_source"] == src]
        sub_human = ann[ann["song_source"] == src]
        row = {"source": src, "n_songeval": len(sub_aes), "n_human": len(sub_human)}
        for ax in SONGEVAL_AXES:
            human_col = SONGEVAL_TO_HUMAN[ax]
            row[f"{ax}_se"] = float(sub_aes[ax].mean()) if len(sub_aes) else np.nan
            row[f"{ax}_human"] = float(
                pd.to_numeric(sub_human[human_col], errors="coerce").mean()
            ) if len(sub_human) else np.nan
        src_rows.append(row)
    src_df = pd.DataFrame(src_rows).set_index("source")
    show = src_df.copy()
    for c in show.columns:
        if c not in ("n_songeval", "n_human"):
            show[c] = show[c].astype(float).round(2)
    report.append(show.to_string())

    report.append("\n  --- Song-level alignment with human song means ---")
    human_song = ann.groupby("song_id")[RATING_COLS].mean()
    se_song = aes_df.set_index("song_id")
    align_rows = []
    for ax in SONGEVAL_AXES:
        human_col = SONGEVAL_TO_HUMAN[ax]
        joined = se_song[[ax]].join(human_song[[human_col]], how="inner").dropna()
        if len(joined) >= 3:
            rho, p_rho = stats.spearmanr(joined[ax], joined[human_col])
            r, p_r = stats.pearsonr(joined[ax], joined[human_col])
            mae = float(np.mean(np.abs(joined[ax] - joined[human_col])))
            mean_diff = float((joined[ax] - joined[human_col]).mean())
        else:
            rho = p_rho = r = p_r = mae = mean_diff = np.nan
        align_rows.append({
            "songeval_axis": ax,
            "axis_label": SONGEVAL_AXIS_LABELS[ax],
            "human_dim": human_col,
            "n_songs": int(len(joined)),
            "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
            "pearson_r": float(r) if pd.notna(r) else np.nan,
            "mae": mae,
            "mean_diff_songeval_minus_human": mean_diff,
            "p_spearman": float(p_rho) if pd.notna(p_rho) else np.nan,
        })
    align_df = pd.DataFrame(align_rows)
    show = align_df.copy()
    for c in ["spearman_rho", "pearson_r", "mae", "mean_diff_songeval_minus_human", "p_spearman"]:
        show[c] = show[c].astype(float).round(3)
    report.append(show.to_string(index=False))

    report.append("\n  --- Per-source song-level Spearman rho ---")
    per_src_rows = []
    for ax in SONGEVAL_AXES:
        human_col = SONGEVAL_TO_HUMAN[ax]
        row = {"axis": f"{ax} ({human_col})"}
        for src in ALL_SOURCES:
            sub_aes = aes_df[aes_df["song_source"] == src].set_index("song_id")[[ax]]
            sub_human = (ann[ann["song_source"] == src]
                         .groupby("song_id")[[human_col]].mean())
            joined = sub_aes.join(sub_human, how="inner").dropna()
            if len(joined) >= 3:
                rho, _ = stats.spearmanr(joined[ax], joined[human_col])
            else:
                rho = np.nan
            row[src] = round(float(rho), 3) if pd.notna(rho) else np.nan
            row[f"n_{src}"] = int(len(joined))
        per_src_rows.append(row)
    per_src_df = pd.DataFrame(per_src_rows)
    report.append(per_src_df.to_string(index=False))

    # ----- Direct comparison to Audiobox on shared human-dimension pairs ---
    audiobox_df = _load_audiobox_scores()
    frontend_rows = []
    if not audiobox_df.empty:
        report.append("\n  --- Direct SongEval vs Audiobox-Aesthetics on matched dimensions ---")
        report.append(
            "  Pairs are matched only when both front-ends map to the same human rating "
            "dimension; SongEval Naturalness has no Audiobox analogue."
        )
        ab_song = audiobox_df.set_index("song_id")
        human_song = ann.groupby("song_id")[RATING_COLS].mean()
        for se_axis, ab_axis, human_col in SONGEVAL_AUDIOBOX_PAIRS:
            joined = (
                se_song[[se_axis]]
                .join(ab_song[[ab_axis]], how="inner")
                .join(human_song[[human_col]], how="inner")
                .dropna()
            )
            if len(joined) >= 3:
                rho_frontends, p_frontends = stats.spearmanr(joined[se_axis], joined[ab_axis])
                r_frontends, _ = stats.pearsonr(joined[se_axis], joined[ab_axis])
                rho_se_human, _ = stats.spearmanr(joined[se_axis], joined[human_col])
                rho_ab_human, _ = stats.spearmanr(joined[ab_axis], joined[human_col])
                mae_frontends = float(np.mean(np.abs(joined[se_axis] - joined[ab_axis])))
                mean_diff = float((joined[se_axis] - joined[ab_axis]).mean())
            else:
                rho_frontends = p_frontends = r_frontends = rho_se_human = rho_ab_human = np.nan
                mae_frontends = mean_diff = np.nan
            frontend_rows.append({
                "human_dim": human_col,
                "songeval_axis": se_axis,
                "audiobox_axis": ab_axis,
                "n_songs": int(len(joined)),
                "rho_songeval_vs_human": float(rho_se_human) if pd.notna(rho_se_human) else np.nan,
                "rho_audiobox_vs_human": float(rho_ab_human) if pd.notna(rho_ab_human) else np.nan,
                "rho_songeval_vs_audiobox": float(rho_frontends) if pd.notna(rho_frontends) else np.nan,
                "pearson_songeval_vs_audiobox": float(r_frontends) if pd.notna(r_frontends) else np.nan,
                "mae_songeval_vs_audiobox": mae_frontends,
                "mean_diff_songeval_minus_audiobox": mean_diff,
                "p_songeval_vs_audiobox": float(p_frontends) if pd.notna(p_frontends) else np.nan,
            })
        frontend_df = pd.DataFrame(frontend_rows)
        show_front = frontend_df.copy()
        for c in [
            "rho_songeval_vs_human", "rho_audiobox_vs_human",
            "rho_songeval_vs_audiobox", "pearson_songeval_vs_audiobox",
            "mae_songeval_vs_audiobox", "mean_diff_songeval_minus_audiobox",
            "p_songeval_vs_audiobox",
        ]:
            show_front[c] = show_front[c].astype(float).round(3)
        report.append(show_front.to_string(index=False))
    else:
        frontend_df = pd.DataFrame()

    # ----- Our REDACTED music-aesthetics front-end --------------------------
    our_df = _load_our_music_aesthetics_scores()
    if our_df.empty:
        report.append(
            "\n  No Our Music-Aesthetics scores found. Run "
            "`uv run python aesthetics/our-music-aesthetics-model/build_input.py` then "
            "`uv run python aesthetics/our-music-aesthetics-model/eval.py "
            "-i aesthetics/our-music-aesthetics-model/input.jsonl "
            "-o aesthetics/our-music-aesthetics-model/output --use_cpu True` first."
        )
    else:
        n_full_our = int((our_df["snippet_kind"] == "full").sum())
        n_crop_our = int((our_df["snippet_kind"] == "cropped").sum())
        report.append("\n  --- Our Music-Aesthetics model vs human song means ---")
        report.append(
            f"  Loaded scores for {len(our_df)} songs "
            f"(v1 full snippets: {n_full_our}; v2-v1 cropped 29s: {n_crop_our}). "
            "Raw 1-5 outputs are endpoint-rescaled to 1-10 for this comparison."
        )
        our_song = our_df.set_index("song_id")
        human_song = ann.groupby("song_id")[RATING_COLS].mean()
        our_align_rows = []
        for ax in SONGEVAL_AXES:
            human_col = SONGEVAL_TO_HUMAN[ax]
            our_axis = _our_music_aesthetics_axis_for_human(ax, human_col)
            joined = our_song[[our_axis]].join(human_song[[human_col]], how="inner").dropna()
            if len(joined) >= 3:
                rho, p_rho = stats.spearmanr(joined[our_axis], joined[human_col])
                r, _ = stats.pearsonr(joined[our_axis], joined[human_col])
                mae = float(np.mean(np.abs(joined[our_axis] - joined[human_col])))
                mean_diff = float((joined[our_axis] - joined[human_col]).mean())
            else:
                rho = p_rho = r = mae = mean_diff = np.nan
            our_align_rows.append({
                "axis": our_axis,
                "mapped_songeval_axis": ax,
                "human_dim": human_col,
                "n_songs": int(len(joined)),
                "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                "pearson_r": float(r) if pd.notna(r) else np.nan,
                "mae": mae,
                "mean_diff_our_minus_human": mean_diff,
                "p_spearman": float(p_rho) if pd.notna(p_rho) else np.nan,
            })
        our_align_df = pd.DataFrame(our_align_rows)
        show_our = our_align_df.copy()
        for c in ["spearman_rho", "pearson_r", "mae", "mean_diff_our_minus_human", "p_spearman"]:
            show_our[c] = show_our[c].astype(float).round(3)
        report.append(show_our.to_string(index=False))

        our_frontend_rows = []
        for ax in SONGEVAL_AXES:
            human_col = SONGEVAL_TO_HUMAN[ax]
            our_axis = _our_music_aesthetics_axis_for_human(ax, human_col)
            joined = (
                our_song[[our_axis]]
                .join(se_song[[ax]].rename(columns={ax: f"SongEval_{ax}"}), how="inner")
                .join(human_song[[human_col]], how="inner")
                .dropna()
            )
            ab_axis = AUDIOBOX_BY_HUMAN.get(human_col)
            if not audiobox_df.empty and ab_axis is not None:
                joined = joined.join(
                    audiobox_df.set_index("song_id")[[ab_axis]],
                    how="inner",
                ).dropna()
            if len(joined) >= 3:
                rho_our_human, _ = stats.spearmanr(joined[our_axis], joined[human_col])
                rho_se_human, _ = stats.spearmanr(joined[f"SongEval_{ax}"], joined[human_col])
                rho_our_se, _ = stats.spearmanr(joined[our_axis], joined[f"SongEval_{ax}"])
                mae_our_se = float(np.mean(np.abs(joined[our_axis] - joined[f"SongEval_{ax}"])))
                if ab_axis is not None and ab_axis in joined.columns:
                    rho_ab_human, _ = stats.spearmanr(joined[ab_axis], joined[human_col])
                    rho_our_ab, _ = stats.spearmanr(joined[our_axis], joined[ab_axis])
                    mae_our_ab = float(np.mean(np.abs(joined[our_axis] - joined[ab_axis])))
                else:
                    rho_ab_human = rho_our_ab = mae_our_ab = np.nan
            else:
                rho_our_human = rho_se_human = rho_our_se = mae_our_se = np.nan
                rho_ab_human = rho_our_ab = mae_our_ab = np.nan
            our_frontend_rows.append({
                "human_dim": human_col,
                "axis": our_axis,
                "mapped_songeval_axis": ax,
                "audiobox_axis": ab_axis,
                "n_songs": int(len(joined)),
                "rho_our_vs_human": float(rho_our_human) if pd.notna(rho_our_human) else np.nan,
                "rho_songeval_vs_human": float(rho_se_human) if pd.notna(rho_se_human) else np.nan,
                "rho_audiobox_vs_human": float(rho_ab_human) if pd.notna(rho_ab_human) else np.nan,
                "rho_our_vs_songeval": float(rho_our_se) if pd.notna(rho_our_se) else np.nan,
                "mae_our_vs_songeval": mae_our_se,
                "rho_our_vs_audiobox": float(rho_our_ab) if pd.notna(rho_our_ab) else np.nan,
                "mae_our_vs_audiobox": mae_our_ab,
            })
        our_frontend_df = pd.DataFrame(our_frontend_rows)
        show_our_front = our_frontend_df.copy()
        for c in [
            "rho_our_vs_human", "rho_songeval_vs_human", "rho_audiobox_vs_human",
            "rho_our_vs_songeval", "mae_our_vs_songeval",
            "rho_our_vs_audiobox", "mae_our_vs_audiobox",
        ]:
            show_our_front[c] = show_our_front[c].astype(float).round(3)
        report.append("\n  --- Our model vs SongEval/Audiobox front-ends ---")
        report.append(show_our_front.to_string(index=False))

        our_align_df.to_csv(OUTPUT_DIR / "our_music_aesthetics_alignment.csv", index=False)
        our_frontend_df.to_csv(OUTPUT_DIR / "our_music_aesthetics_frontend_comparison.csv", index=False)
        our_df.to_csv(OUTPUT_DIR / "our_music_aesthetics_song_scores.csv", index=False)

    model_long = _load_model_eval_long(ann)
    rho_table_rows = []
    se_row = {"evaluator": "SongEval", "kind": "songeval"}
    for ax in SONGEVAL_AXES:
        a_row = align_df[align_df["songeval_axis"] == ax].iloc[0]
        se_row[f"rho_{ax}"] = round(float(a_row["spearman_rho"]), 3) if pd.notna(a_row["spearman_rho"]) else np.nan
        se_row[f"mae_{ax}"] = round(float(a_row["mae"]), 3) if pd.notna(a_row["mae"]) else np.nan
        se_row[f"mean_{ax}"] = round(float(aes_df[ax].mean()), 3) if not aes_df.empty else np.nan
    rho_table_rows.append(se_row)

    human_row = {"evaluator": "Humans", "kind": "human"}
    for ax in SONGEVAL_AXES:
        human_col = SONGEVAL_TO_HUMAN[ax]
        m = pd.to_numeric(ann[human_col], errors="coerce").mean()
        human_row[f"rho_{ax}"] = 1.0
        human_row[f"mae_{ax}"] = 0.0
        human_row[f"mean_{ax}"] = round(float(m), 3) if pd.notna(m) else np.nan
    rho_table_rows.append(human_row)

    if not audiobox_df.empty:
        audiobox_row = {"evaluator": "Audiobox-Aesthetics", "kind": "audiobox"}
        ab_song = audiobox_df.set_index("song_id")
        human_song = ann.groupby("song_id")[RATING_COLS].mean()
        for ax in SONGEVAL_AXES:
            human_col = SONGEVAL_TO_HUMAN[ax]
            ab_axis = AUDIOBOX_BY_HUMAN.get(human_col)
            if ab_axis is None:
                audiobox_row[f"rho_{ax}"] = np.nan
                audiobox_row[f"mae_{ax}"] = np.nan
                audiobox_row[f"mean_{ax}"] = np.nan
                continue
            joined = ab_song[[ab_axis]].join(human_song[[human_col]], how="inner").dropna()
            if len(joined) >= 3:
                rho, _ = stats.spearmanr(joined[ab_axis], joined[human_col])
                mae = float(np.mean(np.abs(joined[ab_axis] - joined[human_col])))
            else:
                rho = np.nan
                mae = np.nan
            audiobox_row[f"rho_{ax}"] = round(float(rho), 3) if pd.notna(rho) else np.nan
            audiobox_row[f"mae_{ax}"] = round(float(mae), 3) if pd.notna(mae) else np.nan
            audiobox_row[f"mean_{ax}"] = round(float(audiobox_df[ab_axis].mean()), 3)
        rho_table_rows.append(audiobox_row)

    if not our_df.empty:
        our_row = {"evaluator": "Our Music-Aesthetics", "kind": "our_music_aesthetics"}
        our_song = our_df.set_index("song_id")
        human_song = ann.groupby("song_id")[RATING_COLS].mean()
        for ax in SONGEVAL_AXES:
            human_col = SONGEVAL_TO_HUMAN[ax]
            our_axis = _our_music_aesthetics_axis_for_human(ax, human_col)
            joined = our_song[[our_axis]].join(human_song[[human_col]], how="inner").dropna()
            if len(joined) >= 3:
                rho, _ = stats.spearmanr(joined[our_axis], joined[human_col])
                mae = float(np.mean(np.abs(joined[our_axis] - joined[human_col])))
            else:
                rho = np.nan
                mae = np.nan
            our_row[f"rho_{ax}"] = round(float(rho), 3) if pd.notna(rho) else np.nan
            our_row[f"mae_{ax}"] = round(float(mae), 3) if pd.notna(mae) else np.nan
            our_row[f"mean_{ax}"] = round(float(our_df[our_axis].mean()), 3)
        rho_table_rows.append(our_row)

    if not model_long.empty:
        for (base, think), g in model_long.groupby(["base_model_alias", "thinking_alias"]):
            row = {"evaluator": f"{base} ({think})", "kind": "model"}
            df_m = g.copy()
            for ax in SONGEVAL_AXES:
                human_col = SONGEVAL_TO_HUMAN[ax]
                if human_col not in df_m.columns:
                    row[f"rho_{ax}"] = np.nan
                    row[f"mae_{ax}"] = np.nan
                    row[f"mean_{ax}"] = np.nan
                    continue
                df_m["_model_pred"] = pd.to_numeric(df_m[human_col], errors="coerce")
                agg = df_m.groupby("song_id")["_model_pred"].mean()
                human_song_axis = ann.groupby("song_id")[human_col].mean()
                joined = agg.to_frame("model_pred").join(human_song_axis, how="inner").dropna()
                if len(joined) >= 3:
                    rho, _ = stats.spearmanr(joined["model_pred"], joined[human_col])
                    mae = float(np.mean(np.abs(joined["model_pred"] - joined[human_col])))
                else:
                    rho = np.nan
                    mae = np.nan
                row[f"rho_{ax}"] = round(float(rho), 3) if pd.notna(rho) else np.nan
                row[f"mae_{ax}"] = round(float(mae), 3) if pd.notna(mae) else np.nan
                row[f"mean_{ax}"] = round(float(agg.mean()), 3) if not agg.empty else np.nan
            rho_table_rows.append(row)
    rho_df = pd.DataFrame(rho_table_rows)

    report.append("\n  --- Cross-evaluator alignment with humans (per SongEval axis) ---")
    show_cols = ["evaluator", "kind"] + [f"rho_{ax}" for ax in SONGEVAL_AXES] + [f"mae_{ax}" for ax in SONGEVAL_AXES]
    report.append(rho_df[show_cols].to_string(index=False))

    tex = []
    tex.append(r"% Auto-generated by analysis/main.py -- songeval_aesthetics_analysis")
    tex.append(r"\begin{tabular}{lccccc ccccc}")
    tex.append(r"\toprule")
    tex.append(
        r" & \multicolumn{5}{c}{\textbf{Spearman} $\rho$ \textbf{vs.\ humans}} & "
        r"\multicolumn{5}{c}{\textbf{Mean abs.\ error (1--10)}} \\"
    )
    tex.append(r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}")
    head_axes = " & ".join(rf"\textbf{{{ax[:3]}}}" for ax in SONGEVAL_AXES)
    tex.append(rf"\textbf{{Evaluator}} & {head_axes} & {head_axes} \\")
    tex.append(r"\midrule")

    def _fmt(v):
        if pd.isna(v):
            return "---"
        return f"${v:.2f}$"

    for kind in ["songeval", "audiobox", "our_music_aesthetics", "human", "model"]:
        sub = rho_df[rho_df["kind"] == kind]
        if sub.empty:
            continue
        if kind == "model":
            tex.append(r"\midrule")
        for _, r in sub.iterrows():
            name = str(r["evaluator"]).replace("&", r"\&").replace("_", r"\_")
            rhos = " & ".join(_fmt(r[f"rho_{ax}"]) for ax in SONGEVAL_AXES)
            maes = " & ".join(_fmt(r[f"mae_{ax}"]) for ax in SONGEVAL_AXES)
            tex.append(f"{name} & {rhos} & {maes} \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex_path = OUTPUT_DIR / "songeval_table.tex"
    tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    report.append("")
    report.append(f"  (LaTeX fragment written to {tex_path})")

    out_csv = OUTPUT_DIR / "songeval_alignment.csv"
    align_df.to_csv(out_csv, index=False)
    rho_df.to_csv(OUTPUT_DIR / "songeval_evaluator_alignment.csv", index=False)
    aes_df.to_csv(OUTPUT_DIR / "songeval_song_scores.csv", index=False)
    if not frontend_df.empty:
        frontend_path = OUTPUT_DIR / "audio_aesthetics_frontend_comparison.csv"
        frontend_df.to_csv(frontend_path, index=False)

        tex = []
        tex.append(r"% Auto-generated by analysis/main.py -- SongEval vs Audiobox-Aesthetics")
        tex.append(r"\begin{tabular}{lllrrrr}")
        tex.append(r"\toprule")
        tex.append(
            r"\textbf{Human dim.} & \textbf{SongEval} & \textbf{Audiobox} & "
            r"\textbf{$\rho$ SE-H} & \textbf{$\rho$ AB-H} & "
            r"\textbf{$\rho$ SE-AB} & \textbf{MAE SE-AB} \\"
        )
        tex.append(r"\midrule")
        for _, r in frontend_df.iterrows():
            dim = str(r["human_dim"]).replace("_", r"\_")
            se_axis = str(r["songeval_axis"]).replace("_", r"\_")
            ab_axis = str(r["audiobox_axis"]).replace("_", r"\_")
            tex.append(
                f"{dim} & {se_axis} & {ab_axis} & "
                f"{_fmt(r['rho_songeval_vs_human'])} & "
                f"{_fmt(r['rho_audiobox_vs_human'])} & "
                f"{_fmt(r['rho_songeval_vs_audiobox'])} & "
                f"{_fmt(r['mae_songeval_vs_audiobox'])} \\\\"
            )
        tex.append(r"\bottomrule")
        tex.append(r"\end{tabular}")
        (OUTPUT_DIR / "audio_aesthetics_frontend_table.tex").write_text(
            "\n".join(tex) + "\n", encoding="utf-8"
        )
        report.append(
            f"  (SongEval/Audiobox paired comparison: {frontend_path}; "
            f"{OUTPUT_DIR / 'audio_aesthetics_frontend_table.tex'})"
        )
    report.append(f"  (Per-song scores: {OUTPUT_DIR / 'songeval_song_scores.csv'})")


def _prettify_base_model(raw_base: str) -> str:
    base_model = raw_base.split("__")[0]
    if "/" in base_model:
        base_model = base_model.split("/", 1)[1]
    for suffix in _BASE_VARIANT_SUFFIXES:
        if base_model.endswith(suffix):
            base_model = base_model[: -len(suffix)]
            break
    pretty = base_model.replace("_", " ").replace("-", " ")
    return " ".join(_BASE_WORD_FIXES.get(w.lower(), w) for w in pretty.split())


def base_model_alias(run_name: str) -> str:
    if "__think-" in run_name:
        base, _ = run_name.rsplit("__think-", 1)
    else:
        base = run_name
    return _prettify_base_model(base)


def model_alias(run_name: str) -> str:
    if "__think-" in run_name:
        base, think = run_name.rsplit("__think-", 1)
    else:
        base, think = run_name, ""
    base_pretty = _prettify_base_model(base)
    think_pretty = _THINKING_ALIAS.get(think.lower(), think)
    return f"{base_pretty} ({think_pretty})" if think_pretty else base_pretty


def _load_model_eval_long(ann):
    """Load model outputs as a single long dataframe aligned to canonical songs."""
    model_files = sorted(MODEL_INPUT_DIR.glob("*.json"))
    if not model_files:
        return pd.DataFrame()

    canonical_song_ids = set(ann["song_id"].unique())
    rows = []
    for path in model_files:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, list) or not raw:
            continue

        df = pd.DataFrame(raw)
        if "song_id" not in df.columns or "song_source" not in df.columns:
            continue
        df = df.drop_duplicates(subset=["song_id"], keep="last").copy()
        df = df[df["song_id"].isin(canonical_song_ids)]
        if df.empty:
            continue

        run_name = str(df.get("model_run_name", pd.Series([path.stem])).iloc[0] or path.stem)
        if "__think-" in run_name:
            base_model, thinking_mode = run_name.rsplit("__think-", 1)
        else:
            base_model, thinking_mode = run_name, "unknown"

        assess_col = (
            "authenticity_assessment_posthoc_uncertain"
            if "authenticity_assessment_posthoc_uncertain" in df.columns
            else "authenticity_assessment"
        )
        out = pd.DataFrame({
            "song_id": df["song_id"],
            "song_source": df["song_source"],
            "assessment": df[assess_col],
        })
        for col in RATING_COLS:
            out[col] = pd.to_numeric(df.get(col), errors="coerce")
        out["model"] = run_name
        out["base_model"] = base_model
        out["thinking_mode"] = str(thinking_mode).lower()
        out["base_model_alias"] = base_model_alias(run_name)
        out["thinking_alias"] = _THINKING_ALIAS.get(
            str(thinking_mode).lower(), str(thinking_mode).lower()
        )
        out["model_alias"] = model_alias(run_name)
        rows.append(out)

    if not rows:
        return pd.DataFrame()
    model_long = pd.concat(rows, ignore_index=True)
    model_long["is_ai_song"] = model_long["song_source"] != "human"
    return model_long


def _summarize_evaluator_metrics(ann, model_long):
    """Return evaluator-level summary for humans + model runs."""
    summary_rows = []

    # Human aggregate baseline
    human_assessment = ann["authenticity_assessment"]
    hm = _compute_detection_metrics(ann, assessment_col="authenticity_assessment")
    human_row = {
        "evaluator": "humans",
        "evaluator_alias": "Humans",
        "base_model_alias": "Humans",
        "kind": "human",
        "base_model": "humans",
        "thinking_mode": "human",
        "thinking_alias": "humans",
        "ai_rate": (human_assessment == "ai-generated").mean(),
        "real_rate": (human_assessment == "real").mean(),
        "uncertain_rate": (human_assessment == "uncertain").mean(),
        "aesthetic_mean": pd.to_numeric(ann["aesthetic_quality"], errors="coerce").mean(),
        "accuracy_non_uncertain": hm["accuracy"],
        "hit_rate_on_ai": hm["hit_rate"],
        "cr_rate_on_real": (1.0 - hm["false_alarm_rate"]) if pd.notna(hm["false_alarm_rate"]) else np.nan,
        "f1_ai": hm["f1_ai"],
        "d_prime": hm["d_prime"],
    }
    summary_rows.append(human_row)

    if model_long.empty:
        return pd.DataFrame(summary_rows)

    for model, g in model_long.groupby("model"):
        m = _compute_detection_metrics(
            g.rename(columns={"assessment": "authenticity_assessment"}),
            assessment_col="authenticity_assessment",
        )
        summary_rows.append({
            "evaluator": model,
            "evaluator_alias": model_alias(model),
            "base_model_alias": g["base_model_alias"].iloc[0],
            "kind": "model",
            "base_model": g["base_model"].iloc[0],
            "thinking_mode": g["thinking_mode"].iloc[0],
            "thinking_alias": g["thinking_alias"].iloc[0],
            "ai_rate": (g["assessment"] == "ai-generated").mean(),
            "real_rate": (g["assessment"] == "real").mean(),
            "uncertain_rate": (g["assessment"] == "uncertain").mean(),
            "aesthetic_mean": g["aesthetic_quality"].mean(),
            "accuracy_non_uncertain": m["accuracy"],
            "hit_rate_on_ai": m["hit_rate"],
            "cr_rate_on_real": (1.0 - m["false_alarm_rate"]) if pd.notna(m["false_alarm_rate"]) else np.nan,
            "f1_ai": m["f1_ai"],
            "d_prime": m["d_prime"],
        })

    out = pd.DataFrame(summary_rows)
    return out


def _summarize_evaluator_metrics_binary(ann, model_long):
    """Aggregate model runs that share the same (base_model, binary thinking) alias."""
    rows = []
    human_assessment = ann["authenticity_assessment"]
    hm = _compute_detection_metrics(ann, assessment_col="authenticity_assessment")
    human_cr = (1.0 - hm["false_alarm_rate"]) if pd.notna(hm["false_alarm_rate"]) else np.nan
    human_bacc = (
        0.5 * (hm["hit_rate"] + human_cr)
        if (pd.notna(hm["hit_rate"]) and pd.notna(human_cr)) else np.nan
    )
    rows.append({
        "evaluator_alias": "Humans",
        "base_model_alias": "Humans",
        "kind": "human",
        "thinking_alias": "humans",
        "n_runs": 1,
        "ai_rate": (human_assessment == "ai-generated").mean(),
        "real_rate": (human_assessment == "real").mean(),
        "uncertain_rate": (human_assessment == "uncertain").mean(),
        "aesthetic_mean": pd.to_numeric(ann["aesthetic_quality"], errors="coerce").mean(),
        "accuracy_non_uncertain": hm["accuracy"],
        "balanced_accuracy": human_bacc,
        "hit_rate_on_ai": hm["hit_rate"],
        "cr_rate_on_real": human_cr,
        "f1_ai": hm["f1_ai"],
        "d_prime": hm["d_prime"],
    })

    if model_long is None or model_long.empty:
        return pd.DataFrame(rows)

    for alias, g in model_long.groupby("model_alias"):
        m = _compute_detection_metrics(
            g.rename(columns={"assessment": "authenticity_assessment"}),
            assessment_col="authenticity_assessment",
        )
        cr = (1.0 - m["false_alarm_rate"]) if pd.notna(m["false_alarm_rate"]) else np.nan
        bacc = (
            0.5 * (m["hit_rate"] + cr)
            if (pd.notna(m["hit_rate"]) and pd.notna(cr)) else np.nan
        )
        rows.append({
            "evaluator_alias": alias,
            "base_model_alias": g["base_model_alias"].iloc[0],
            "kind": "model",
            "thinking_alias": g["thinking_alias"].iloc[0],
            "n_runs": g["model"].nunique(),
            "ai_rate": (g["assessment"] == "ai-generated").mean(),
            "real_rate": (g["assessment"] == "real").mean(),
            "uncertain_rate": (g["assessment"] == "uncertain").mean(),
            "aesthetic_mean": g["aesthetic_quality"].mean(),
            "accuracy_non_uncertain": m["accuracy"],
            "balanced_accuracy": bacc,
            "hit_rate_on_ai": m["hit_rate"],
            "cr_rate_on_real": cr,
            "f1_ai": m["f1_ai"],
            "d_prime": m["d_prime"],
        })

    return pd.DataFrame(rows)


def _build_aesthetic_long_df(ann, model_long):
    """Combine per-annotation aesthetic ratings from humans + models into long form."""
    human_df = pd.DataFrame({
        "aesthetic_quality": pd.to_numeric(ann["aesthetic_quality"], errors="coerce"),
    })
    human_df["evaluator_alias"] = "Humans"
    human_df["base_model_alias"] = "Humans"
    human_df["thinking_alias"] = "humans"
    human_df["kind"] = "human"
    human_df = human_df.dropna(subset=["aesthetic_quality"])

    if model_long is None or model_long.empty:
        return human_df

    model_df = model_long[[
        "aesthetic_quality", "model_alias", "base_model_alias", "thinking_alias",
    ]].copy()
    model_df = model_df.rename(columns={"model_alias": "evaluator_alias"})
    model_df["kind"] = "model"
    model_df = model_df.dropna(subset=["aesthetic_quality"])

    return pd.concat([human_df, model_df], ignore_index=True)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def create_figures(ann, pca_model, scaler):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Fig 1: Rating distributions by source (real + four AI platforms)
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
    for i, col in enumerate(RATING_COLS):
        sns.boxplot(data=ann, x="song_source", y=col, order=ALL_SOURCES,
                    palette=PALETTE_SOURCE, ax=axes[i], fliersize=2)
        axes[i].set_title(RATING_LABELS[col], fontsize=11)
        axes[i].set_xlabel("")
        if i == 0:
            axes[i].set_ylabel("Rating")
        else:
            axes[i].set_ylabel("")
        axes[i].tick_params(axis="x", rotation=30)
        # Add n, median, and percentiles below each box
        for j, src in enumerate(ALL_SOURCES):
            vals = ann.loc[ann["song_source"] == src, col].dropna()
            n = len(vals)
            med = vals.median() if n else float("nan")
            p25 = vals.quantile(0.25) if n else float("nan")
            p75 = vals.quantile(0.75) if n else float("nan")
            axes[i].text(j, axes[i].get_ylim()[0] - 0.15, f"n={n}\nmed={med:.1f}\nP25={p25:.1f}\nP75={p75:.1f}",
                         ha="center", va="top", fontsize=6.5, color="0.3")
    fig.suptitle("Rating Distributions by Source (real + AI platforms)", fontsize=14, y=1.02)
    fig.subplots_adjust(bottom=0.22)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(FIG_DIR / "fig1_ratings_by_source.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "fig1_ratings_by_source.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Fig 2: Authenticity assessment distribution by source (includes human)
    ct = pd.crosstab(ann["song_source"], ann["authenticity_assessment"], normalize="index")
    ct = ct.reindex(index=ALL_SOURCES, columns=AUTH_ORDER).fillna(0)
    ct_abs = pd.crosstab(ann["song_source"], ann["authenticity_assessment"])
    ct_abs = ct_abs.reindex(index=ALL_SOURCES, columns=AUTH_ORDER).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(9, 5))
    ct.plot(kind="bar", stacked=True, color=[PALETTE_AUTH[c] for c in AUTH_ORDER], ax=ax)
    for i, src in enumerate(ALL_SOURCES):
        cumulative = 0
        for auth in AUTH_ORDER:
            prop = ct.loc[src, auth] if auth in ct.columns else 0
            count = ct_abs.loc[src, auth] if auth in ct_abs.columns else 0
            if prop > 0.06:
                ax.text(i, cumulative + prop / 2, f"{count}\n({prop:.0%})",
                        ha="center", va="center", fontsize=8, fontweight="bold", color="white")
            cumulative += prop
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Source (real + AI)")
    ax.set_title("Authenticity Assessment Distribution by Source")
    ax.legend(title="Assessment", bbox_to_anchor=(1.05, 1))
    ax.tick_params(axis="x", rotation=30)
    totals = ann["song_source"].value_counts().reindex(ALL_SOURCES).fillna(0).astype(int)
    labels = [
        f"{src}\n({'real' if src == 'human' else 'AI'}, n={totals[src]})"
        for src in ALL_SOURCES
    ]
    ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_authenticity_by_source.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "fig2_authenticity_by_source.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Fig 3: Ratings by authenticity assessment (what people thought)
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
    for i, col in enumerate(RATING_COLS):
        sns.violinplot(data=ann, x="authenticity_assessment", y=col, order=AUTH_ORDER,
                       palette=PALETTE_AUTH, ax=axes[i], inner="box", cut=0)
        axes[i].set_title(RATING_LABELS[col], fontsize=11)
        axes[i].set_xlabel("")
        if i == 0:
            axes[i].set_ylabel("Rating")
        else:
            axes[i].set_ylabel("")
        axes[i].tick_params(axis="x", rotation=30)
        # Add n, median, and percentiles below each violin
        for j, auth in enumerate(AUTH_ORDER):
            vals = ann.loc[ann["authenticity_assessment"] == auth, col].dropna()
            n = len(vals)
            med = vals.median()
            p25, p75 = vals.quantile(0.25), vals.quantile(0.75)
            axes[i].text(j, axes[i].get_ylim()[0] - 0.15, f"n={n}\nmed={med:.1f}\nP25={p25:.1f}\nP75={p75:.1f}",
                         ha="center", va="top", fontsize=6.5, color="0.3")
    fig.suptitle("Ratings by Perceived Authenticity", fontsize=14, y=1.02)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(FIG_DIR / "fig3_ratings_by_assessment.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "fig3_ratings_by_assessment.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Fig 4: Correlation heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = ann[RATING_COLS].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    labels = [RATING_LABELS[c] for c in RATING_COLS]
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                xticklabels=labels, yticklabels=labels, ax=ax, vmin=-1, vmax=1,
                square=True, linewidths=0.5)
    ax.set_title("Rating Dimension Correlations")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_correlation_heatmap.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "fig4_correlation_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Fig 5: PCA biplot
    ratings = ann[RATING_COLS].dropna()
    ratings_scaled = scaler.transform(ratings)
    scores = pca_model.transform(ratings_scaled)
    auth_labels = ann.loc[ratings.index, "authenticity_assessment"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for auth, color in PALETTE_AUTH.items():
        mask = auth_labels == auth
        ax.scatter(scores[mask, 0], scores[mask, 1], c=color, alpha=0.5, s=30, label=auth, edgecolors="none")

    loading_scale = 3
    for j, col in enumerate(RATING_COLS):
        ax.arrow(0, 0,
                 pca_model.components_[0, j] * loading_scale,
                 pca_model.components_[1, j] * loading_scale,
                 head_width=0.08, head_length=0.05, fc="black", ec="black", linewidth=1.2)
        ax.text(pca_model.components_[0, j] * loading_scale * 1.15,
                pca_model.components_[1, j] * loading_scale * 1.15,
                RATING_LABELS[col], fontsize=9, ha="center")

    ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA Biplot of Rating Dimensions")
    ax.legend(title="Assessment")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_pca_biplot.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "fig5_pca_biplot.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Fig 6: "Said AI" rate by source (hits on AI; false alarms on human) and by AI experience
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    said_ai_by_source = ann.groupby("song_source")["authenticity_assessment"].apply(
        lambda x: (x == "ai-generated").mean()
    ).reindex(ALL_SOURCES)
    n_by_source = ann.groupby("song_source").size().reindex(ALL_SOURCES).fillna(0).astype(int)
    said_ai_count = ann.groupby("song_source")["authenticity_assessment"].apply(
        lambda x: (x == "ai-generated").sum()
    ).reindex(ALL_SOURCES).fillna(0).astype(int)
    bars = axes[0].bar(said_ai_by_source.index, said_ai_by_source.values,
                       color=[PALETTE_SOURCE[s] for s in said_ai_by_source.index])
    axes[0].set_ylabel("P(response = 'AI-generated')")
    axes[0].set_title("'Said AI' Rate by Source  (hit rate on AI, FA rate on real)")
    axes[0].set_ylim(0, 1.15)
    for bar, val, src in zip(bars, said_ai_by_source.values, ALL_SOURCES):
        kind = "FA" if src == "human" else "HR"
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{kind}={val:.1%}\n({said_ai_count[src]}/{n_by_source[src]})",
                     ha="center", fontsize=9)
    axes[0].tick_params(axis="x", rotation=30)

    # Right panel: accuracy (correct detection) by AI experience level
    ai_exp_order = ["Heard about it but never tried", "Tried once or twice",
                    "Use occasionally", "Use regularly", "Professional experience with AI music"]
    acc_exp = ann.groupby("participant_ai_music_experience")["is_correct"].mean()
    acc_exp = acc_exp.reindex([e for e in ai_exp_order if e in acc_exp.index])
    n_by_exp = ann.groupby("participant_ai_music_experience").size()
    n_by_exp = n_by_exp.reindex([e for e in ai_exp_order if e in n_by_exp.index])
    correct_by_exp = ann.groupby("participant_ai_music_experience")["is_correct"].sum()
    correct_by_exp = correct_by_exp.reindex([e for e in ai_exp_order if e in correct_by_exp.index])
    short_labels = ["Never tried", "Tried 1-2x", "Occasional", "Regular", "Professional"]
    short_labels = short_labels[:len(acc_exp)]
    bars2 = axes[1].bar(range(len(acc_exp)), acc_exp.values, color="#457B9D")
    axes[1].set_xticks(range(len(acc_exp)))
    axes[1].set_xticklabels(short_labels, rotation=30, ha="right")
    axes[1].set_ylabel("Accuracy (correct real/AI, excl. uncertain)")
    axes[1].set_title("Detection Accuracy by AI Experience")
    axes[1].set_ylim(0, 1.15)
    for bar, val, exp_key in zip(bars2, acc_exp.values, acc_exp.index):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.1%}\n({int(correct_by_exp[exp_key])}/{int(n_by_exp[exp_key])})",
                     ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_detection_rates.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "fig6_detection_rates.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Fig 7: Mean ratings radar chart by source (real + AI platforms)
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))
    means = ann.groupby("song_source")[RATING_COLS].mean().reindex(ALL_SOURCES)
    n_per_source = ann.groupby("song_source").size().reindex(ALL_SOURCES).fillna(0).astype(int)
    angles = np.linspace(0, 2 * np.pi, len(RATING_COLS), endpoint=False).tolist()
    angles += angles[:1]

    for source in ALL_SOURCES:
        if means.loc[source].isna().all():
            continue
        values = means.loc[source].values.tolist()
        values += values[:1]
        kind = "real" if source == "human" else "AI"
        ax.plot(angles, values, "o-", linewidth=2,
                label=f"{source} ({kind}, n={n_per_source[source]})",
                color=PALETTE_SOURCE[source])
        ax.fill(angles, values, alpha=0.1, color=PALETTE_SOURCE[source])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([RATING_LABELS[c] for c in RATING_COLS], fontsize=9)
    ax.set_title("Mean Ratings by Source (real + AI platforms)", pad=20, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_radar_by_source.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "fig7_radar_by_source.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Fig 8: Annotation duration distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    dur = ann["annotation_duration_ms"].dropna() / 1000
    dur_clipped = dur[dur < dur.quantile(0.99)]

    sns.histplot(dur_clipped, bins=40, ax=axes[0], color="#457B9D", edgecolor="white")
    axes[0].axvline(dur.median(), color="#E63946", linestyle="--", label=f"Median: {dur.median():.0f}s")
    axes[0].set_xlabel("Duration (seconds)")
    axes[0].set_title(f"Annotation Duration Distribution (n={len(dur)})")
    dur_stats = (
        f"P25={dur.quantile(0.25):.0f}s\n"
        f"P50={dur.median():.0f}s\n"
        f"P75={dur.quantile(0.75):.0f}s\n"
        f"P95={dur.quantile(0.95):.0f}s\n"
        f"Mean={dur.mean():.0f}s"
    )
    axes[0].text(0.97, 0.95, dur_stats, transform=axes[0].transAxes, fontsize=9,
                 va="top", ha="right", bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))
    axes[0].legend()

    sns.boxplot(data=ann, x="song_source", y=ann["annotation_duration_ms"] / 1000,
                order=ALL_SOURCES, palette=PALETTE_SOURCE, ax=axes[1], fliersize=2,
                showfliers=False)
    axes[1].set_xlabel("Source (real + AI)")
    axes[1].set_ylabel("Duration (seconds)")
    axes[1].set_title("Annotation Duration by Source")
    axes[1].tick_params(axis="x", rotation=30)
    for j, src in enumerate(ALL_SOURCES):
        src_dur = ann.loc[ann["song_source"] == src, "annotation_duration_ms"].dropna() / 1000
        n = len(src_dur)
        if n == 0:
            continue
        med = src_dur.median()
        p25, p75 = src_dur.quantile(0.25), src_dur.quantile(0.75)
        axes[1].text(j, axes[1].get_ylim()[0] - 5, f"n={n}\nmed={med:.0f}s\nP25={p25:.0f}s\nP75={p75:.0f}s",
                     ha="center", va="top", fontsize=7, color="0.3")
    fig.subplots_adjust(bottom=0.2)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(FIG_DIR / "fig8_duration.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "fig8_duration.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Fig 10: Raters per song distribution
    ratings_per_song = ann.groupby("song_id").size()
    rater_counts = ratings_per_song.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(rater_counts.index.astype(str), rater_counts.values, color="#457B9D", edgecolor="white")
    total_songs = ratings_per_song.count()
    for bar, (n_raters, count) in zip(bars, rater_counts.items()):
        pct = count / total_songs * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(rater_counts.values) * 0.02,
                f"{count} songs\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Number of Individual Raters")
    ax.set_ylabel("Number of Songs")
    ax.set_title(f"How Many Individuals Rated Each Song? (n={total_songs} songs, {ann['participant_id'].nunique()} unique raters)")
    # Add summary stats as text box
    summary_text = (
        f"Median raters/song: {ratings_per_song.median():.0f}\n"
        f"Mean raters/song: {ratings_per_song.mean():.2f}\n"
        f"Max raters/song: {ratings_per_song.max():.0f}\n"
        f"Songs with 1 rater: {(ratings_per_song == 1).sum()} ({(ratings_per_song == 1).sum()/total_songs*100:.1f}%)\n"
        f"Songs with 2+ raters: {(ratings_per_song >= 2).sum()} ({(ratings_per_song >= 2).sum()/total_songs*100:.1f}%)"
    )
    ax.text(0.97, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig10_raters_per_song.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "fig10_raters_per_song.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Fig 11: Songs rated per annotator
    songs_per_annotator = ann.groupby("participant_id").size().sort_values(ascending=False)
    n_annotators = len(songs_per_annotator)

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = range(0, int(songs_per_annotator.max()) + 3, 2)
    counts, bin_edges, patches = ax.hist(songs_per_annotator.values, bins=bins,
                                          color="#2A9D8F", edgecolor="white", rwidth=0.9)
    # Label each bar with count + percentage
    for count_val, patch in zip(counts, patches):
        if count_val > 0:
            pct = count_val / n_annotators * 100
            ax.text(patch.get_x() + patch.get_width() / 2, count_val + max(counts) * 0.02,
                    f"{int(count_val)} ({pct:.0f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Number of Songs Rated")
    ax.set_ylabel("Number of Annotators")
    ax.set_title(f"Songs Rated per Annotator (n={n_annotators} annotators, {len(ann)} total ratings)")

    # Percentile / summary stats box
    stats_text = (
        f"Mean: {songs_per_annotator.mean():.1f}\n"
        f"Median (P50): {songs_per_annotator.median():.0f}\n"
        f"P25: {songs_per_annotator.quantile(0.25):.0f}\n"
        f"P75: {songs_per_annotator.quantile(0.75):.0f}\n"
        f"P90: {songs_per_annotator.quantile(0.90):.0f}\n"
        f"Max: {songs_per_annotator.max()}\n"
        f"Rated 1 song: {(songs_per_annotator == 1).sum()} ({(songs_per_annotator == 1).sum()/n_annotators*100:.0f}%)\n"
        f"Rated 10+: {(songs_per_annotator >= 10).sum()} ({(songs_per_annotator >= 10).sum()/n_annotators*100:.0f}%)\n"
        f"Rated 20+: {(songs_per_annotator >= 20).sum()} ({(songs_per_annotator >= 20).sum()/n_annotators*100:.0f}%)"
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig11_songs_per_annotator.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "fig11_songs_per_annotator.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Fig 9: Participant demographics
    unique_participants = ann.drop_duplicates("participant_id")
    total_p = len(unique_participants)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    eng_order = ["casual", "enthusiast", "musician", "professional"]
    eng_data = unique_participants["participant_musical_engagement"].value_counts()
    eng_data = eng_data.reindex([e for e in eng_order if e in eng_data.index])
    bar_eng = axes[0].bar(eng_data.index, eng_data.values, color="#2A9D8F")
    axes[0].set_title(f"Musical Engagement (n={total_p})")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=30)
    for bar, val in zip(bar_eng, eng_data.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val} ({val/total_p*100:.0f}%)", ha="center", fontsize=9, fontweight="bold")

    ai_exp_data = unique_participants["participant_ai_music_experience"].value_counts()
    ai_exp_data = ai_exp_data.reindex([e for e in ai_exp_order if e in ai_exp_data.index])
    bar_ai = axes[1].barh(range(len(ai_exp_data)), ai_exp_data.values, color="#457B9D")
    axes[1].set_yticks(range(len(ai_exp_data)))
    axes[1].set_yticklabels(ai_exp_data.index, fontsize=9)
    axes[1].set_title(f"AI Music Experience (n={total_p})")
    axes[1].set_xlabel("Count")
    for bar, val in zip(bar_ai, ai_exp_data.values):
        axes[1].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                     f"{val} ({val/total_p*100:.0f}%)", va="center", fontsize=9, fontweight="bold")

    training = unique_participants["participant_formal_training_years"].dropna()
    sns.histplot(training, bins=15, ax=axes[2], color="#E9C46A", edgecolor="white")
    axes[2].set_title(f"Formal Musical Training (n={len(training)})")
    axes[2].set_xlabel("Years")
    train_stats = (
        f"P25={training.quantile(0.25):.0f}y\n"
        f"P50={training.median():.0f}y\n"
        f"P75={training.quantile(0.75):.0f}y\n"
        f"Mean={training.mean():.1f}y"
    )
    axes[2].text(0.97, 0.95, train_stats, transform=axes[2].transAxes, fontsize=9,
                 va="top", ha="right", bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Participant Demographics", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_demographics.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "fig9_demographics.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Fig 12: Confusion matrix — ground truth x response
    cm = pd.crosstab(ann["ground_truth"], ann["authenticity_assessment"]).reindex(
        index=GROUND_TRUTH_ORDER, columns=AUTH_ORDER, fill_value=0
    )
    cm_prop = cm.div(cm.sum(axis=1), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False,
                linewidths=0.5, linecolor="white")
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_xlabel("Participant response")
    axes[0].set_ylabel("Ground truth")
    sns.heatmap(cm_prop, annot=True, fmt=".2f", cmap="Purples", ax=axes[1], vmin=0, vmax=1,
                linewidths=0.5, linecolor="white")
    axes[1].set_title("Confusion Matrix (row-normalized proportions)")
    axes[1].set_xlabel("Participant response")
    axes[1].set_ylabel("Ground truth")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig12_confusion_matrix.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "fig12_confusion_matrix.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Fig 13: Snippet-condition comparison (ratings + hit rate on AI songs)
    ai_ann = ann[ann["is_ai_song"]].copy()
    if ai_ann["snippet_condition"].nunique() >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: mean ratings by snippet condition on AI songs
        mean_by_cond = (ai_ann.groupby("snippet_condition")[RATING_COLS]
                              .mean().reindex(SNIPPET_ORDER))
        x = np.arange(len(RATING_COLS))
        width = 0.38
        for i, cond in enumerate(SNIPPET_ORDER):
            if cond in mean_by_cond.index:
                vals = mean_by_cond.loc[cond].values
                axes[0].bar(x + (i - 0.5) * width, vals, width,
                            label=f"{cond} (n={(ai_ann['snippet_condition'] == cond).sum()})",
                            color=PALETTE_SNIPPET[cond])
                for xi, v in zip(x, vals):
                    axes[0].text(xi + (i - 0.5) * width, v + 0.05, f"{v:.2f}",
                                 ha="center", va="bottom", fontsize=8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([RATING_LABELS[c] for c in RATING_COLS],
                                rotation=30, ha="right", fontsize=9)
        axes[0].set_ylabel("Mean rating (AI songs only)")
        axes[0].set_title("Ratings by Snippet Condition (AI songs)")
        axes[0].legend(title="Snippet")

        # Right: hit / miss / uncertain rates on AI songs by snippet condition
        resp_props = pd.crosstab(ai_ann["snippet_condition"],
                                 ai_ann["authenticity_assessment"],
                                 normalize="index").reindex(
            index=SNIPPET_ORDER, columns=AUTH_ORDER, fill_value=0
        )
        resp_props.plot(kind="bar", stacked=True, ax=axes[1],
                        color=[PALETTE_AUTH[c] for c in AUTH_ORDER])
        axes[1].set_ylabel("Proportion of responses")
        axes[1].set_xlabel("Snippet condition")
        axes[1].set_title("Response Distribution on AI Songs by Snippet Condition")
        axes[1].legend(title="Response", bbox_to_anchor=(1.02, 1))
        axes[1].tick_params(axis="x", rotation=0)
        for i, cond in enumerate(SNIPPET_ORDER):
            if cond not in resp_props.index:
                continue
            cumulative = 0
            for auth in AUTH_ORDER:
                prop = resp_props.loc[cond, auth]
                if prop > 0.04:
                    axes[1].text(i, cumulative + prop / 2, f"{prop:.0%}",
                                 ha="center", va="center", fontsize=9,
                                 fontweight="bold", color="white")
                cumulative += prop

        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig13_snippet_condition.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(FIG_DIR / "fig13_snippet_condition.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

    # Model comparison figures (humans vs. model runs, aggregated at the
    # binary thinking / non-thinking grouping defined by _THINKING_ALIAS).
    model_long = _load_model_eval_long(ann)
    evaluator_df = _summarize_evaluator_metrics_binary(ann, model_long)
    if not evaluator_df.empty:
        # Fig 14: Base-rate-corrected detection performance per evaluator alias.
        # Raw accuracy is confounded by the 86/14 AI/real class imbalance
        # (a trivial "always AI" classifier would score ~0.86). We therefore
        # foreground balanced accuracy = 0.5 * (hit_rate + correct_rejection)
        # and F1 for the AI class, with overall accuracy shown as a reference
        # bar to illustrate the base-rate gap.
        det_df = evaluator_df.copy()
        det_df["sort_key"] = det_df["kind"].map({"human": -1, "model": 1}).fillna(1)
        det_df = det_df.sort_values(
            ["sort_key", "balanced_accuracy"], ascending=[True, False]
        )

        metric_cols = [
            ("balanced_accuracy", "Balanced accuracy", "#1D3557"),
            ("hit_rate_on_ai", "Hit rate (AI songs)", "#E63946"),
            ("cr_rate_on_real", "Correct rejection (real)", "#2A9D8F"),
            ("f1_ai", "F1 (AI class)", "#F4A261"),
            ("accuracy_non_uncertain", "Overall accuracy", "#B8B8B8"),
        ]
        n_eval = len(det_df)
        n_metrics = len(metric_cols)
        bar_width = 0.82 / n_metrics
        x = np.arange(n_eval)

        fig, ax = plt.subplots(figsize=(max(10, n_eval * 0.9), 6))
        for i, (col, label, color) in enumerate(metric_cols):
            offsets = x + (i - (n_metrics - 1) / 2) * bar_width
            ax.bar(offsets, det_df[col].values, width=bar_width, label=label, color=color)

        human_bacc = det_df.loc[det_df["kind"] == "human", "balanced_accuracy"].iloc[0]
        # Base-rate floor: trivial "always-AI" classifier on this dataset.
        ai_base_rate = float((ann["song_source"] != "human").mean())
        ax.axhline(human_bacc, linestyle="--", color="#1D3557", linewidth=1.2,
                   label=f"Human balanced acc. = {human_bacc:.2f}")
        ax.axhline(ai_base_rate, linestyle=":", color="grey", linewidth=1.0,
                   label=f"'Always AI' baseline = {ai_base_rate:.2f}")
        ax.axhline(0.5, linestyle=":", color="#999999", linewidth=0.8,
                   label="Chance (balanced)")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Rate (non-uncertain judgments)")
        ax.set_title(
            "AI-or-Real Detection: Humans vs Models\n"
            "(sorted by balanced accuracy; humans fixed leftmost)"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(det_df["evaluator_alias"].tolist(), rotation=35, ha="right", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig14_models_detection_accuracy_vs_humans.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(FIG_DIR / "fig14_models_detection_accuracy_vs_humans.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

        # Fig 15: Aesthetic rating distributions as a faceted "halo" view.
        #
        # Three rows: (top) all trials, (middle) trials the evaluator judged
        # "real", (bottom) trials the evaluator judged "AI-generated".
        # Two columns: humans (left) and AI models grouped by base model ×
        # binary thinking mode (right).
        #
        # The row-to-row vertical shift within a single evaluator is the
        # halo effect (perceived "real" → higher ratings). Each violin is
        # annotated with its mean and a 95% bootstrap CI so the within-
        # evaluator delta is legible at a glance.
        aes_long = _build_aesthetic_long_df(ann, model_long)
        if not aes_long.empty:
            # Attach per-trial authenticity judgments to the rating rows.
            human_aes = pd.DataFrame({
                "aesthetic_quality": pd.to_numeric(ann["aesthetic_quality"], errors="coerce"),
                "assessment": ann["authenticity_assessment"].astype(str),
                "base_model_alias": "Humans",
                "thinking_binary": "humans",
                "kind": "human",
            }).dropna(subset=["aesthetic_quality"])

            if not model_long.empty:
                order_think = ["non-thinking", "thinking"]
                model_aes = pd.DataFrame({
                    "aesthetic_quality": model_long["aesthetic_quality"],
                    "assessment": model_long["assessment"].astype(str),
                    "base_model_alias": model_long["base_model_alias"],
                    "thinking_binary": model_long["thinking_alias"].where(
                        model_long["thinking_alias"].isin(order_think),
                        model_long["thinking_alias"],
                    ),
                    "kind": "model",
                }).dropna(subset=["aesthetic_quality"])
                model_aes = model_aes[model_aes["thinking_binary"].isin(order_think)]
                base_order = sorted(model_aes["base_model_alias"].unique())
            else:
                model_aes = pd.DataFrame(columns=human_aes.columns)
                base_order = []

            human_mean = human_aes["aesthetic_quality"].mean()
            n_base = max(len(base_order), 1)

            row_specs = [
                ("All trials",       None,          "#6A4C93"),
                ("Judged 'real'",    "real",        "#2A9D8F"),
                ("Judged 'AI'",      "ai-generated", "#E63946"),
            ]
            model_thinking_palette = {
                "non-thinking": "#457B9D",
                "thinking": "#E76F51",
            }

            fig, axes = plt.subplots(
                3, 2,
                figsize=(max(11, n_base * 1.8 + 3.0), 12),
                gridspec_kw={"width_ratios": [0.9, max(n_base, 3)]},
                sharey=True, sharex="col",
            )

            def _ci95(vals, n_boot=1000, rng=np.random.default_rng(42)):
                vals = np.asarray(vals)
                vals = vals[~np.isnan(vals)]
                if len(vals) < 2:
                    return (np.nan, np.nan)
                boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
                return (float(np.percentile(boots, 2.5)),
                        float(np.percentile(boots, 97.5)))

            for row_idx, (row_label, filter_val, human_col) in enumerate(row_specs):
                ax_h = axes[row_idx, 0]
                ax_m = axes[row_idx, 1]

                # --- Left column: humans --------------------------------
                hh = human_aes if filter_val is None else human_aes[human_aes["assessment"] == filter_val]
                if len(hh):
                    sns.violinplot(
                        x=["Humans"] * len(hh),
                        y=hh["aesthetic_quality"].values,
                        color=human_col,
                        inner="quartile",
                        cut=0,
                        linewidth=0.9,
                        ax=ax_h,
                    )
                    mean_h = hh["aesthetic_quality"].mean()
                    lo_h, hi_h = _ci95(hh["aesthetic_quality"].values)
                    ax_h.text(
                        0, 10.25,
                        f"μ={mean_h:.2f}\n[{lo_h:.2f}, {hi_h:.2f}]\nn={len(hh)}",
                        ha="center", va="bottom", fontsize=8,
                    )
                ax_h.axhline(human_mean, linestyle="--", color="#6A4C93",
                             linewidth=1.0, alpha=0.6)
                ax_h.set_ylabel(f"{row_label}\n\nAesthetic rating")
                ax_h.set_xlabel("")
                if row_idx == 0:
                    ax_h.set_title("Humans")
                ax_h.grid(axis="y", alpha=0.25)
                ax_h.set_ylim(0.5, 11.2)

                # --- Right column: AI models ----------------------------
                mm = model_aes if filter_val is None else model_aes[model_aes["assessment"] == filter_val]
                if len(mm) and base_order:
                    sns.violinplot(
                        data=mm,
                        x="base_model_alias",
                        y="aesthetic_quality",
                        hue="thinking_binary",
                        order=base_order,
                        hue_order=order_think,
                        palette=model_thinking_palette,
                        inner="quartile",
                        cut=0,
                        linewidth=0.8,
                        ax=ax_m,
                    )
                    # Annotate mean ± 95% CI above each violin.
                    group_width = 0.8
                    inner = group_width / len(order_think)
                    for xi, base in enumerate(base_order):
                        for hj, think in enumerate(order_think):
                            sub = mm[(mm["base_model_alias"] == base) &
                                     (mm["thinking_binary"] == think)]
                            if sub.empty:
                                continue
                            mu = sub["aesthetic_quality"].mean()
                            lo, hi = _ci95(sub["aesthetic_quality"].values)
                            x_pos = xi + (hj - (len(order_think) - 1) / 2) * inner
                            ax_m.text(
                                x_pos, 10.25,
                                f"{mu:.1f}\n[{lo:.1f},{hi:.1f}]",
                                ha="center", va="bottom", fontsize=7,
                                color=model_thinking_palette[think],
                            )
                ax_m.axhline(human_mean, linestyle="--", color="#6A4C93",
                             linewidth=1.0, alpha=0.6,
                             label=f"Human overall mean = {human_mean:.2f}"
                                   if row_idx == 0 else None)
                ax_m.set_ylabel("")
                ax_m.set_xlabel("Base model" if row_idx == 2 else "")
                ax_m.tick_params(axis="x", rotation=20)
                for lbl in ax_m.get_xticklabels():
                    lbl.set_ha("right")
                if row_idx == 0:
                    ax_m.set_title("AI Models (grouped by thinking mode)")
                    handles, labels = ax_m.get_legend_handles_labels()
                    ax_m.legend(handles=handles, labels=labels,
                                title="Thinking mode", loc="lower right",
                                fontsize=8)
                else:
                    leg = ax_m.get_legend()
                    if leg is not None:
                        leg.remove()
                ax_m.grid(axis="y", alpha=0.25)
                ax_m.set_ylim(0.5, 11.2)

            fig.suptitle(
                "Aesthetic Ratings by Evaluator and Perceived Authenticity\n"
                "(row-to-row gap = within-evaluator halo effect)",
                fontsize=13, y=0.995,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.975))
            fig.savefig(FIG_DIR / "fig15_models_aesthetic_vs_humans.pdf", bbox_inches="tight", dpi=300)
            fig.savefig(FIG_DIR / "fig15_models_aesthetic_vs_humans.png", bbox_inches="tight", dpi=300)
            plt.close(fig)

        # Fig 16: Thinking-mode effect per base model.
        #   Left column:   top = detection accuracy, bottom = F1 (AI class).
        #   Right column:  per-song aesthetic rating distribution (violin),
        #                  spanning both rows.
        thinking_df = evaluator_df[evaluator_df["kind"] == "model"].copy()
        if not thinking_df.empty and thinking_df["base_model_alias"].nunique() > 0:
            order_think = [m for m in THINKING_ALIAS_ORDER
                           if m in thinking_df["thinking_alias"].unique()]
            if not order_think:
                order_think = sorted(thinking_df["thinking_alias"].dropna().unique())
            thinking_df["thinking_alias"] = pd.Categorical(
                thinking_df["thinking_alias"], categories=order_think, ordered=True
            )

            fig, ax_roc_only = plt.subplots(figsize=(9.5, 8.5))
            axd = {"roc": ax_roc_only}

            # ROC-space scatter — one point per (base model × thinking mode).
            # x = false-alarm rate (real → 'AI'), y = hit rate (AI → 'AI').
            # Color encodes base model; shape encodes thinking mode so that the
            # effect is readable without relying on color alone (colorblind-friendly):
            #   ○  circle  = non-thinking
            #   ◆  diamond = thinking
            # Arrows connect each model's non-thinking → thinking point.
            _shape_map = {"non-thinking": "o", "thinking": "D"}
            _base_models = sorted(thinking_df["base_model_alias"].unique())
            _model_colors = dict(zip(_base_models,
                                     sns.color_palette("tab10", len(_base_models))))
            ax_roc = axd["roc"]

            # Iso-d' contours: curves where Φ⁻¹(HR) - Φ⁻¹(FAR) = d' = constant.
            # Standard SDT ROC annotation — points on the same curve have equal
            # discriminability regardless of response bias.
            _far_grid = np.linspace(0.001, 0.999, 500)
            for _dprime, _alpha, _lw in [(0.5, 0.18, 0.8),
                                          (1.0, 0.28, 1.0),
                                          (1.5, 0.28, 1.0),
                                          (2.0, 0.28, 1.0)]:
                _tpr_grid = stats.norm.cdf(stats.norm.ppf(_far_grid) + _dprime)
                _mask = (_tpr_grid >= 0) & (_tpr_grid <= 1)
                ax_roc.plot(_far_grid[_mask], _tpr_grid[_mask],
                            color="steelblue", alpha=_alpha, linewidth=_lw,
                            linestyle="--", zorder=0)
                # Label at FAR ≈ 0.55 (right side of plot, clear of most points)
                _label_far = 0.55
                _label_tpr = stats.norm.cdf(stats.norm.ppf(_label_far) + _dprime)
                if 0.0 < _label_tpr < 1.0:
                    ax_roc.text(_label_far + 0.01, _label_tpr + 0.01,
                                f"d′={_dprime:.1f}",
                                fontsize=7, color="steelblue", alpha=0.6,
                                va="bottom", ha="left")

            # Iso-F1 contours: straight lines in ROC space for the AI-class F1
            # at the actual AI/real class ratio in this benchmark.
            # Derivation: F1 = 2·TPR·P / (P·(1+TPR) + FPR·N)
            # → TPR = F1/(2−F1) × (1 + FPR × N/P)
            # The slope scales with N/P (real/AI). When AI ≫ real the lines are
            # nearly flat — meaning F1 ≈ recall in that regime. A model that
            # always says "AI" lands at (FAR=1, HR=1), which sits far above the
            # F1=0.9 line, exposing the trivial bias. This is the standard approach
            # for imbalanced binary detection tasks at NeurIPS/ICML.
            _n_ai_roc = int(ann["is_ai_song"].sum())
            _n_real_roc = int((~ann["is_ai_song"]).sum())
            _inv_ratio = _n_real_roc / max(_n_ai_roc, 1)  # N/P = real/AI
            _fpr_f1 = np.linspace(0.0, 1.0, 500)
            for _f1_val, _f1_alpha, _f1_lw in [(0.5, 0.22, 0.9),
                                                (0.7, 0.30, 1.0),
                                                (0.9, 0.30, 1.0)]:
                _k = _f1_val / (2.0 - _f1_val)
                _tpr_f1 = _k * (1.0 + _fpr_f1 * _inv_ratio)
                _valid = (_tpr_f1 >= 0.0) & (_tpr_f1 <= 1.0)
                ax_roc.plot(_fpr_f1[_valid], _tpr_f1[_valid],
                            color="darkorange", alpha=_f1_alpha, linewidth=_f1_lw,
                            linestyle="-.", zorder=0)
                _lt0 = _k  # y-intercept at FPR=0
                if 0.0 <= _lt0 <= 1.0:
                    ax_roc.text(0.01, _lt0 + 0.01, f"F1={_f1_val:.1f}",
                                fontsize=7, color="darkorange", alpha=0.7,
                                va="bottom", ha="left")

            # Chance diagonal
            ax_roc.plot([0, 1], [0, 1], linestyle=":", color="lightgrey",
                        linewidth=1.0, zorder=0)

            # Scatter points + arrows per base model
            for bm in _base_models:
                grp = (thinking_df[thinking_df["base_model_alias"] == bm]
                       .set_index("thinking_alias"))
                color = _model_colors[bm]
                for alias in order_think:
                    if alias not in grp.index:
                        continue
                    row = grp.loc[alias]
                    fpr = 1.0 - row["cr_rate_on_real"]
                    tpr = row["hit_rate_on_ai"]
                    marker = _shape_map.get(alias, "s")
                    ax_roc.scatter(fpr, tpr, color=color, marker=marker,
                                   s=160, zorder=3, edgecolors="white", linewidths=0.6)
                # Arrow: non-thinking → thinking.
                # shrinkA/shrinkB pull the tail/head back in points so the
                # arrowhead doesn't disappear behind the destination marker.
                if "non-thinking" in grp.index and "thinking" in grp.index:
                    nt_fpr = 1.0 - grp.loc["non-thinking", "cr_rate_on_real"]
                    nt_tpr = grp.loc["non-thinking", "hit_rate_on_ai"]
                    t_fpr = 1.0 - grp.loc["thinking", "cr_rate_on_real"]
                    t_tpr = grp.loc["thinking", "hit_rate_on_ai"]
                    ax_roc.annotate(
                        "", xy=(t_fpr, t_tpr), xytext=(nt_fpr, nt_tpr),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=1.8, mutation_scale=14,
                                        shrinkA=6, shrinkB=10),
                        zorder=2,
                    )

            # ---- Helper: compute HR / FAR for a subset of annotations ----
            def _roc_point(subset):
                """Return (fpr, tpr, n_real) for a subset, excluding uncertain
                responses. If n_real=0 the fpr is undefined; we return fpr=None
                in that case so the caller can decide how to render the point
                (e.g. as a hit-rate-only marker on the FPR=0 axis)."""
                _d = subset[subset["authenticity_assessment"] != "uncertain"]
                _ai_n = int(_d["is_ai_song"].sum())
                _re_n = int((~_d["is_ai_song"]).sum())
                if _ai_n == 0:
                    return None, None, _re_n
                _hr = int((_d["is_ai_song"] & (_d["detected_ai"] == 1)).sum()) / _ai_n
                if _re_n == 0:
                    return None, _hr, 0
                _fr = int((~_d["is_ai_song"] & (_d["detected_ai"] == 1)).sum()) / _re_n
                return _fr, _hr, _re_n

            def _plot_human_group(subsets, labels, marker, palette_cm, marker_zorder=4):
                """Plot markers for one human-group series.
                Returns (legend_handles, text_objects, xs, ys) — caller collects
                all texts and calls adjust_text once so labels for all groups are
                repositioned together without overlap.

                Groups with no real-song trials (FAR undefined, e.g. the small
                professional musical-engagement cell) are still plotted: they
                are rendered as an open marker at FPR=0 so the hit rate is
                still visible, with the legend noting ``n_real=0``."""
                _colors = palette_cm(np.linspace(0.40, 0.88, max(len(subsets), 1)))
                _leg_h, _texts, _xs, _ys = [], [], [], []
                for _lbl, _col, _sub in zip(labels, _colors, subsets):
                    _fr, _hr, _re_n = _roc_point(_sub)
                    if _hr is None:
                        continue
                    if _fr is None:
                        # No real trials: place at FPR=0 with a hollow marker.
                        _fr_plot = 0.0
                        ax_roc.scatter(_fr_plot, _hr, facecolors="none",
                                       edgecolors=_col, marker=marker,
                                       s=240, linewidths=1.6, zorder=marker_zorder)
                        _label_txt = f"{_lbl} *"
                        _leg_text = (f"{_lbl}  (HR={_hr:.2f}, FAR undef., "
                                     f"n_real=0)")
                    else:
                        _fr_plot = _fr
                        ax_roc.scatter(_fr_plot, _hr, color=_col, marker=marker,
                                       s=220, zorder=marker_zorder,
                                       edgecolors="white", linewidths=0.5)
                        _label_txt = _lbl
                        _leg_text = f"{_lbl}  (HR={_hr:.2f}, FAR={_fr:.2f})"
                    _t = ax_roc.text(_fr_plot, _hr, _label_txt,
                                     fontsize=7, color=_col, va="center",
                                     zorder=6)
                    _texts.append(_t)
                    _xs.append(_fr_plot)
                    _ys.append(_hr)
                    _leg_h.append(matplotlib.lines.Line2D(
                        [0], [0], color=_col, marker=marker, linestyle="None",
                        markersize=9,
                        markerfacecolor=("none" if _fr is None else _col),
                        markeredgecolor=_col,
                        label=_leg_text))
                return _leg_h, _texts, _xs, _ys

            # -- Group 1: Musical engagement  ★  purple --
            _eng_levels = ["casual", "enthusiast", "musician", "professional"]
            _eng_labels = ["Casual", "Enthusiast", "Musician", "Professional"]
            _eng_subsets = [ann[ann["participant_musical_engagement"] == e]
                            for e in _eng_levels
                            if e in ann["participant_musical_engagement"].values]
            _eng_labels_present = [_eng_labels[_eng_levels.index(e)]
                                   for e in _eng_levels
                                   if e in ann["participant_musical_engagement"].values]
            _eng_leg, _eng_texts, _eng_xs, _eng_ys = _plot_human_group(
                _eng_subsets, _eng_labels_present, "*", plt.cm.Purples)

            # -- Group 2: AI music experience  ■  teal — raw labels in frequency order --
            _ae_order = [
                "Heard about it but never tried",
                "Tried once or twice",
                "Use occasionally",
                "Use regularly",
                "Professional experience with AI music",
            ]
            _ae_present = [lbl for lbl in _ae_order
                           if lbl in ann["participant_ai_music_experience"].values]
            _ae_subsets = [ann[ann["participant_ai_music_experience"] == lbl]
                           for lbl in _ae_present]
            _ae_leg, _ae_texts, _ae_xs, _ae_ys = _plot_human_group(
                _ae_subsets, _ae_present, "s", plt.cm.GnBu)

            # Reposition all labels jointly so they don't overlap each other or
            # the markers. Thin grey lines connect each label back to its point.
            _all_texts = _eng_texts + _ae_texts
            _all_xs = np.array(_eng_xs + _ae_xs)
            _all_ys = np.array(_eng_ys + _ae_ys)
            if _all_texts:
                _adjust_text(
                    _all_texts,
                    x=_all_xs, y=_all_ys,
                    ax=ax_roc,
                    expand=(1.3, 1.6),
                    force_text=(0.4, 0.6),
                    force_points=(0.3, 0.4),
                    arrowprops=dict(arrowstyle="-", color="grey",
                                   lw=0.6, alpha=0.7),
                )

            ax_roc.set_xlim(-0.02, 1.02)
            ax_roc.set_ylim(-0.02, 1.02)
            ax_roc.set_aspect("equal", adjustable="box")
            ax_roc.set_xlabel("False Alarm Rate  (real → 'AI')")
            ax_roc.set_ylabel("Hit Rate  (AI → 'AI')")
            ax_roc.set_title("Thinking Mode Effect: ROC Space")
            ax_roc.grid(alpha=0.25)

            # Legend — model colors | two human-group series | shape + line keys
            _leg = []
            for bm, c in _model_colors.items():
                _leg.append(matplotlib.lines.Line2D(
                    [0], [0], color=c, marker="o", linestyle="-",
                    markersize=8, label=bm))
            _leg.append(matplotlib.patches.Patch(
                color="none", label="— ★ Engagement (purple) —"))
            _leg.extend(_eng_leg)
            _leg.append(matplotlib.patches.Patch(
                color="none", label="— ■ AI experience (teal) —"))
            _leg.extend(_ae_leg)
            _leg.append(matplotlib.lines.Line2D(
                [0], [0], color="k", marker="o", linestyle="None",
                markersize=8, label="○  non-thinking"))
            _leg.append(matplotlib.lines.Line2D(
                [0], [0], color="k", marker="D", linestyle="None",
                markersize=8, label="◆  thinking"))
            _leg.append(matplotlib.lines.Line2D(
                [0], [0], color="darkorange", linestyle="-.", linewidth=1.0,
                label=f"iso-F1 (AI class, P/N={_n_ai_roc}/{_n_real_roc})"))
            _leg.append(matplotlib.lines.Line2D(
                [0], [0], color="steelblue", linestyle="--", linewidth=1.0,
                label="iso-d′"))
            ax_roc.legend(handles=_leg, loc="lower right",
                          fontsize=7.5, title="Models  |  Human subgroups",
                          ncol=2)

            fig.tight_layout()
            fig.savefig(FIG_DIR / "fig16_models_thinking_mode_effects.pdf", bbox_inches="tight", dpi=300)
            fig.savefig(FIG_DIR / "fig16_models_thinking_mode_effects.png", bbox_inches="tight", dpi=300)
            plt.close(fig)

    # -- Fig 17: Audiobox-aesthetics vs humans -------------------------------
    aes_df = _load_audiobox_scores()
    if not aes_df.empty:
        human_song = ann.groupby("song_id")[RATING_COLS].mean()
        aes_song = aes_df.set_index("song_id")
        joined_full = aes_song.join(human_song, how="inner")

        fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
        for i, ax_name in enumerate(AUDIOBOX_AXES):
            ax = axes[i]
            human_col = AUDIOBOX_TO_HUMAN[ax_name]
            sub = joined_full[[ax_name, human_col, "song_source"]].dropna()
            if sub.empty:
                ax.set_visible(False)
                continue
            for src in ALL_SOURCES:
                ss = sub[sub["song_source"] == src]
                if ss.empty:
                    continue
                ax.scatter(
                    ss[human_col], ss[ax_name],
                    s=18, alpha=0.65,
                    color=PALETTE_SOURCE.get(src, "#888888"),
                    edgecolor="white", linewidth=0.4,
                    label=src,
                )
            # Identity line
            ax.plot([1, 10], [1, 10], "k--", lw=0.8, alpha=0.5, label="y = x")
            # OLS fit
            xs = sub[human_col].to_numpy()
            ys = sub[ax_name].to_numpy()
            if len(xs) >= 3 and np.std(xs) > 0:
                slope, intercept = np.polyfit(xs, ys, 1)
                xs_line = np.array([xs.min(), xs.max()])
                ax.plot(xs_line, slope * xs_line + intercept,
                        color="#264653", lw=1.2, label="OLS")
            rho, _ = stats.spearmanr(sub[human_col], sub[ax_name])
            r, _ = stats.pearsonr(sub[human_col], sub[ax_name])
            mae = float(np.mean(np.abs(sub[human_col] - sub[ax_name])))
            ax.set_title(
                f"{ax_name} ({AUDIOBOX_AXIS_LABELS[ax_name]})  vs.\n"
                f"{RATING_LABELS[human_col]}\n"
                rf"$\rho$={rho:.2f}, r={r:.2f}, MAE={mae:.2f}",
                fontsize=10,
            )
            ax.set_xlabel(f"Human mean: {RATING_LABELS[human_col]}")
            ax.set_ylabel(f"Audiobox: {ax_name}")
            ax.set_xlim(0.5, 10.5)
            ax.set_ylim(0.5, 10.5)
            ax.grid(alpha=0.25)
            if i == len(AUDIOBOX_AXES) - 1:
                ax.legend(loc="lower right", fontsize=7, framealpha=0.85)
        fig.suptitle(
            "Audiobox-Aesthetics vs. human ratings (song-level means, 1-10)",
            fontsize=13, y=1.02,
        )
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig17_audiobox_vs_humans.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(FIG_DIR / "fig17_audiobox_vs_humans.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

    # -- Fig 18: custom audio models vs humans (+ compact LLM reference) ----
    custom_align = custom_model_human_alignment(ann)
    if not custom_align.empty:
        custom_align.to_csv(
            OUTPUT_DIR / "aesthetics_llm_vs_custom_models.csv",
            index=False,
        )
        _write_aesthetics_llm_custom_summary_table(custom_align)

        panel_order = [
            "aesthetic_quality",
            "production_quality",
            "playlist_likelihood",
            "musical_creativity",
            "emotional_engagement",
        ]
        family_colors = {
            "custom_aesthetics": "#5E60CE",
            "our_aesthetics": "#00A6A6",
            "our_popularity": "#2A9D8F",
            "llm_non_thinking": "#F4A261",
            "llm_thinking": "#E76F51",
        }
        fig, axes = plt.subplots(
            1,
            len(panel_order),
            figsize=(27, 13.0),
            sharex=True,
            sharey=False,
        )

        for ax, human_col in zip(axes, panel_order):
            sub = custom_align[
                (custom_align["human_dim"] == human_col)
                & custom_align["spearman_rho"].notna()
            ].copy()

            def _short_model_label(row):
                evaluator = row["evaluator"]
                score = str(row["score_label"]).split(":")[0]
                if evaluator == "Audiobox-Aesthetics":
                    full_name = str(row["score_label"]).split(":", 1)[-1].strip()
                    return f"Audiobox\n{full_name}"
                if evaluator == "SongEval":
                    return f"SongEval\n{score}"
                if evaluator == "Our Music-Aesthetics":
                    if score == "Overall_Aesthetics":
                        score = "Overall Aesthetics"
                    return f"Our Aesthetics\n{score}"
                if evaluator == "Our Music-Popularity":
                    target = "plays" if "play" in str(row["score_label"]) else "upvotes"
                    return f"Our Popularity\n{target}"
                if row["family"] in ("llm_non_thinking", "llm_thinking"):
                    return f"{evaluator}\n{row['score_label']}"
                return str(evaluator)

            sub["plot_label"] = sub.apply(_short_model_label, axis=1)
            # With matplotlib.barh, later rows appear higher on the plot.
            # Ascending sort therefore places the strongest alignment at top.
            sub = sub.sort_values("spearman_rho", ascending=True)

            if sub.empty:
                ax.set_visible(False)
                continue

            y_pos = np.arange(len(sub))
            colors = [family_colors.get(k, "#A8DADC") for k in sub["family"]]
            for idx, (_, row) in enumerate(sub.iterrows()):
                low = row.get("ci95_low", np.nan)
                high = row.get("ci95_high", np.nan)
                if pd.notna(low) and pd.notna(high):
                    ax.barh(
                        y_pos[idx],
                        float(high) - float(low),
                        left=float(low),
                        height=0.82,
                        color=family_colors.get(row["family"], "#A8DADC"),
                        alpha=0.28,
                        edgecolor="none",
                        zorder=1,
                    )
            ax.barh(
                y_pos,
                sub["spearman_rho"],
                color=colors,
                edgecolor="white",
                linewidth=0.7,
                height=0.56,
                zorder=2,
            )

            for idx, (_, row) in enumerate(sub.iterrows()):
                y = y_pos[idx]
                val = float(row["spearman_rho"])
                ax.text(
                    val + (0.008 if val >= 0 else -0.008),
                    y,
                    f"{val:+.2f}",
                    va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=8,
                    color="#222222",
                    zorder=3,
                )

            ax.axvline(0, color="0.45", lw=0.8)
            ax.set_yticks(y_pos)
            tick_labels = ax.set_yticklabels(sub["plot_label"], fontsize=5.7)
            for tick_label, (_, row) in zip(tick_labels, sub.iterrows()):
                if str(row["evaluator"]).startswith("Our Music-"):
                    tick_label.set_fontweight("bold")
                    tick_label.set_color(family_colors.get(row["family"], "#222222"))
            ax.set_title(RATING_LABELS[human_col], fontsize=11)
            ax.set_xlabel(r"Spearman $\rho$ vs. humans")
            ax.grid(alpha=0.22, axis="x")

        all_bounds = pd.concat([
            custom_align["spearman_rho"],
            custom_align.get("ci95_low", pd.Series(dtype=float)),
            custom_align.get("ci95_high", pd.Series(dtype=float)),
        ]).dropna()
        if not all_bounds.empty:
            xmin = min(-0.15, float(all_bounds.min()) - 0.03)
            xmax = max(0.25, float(all_bounds.max()) + 0.04)
            for ax in axes:
                ax.set_xlim(xmin, xmax)

        legend_handles = [
            matplotlib.patches.Patch(
                color=family_colors["custom_aesthetics"],
                label="External custom aesthetics models",
            ),
            matplotlib.patches.Patch(
                color=family_colors["our_aesthetics"],
                label="Our music-aesthetics model",
            ),
            matplotlib.patches.Patch(
                color=family_colors["our_popularity"],
                label="Our music-popularity proxies",
            ),
            matplotlib.patches.Patch(
                color=family_colors["llm_non_thinking"],
                label="LLM judges: non-thinking",
            ),
            matplotlib.patches.Patch(
                color=family_colors["llm_thinking"],
                label="LLM judges: thinking",
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )
        fig.suptitle(
            "Custom audio models vs. human song-level ratings",
            fontsize=14,
            y=1.02,
        )
        legend_text = [
            "Figure 18 legend and mapping rationale",
            "======================================",
            "",
            "Metric:",
            "  Each bar is a song-level Spearman correlation between one model score and the human song-level mean for the panel's questionnaire dimension.",
            "  Bars are sorted within each panel so the highest correlation with humans appears at the top.",
            "  Translucent bands show approximate 95% confidence intervals for Spearman rho, using Fisher's z transform.",
            "",
            "Color legend:",
            "  Purple: external custom audio aesthetics models (Audiobox-Aesthetics and SongEval).",
            "  Cyan: our music-aesthetics model.",
            "  Green: our music-popularity model, shown as preference proxies rather than direct aesthetics scores.",
            "  Orange: LLM judge configuration without extended thinking.",
            "  Red: LLM judge configuration with extended thinking.",
            "  Bold y-axis labels mark our models.",
            "",
            "Audiobox-Aesthetics axes:",
            "  CE: Content Enjoyment",
            "  CU: Content Usefulness",
            "  PC: Production Complexity",
            "  PQ: Production Quality",
            "",
            "Why mappings are needed:",
            "  The systems do not share a common output schema. Humans rated five questionnaire dimensions on a 1-10 scale; Audiobox emits four 1-10 audio-quality axes; SongEval and our music-aesthetics model emit five music-aesthetics axes; the popularity model emits log1p play/upvote counts; the LLM judges emit the same five questionnaire dimensions as the human study.",
            "  Therefore Figure 18 uses nearest-concept mappings, and the CSV preserves the exact score_label used for every bar.",
            "",
            "Mappings used:",
            "  Human aesthetic_quality <- Audiobox CE (Content Enjoyment); SongEval Musicality; Our Music-Aesthetics Overall_Aesthetics; Our Music-Popularity log1p play/upvote counts; LLM aesthetic_quality.",
            "  Human production_quality <- Audiobox PQ (Production Quality); SongEval Coherence; Our Music-Aesthetics Coherence; Our Music-Popularity log1p play/upvote counts; LLM production_quality.",
            "  Human playlist_likelihood <- Audiobox CU (Content Usefulness); SongEval Memorability; Our Music-Aesthetics Memorability; Our Music-Popularity log1p play/upvote counts; LLM playlist_likelihood.",
            "  Human musical_creativity <- Audiobox PC (Production Complexity); SongEval Clarity; Our Music-Aesthetics Clarity; Our Music-Popularity log1p play/upvote counts; LLM musical_creativity.",
            "  Human emotional_engagement <- SongEval Naturalness; Our Music-Aesthetics Naturalness; Our Music-Popularity log1p play/upvote counts; LLM emotional_engagement. Audiobox has no corresponding axis for this panel.",
            "",
            "LLM handling:",
            "  LLM judge outputs are compared dimension-by-dimension against the matching human questionnaire dimension. No cross-dimension mapping is applied for LLMs.",
            "",
            "Popularity handling:",
            "  Our Music-Popularity is not an aesthetics model. It is included to test whether audio-only predicted popularity behaves like human preference/quality ratings. Its play/upvote bars should be interpreted as preference-proxy correlations, not as direct quality-score accuracy.",
        ]
        (OUTPUT_DIR / "fig18_aesthetics_llm_vs_custom_models.txt").write_text(
            "\n".join(legend_text) + "\n",
            encoding="utf-8",
        )
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(
            FIG_DIR / "fig18_aesthetics_llm_vs_custom_models.pdf",
            bbox_inches="tight",
            dpi=300,
        )
        fig.savefig(
            FIG_DIR / "fig18_aesthetics_llm_vs_custom_models.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)

    # -- Fig 19: SongEval vs humans -----------------------------------------
    se_df = _load_songeval_scores()
    if not se_df.empty:
        human_song = ann.groupby("song_id")[RATING_COLS].mean()
        se_song = se_df.set_index("song_id")
        joined_full = se_song.join(human_song, how="inner")

        fig, axes = plt.subplots(1, len(SONGEVAL_AXES), figsize=(25, 5), sharey=False)
        for i, ax_name in enumerate(SONGEVAL_AXES):
            ax = axes[i]
            human_col = SONGEVAL_TO_HUMAN[ax_name]
            sub = joined_full[[ax_name, human_col, "song_source"]].dropna()
            if sub.empty:
                ax.set_visible(False)
                continue
            for src in ALL_SOURCES:
                ss = sub[sub["song_source"] == src]
                if ss.empty:
                    continue
                ax.scatter(
                    ss[human_col], ss[ax_name],
                    s=18, alpha=0.65,
                    color=PALETTE_SOURCE.get(src, "#888888"),
                    edgecolor="white", linewidth=0.4,
                    label=src,
                )
            ax.plot([1, 10], [1, 10], "k--", lw=0.8, alpha=0.5, label="y = x")
            xs = sub[human_col].to_numpy()
            ys = sub[ax_name].to_numpy()
            if len(xs) >= 3 and np.std(xs) > 0:
                slope, intercept = np.polyfit(xs, ys, 1)
                xs_line = np.array([xs.min(), xs.max()])
                ax.plot(xs_line, slope * xs_line + intercept,
                        color="#264653", lw=1.2, label="OLS")
            rho, _ = stats.spearmanr(sub[human_col], sub[ax_name])
            r, _ = stats.pearsonr(sub[human_col], sub[ax_name])
            mae = float(np.mean(np.abs(sub[human_col] - sub[ax_name])))
            ax.set_title(
                f"{ax_name} ({SONGEVAL_AXIS_LABELS[ax_name]})  vs.\n"
                f"{RATING_LABELS[human_col]}\n"
                rf"$\rho$={rho:.2f}, r={r:.2f}, MAE={mae:.2f}",
                fontsize=9,
            )
            ax.set_xlabel(f"Human mean: {RATING_LABELS[human_col]}")
            ax.set_ylabel(f"SongEval: {ax_name}")
            ax.set_xlim(0.5, 10.5)
            ax.set_ylim(0.5, 10.5)
            ax.grid(alpha=0.25)
            if i == len(SONGEVAL_AXES) - 1:
                ax.legend(loc="lower right", fontsize=7, framealpha=0.85)
        fig.suptitle(
            "SongEval vs. human ratings (song-level means, 1-10)",
            fontsize=13, y=1.02,
        )
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig19_songeval_vs_humans.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(FIG_DIR / "fig19_songeval_vs_humans.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

        # -- Fig 20: SongEval vs Audiobox on matched dimensions ------------
        ab_df = _load_audiobox_scores()
        if not ab_df.empty:
            se_song = se_df.set_index("song_id")
            ab_song = ab_df.set_index("song_id")
            n_pairs = len(SONGEVAL_AUDIOBOX_PAIRS)
            fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 5), sharey=False)
            if n_pairs == 1:
                axes = [axes]
            for i, (se_axis, ab_axis, human_col) in enumerate(SONGEVAL_AUDIOBOX_PAIRS):
                ax = axes[i]
                sub = (
                    se_song[[se_axis, "song_source"]]
                    .join(ab_song[[ab_axis]], how="inner")
                    .dropna()
                )
                if sub.empty:
                    ax.set_visible(False)
                    continue
                for src in ALL_SOURCES:
                    ss = sub[sub["song_source"] == src]
                    if ss.empty:
                        continue
                    ax.scatter(
                        ss[ab_axis], ss[se_axis],
                        s=18, alpha=0.65,
                        color=PALETTE_SOURCE.get(src, "#888888"),
                        edgecolor="white", linewidth=0.4,
                        label=src,
                    )
                ax.plot([1, 10], [1, 10], "k--", lw=0.8, alpha=0.5, label="y = x")
                xs = sub[ab_axis].to_numpy()
                ys = sub[se_axis].to_numpy()
                if len(xs) >= 3 and np.std(xs) > 0:
                    slope, intercept = np.polyfit(xs, ys, 1)
                    xs_line = np.array([xs.min(), xs.max()])
                    ax.plot(xs_line, slope * xs_line + intercept,
                            color="#264653", lw=1.2, label="OLS")
                rho, _ = stats.spearmanr(sub[ab_axis], sub[se_axis])
                r, _ = stats.pearsonr(sub[ab_axis], sub[se_axis])
                mae = float(np.mean(np.abs(sub[ab_axis] - sub[se_axis])))
                ax.set_title(
                    f"{RATING_LABELS[human_col]}\n"
                    f"SongEval {se_axis} vs. Audiobox {ab_axis}\n"
                    rf"$\rho$={rho:.2f}, r={r:.2f}, MAE={mae:.2f}",
                    fontsize=10,
                )
                ax.set_xlabel(f"Audiobox: {ab_axis}")
                ax.set_ylabel(f"SongEval: {se_axis}")
                ax.set_xlim(0.5, 10.5)
                ax.set_ylim(0.5, 10.5)
                ax.grid(alpha=0.25)
                if i == n_pairs - 1:
                    ax.legend(loc="lower right", fontsize=7, framealpha=0.85)
            fig.suptitle(
                "SongEval vs. Audiobox-Aesthetics on matched perceptual dimensions",
                fontsize=13, y=1.02,
            )
            fig.tight_layout()
            fig.savefig(FIG_DIR / "fig20_songeval_vs_audiobox.pdf", bbox_inches="tight", dpi=300)
            fig.savefig(FIG_DIR / "fig20_songeval_vs_audiobox.png", bbox_inches="tight", dpi=300)
            plt.close(fig)

    # -- Fig 21: feature-based detector experiments ------------------------
    feature_baselines = _load_feature_baseline_comparison()
    feature_invariants = _load_feature_invariants()
    feature_augmentations = _load_feature_augmentations()
    if not feature_baselines.empty or not feature_invariants.empty or not feature_augmentations.empty:
        family_colors = {
            "proposed": "#00A6A6",
            "foundation": "#2A9D8F",
            "cnn": "#6A4C93",
            "vision": "#F4A261",
            "sonics": "#E76F51",
        }
        fig, axes = plt.subplots(3, 1, figsize=(12.5, 15.5))

        ax = axes[0]
        if not feature_baselines.empty:
            sub = feature_baselines.sort_values("auc", ascending=True).copy()
            y_pos = np.arange(len(sub))
            colors = [family_colors.get(f, "#999999") for f in sub["family"]]
            for i, (_, row) in enumerate(sub.iterrows()):
                if pd.notna(row["auc_ci_low"]) and pd.notna(row["auc_ci_high"]):
                    ci_low = float(row["auc_ci_low"])
                    ci_high = float(row["auc_ci_high"])
                    ax.barh(
                        y_pos[i],
                        ci_high - ci_low,
                        left=ci_low,
                        color=family_colors.get(row["family"], "#999999"),
                        edgecolor="none",
                        height=0.86,
                        alpha=0.28,
                        zorder=1,
                    )
            ax.barh(
                y_pos,
                sub["auc"],
                color=colors,
                edgecolor="white",
                linewidth=0.7,
                height=0.58,
                zorder=2,
            )
            for i, (_, row) in enumerate(sub.iterrows()):
                ax.text(
                    min(float(row["auc"]) + 0.006, 1.01),
                    y_pos[i],
                    f"{float(row['auc']):.3f}",
                    va="center",
                    ha="left",
                    fontsize=8.5,
                    zorder=3,
                )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sub["model"], fontsize=9)
            for label, (_, row) in zip(ax.get_yticklabels(), sub.iterrows()):
                if row["family"] == "proposed":
                    label.set_fontweight("bold")
                    label.set_color(family_colors["proposed"])
            ax.set_xlim(0.60, 1.02)
            ax.set_xlabel("OOD AUC")
            ax.set_title("A. Detector head-to-head on out-of-distribution generators", loc="left")
            ax.grid(axis="x", alpha=0.22)
            legend_handles = [
                matplotlib.patches.Patch(color=family_colors[k], label=v)
                for k, v in [
                    ("proposed", "Proposed multiview"),
                    ("foundation", "Frozen foundation head"),
                    ("cnn", "CNN baseline"),
                    ("vision", "Vision backbone"),
                    ("sonics", "SONICS SpecTTTra"),
                ]
            ]
            ax.legend(handles=legend_handles, loc="lower right", fontsize=8, frameon=True, ncol=2)
        else:
            ax.set_visible(False)

        ax = axes[1]
        if not feature_invariants.empty:
            sub = feature_invariants.sort_values("ood_auc", ascending=True).copy()
            y_pos = np.arange(len(sub))
            colors = ["#00A6A6" if p == "bicoherence" else "#6A4C93" for p in sub["probe"]]
            ax.barh(y_pos, sub["ood_auc"], color=colors, edgecolor="white", linewidth=0.7, height=0.68)
            for i, (_, row) in enumerate(sub.iterrows()):
                ax.text(
                    float(row["ood_auc"]) + 0.008,
                    y_pos[i],
                    f"AUC {row['ood_auc']:.3f} | TPR {row['ood_tpr_at_1pct_fpr']:.3f}",
                    va="center",
                    ha="left",
                    fontsize=8.5,
                )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sub["probe"], fontsize=9)
            for label, probe in zip(ax.get_yticklabels(), sub["probe"]):
                if probe == "bicoherence":
                    label.set_fontweight("bold")
                    label.set_color("#00A6A6")
            ax.set_xlim(0.35, 1.04)
            ax.set_xlabel("OOD AUC")
            ax.set_title("B. Standalone invariant probes", loc="left")
            ax.grid(axis="x", alpha=0.22)
        else:
            ax.set_visible(False)

        ax = axes[2]
        if not feature_augmentations.empty:
            sub = feature_augmentations.copy()
            sub["destroys"] = sub["verdict"].str.contains("destroys", case=False, na=False)
            sub = sub.sort_values(["destroys", "ood_auc"], ascending=[True, True])
            y_pos = np.arange(len(sub))
            colors = np.where(sub["destroys"], "#E76F51", "#2A9D8F")
            ax.barh(y_pos, sub["ood_auc"], color=colors, edgecolor="white", linewidth=0.7, height=0.68)
            baseline = sub.loc[sub["regime"] == "none", "ood_auc"]
            if not baseline.empty:
                ax.axvline(
                    float(baseline.iloc[0]),
                    color="#222222",
                    linestyle="--",
                    linewidth=1.0,
                    label="no-aug baseline",
                )
            for i, (_, row) in enumerate(sub.iterrows()):
                ax.text(
                    float(row["ood_auc"]) + 0.0025,
                    y_pos[i],
                    f"{row['ood_auc']:.3f} | TPR {row['ood_tpr_at_1pct_fpr']:.3f}",
                    va="center",
                    ha="left",
                    fontsize=8.5,
                )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sub["regime"], fontsize=9)
            ax.set_xlim(0.92, 1.01)
            ax.set_xlabel("OOD AUC")
            ax.set_title("C. LCNN augmentation ablation", loc="left")
            ax.grid(axis="x", alpha=0.22)
            aug_handles = [
                matplotlib.patches.Patch(color="#2A9D8F", label="preserves signal"),
                matplotlib.patches.Patch(color="#E76F51", label="destroys signal"),
                matplotlib.lines.Line2D([0], [0], color="#222222", linestyle="--", label="no-aug baseline"),
            ]
            ax.legend(handles=aug_handles, loc="lower right", fontsize=8, frameon=True, ncol=3)
        else:
            ax.set_visible(False)

        fig.suptitle(
            "Feature-based AI-music detection experiments",
            fontsize=14,
            y=0.995,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98], h_pad=2.0)
        fig.savefig(FIG_DIR / "fig21_feature_analysis.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(FIG_DIR / "fig21_feature_analysis.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

    # -- Fig 22: mood tags by provenance and perceived authenticity --------
    _, mood_stats, mood_summary, mood_meta = _compute_mood_tag_analysis(ann)
    if not mood_stats.empty:
        gt_label = "Ground truth: AI songs - real songs"
        perc_label = "Perceived: judged AI - judged real"
        order_df = (
            mood_stats.assign(abs_diff=mood_stats["diff"].abs())
            .groupby("mood_tag", as_index=False)["abs_diff"].max()
            .sort_values("abs_diff", ascending=True)
        )
        mood_order = order_df["mood_tag"].tolist()
        y_pos = np.arange(len(mood_order))
        contrast_specs = [
            (gt_label, "A. Ground truth provenance", "#457B9D", "AI songs - real songs"),
            (perc_label, "B. Perceived authenticity", "#E76F51", "judged AI - judged real"),
        ]
        max_ci = float(
            np.nanmax(np.abs(mood_stats[["diff_ci_low", "diff_ci_high"]].to_numpy()))
        )
        x_lim = max(0.18, min(0.42, np.ceil((max_ci + 0.025) * 20) / 20))

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.1), sharey=True)
        for ax, (contrast, title, color, xlabel) in zip(axes, contrast_specs):
            sub = mood_stats[mood_stats["contrast"] == contrast].set_index("mood_tag").loc[mood_order]
            for i, (_, row) in enumerate(sub.iterrows()):
                ci_low = float(row["diff_ci_low"])
                ci_high = float(row["diff_ci_high"])
                ax.barh(
                    y_pos[i],
                    ci_high - ci_low,
                    left=ci_low,
                    color=color,
                    alpha=0.23,
                    height=0.82,
                    edgecolor="none",
                    zorder=1,
                )
            bars = ax.barh(
                y_pos,
                sub["diff"],
                color=color,
                alpha=0.92,
                height=0.54,
                edgecolor="white",
                linewidth=0.8,
                zorder=2,
            )
            ax.axvline(0, color="#222222", linewidth=1.0, zorder=3)
            for bar, (_, row) in zip(bars, sub.iterrows()):
                diff = float(row["diff"])
                q = float(row["q_fdr"])
                star = "*" if q < 0.05 else ""
                xpos = diff + (0.008 if diff >= 0 else -0.008)
                ax.text(
                    xpos,
                    bar.get_y() + bar.get_height() / 2,
                    f"{100 * diff:+.1f}{star}",
                    va="center",
                    ha="left" if diff >= 0 else "right",
                    fontsize=8.5,
                    color="#222222",
                    zorder=4,
                )
            ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
            ax.set_xlabel(f"Prevalence difference, {xlabel} (percentage points)")
            ax.set_xlim(-x_lim, x_lim)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{100 * x:.0f}"))
            ax.grid(axis="x", alpha=0.24)
            ax.text(
                0.01,
                0.02,
                "Shaded bars: approx. 95% CI\n* FDR q<0.05 within panel",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="#555555",
            )

        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(mood_order, fontsize=10)
        ai_row = mood_summary[mood_summary["group"] == "AI songs"].iloc[0]
        real_row = mood_summary[mood_summary["group"] == "real songs"].iloc[0]
        judged_ai_row = mood_summary[mood_summary["group"] == "judged AI"].iloc[0]
        judged_real_row = mood_summary[mood_summary["group"] == "judged real"].iloc[0]
        fig.suptitle(
            "Human mood-tag shifts: actual provenance vs perceived authenticity",
            fontsize=14,
            y=0.985,
        )
        fig.text(
            0.5,
            0.925,
            (
                f"Canonical tags cover {100 * mood_meta['canonical_tag_share']:.1f}% of selections. "
                f"Mean tags/trial: AI songs {ai_row['mean_mood_tags']:.2f}, real songs {real_row['mean_mood_tags']:.2f}; "
                f"judged AI {judged_ai_row['mean_mood_tags']:.2f}, judged real {judged_real_row['mean_mood_tags']:.2f}."
            ),
            ha="center",
            va="center",
            fontsize=9.5,
            color="#444444",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.90], w_pad=3.0)
        fig.savefig(FIG_DIR / "fig22_mood_tag_shifts.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(FIG_DIR / "fig22_mood_tag_shifts.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

    # -- Fig 23: model mood tags vs human mood-tag profile -----------------
    model_mood_comp, model_mood_summary, _, model_mood_meta = _compute_model_human_mood_tag_analysis(ann)
    if not model_mood_comp.empty and not model_mood_summary.empty:
        order = model_mood_summary.sort_values(
            ["mean_song_dice", "median_abs_diff_pp"],
            ascending=[False, True],
        )["model"].tolist()
        diff_mat = (
            model_mood_comp
            .assign(diff_pp=100.0 * model_mood_comp["diff_model_minus_human"])
            .pivot(index="model", columns="mood_tag", values="diff_pp")
            .loc[order, MOOD_TAGS_CANONICAL]
        )
        q_mat = (
            model_mood_comp
            .pivot(index="model", columns="mood_tag", values="q_fdr_global")
            .loc[order, MOOD_TAGS_CANONICAL]
        )
        max_abs = float(np.nanmax(np.abs(diff_mat.to_numpy())))
        vmax = max(20.0, min(75.0, np.ceil((max_abs + 2.0) / 5.0) * 5.0))

        fig = plt.figure(figsize=(17.4, 8.4))
        gs = fig.add_gridspec(1, 3, width_ratios=[4.65, 1.18, 1.38], wspace=0.10)
        ax_hm = fig.add_subplot(gs[0, 0])
        ax_overlap = fig.add_subplot(gs[0, 1])
        ax_dist = fig.add_subplot(gs[0, 2])

        annot = diff_mat.map(lambda x: "" if pd.isna(x) else f"{x:+.0f}")
        sns.heatmap(
            diff_mat,
            ax=ax_hm,
            cmap="RdBu_r",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            linewidths=0.7,
            linecolor="white",
            annot=annot,
            fmt="",
            annot_kws={"fontsize": 7.5},
            cbar_kws={"label": "Model - human prevalence (pp)", "shrink": 0.68},
        )
        cbar = ax_hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Model - human prevalence (pp)", fontsize=9)
        ax_hm.set_xlabel("")
        ax_hm.set_ylabel("")
        ax_hm.set_title(
            "A. Canonical mood-tag prevalence deviation from humans",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        ax_hm.set_xticklabels(ax_hm.get_xticklabels(), rotation=35, ha="right", fontsize=9)
        ax_hm.set_yticklabels(ax_hm.get_yticklabels(), rotation=0, fontsize=8.6)

        summary_idx = model_mood_summary.set_index("model").loc[order]
        thinking_colors = {"non-thinking": "#F4A261", "thinking": "#E76F51"}
        for tick in ax_hm.get_yticklabels():
            model_name = tick.get_text()
            thinking = summary_idx.loc[model_name, "thinking_alias"]
            tick.set_color(thinking_colors.get(thinking, "#555555"))
            if model_name == order[0]:
                tick.set_fontweight("bold")

        sig_count = 0
        for i, model_name in enumerate(order):
            for j, tag in enumerate(MOOD_TAGS_CANONICAL):
                q = q_mat.loc[model_name, tag]
                if pd.notna(q) and q < 0.05:
                    sig_count += 1
                    ax_hm.text(
                        j + 0.87,
                        i + 0.22,
                        "*",
                        ha="center",
                        va="center",
                        fontsize=11,
                        fontweight="bold",
                        color="#111111",
                    )

        y = np.arange(len(order)) + 0.5
        colors = [thinking_colors.get(summary_idx.loc[m, "thinking_alias"], "#888888") for m in order]
        ax_overlap.barh(
            y,
            100.0 * summary_idx["mean_song_dice"],
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            height=0.70,
        )
        for yi, (_, row) in zip(y, summary_idx.iterrows()):
            ax_overlap.text(
                100.0 * float(row["mean_song_dice"]) + 0.9,
                yi,
                f"{100.0 * row['mean_song_dice']:.1f}",
                va="center",
                ha="left",
                fontsize=8.5,
            )
        ax_overlap.set_ylim(len(order), 0)
        ax_overlap.set_yticks([])
        ax_overlap.set_xlabel("Mean per-song Dice (%)")
        ax_overlap.set_title("B. Per-song\nmood overlap", loc="left", fontsize=11, fontweight="bold")
        ax_overlap.grid(axis="x", alpha=0.24)
        ax_overlap.set_xlim(0, max(8.0, float(100.0 * summary_idx["mean_song_dice"].max()) * 1.22))

        ax_dist.barh(
            y,
            summary_idx["median_abs_diff_pp"],
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            height=0.70,
            alpha=0.92,
        )
        for yi, (_, row) in zip(y, summary_idx.iterrows()):
            mean_abs = float(row["mean_abs_diff_pp"])
            median_abs = float(row["median_abs_diff_pp"])
            ax_dist.plot(
                mean_abs,
                yi,
                marker="D",
                markersize=4.2,
                color="#222222",
                zorder=4,
            )
            ax_dist.text(
                median_abs + 0.5,
                yi,
                f"{median_abs:.1f}",
                va="center",
                ha="left",
                fontsize=8.5,
            )
        ax_dist.set_ylim(len(order), 0)
        ax_dist.set_yticks([])
        ax_dist.set_xlabel("Median |Δ| pp")
        ax_dist.set_title("C. Prevalence\nbias distance", loc="left", fontsize=11, fontweight="bold")
        ax_dist.grid(axis="x", alpha=0.24)
        ax_dist.set_xlim(0, max(5.0, float(summary_idx["mean_abs_diff_pp"].max()) * 1.22))
        legend_handles = [
            matplotlib.patches.Patch(color="#F4A261", label="non-thinking"),
            matplotlib.patches.Patch(color="#E76F51", label="thinking"),
            matplotlib.lines.Line2D(
                [0], [0], marker="D", color="#222222", linestyle="",
                markersize=4.2, label="mean |Δ| pp",
            ),
        ]
        ax_dist.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.02),
            fontsize=8,
            frameon=True,
        )

        best = summary_idx.sort_values("mean_song_dice", ascending=False).iloc[0]
        worst = summary_idx.sort_values("mean_song_dice", ascending=True).iloc[0]
        closest_robust = summary_idx.sort_values("median_abs_diff_pp", ascending=True).iloc[0]
        sig_note = (
            f"{sig_count} model-tag cells survive global FDR q<0.05."
            if sig_count
            else "No model-tag cell survives global FDR q<0.05."
        )
        fig.suptitle(
            "Model mood-tag profiles compared with human annotators",
            fontsize=14,
            y=0.99,
        )
        fig.text(
            0.5,
            0.935,
            (
                f"Human reference: n={model_mood_meta['human_n']}, "
                f"{model_mood_meta['human_mean_tags_per_trial']:.2f} tags/trial. "
                f"Rows sorted by per-song Dice: best {best.name} ({100.0 * best['mean_song_dice']:.1f}%), "
                f"weakest {worst.name} ({100.0 * worst['mean_song_dice']:.1f}%). "
                f"Closest robust prevalence profile: {closest_robust.name} "
                f"(median |Δ|={closest_robust['median_abs_diff_pp']:.1f} pp). "
                f"{sig_note}"
            ),
            ha="center",
            va="center",
            fontsize=9.2,
            color="#444444",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        fig.savefig(FIG_DIR / "fig23_model_human_mood_tags.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(FIG_DIR / "fig23_model_human_mood_tags.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

    print(f"  All figures saved to {FIG_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    ann, par = load_data()
    print(f"  Loaded {len(ann)} annotations, {len(par)} participants\n")

    report = []
    report.append("=" * 80)
    report.append("NEURIPS ANALYSIS: PERCEPTUAL EVALUATION OF AI-GENERATED MUSIC")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    n_ai = int(ann["is_ai_song"].sum())
    n_real = int((~ann["is_ai_song"]).sum())
    n_orig = int((ann["snippet_condition"] == "original").sum())
    n_30s = int((ann["snippet_condition"] == "30s").sum())
    report.append(
        f"Annotations: {len(ann)} (AI: {n_ai}, real: {n_real} | "
        f"full-snippet: {n_orig}, 30s-snippet: {n_30s}) | "
        f"Participants: {par['participant_id'].nunique()}"
    )
    report.append("=" * 80)

    print("1. Computing descriptive statistics...")
    descriptive_statistics(ann, par, report)

    print("2. Running signal detection / AI detection analysis...")
    sdt_analysis(ann, report)

    print("3. Analyzing ratings across sources...")
    rating_analysis(ann, report)

    print("4. Fitting mixed-effects models...")
    mixed_effects_models(ann, report)

    print("5. Fitting logistic regression for detection predictors...")
    detection_model(ann, report)

    print("6. Computing inter-rater reliability...")
    inter_rater_reliability(ann, report)

    print("7. Running PCA on rating dimensions...")
    pca_model, scaler = pca_analysis(ann, report)

    print("8. Analyzing annotation durations...")
    duration_analysis(ann, report)

    print("9. Analyzing AI aspects...")
    ai_aspects_analysis(ann, report)

    print("10. Computing effect sizes...")
    effect_size_summary(ann, report)

    print("11. Computing per-song detection rates...")
    per_song_analysis(ann, report)

    print("12. Checking trial order / fatigue effects...")
    trial_order_analysis(ann, report)

    print("13. Comparing snippet conditions (full-length v1 vs 30s v2)...")
    snippet_condition_analysis(ann, report)

    print("14. Benchmarking model judgments vs humans...")
    model_vs_human_analysis(ann, report)

    print("15. Computing per-evaluator halo effect on aesthetic ratings...")
    halo_effect_by_evaluator(ann, report)

    print("16. Computing ground-truth aesthetic ratings by evaluator...")
    ground_truth_rating_by_evaluator(ann, report)

    print("17. Fitting crossed mixed-effects model for aesthetic predictors...")
    aesthetic_predictors_model(ann, report)

    print("17b. Fitting logistic detection-predictors model (parallel to 17)...")
    detection_predictors_model(ann, report)

    print("18. Personalized perception (age, taste, device, environment)...")
    personalization_analysis(ann, report)

    print("19. Comparing audiobox-aesthetics scores to human + LLM judgments...")
    audiobox_aesthetics_analysis(ann, report)

    print("20. Comparing SongEval scores to human + LLM judgments...")
    songeval_aesthetics_analysis(ann, report)

    print("21. Comparing music-popularity predictions to human + audio aesthetics signals...")
    popularity_model_analysis(ann, report)

    print("22. Integrating feature-based detector analysis...")
    feature_detection_analysis(report)

    print("23. Comparing human mood tags by provenance and perceived authenticity...")
    mood_tag_analysis(ann, report)

    print("24. Comparing model mood tags against humans...")
    model_human_mood_tag_analysis(ann, report)

    report_text = "\n".join(report)
    report_path = OUTPUT_DIR / "analysis_report.txt"
    report_path.write_text(report_text)
    print(f"\nFull report saved to {report_path}")
    print("\n" + report_text)

    print("\nGenerating figures...")
    create_figures(ann, pca_model, scaler)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
