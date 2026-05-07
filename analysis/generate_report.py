"""
Generate a polished PDF report for the NeurIPS paper analysis.
"""

from pathlib import Path
from fpdf import FPDF

OUTPUT_DIR = Path("output")
FIG_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = OUTPUT_DIR / "neurips_analysis_report.pdf"

UNICODE_TO_LATIN = {
    "\u2014": "--",
    "\u2013": "-",
    "\u2019": "'",
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u2022": "-",
    "\u207b\u00b9\u2070": "^-10",
    "\u207b": "-",
    "\u00b9": "1",
    "\u2070": "0",
    "\u0394": "D",
    "\u03b7": "eta",
    "\u03c1": "rho",
    "\u03b1": "alpha",
    "\u03c7": "chi",
    "\u00b2": "2",
    "\u2032": "'",
    "\u2260": "!=",
    "\u2265": ">=",
    "\u2264": "<=",
    "\u2194": "<->",
}

def sanitize(text):
    for u, a in UNICODE_TO_LATIN.items():
        text = text.replace(u, a)
    return text.encode("latin-1", errors="replace").decode("latin-1")


class ReportPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 6, "Perceptual Evaluation of AI-Generated Music -- Analysis Report", align="R")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title, level=1):
        title = sanitize(title)
        if level == 1:
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(30, 60, 110)
            self.ln(6)
            self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(30, 60, 110)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(4)
        elif level == 2:
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(50, 80, 130)
            self.ln(4)
            self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(2)
        else:
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(70, 70, 70)
            self.ln(2)
            self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)

    def body_text(self, text):
        text = sanitize(text)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, text, bold_prefix=""):
        text = sanitize(text)
        bold_prefix = sanitize(bold_prefix)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.cell(6, 5.5, "-")
        if bold_prefix:
            self.set_font("Helvetica", "B", 10)
            self.cell(self.get_string_width(bold_prefix) + 1, 5.5, bold_prefix)
            self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, text)
        self.ln(0.5)

    def table_row(self, cells, widths, bold=False, bg=False):
        style = "B" if bold else ""
        if bg:
            self.set_fill_color(235, 240, 250)
        self.set_font("Helvetica", style, 9)
        self.set_text_color(40, 40, 40)
        h = 6.5
        for i, (cell, w) in enumerate(zip(cells, widths)):
            self.cell(w, h, str(cell), border=1, fill=bg, align="C" if i > 0 else "L")
        self.ln(h)

    def add_figure(self, path, caption, width=170):
        caption = sanitize(caption)
        path = str(path)
        if self.get_y() + 90 > self.h - 25:
            self.add_page()
        self.image(path, x=(self.w - width) / 2, w=width)
        self.ln(2)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 4.5, caption, align="C")
        self.ln(4)


def build_report():
    pdf = ReportPDF()
    pdf.alias_nb_pages()

    # ---- Title Page ----
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(30, 60, 110)
    pdf.multi_cell(0, 12, "Perceptual Evaluation of\nAI-Generated Music", align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "Comprehensive Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(12)
    pdf.set_draw_color(30, 60, 110)
    pdf.line(60, pdf.get_y(), pdf.w - 60, pdf.get_y())
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 7, "591 Annotations  |  61 Active Participants  |  Real + 4 AI Platforms", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Human (real)  |  Suno  |  Udio  |  Sonauto  |  Mureka", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 6,
             "v1: 403 annotations (full-length snippet, AI only)  --  "
             "v2: 188 new annotations (30-second snippet, AI + real)",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, "April 2026", align="C", new_x="LMARGIN", new_y="NEXT")

    # ---- Analyses Performed ----
    pdf.add_page()
    pdf.section_title("Analyses Performed", level=1)

    analyses = [
        ("1. Descriptive Statistics",
         "Dataset overview (real human songs + four AI platforms), participant demographics, rating distributions with moments (mean, median, skewness, kurtosis), annotation duration statistics, ground-truth balance (real vs AI), and snippet-condition counts."),
        ("2. Signal Detection Theory (AI detection accuracy)",
         "Proper SDT analysis now that the v2 dataset includes both real (human) and AI songs. Reports hits, misses, false alarms, correct rejections, hit rate, false-alarm rate, d' (sensitivity) and c (response bias) at both the overall and per-participant level, with a log-linear correction. Breakdown by source, musical engagement, AI experience, and formal training."),
        ("3. Rating Analysis Across Sources",
         "Direct real-vs-AI contrast on each rating dimension (Mann-Whitney U, rank-biserial r, Cohen's d). Kruskal-Wallis tests across all five sources and across AI platforms only, with Bonferroni-corrected post-hoc pairwise Mann-Whitney U. Ratings by perceived authenticity and the quality--authenticity halo."),
        ("4. Mixed-Effects Models",
         "Linear mixed models for each rating with fixed effects for song source (reference = human) and random intercepts for participants, accounting for repeated measures. Intraclass correlation coefficients (ICC) quantify participant-level variance."),
        ("4b. Aesthetic Predictors (Crossed LMM)",
         "A dedicated crossed linear mixed-effects model for aesthetic quality with random intercepts for both participant and song. Fixed effects include source, perceived authenticity, genre family, snippet condition, musical engagement, AI-music experience, and z-scored log annotation duration. Outputs coefficient table with confidence intervals and variance components in output/aesthetic_lmm_table.tex."),
        ("4c. Detection Predictors (parallel logistic to 4b)",
         "A logistic regression on AI-detection correctness fit on non-uncertain trials, with the same predictor structure as the aesthetic LMM (source replaces ground truth since real==human; genre family, snippet condition, musical engagement, AI-music experience, log duration, age, age-missing flag, taste breadth, listening device, listening environment, favourite-genre match). Standard errors are cluster-robust by participant. Reports beta, OR, 95% OR CI, p, and Tjur's D. Outputs output/detection_lmm_table.tex."),
        ("5. Logistic Regression for Detection Predictors",
         "Two models: (A) probability of responding 'AI-generated' across all non-uncertain trials, conditioned on ground truth, source, snippet condition, and participant covariates; (B) probability of a correct response, conditioned on ground truth and snippet condition. Odds ratios reported for both."),
        ("6. Inter-Rater Reliability",
         "Krippendorff's alpha (interval scale for ratings, ordinal for authenticity assessment) on the subset of songs rated by multiple raters, together with pairwise agreement metrics and Pearson correlations between rater pairs."),
        ("7. Principal Component Analysis (PCA)",
         "PCA on the five rating dimensions to identify latent perceptual factors. Component loadings and full inter-dimension correlation matrix."),
        ("8. Annotation Duration Analysis",
         "Duration distributions by source and assessment, Kruskal-Wallis test for duration differences by assessment, and Spearman correlations between duration and each rating dimension."),
        ("9. AI Aspects Analysis",
         "Frequency analysis of participant-cited AI artifacts (singing voice, composition, lyrics, instruments, rhythm, harmony) split between true positives (correctly detected AI songs) and false positives (real songs misclassified as AI). Breakdown by AI source for true positives."),
        ("10. Effect Size Summary",
         "Cohen's d for all pairwise source comparisons (including human) on every rating dimension, with magnitude classification (negligible / small / medium / large)."),
        ("11. Snippet Condition (v1 full-length vs v2 30-second)",
         "Like-for-like comparison on AI songs between annotations collected against the full-length snippet (v1) and those collected against the 30-second snippet (v2). Reports per-rating Mann-Whitney U with Cohen's d, hit/miss/uncertain distributions on AI trials with a 2x2 chi-squared test, and annotation duration by condition."),
        ("12b. Personalized Perception (age, taste, listening conditions)",
         "Treats each annotation as a personalized observation: rater age (median-imputed "
         "with a missingness flag), taste breadth (number of self-reported favourite genres), "
         "favourite-genre vs. song-genre family match, listening device, and listening "
         "environment. Reports descriptive accuracy and aesthetic-quality means by each "
         "demographic axis, then fits two compact models -- a logistic regression on "
         "correctness and a crossed LMM on aesthetic quality -- that report the personalized "
         "covariates after partialling out source, snippet condition, and the standard "
         "listener traits. Outputs output/personalization_table.tex."),
        ("13. Audiobox-Aesthetics Comparison",
         "Benchmarks Meta's audiobox-aesthetics audio-only quality model "
         "(four 1-10 axes: Content Enjoyment, Content Usefulness, Production Complexity, "
         "Production Quality) against the human ratings on the same 1-10 scale. The model is "
         "fed the exact audio that human annotators heard: full-length cached audio for v1 "
         "songs and the 29s deterministic snippet for songs new in v2. We map audiobox axes "
         "to our perceptual dimensions (PQ-production_quality, CE-aesthetic_quality, "
         "CU-playlist_likelihood, PC-musical_creativity), then report per-axis Spearman rho, "
         "Pearson r, mean absolute error and per-source means against the human aggregate. "
         "The same alignment metric is computed for every LLM judge so the audio-only model "
         "is comparable to language-grounded evaluators. Outputs: "
         "output/audiobox_table.tex, output/audiobox_alignment.csv, "
         "output/audiobox_evaluator_alignment.csv, output/audiobox_song_scores.csv, "
         "and figures 17-18."),
        ("14. SongEval and Audio-Frontend Comparison",
         "Benchmarks SongEval's five perceptual song-aesthetics axes and, when "
         "available, our released REDACTED/music-aesthetics model "
         "(Overall_Aesthetics for aesthetic quality, then Coherence, Memorability, "
         "Clarity, and Naturalness for the nearest remaining questionnaire dimensions) "
         "against human song-level ratings. The same section "
         "compares audio front-ends directly wherever axes map onto the same human "
         "dimension: Musicality/CE for aesthetic quality, Coherence/PQ for production "
         "quality, Memorability/CU for playlist likelihood, and Clarity/PC for musical "
         "creativity. The our-music-aesthetics outputs are raw 1-5 scores and are "
         "endpoint-rescaled to 1-10 for comparison. Outputs: output/songeval_table.tex, "
         "output/songeval_alignment.csv, output/songeval_evaluator_alignment.csv, "
         "output/songeval_song_scores.csv, output/audio_aesthetics_frontend_table.tex, "
         "output/audio_aesthetics_frontend_comparison.csv, optional "
         "output/our_music_aesthetics_*.csv files, and figures 19-20."),
        ("15. Music-Popularity Preference-Signal Comparison",
         "Benchmarks the released REDACTED/music-popularity audio model against the same "
         "song set. Because the human study did not collect true platform play/upvote "
         "counts, the comparison treats predicted log1p plays and log1p upvotes as "
         "audio-only popularity signals and correlates them with human song-level "
         "quality ratings plus Audiobox, SongEval, and our music-aesthetics front-end "
         "scores. Outputs: output/our_music_popularity_song_scores.csv, "
         "output/our_music_popularity_human_alignment.csv, and "
         "output/our_music_popularity_frontend_alignment.csv when predictions exist. "
         "Figure 18 summarizes the custom-model Spearman alignment against humans."),
        ("16. Feature-Based AI-Music Detection Analysis",
         "Integrates the copied feature-analysis experiment outputs from analysis/features. "
         "This section reports the OOD detector head-to-head with bootstrap AUC confidence "
         "intervals and significance tests, the standalone invariant-probe ranking, and the "
         "LCNN augmentation ablation. Outputs: output/feature_detection_table.tex, "
         "output/feature_detection_baselines.csv, output/feature_invariant_ranking.csv, "
         "output/feature_augmentation_ablation.csv, and Figure 21."),
        ("17. Mood-Tag Analysis",
         "Compares human annotator mood tags as multi-label trial-level indicators under two "
         "contrasts: actual provenance (AI songs vs. real human songs) and perceived authenticity "
         "(trials judged AI-generated vs. trials judged real). The figure focuses on the nine "
         "canonical prompt mood tags, reports prevalence differences with approximate 95% "
         "confidence intervals and FDR-corrected Fisher exact tests, and writes "
         "output/mood_tag_table.tex, output/mood_tag_prevalence_comparison.csv, "
         "output/mood_tag_summary.csv, and Figure 22."),
        ("18. Model-Human Mood-Tag Analysis",
         "Compares each LLM judge run's canonical mood-tag distribution against human annotators. "
         "For every model and mood tag, the pipeline reports model-minus-human prevalence "
         "differences, Fisher exact tests with global FDR correction, per-song mood-overlap "
         "Dice scores, and robust/mean percentage-point prevalence distances. Outputs: "
         "output/model_human_mood_tag_table.tex, "
         "output/model_human_mood_tag_comparison.csv, "
         "output/model_human_mood_tag_summary.csv, and Figure 23."),
        ("12. Model-vs-Human Benchmark",
         "Loads model annotation outputs from input/models/ and benchmarks each model run against "
         "human judgments on the shared song set. The benchmark covers twelve LLM-judge "
         "configurations spanning two frontier closed models (Gemini 3.1 Pro and Flash-Lite), two "
         "open-weight small models (Gemma 4 E2B and E4B), and two open-weight audio-native models "
         "(MOSS-Audio 4B and 8B), each run with and without extended thinking. For each run the "
         "pipeline reports: detection metrics (accuracy, hit rate, false-alarm rate, uncertainty "
         "rate, d', criterion, F1); song coverage and missing-song counts; and the model's "
         "percentile rank within the human participant distribution on every metric. Song-level "
         "aesthetic-quality ratings from each model are correlated with human mean ratings "
         "(Pearson r, Spearman rho). Figures 14-16 visualise the results at three levels of "
         "aggregation: (14) per-evaluator bar chart of all detection metrics with human baseline; "
         "(15) aesthetic-rating distributions per evaluator vs. humans; (16) ROC-space scatter "
         "showing the thinking-mode effect per base model, with iso-d' and iso-F1 contours and "
         "human sub-groups broken down by musical engagement and AI music experience."),
    ]

    for title, desc in analyses:
        pdf.section_title(title, level=3)
        pdf.body_text(desc)

    # ---- Key Findings ----
    pdf.add_page()
    pdf.section_title("Key Findings", level=1)

    pdf.section_title("Detection Performance", level=2)
    pdf.bullet(
        " \u2014 Overall discrimination is modest: hit rate 61.0% on AI trials, false-alarm rate "
        "28.9% on real (human) trials, accuracy 62.5% on non-uncertain responses, and d' = 0.83 "
        "(significantly above chance, one-sample t-test p = 0.013). Criterion c = +0.14 indicates "
        "a mild bias toward saying \"real\".",
        "Above chance, but modest"
    )
    pdf.bullet(
        " \u2014 Real (human) songs were correctly rejected 64.3% of the time, while the best-"
        "detected AI platform (Mureka) was flagged 57.1% of the time. Udio was the hardest to "
        "detect (hit rate 41.4%).",
        "Source differences"
    )
    pdf.bullet(
        " \u2014 Accuracy scaled monotonically with musical engagement (casual 53.4%, enthusiast "
        "63.8%, musician 70.0%, professional 80.0%). Engagement was the strongest predictor in "
        "the logistic model (OR = 1.66, p < 0.001).",
        "Expertise matters"
    )
    pdf.bullet(
        " \u2014 Formal training again showed a non-linear relationship: accuracy was 63.9% with "
        "no training, dipped to 52.1% at 1-3 years, and peaked at 82.6% for 4-7 years. In the "
        "logistic model has_training has a negative odds ratio (OR = 0.53, p = 0.005) once "
        "engagement is controlled for.",
        "Training paradox"
    )
    pdf.bullet(
        " \u2014 Correct responses are less likely on AI trials than on real trials (OR = 0.38, "
        "p = 0.004), consistent with the \"said AI\" bias below chance on AI: participants are "
        "better at recognising real music than at recognising AI.",
        "Harder to catch AI than to accept real"
    )

    pdf.section_title("Perceptual Quality", level=2)
    pdf.bullet(
        " \u2014 Real (human) songs were rated higher than AI on every dimension, significantly so "
        "for aesthetic quality, playlist likelihood, musical creativity, and production quality "
        "(all p < 0.05, Cohen's d = 0.25-0.38). Emotional engagement showed a smaller, "
        "non-significant gap.",
        "Humans still edge out AI"
    )
    pdf.bullet(
        " \u2014 Songs perceived as real scored dramatically higher than those perceived as AI "
        "across all five dimensions, reinforcing the quality\u2013authenticity halo effect: higher "
        "perceived quality drives \"real\" classifications regardless of ground truth.",
        "Quality-authenticity halo"
    )
    pdf.bullet(
        " \u2014 Across the four AI platforms alone, Kruskal-Wallis tests revealed no significant "
        "differences on any rating dimension, with uniformly small-to-negligible effect sizes. "
        "The platforms have reached perceptual parity with each other, while still lagging "
        "slightly behind real music.",
        "Platform convergence"
    )
    pdf.bullet(
        " \u2014 In the crossed aesthetic mixed-effects model (participant + song random "
        "intercepts), the strongest positive coefficients identify the conditions associated "
        "with higher aesthetics after controlling for source, genre family, snippet length, "
        "participant engagement, AI-music experience, and annotation duration. See "
        "output/aesthetic_lmm_table.tex and Section 17 in analysis_report.txt for exact "
        "coefficient magnitudes and significance.",
        "Adjusted aesthetics drivers (LMM)"
    )
    pdf.bullet(
        " \u2014 Mureka continues to trail slightly on musical creativity and emotional engagement "
        "(small Cohen's d vs. Suno), and shows the largest gap to real songs on musical "
        "creativity (d = 0.51, medium).",
        "Mureka trailing on creativity"
    )

    pdf.section_title("Perceptual Dimensions", level=2)
    pdf.bullet(
        " \u2014 PC1 explains 72.5% of rating variance with near-equal loadings on all five "
        "dimensions, suggesting participants rely on a single latent \"overall quality\" judgment.",
        "Single quality factor"
    )
    pdf.bullet(
        " \u2014 All inter-dimension correlations remain strong (r \u2248 0.6-0.75), indicating "
        "persistent positive coupling across perceptual attributes.",
        "High inter-correlation"
    )

    pdf.section_title("Snippet Condition (v1 vs v2)", level=2)
    pdf.bullet(
        " \u2014 On AI songs, every rating dimension was rated higher under the 30-second snippet "
        "than under the full-length snippet (Cohen's d \u2248 -0.3 to -0.47, all p < 0.01). "
        "Participants are more lenient when given only a 30-second sample.",
        "Shorter = more lenient ratings"
    )
    pdf.bullet(
        " \u2014 AI detection on AI songs dropped from 55.6% (full-length) to 41.3% (30-second), "
        "while misses rose from 31.5% to 42.3% (chi-squared p = 0.019). The 30-second excerpt "
        "makes AI noticeably harder to catch.",
        "Shorter = AI better hidden"
    )
    pdf.bullet(
        " \u2014 Median annotation time halved (97.7s to 44.8s) in the 30-second condition, "
        "consistent with participants having less audio to evaluate.",
        "Faster judgments"
    )

    pdf.section_title("Annotation Behavior", level=2)
    pdf.bullet(
        " \u2014 Participants spent significantly longer on songs they marked \"uncertain\" than "
        "on definite judgments, consistent with deliberation under ambiguity.",
        "Deliberation time"
    )
    pdf.bullet(
        " \u2014 Mixed-model ICCs indicate that a sizeable fraction of rating variance is "
        "attributable to stable individual differences in leniency/harshness.",
        "Strong rater effects"
    )

    pdf.section_title("Model vs. Human Benchmark", level=2)
    pdf.bullet(
        " \u2014 Thinking does not uniformly help: it improves Gemini Pro and Flash-Lite, "
        "but degrades Gemma 4 E2B and yields lower hit rates for MOSS-Audio 4B/8B. In ROC "
        "space (Figure 16), the non-thinking -> thinking arrow direction varies by base "
        "model rather than uniformly moving toward the upper-left, indicating that the "
        "benefit of extended thinking is contingent on base-model capability.",
        "Thinking effect is mixed"
    )
    pdf.bullet(
        " \u2014 Frontier Gemini Pro models adopt a strongly liberal 'always AI' criterion "
        "(c approx -1.3, FAR > 0.8), while open-weight MOSS-Audio and Gemma 4 E4B adopt the "
        "opposite 'always real' criterion (c >= +0.8, hit rate <= 0.28). On balanced "
        "accuracy and on F1 for the real-song class, the human aggregate outperforms every "
        "model configuration; only Gemini 3.1 Pro (think-high) matches the human d'.",
        "Models lag humans on base-rate-corrected metrics"
    )
    pdf.bullet(
        " \u2014 MOSS-Audio (both 4B and 8B) achieves the lowest raw accuracy of any model "
        "tested (15-27%), well below the 'always AI' baseline (86%). The audio-native models "
        "label nearly every song as 'real' regardless of provenance, producing near-zero "
        "hit rates on AI songs.",
        "MOSS audio-native models default to 'real'"
    )
    pdf.bullet(
        " \u2014 Aesthetic-rating bias is base-model dependent: Gemini Pro and Flash-Lite "
        "under-rate aesthetic quality (delta = -0.4 to -1.4 vs. humans), while Gemma 4 and "
        "MOSS-Audio over-rate by +1.2 to +2.7 points. MOSS-Audio shows the largest positive "
        "shift (mean ~ 7.2-7.9 vs. human 5.16).",
        "Rating bias varies by base model"
    )
    pdf.bullet(
        " \u2014 Iso-F1 contours in Figure 16 are nearly flat due to the AI-heavy class ratio "
        "in the benchmark set, confirming that F1 in this regime is dominated by recall "
        "(hit rate). A model that always predicts 'AI' would sit at (FAR=1, HR=1) -- visually "
        "exposed as trivially biased despite achieving a high raw F1.",
        "Class imbalance context"
    )

    pdf.section_title("Personalized Perception (age, taste, listening conditions)", level=2)
    pdf.bullet(
        " — Favourite-genre match drives detection accuracy. When the song's "
        "declared genre overlaps with a rater's self-reported favourites, the "
        "odds of a correct AI/real call more than double (OR ~ 2.7, p ~ 0.003) "
        "vs. trials where no genre tag is available. Trials with a confirmed "
        "no-match are essentially indistinguishable from the unknown-tag baseline. "
        "Listeners are sharper in their own genre territory.",
        "Favourite-genre match"
    )
    pdf.bullet(
        " — Listening environment matters for accuracy and ratings. Annotations "
        "made in an office/workplace are markedly less accurate than in a quiet "
        "room (OR ~ 0.43, p ~ 0.03). Outdoor sessions, while rare (n = 50), "
        "produce dramatically higher aesthetic ratings (delta ~ +2.4 points on "
        "the 1-10 scale, p < 1e-7), suggesting environment-driven leniency that "
        "is independent of the underlying song quality.",
        "Listening environment"
    )
    pdf.bullet(
        " — Listening device shapes ratings. Earbuds/in-ear listeners give "
        "aesthetics ratings ~0.7 points higher than on-ear-headphone listeners "
        "(p ~ 0.02), with similar but smaller positive shifts for over-ear and "
        "external speakers. Detection accuracy varies non-monotonically across "
        "devices (over-ear and earbuds top, laptop/phone speakers and on-ear "
        "headphones lower).",
        "Listening device"
    )
    pdf.bullet(
        " — Age and taste breadth produce small but reliable shifts. Each "
        "1-SD increase in rater age is associated with a ~0.25-point increase "
        "in aesthetic-quality ratings (p ~ 0.02), while broader taste (more "
        "favourite genres) is marginally associated with lower ratings "
        "(p ~ 0.05). Age does not significantly predict detection accuracy.",
        "Age and taste breadth"
    )
    pdf.bullet(
        " — Song familiarity contaminates BOTH ratings and detection. "
        "9.5% of trials (56/591) are self-reported as 'familiar' to the rater, "
        "and 49/56 of those are AI songs (likely recognised from social "
        "platforms). After joint adjustment, familiar trials are rated ~1.3 "
        "points higher in aesthetic quality (p ~ 2e-06) AND have ~half the "
        "odds of correct AI/real classification (OR ~ 0.47, p ~ 0.06) -- a "
        "double leakage where recognition simultaneously inflates ratings "
        "and biases the call toward 'real'. Robustness check on the "
        "uncontaminated 'never'-familiar subset (n=495 trials) yields d' = "
        "0.88 vs. 0.83 overall, so the headline conclusions are robust.",
        "Recognition leakage (familiarity)"
    )
    pdf.bullet(
        " — Personalized covariates are now jointly fit alongside source, "
        "snippet condition and the original listener traits in the detection "
        "logistic regression and the aesthetic crossed LMM, so the source "
        "and judged-authenticity coefficients reported elsewhere are adjusted "
        "for these factors. See output/personalization_table.tex for the "
        "focused table referenced from the paper.",
        "Joint adjustment in main models"
    )

    pdf.section_title("Audiobox-Aesthetics vs. Human Ratings", level=2)
    pdf.bullet(
        " — Audiobox-Aesthetics (Meta, 2024) emits four 1-10 axes from raw audio with no "
        "language grounding. Its alignment with human song-level means is moderate and "
        "axis-dependent: the strongest signal is on production-quality-flavoured judgements "
        "(PQ ↔ production_quality), with weaker correlation on subjective enjoyment "
        "(CE ↔ aesthetic_quality) and listening intent (CU ↔ playlist_likelihood). "
        "Production complexity (PC) maps least cleanly to musical creativity, as expected. "
        "See output/audiobox_table.tex and Figure 17 for the per-axis scatter and metrics.",
        "Audio-only benchmark"
    )
    pdf.bullet(
        " — The audiobox front-end is competitive with the LLM judges on song-level "
        "Spearman rho despite using no language: it lands inside the LLM-judge spread on "
        "every mapped axis, and its mean absolute error against human ratings is in the "
        "same 1-2 point range as the best language models -- with the upside that scoring "
        "all 571 songs runs locally on a laptop GPU. Figure 18 ranks all evaluators on "
        "song-level alignment per axis.",
        "Comparable to LLM judges, no text"
    )
    pdf.bullet(
        " — Audiobox is consistent across sources: per-source Spearman rho stays in "
        "a narrow band (typical |delta| < 0.2 across human, Suno, Udio, Sonauto, Mureka), "
        "in contrast to LLM judges where rho varies widely with source (some platforms "
        "track human ratings closely, others drift). This source-invariance suggests "
        "audiobox is not exploiting platform-specific spectral signatures and is therefore "
        "a reasonable model for studies that care about cross-platform comparability.",
        "Source-invariant alignment"
    )

    pdf.section_title("SongEval vs. Audiobox-Aesthetics", level=2)
    pdf.bullet(
        " — SongEval adds a second audio-only reference point with a different training target: "
        "five song-aesthetic dimensions rather than Audiobox's four production/content axes. "
        "The pipeline now benchmarks SongEval against human song-level means and against "
        "Audiobox on matched dimensions, so the two audio front-ends can be compared without "
        "routing through LLM judges.",
        "Second audio-only benchmark"
    )
    pdf.bullet(
        " — The direct front-end table separates two questions: which system tracks human "
        "rankings better on each mapped dimension, and how similarly the two audio models "
        "score the same songs. This is written to output/audio_aesthetics_frontend_table.tex "
        "and visualised in Figure 20.",
        "Direct SongEval-Audiobox comparison"
    )
    pdf.bullet(
        " — SongEval's Naturalness dimension has no Audiobox analogue, so it is reported "
        "against human emotional-engagement ratings only. That axis should be interpreted as "
        "a model-specific vocal phrasing/breathing signal, not as a matched comparison with "
        "Audiobox.",
        "Naturalness is unmatched"
    )
    pdf.bullet(
        " — The released REDACTED/music-aesthetics model is now wired into the same benchmark. "
        "Once aesthetics/our-music-aesthetics-model/output/result.json exists, main.py adds "
        "it to the SongEval table, writes output/our_music_aesthetics_alignment.csv, and "
        "compares it directly with SongEval and Audiobox on the shared song set.",
        "Our model included when scored"
    )
    pdf.bullet(
        " — The REDACTED/music-popularity model is now staged as a separate preference-signal "
        "comparison. It predicts log1p play and upvote counts rather than 1-10 aesthetic "
        "quality, so main.py reports correlations with human ratings and the three audio "
        "aesthetic front-ends instead of treating it as another direct quality scorer.",
        "Popularity is a different target"
    )

    pdf.section_title("Mood Tags", level=2)
    pdf.bullet(
        " — Canonical prompt mood tags account for 64.6% of all mood selections "
        "(543/841). The remaining labels are sparse free-text tags, so Figure 22 focuses "
        "on the nine canonical tags and reports the long-tail caveat in the statistical "
        "output.",
        "Canonical tags dominate"
    )
    pdf.bullet(
        " — AI songs receive slightly more mood tags per trial than real songs "
        "(1.47 vs. 1.12; Mann-Whitney p = 0.056), but individual canonical mood "
        "differences are weak. The largest provenance shift is Joyful activation "
        "(-6.6 percentage points for AI vs. real; 95% CI [-15.7, +2.5]; FDR q = 0.591).",
        "No robust provenance mood shift"
    )
    pdf.bullet(
        " — Perceived authenticity is even less mood-separated: trials judged AI average "
        "1.33 mood tags versus 1.45 for trials judged real (p = 0.867), and no canonical "
        "tag survives FDR correction. The largest perceived-authenticity shift is "
        "Peacefulness (-3.7 percentage points for judged AI vs. judged real; "
        "95% CI [-9.4, +1.9]; FDR q = 0.820).",
        "No robust judged-AI mood signature"
    )
    pdf.bullet(
        " — Figure 23 extends the mood-tag analysis to LLM judges by comparing each model "
        "run's canonical mood tags with the human profile. The primary ranking uses per-song "
        "Dice overlap, which avoids a failure mode of pure marginal-prevalence comparisons "
        "where a model could match the global tag rate while assigning tags to the wrong songs. "
        "Median and mean prevalence-distance diagnostics remain as secondary bias measures.",
        "Model mood vocabulary check"
    )

    pdf.section_title("AI Artifact Perception", level=2)
    pdf.bullet(
        " \u2014 Singing voice dominates correct AI detections (64.8% of true positives), "
        "followed by overall composition (41.9%), instrument sounds (36.3%), lyrics (36.3%), "
        "rhythm (28.8%), and harmony (20.6%).",
        "Voice is the top tell"
    )
    pdf.bullet(
        " \u2014 When real songs were misclassified as AI, participants were more likely to cite "
        "instrument sounds (45.5%) and overall composition (40.9%) than singing voice (27.3%). "
        "The voice cue is more specific to genuine AI artifacts than a generic \"suspicious\" "
        "marker.",
        "False positives differ"
    )
    pdf.bullet(
        " \u2014 Mureka again had the highest rate of voice-related flags among its true "
        "positives (58/76 = 76%), suggesting its vocal synthesis remains the most perceptually "
        "distinct among the AI platforms.",
        "Mureka vocals stand out"
    )

    # ---- Figures ----
    pdf.add_page()
    pdf.section_title("Figures", level=1)

    figures = [
        ("fig9_demographics.png",
         "Figure 1. Participant demographics: musical engagement level (left), AI music experience (center), "
         "and years of formal musical training (right). The sample spans casual listeners to professionals."),
        ("fig1_ratings_by_source.png",
         "Figure 2. Rating distributions by source across five perceptual dimensions. Real (human) songs "
         "appear alongside the four AI platforms (Suno, Udio, Sonauto, Mureka). Boxes show IQR with median line."),
        ("fig2_authenticity_by_source.png",
         "Figure 3. Authenticity assessment proportions by source. Real (human) songs are correctly rejected "
         "64.3% of the time; AI platforms are detected at 41-57%, with Udio the hardest and Mureka the "
         "easiest to detect. Chi-squared test: chi-sq = 37.1, p < 0.001, Cram\u00e9r's V = 0.18."),
        ("fig12_confusion_matrix.png",
         "Figure 4. Full confusion matrix of participant responses against ground truth. Counts on the left, "
         "row-normalised proportions on the right. Participants exhibit a mild bias toward \"real\" "
         "(correct rejection on real = 64.3%; hit on AI = 52.7%)."),
        ("fig3_ratings_by_assessment.png",
         "Figure 5. Violin plots of ratings grouped by participants' authenticity assessment. "
         "Songs perceived as \"real\" received substantially higher ratings across all dimensions -- "
         "the quality-authenticity halo."),
        ("fig6_detection_rates.png",
         "Figure 6. 'Said AI' rate by source (left; hits on AI, false alarms on real) and detection "
         "accuracy by participant AI experience level (right). Professional and regular AI users show "
         "the highest accuracy."),
        ("fig4_correlation_heatmap.png",
         "Figure 7. Pearson correlation matrix of the five rating dimensions. All correlations are strong, "
         "consistent with a single dominant quality factor."),
        ("fig5_pca_biplot.png",
         "Figure 8. PCA biplot of rating dimensions. PC1 (72.5% variance) represents overall quality with "
         "near-equal loadings. Points coloured by perceived authenticity show that \"real\" classifications "
         "cluster in the high-quality region."),
        ("fig7_radar_by_source.png",
         "Figure 9. Radar chart of mean ratings by source (real + AI platforms). Real (human) songs occupy "
         "a slightly larger polygon than any AI platform on every dimension, with the four AI platforms "
         "showing near-identical perceptual profiles."),
        ("fig13_snippet_condition.png",
         "Figure 10. Snippet-condition comparison on AI songs only. Left: mean ratings are uniformly higher "
         "under the 30-second snippet (v2) than under the full-length snippet (v1). Right: hit rate on AI "
         "drops from 55.6% to 41.3% (misses rise to 42.3%), chi-squared p = 0.019."),
        ("fig8_duration.png",
         "Figure 11. Annotation duration: overall distribution (left) and by source (right). Median "
         "annotation time is substantially shorter in the 30-second condition."),
        ("fig10_raters_per_song.png",
         "Figure 12. Distribution of individual raters per song. The vast majority of songs are rated by "
         "a single rater; a small subset of 19 songs received two or more independent ratings."),
        ("fig11_songs_per_annotator.png",
         "Figure 13. Distribution of songs rated per annotator. The sample shows considerable variation "
         "in participation with a long right tail."),
        ("fig14_models_detection_accuracy_vs_humans.png",
         "Figure 14. AI-or-real detection per evaluator (humans + twelve LLM-judge configurations: "
         "Gemini 3.1 Pro/Flash-Lite, Gemma 4 E2B/E4B, and MOSS-Audio 4B/8B, each in non-thinking "
         "and thinking modes). Bars show balanced accuracy, hit rate on AI songs, correct-rejection "
         "rate on real songs, AI-class F1, and overall accuracy; dashed line marks the human balanced "
         "accuracy, dotted line the 'always AI' base-rate floor, light grey the chance baseline. "
         "Model runs are grouped at the binary thinking level (non-thinking = off/minimal/low; "
         "thinking = on/medium/high)."),
        ("fig15_models_aesthetic_vs_humans.png",
         "Figure 15. Aesthetic-quality rating distributions per evaluator. Humans (left, purple) "
         "compared with each base model (right) as violin plots over per-trial ratings, coloured by "
         "binary thinking mode; the dashed line shows the human overall mean. Three rows split by "
         "perceived authenticity: all trials, trials judged 'real', trials judged 'AI'. The new "
         "MOSS-Audio rows show extreme positive bias: ratings concentrate at the high end of the "
         "scale regardless of perceived authenticity."),
        ("fig17_audiobox_vs_humans.png",
         "Figure 17. Audiobox-Aesthetics (Meta, 2024) vs. human song-level mean ratings on the "
         "1-10 scale. Each panel pairs one of the four audiobox axes with the conceptually closest "
         "human dimension (PQ-production_quality, CE-aesthetic_quality, CU-playlist_likelihood, "
         "PC-musical_creativity). Points are individual songs coloured by source; the dashed "
         "diagonal is y=x and the dark line is an ordinary least-squares fit. Per-panel "
         "Spearman rho, Pearson r and mean absolute error are reported. Audiobox uses no "
         "language signal; humans heard the same audio (v1 full, v2-v1 cropped to 29s)."),
        ("fig18_aesthetics_llm_vs_custom_models.png",
         "Figure 18. Custom audio models vs. human song-level mean ratings. Each panel is one "
         "human questionnaire dimension and bars show Spearman rho for Audiobox-Aesthetics, "
         "SongEval, our music-aesthetics model, and our music-popularity preference proxies "
         "(log1p plays and log1p upvotes). Individual LLM judge configurations are shown "
         "on every matching questionnaire dimension, with separate colors for thinking and "
         "non-thinking runs. Our music-aesthetics model uses Overall_Aesthetics for the "
         "aesthetic-quality panel. The mapping rationale is written to "
         "output/fig18_aesthetics_llm_vs_custom_models.txt, and the compact LaTeX "
         "summary table is written to output/aesthetics_llm_custom_summary_table.tex."),
        ("fig19_songeval_vs_humans.png",
         "Figure 19. SongEval vs. human song-level mean ratings on the 1-10 scale. Each panel "
         "pairs one SongEval axis with the closest human questionnaire dimension: "
         "Musicality-aesthetic_quality, Coherence-production_quality, "
         "Memorability-playlist_likelihood, Clarity-musical_creativity, and "
         "Naturalness-emotional_engagement. Points are individual songs coloured by source; "
         "the dashed diagonal is y=x and the dark line is an ordinary least-squares fit."),
        ("fig20_songeval_vs_audiobox.png",
         "Figure 20. Direct SongEval vs. Audiobox-Aesthetics comparison on matched perceptual "
         "dimensions. Four panels compare the paired audio-front-end scores for the same songs: "
         "Musicality vs CE, Coherence vs PQ, Memorability vs CU, and Clarity vs PC. "
         "Naturalness is omitted because Audiobox has no corresponding axis."),
        ("fig21_feature_analysis.png",
         "Figure 21. Feature-based AI-music detection experiments. Left: OOD detector "
         "head-to-head AUC with 95% bootstrap confidence intervals from the copied feature "
         "analysis; the proposed multiview ensemble is highlighted against foundation-model, "
         "CNN, vision-backbone, and SONICS SpecTTTra baselines. Middle: standalone invariant "
         "probes ranked by OOD AUC, highlighting bicoherence as the strongest linear probe. "
         "Right: LCNN augmentation ablation, with the no-augmentation OOD baseline shown as "
         "a dashed line and signal-destroying regimes colored separately."),
        ("fig22_mood_tag_shifts.png",
         "Figure 22. Human mood-tag prevalence shifts. Bars show the percentage-point change "
         "for each canonical mood tag under two contrasts: actual provenance (AI songs minus "
         "real human songs) and perceived authenticity (trials judged AI-generated minus trials "
         "judged real). Shaded bands show approximate 95% confidence intervals, and asterisks "
         "mark mood tags that survive FDR correction within the panel."),
        ("fig23_model_human_mood_tags.png",
         "Figure 23. Model mood-tag profiles vs. human annotators. The heatmap shows "
         "model-minus-human prevalence differences in percentage points for the nine canonical "
         "mood tags, sorted by per-song mood-overlap Dice score. Asterisks mark model-tag cells "
         "that survive global FDR correction. The middle panel reports per-song Dice overlap, "
         "while the right panel reports robust marginal prevalence distance (median absolute "
         "percentage-point deviation; diamonds mark the mean distance)."),
        ("fig16_models_thinking_mode_effects.png",
         "Figure 16. Thinking-mode effect per base model in ROC space. Each base model contributes "
         "two points (circles = non-thinking, diamonds = thinking), connected by an arrow showing "
         "the non-thinking -> thinking shift. Frontier Gemini models occupy the upper-right (liberal "
         "'AI' bias); MOSS-Audio and Gemma 4 E4B occupy the lower-left (conservative 'real' bias). "
         "Dashed blue curves are iso-d' contours (d'=0.5/1.0/1.5/2.0); dash-dot orange lines are "
         "iso-F1 contours at the actual AI/real class ratio. Purple stars (musical engagement) and "
         "teal squares (AI music experience) show human sub-group reference points."),
    ]

    for filename, caption in figures:
        fig_path = FIG_DIR / filename
        if fig_path.exists():
            pdf.add_figure(fig_path, caption, width=170)
        else:
            pdf.body_text(f"[Figure missing: {filename}]")

    # ---- Statistical Details (from report.txt) ----
    pdf.add_page()
    pdf.section_title("Full Statistical Output", level=1)
    pdf.body_text(
        "The complete statistical output is reproduced below from the automated analysis pipeline. "
        "All tests, model fits, and effect sizes are reported for reproducibility."
    )
    pdf.ln(2)

    report_path = OUTPUT_DIR / "analysis_report.txt"
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        pdf.set_font("Courier", "", 7)
        pdf.set_text_color(30, 30, 30)
        for line in text.split("\n"):
            safe = line.encode("latin-1", errors="replace").decode("latin-1")
            pdf.cell(0, 3.5, safe, new_x="LMARGIN", new_y="NEXT")

    pdf.output(str(REPORT_PATH))
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    build_report()
