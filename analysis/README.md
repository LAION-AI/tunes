# Paper analysis and camera-ready updates

Run commands from the repository root unless shown otherwise. Existing analyses
and their tables live in `analysis/output/`; the manuscript includes those tables
directly. Training the companion models and administering the listener study are
separate workflows.

## LLM-judge prompts

```bash
python3 analysis/export_judge_prompts.py
```

This reads the prompt constants in `model_eval.py` without importing or running
the API client. It writes exact base system/user strings to `analysis/prompts/`
and a verbatim TeX version to `analysis/output/benchmark_judge_prompts.tex`.
The manuscript also documents model-specific reasoning wrappers and the
confidence threshold used for post-hoc uncertainty. Change the runner's prompt
constants first, then regenerate the exports. The arXiv builder runs this export
automatically. Archived evaluation outputs remain the record of the past runs;
editing a prompt does not change those outputs.

## Annotation audit

Read `annotation_audit/PROTOCOL.md` before sampling or labelling. With the existing
analysis virtual environment:

```bash
analysis/.venv/bin/python analysis/annotation_audit.py --help
analysis/.venv/bin/python analysis/annotation_audit.py selftest
```

The `schema`, `sample` and `analyze` subcommands inspect released parquets, draw
stratified samples, and score completed human labels. The synthetic self-test
checks the pipeline; it is not evidence that the human audit has been completed.

## Manuscript builds

Compile `neurips_2026.tex` on Overleaf with pdfLaTeX, or with the existing local
Tectonic installation. Rebuild the flattened arXiv source and preview with:

```bash
python3 arxiv/build_arxiv.py
```

See `TODO.md` for remaining team-supplied facts and `arxiv/README.md` for upload
metadata. Generated PDFs are git-ignored; `arxiv/src/` is tracked.
