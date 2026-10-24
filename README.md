# Voice-clone detection for phone scams (AI-01)

Hackathon project: Android app that flags synthetic speech in imported recordings and in-app calls.
Current stage: two-clip smoke test passed with a **third-party** checkpoint (Dhwani, MIT); API prepared for Colab. Nothing trained by us yet.

- `docs/PROJECT_CONTEXT.md` — brief, product, rules
- `docs/UPSTREAM_AUDIT.md` — pinned revisions, evidence for the preprocessing conflict, known risks
- `docs/PROGRESS.md` — what is done / verified / pending
- `src/vcd/dhwani_infer.py` — inference + validation + windowing (single source of truth)
- `notebooks/01_dhwani_baseline_inference.ipynb`: **executed** baseline notebook (do not overwrite); a fresh copy is generated into `notebooks/build/` by `scripts/build_inference_notebook.py`
- `notebooks/02_detector_api_colab.ipynb`: FastAPI server + tests + HTTPS tunnel on Colab (`scripts/build_api_notebook.py`)
- `src/vcd/vcd_service.py` (inference service, no HTTP) and `src/vcd/vcd_api.py` (FastAPI layer)
- `runs/`: preserved executed notebooks, CSVs and provenance per Colab run

Models, audio and results are git-ignored. Execution happens in Google Colab.
