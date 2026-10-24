# Development evidence (burned for tuning)

- `itw_pilot_clip_ids.txt`: the 200 In-the-Wild clip IDs of pilot run 20261024T123319Z (100 genuine + 100 spoof,
  all 54 speakers in meta.csv). They were used to diagnose the Dhwani baseline and may be used for implementation
  sanity checks (e.g. does a converted checkpoint behave at all), but **never** for threshold tuning, model selection
  or training. Every fresh held-out subset must exclude these IDs (`eval_lib.select_subset(..., exclude_clip_ids=...)`).
- Speaker-disjoint held-out is **impossible** on In-the-Wild: the pilot touched all 54 speakers. Held-out subsets are
  clip-disjoint only; this is a known limitation and must be stated with every result.
