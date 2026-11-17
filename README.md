# Calibrated evaluation of synthetic data in small clinical cohorts

Code for the paper *Calibrated Evaluation of Synthetic Data in Small Clinical Cohorts:
Membership Leakage, Utility and the Limits of Fidelity Scores*.

The pipeline fits nine generators to a small rare-disease cohort drawn from MIMIC-IV
and measures each one on three axes (fidelity, utility, membership leakage). Two of the
nine are reference generators that fix the ends of every scale: one resamples the real
training rows, the other draws uniform noise and sees no patient data at all.

## Data

No patient data is included here, and none may be. MIMIC-IV v3.1 is distributed by
PhysioNet under a data use agreement that does not permit redistribution. To reproduce
the results you need your own credentialed access
(https://physionet.org/content/mimiciv/3.1/) and the BigQuery release of the database.

The scripts expect four files in the working directory, built from that database:

| File | Contents |
|---|---|
| `patient_matrix_24h.parquet` | one row per tamponade admission, first-24-hour feature means, before imputation |
| `matrix_imputed_with_subject.parquet` | the same rows with `subject_id` and the outcome |
| `aortic_train.parquet`, `aortic_test.parquet` | the aortic dissection replication cohort, already split |

## Scripts

Run them in this order. Each writes a JSON file of aggregate results; those JSON files
are committed here, so the tables in the paper can be checked without rebuilding the
cohort.

| Script | What it does | Output |
|---|---|---|
| `phase_ab.py` | nine generators, ten generator seeds, fidelity and both attacks, record-level and seed-level bootstrap intervals | `phase_ab.json` |
| `phase_c.py` | utility (PR-AUC on held-out real data) and the first noisy-training pass | `phase_c.json` |
| `phase_d.py` | proximity ratio and near-duplicate rate | `phase_d.json` |
| `phase_e.py` | the whole analysis rerun unchanged on the aortic dissection cohort | `phase_e.json` |
| `phase_f.py` | classifier two-sample test and mutual-information error, both cohorts | `phase_f.json` |
| `phase_g.py` | training-budget sweep for the deep generators | `phase_g.json` |
| `phase_h.py` | ten independent patient-level splits, pipeline rebuilt from the imputer onward in each | `phase_h.json` |
| `phase_i.py` | clipping-only ablation and noisy training over ten seeds, with the Rényi accountant | `phase_i.json` |
| `consolidate.py` | merges every output, computes the sampling floor and the exact permutation tests | `numbers_of_record.json` |
| `dp_fast.py` | checks the vectorised per-sample gradients against a per-sample loop | prints the agreement |

`numbers_of_record.json` is the single source for every number in the paper.

## A note on two corrections

Two defects in the first version of the noisy-training code are fixed in `phase_i.py`
and the paper reports the corrected results. The standardisation statistics were taken
from the private member set and carried into the released output; they now come from
the reference population of surviving admissions. The clipped gradient sum was divided
by the realised batch size; under Poisson subsampling the correct normaliser is the
expected batch size `q*N`.

`phase_c.py` is kept as it was run, so its noisy-training section still contains the
earlier version. Use `phase_i.py` for that part.

## Environment

Python 3.13.

```
pip install -r requirements.txt
```

Database queries were run through the BigQuery client. Architecture comparisons used ten
generator seeds, the training-budget sweep three, the noisy and clipped runs ten, and the
classifier two-sample test five. On two cores, one split of `phase_h.py` takes about four
minutes for all nine generators and one run of `phase_i.py` about eight seconds.
