# Contributor record — family contributor #1

*Chain-of-custody for third-party works contributed to the project. Per docs/plan.md §10 governance. This is a record of CUSTODY, **not** a rights grant: copyrighted / eval-only items stay quarantined from training regardless of possession. No personal name is recorded beyond the placeholder the contributor has approved.*

| field | value |
|---|---|
| Contributor | family contributor #1 (placeholder; real name withheld pending approval) |
| Role | professional interpreter / translator (reference collection) |
| Date received | 2026-07-23 (files copied to ml/data/local/family-v1/, untracked) |
| Nature | chain-of-custody of third-party works — NOT a rights grant |

## Per-file provenance + rights outcome

| file | producer | rights verdict | outcome |
|---|---|---|---|
| cfpb_adult-fin-ed…glossary.pdf | Consumer Financial Protection Bureau (federal) | TRAIN-OK (17 U.S.C. §105 PD) | ingested (train) + EN↔HT pairs → committable federal pairs file |
| 11783_glossary…COURT TERMS.pdf | New Jersey Judiciary (state) | PERMISSION-ROUTE (state work ≠ PD) | EVAL-ONLY legal-terminology probe pool; terms-check email pending |
| iep-form-haitiancreole.pdf | MA DESE (state) | PERMISSION-ROUTE | EVAL-ONLY education probe pool (form labels); tiny |
| Clinical_Trial_Glossary…pdf | Boston Medical Center ("Courtesy") | PERMISSION-ROUTE (courtesy ≠ license) | EVAL-ONLY clinical probe pool; **nonstandard orthography tagged** |
| FILE_7167.pdf | **Beverly Hospital (Beth Israel Lahey Health)** — identified from page-1 logo | PRIVATE, not federal PD | EVAL-ONLY pending permission; **not extracted** into any artifact |
| Haitian-English Dictionary (Bryant C. Freeman) | published, copyrighted | QUARANTINE | recorded only; **not opened/extracted** |

## Notes

- Only the **CFPB** glossary (federal PD) enters training; its EN↔HT pairs are the sole family-set text that becomes a committed artifact (`corpus/glossary_pairs_federal.json`).
- The three state/private/courtesy items are held **EVAL-ONLY** as terminology probe pools (registered in `splits.yaml` `eval_slices_v0_2.terminology_probes`); the permission emails are a J5 human action.
- **BMC orthography audit:** 4.0% of words carry French-style accents (e.g. Doné, Dézako, Santé, Sélman, antré, apré) → tagged `nonstandard_orthography`, **not** silently normalized (measure-don't-filter).
- The Freeman dictionary was **not opened**; its quarantine is independent of possession.