# EP24 stage audit (#333)

**Export:** `/mnt/workspace/LaclauGPT-Private/analysis/ep24/mongo_export_210`  
**Method:** the exporter's own `stage_coverage()`, per country. No second opinion.

## Records

| Country | Rows in frame | dna_statements | entities | sna | themes |
|---|---|---|---|---|---|
| finland | 2125 | 1583 | 1302 | 3359 | 1460 |
| poland | 1781 | 0 | 1268 | 0 | 1440 |
| portugal | 960 | 4 | 1564 | 0 | 1507 |

## finland (2125 records)

| Step | Stage | With output | % | Failed / other | Available |
|---|---|---|---|---|---|
| 0 | Ingest | 2125 | 100.0% | 544 | yes |
| 1 | Preprocess | 2125 | 100.0% | 19 | yes |
| 2 | Frame analysis | 2125 | 100.0% | 0 | yes |
| 3 | Video analysis | 0 | 0.0% | 0 | **no** |
| 4 | Summary | 1047 | 49.3% | 337 | yes |
| 5 | Postprocess | 1034 | 48.7% | 0 | yes |
| 6 | Discourse | 80 | 3.8% | 31 | yes |
| 7 | DNA statements | 575 | 27.1% | 0 | yes |
| 8 | Social network | 261 | 12.3% | 0 | yes |
| 9 | RDF export | 0 | 0.0% | 0 | **no** |

**DNA grain:** **1583** in children vs **1490** in frame — different grains, do not mix


## poland (1781 records)

| Step | Stage | With output | % | Failed / other | Available |
|---|---|---|---|---|---|
| 0 | Ingest | 1781 | 100.0% | 239 | yes |
| 1 | Preprocess | 1781 | 100.0% | 639 | yes |
| 2 | Frame analysis | 1781 | 100.0% | 16 | yes |
| 3 | Video analysis | 0 | 0.0% | 0 | **no** |
| 4 | Summary | 1099 | 61.7% | 139 | yes |
| 5 | Postprocess | 1763 | 99.0% | 0 | yes |
| 6 | Discourse | 1 | 0.1% | 0 | yes |
| 7 | DNA statements | 0 | 0.0% | 0 | **no** |
| 8 | Social network | 0 | 0.0% | 0 | **no** |
| 9 | RDF export | 0 | 0.0% | 0 | **no** |

**DNA grain:** **0** (frame and children agree)


## portugal (960 records)

| Step | Stage | With output | % | Failed / other | Available |
|---|---|---|---|---|---|
| 0 | Ingest | 960 | 100.0% | 255 | yes |
| 1 | Preprocess | 960 | 100.0% | 300 | yes |
| 2 | Frame analysis | 960 | 100.0% | 66 | yes |
| 3 | Video analysis | 0 | 0.0% | 0 | **no** |
| 4 | Summary | 600 | 62.5% | 153 | yes |
| 5 | Postprocess | 960 | 100.0% | 0 | yes |
| 6 | Discourse | 84 | 8.8% | 0 | yes |
| 7 | DNA statements | 16 | 1.7% | 0 | yes |
| 8 | Social network | 0 | 0.0% | 0 | **no** |
| 9 | RDF export | 0 | 0.0% | 0 | **no** |

**DNA grain:** **4** (frame and children agree)

## How to read this

* **Available = no** means no row carries any output field for that step: the export cannot express
  it. That is different from `0` with output, which means the step ran and produced nothing.
* Child-table counts are file rows, not collapsed records. They are shown separately on purpose.
* The grain line exists because a manifest number and a frame number can both be true and still
  describe different populations (#331).
