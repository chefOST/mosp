# Candidate original feature sources

Inspected November 7, 2026. These are source leads, not a completed downloaded
feature release or a claim of exact equivalence to MOSCATO's training inputs.

## CMU and EGTEA

The [official FACT repository](https://github.com/ZijiaLewisLu/CVPR2024-FACT#3-prepare-data)
links an [EgoProceL data release](https://drive.google.com/drive/folders/1lAf9qGsCCZmtibVz3KplC4V9L_w4I2Ag).
Its [features directory](https://drive.google.com/drive/folders/13aY0pp7_vrILBwRows8nSNFS6TPgsakw)
contains CMU and EGTEA `.npy` files.

The [FACT supplement, §4](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Lu_FACT_Frame-Action_Cross-Attention_CVPR_2024_supplemental.pdf)
explicitly identifies its EgoProceL features as I3D. That passage does not identify
the pretraining checkpoint/hash, so exact equivalence to MOSCATO remains unverified.

A 256-byte header read of
[`S30_Eggs_7150991-1.npy`](https://drive.google.com/file/d/1Jbz--8mqx6Ga7n6qfQZkXWuurEXwGryC/view)
confirmed float32 shape `[2727, 2048]`. MOSCATO's CMU `files_dict.json` names the
same file under `egoprocel/features` and records `feature_len: 2727`.

The full 22,339,712-byte file was subsequently downloaded for a diagnostic.
The loader and a short, randomly initialized full-architecture forward pass
succeeded; see [implementation checks](checks.md). Matching names and lengths support
the input contract but do not prove the original extraction checkpoint or exact
timestamp convention.

For [`P02-R01-PastaSalad.npy`](https://drive.google.com/file/d/197Re-sASxAryKEpxZ_p-o3DJh_taKfGA/view),
the published file size is consistent with float32 `[17957, 2048]` and MOSCATO
records `feature_len: 17957`. This shape was inferred from file size; its header
was not retrieved successfully. Coverage checks were incomplete, so missing
search results do not establish missing source files.

These matches justify validating existing arrays before re-extraction. Retrieve
only selected files after checking feature provenance, inventory coverage, hashes
and separate state/action timeline mappings. FACT's action-segmentation splits
must not be assumed to be MOSCATO's manually annotated state-evaluation splits.

## EgoPER

[EgoPER's official repository](https://github.com/robert80203/EgoPER_official)
advertises `{task_name}_i3d_2048.zip` archives. Its OneDrive link is request-gated
and expires after two weeks. No access request was sent.

The public [extraction script](https://github.com/robert80203/EgoPER_official/blob/main/I3D_extractor/src/feature_extract.py)
uses RGB I3D-ResNet50, Kinetics400 weights, 10-fps decoded frames and `[T,2048]`
output. It is a candidate recipe; MOSCATO's paper does not establish that it used
these exact extraction settings or archives.
