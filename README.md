# DRBP

## Title
Interpretable prediction of nucleic acid-binding proteins using a protein language model

## Authors and Affiliations
Hanjin Kim, Sung-Gwon Lee, Jooseong Oh, Kee K. Kim, Eun-Mi Kim, and Chungoo Park


## Abstract
Nucleic acid-binding proteins (DBPs and RBPs) regulate transcription, RNA processing, and translation, but experimental discovery is incomplete and traditional predictors often depend on domain annotations or handcrafted features. We present a sequence-only deep learning framework built on ESM-2 with LoRA fine-tuning to classify DNA- and RNA-binding proteins. Using experimentally validated human datasets, the DBP model achieved AUC-ROC 0.84 and the RBP model achieved AUC-ROC 0.92 in 20-fold cross-validation. To interpret predictions, value-aware attention (VAT) analysis showed that model attention concentrates within annotated nucleic acid-binding domains at both domain and residue levels, despite no explicit domain inputs. These results demonstrate that protein language models can capture biologically meaningful binding signals directly from sequence and provide interpretable, scalable functional annotation.

## Keywords
DNA-binding proteins; RNA-binding proteins; protein language model; ESM-2; deep learning; attention mechanism

## Problem Statement
This project classifies protein sequences as DNA-binding proteins (DBP) or RNA-binding proteins (RBP) using a transformer-based sequence model with lightweight fine-tuning and interpretable attention analysis.

## Method Overview
- Backbone model: `facebook/esm2_t33_650M_UR50D`
- Fine-tuning: LoRA on attention projections (`query`, `key`, `value`)
- Tasks: binary classification for D (DNA) and R (RNA)
- Interpretability: value-aware attention (VAT) token scoring

## Dataset
Human protein sequences (18,221 protein-coding genes; GRCh38.p12) with experimental annotations:
- DBP positives: 1,447 (ChIP-seq, microarray) from hPDI and hTFtarget
- RBP positives: 351 (eCLIP) from ENCODE
- Dual-binding proteins: 90
- Negatives: 16,774 (DBP) and 17,870 (RBP)

Raw files are under `data/raw_data/`. Cross-validation outputs are under `data/DRBP_cv_results/`.

## Experimental Setup
- 20-fold cross-validation with 95% train / 5% test per fold
- Sequence length: max 1,024 residues (truncate/pad)
- Positive oversampling: 12x (DBP) and 50x (RBP)
- Threshold selection: MCC-maximizing cutoffs (DBP 0.7, RBP 0.9)

## Interpretability
Value-aware attention (VAT) scores localize to known nucleic acid-binding domains, including zinc finger and homeodomain regions for DBPs and RNA recognition motifs (RRM) for RBPs. Domain-level enrichment and residue-level profiles show consistent attention concentration in annotated binding regions.

## Repository Structure
- `src/scripts/DRBP_train.py`: train LoRA models for D/R tasks (CV-based)
- `src/scripts/DRBP_predicton.py`: evaluate models and write prediction CSVs
- `src/scripts/DRBP_att_score.py`: compute VAT token scores for validation sequences
- `src/notebooks/`: data EDA and result analysis notebooks
- `data/`: raw FASTA files and cross-validation results

## How to Run
Training (single CV):
```bash
python src/scripts/DRBP_train.py
```

Evaluation (single CV):
```bash
python src/scripts/DRBP_predicton.py --kind D --cv 1
```

Evaluation (CV range):
```bash
python src/scripts/DRBP_predicton.py --kind R --cv-start 1 --cv-end 20
```

VATP scoring:
```bash
python src/scripts/DRBP_att_score.py --kind D --cv 1 --pad-len 40 --prefix date
```

## Outputs
- Predictions: `New_new_DRBP/RBP_result/*.csv` (see `DRBP_predicton.py`)
- VAT scores: `New_DRBP/*VATP_result*/`

## Reproducibility
Please document exact versions and hardware in the final submission:
- OS, CUDA, GPU model and driver
- Python version
- `torch`, `transformers`, `peft`, `scikit-learn`, `pandas`, `tqdm`

## Data Availability
All scripts for preprocessing, training, and evaluation are provided in this repository. Trained model weights are included here. For paper submission, add the public repository link and DOI as needed.


## Author Contributions
CP and SGL, HK designed the research and wrote the paper. CP, EMK and KK coordinated the research. SGL, HK and JO performed the research and analyzed the data. All authors read and approved the final manuscript.

## Competing Interests
The authors declare no competing interests.
