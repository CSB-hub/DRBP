# DRBP

## Title
Interpretable prediction of nucleic acid-binding proteins using a protein language model

## Authors and Affiliations
Hanjin Kim, Sung-Gwon Lee, Jooseong Oh, Kee K. Kim, Eun-Mi Kim, and Chungoo Park


## Abstract
Accurate identification of DNA-binding proteins (DBPs) and RNA-binding proteins (RBPs) is critical for elucidating transcriptional and post-transcriptional regulatory mechanisms. However, existing computational approaches often rely on inferred labels or domain-specific annotations, which limit the subsequent generalizability. Thus, this study aimed to introduce transformer-based classifiers for human DBPs and RBPs that rely solely on protein sequence information without engineered features or domain constraints. The models were implemented using ESM-2 with low-rank adaptation (LoRA) fine-tuning and trained on experimentally validated datasets, including chromatin immunoprecipitation sequencing (ChIP-seq) annotations for DBPs and eCLIP annotations for RBPs. Next, to evaluate biological relevance, we computed value-aware attention (VAT) scores aggregated across transformer layers to interpret model focus. In 20-fold cross-validation, the DBP model achieved an area under the receiver operating curve (AUROC) of 0.84 with a Matthews correlation coefficient (MCC) of 0.40, while the RBP model achieved an AUROC of 0.92 with an MCC of 0.46. Proteins predicted as nucleic acid-binding were enriched for known binding domains, and inspection of attention distributions revealed preferential focus on annotated functional regions rather than non-binding segments. These results demonstrate that attention-based protein language models can accurately identify nucleic acid-binding proteins directly from sequence data. Moreover, these models reveal biologically meaningful sequence determinants of binding, establishing an interpretable and scalable framework for proteome-wide characterization of protein–nucleic acid interactions


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
