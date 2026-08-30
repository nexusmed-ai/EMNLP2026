# Attention-Weighted Retrieval-Augmented Data Engineering for ADR Severity Classification

Predicting whether an adverse drug reaction (ADR) case report will be **severe** —
resulting in death or a life-threatening event — from the patient, drug, and indication
fields of an FDA Adverse Event Reporting System (FAERS) report, *before* the reaction
outcome is known.

The central idea: instead of training a classifier over twenty years of case reports,
keep the corpus in a **retrieval index** and give the classifier three vectors per case —
the query, plus one attention-pooled summary from each half of a hybrid (BM25 + dense)
retriever. The corpus does not disappear; it moves out of the training tensor and into
an index. See [`main README`](../README.md) for the full accounting of what that trade buys and what it costs.

---

## 1. Task

|                     |                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Input**     | `inst` — a structured string with patient demographics (age, sex, weight), each drug (name, route, dose), and the indication for use                                                                                                                                                                                                                                              |
| **Target**    | `severity` ∈ {`YES`, `NO`} — `YES` iff the case's FAERS `OUTC` set contains `DE` (death) or `LT` (life-threatening). The milder codes `HO` (hospitalization), `DS` (disability), `CA` (congenital anomaly), `RI` (required intervention), `OT` (other) map to `NO` on their own. Verified to reproduce the stored label exactly on all 118,953 OOT cases |
| **Split**     | **Temporal.** Train on 2004q1–2024q3, evaluate on 2024q4–2025q2. No future information reaches retrieval                                                                                                                                                                                                                                                                     |
| **Imbalance** | 13.3% positive in train, 17.5% in the out-of-time (OOT) set                                                                                                                                                                                                                                                                                                                          |

Because the positive class is both rare and the costly one to miss, evaluation reports
**F1, recall, ROC-AUC, and PR-AUC** alongside accuracy, and an **F2** variant
(`fbeta_score(..., beta=1)`) that weights recall over precision. Accuracy alone is
misleading here — a constant `NO` predictor scores 82.5% on the OOT set.

---

## 2. Data

### 2.1 Source

Public **FAERS quarterly ASCII extracts** from the FDA:

```
https://fis.fda.gov/content/Exports/faers_ascii_<YYYY>q<N>.zip   # 2012q4 onward
https://fis.fda.gov/content/Exports/aers_ascii_<YYYY>q<N>.zip    # 2004q1 – 2012q3 (legacy AERS)
```

Five of the seven `$`-delimited tables per quarter are used:

| Table    | Fields used                                                                                                    |
| -------- | -------------------------------------------------------------------------------------------------------------- |
| `DEMO` | age, age unit, sex, weight, weight unit                                                                        |
| `DRUG` | drug name, route, dose                                                                                         |
| `INDI` | indication for use                                                                                             |
| `REAC` | MedDRA Preferred Terms (used to build the retrieval corpus, not as input)                                      |
| `OUTC` | outcome codes (`DE`, `LT`, `HO`, `DS`, `CA`, `RI`, `OT`) are grouped into binary severity labels |

`RPSR` (report source) is excluded — populated for under 3% of reports.
Download and merge live in [`FAERS_Prep.ipynb`](FAERS_Prep.ipynb); raw quarters are
expected at `faers/<yyyyqN>/ASCII/`.

### 2.2 Prepared splits

Pandas DataFrames pickled under [`key_data/`](key_data/):

| File                         | Rows                    | Period           | Role                           |
| ---------------------------- | ----------------------- | ---------------- | ------------------------------ |
| `key_data/adr_trn_new.pkl` | 3,387,476               | 2004q1 – 2024q3 | Retrieval corpus + supervision |
| `key_data/adr_oot_new.pkl` | 118,953 (78657 + 40296) | 2024q4 – 2025q2 | Training and testing sets      |

**Schema**

| Column                      | Description                                                               |
| --------------------------- | ------------------------------------------------------------------------- |
| `caseid`                  | FAERS case identifier, deduplicated across quarters                       |
| `inst`                    | Model input — patient / treatment / indication as a JSON-ish string      |
| `outcome`                 | `{"pt": "<PT; PT; …>", "uni_code": "<code>"}` — the observed reaction |
| `pt`                      | Semicolon-joined MedDRA Preferred Terms                                   |
| `uni_code`                | Most-severe single outcome code                                           |
| `outc_code`               | Full semicolon-joined set of outcome codes                                |
| `Severity` / `severity` | **The binary target**, `YES` / `NO`                             |
| `yr_qtr`                  | Source quarter, e.g.`2025q1`                                            |
| `id`                      | Row index used as the retrieval document key (train only)                 |

Label counts: train 450,511 `YES` / 2,936,965 `NO`; OOT 20,871 `YES` / 98,082 `NO`.

### 2.3 Retrieval indexes

| Artifact              | Built with | Contents                                                                    |
| --------------------- | ---------- | --------------------------------------------------------------------------- |
| `bm25s_adr_trn/`    | `bm25s`  | Sparse BM25 index +`corpus.jsonl` over the 3.4M training `inst` strings |
| `bge_m3_adr.lance/` | LanceDB    | Dense 1024-d case embeddings over the same corpus                           |

The two indexes are complementary: BM25 catches verbatim drug and reaction terms, dense
search catches paraphrase and misspelling. The results below quantify how differently they behave.

### 2.4 Model input tensors

The retrieval step compiles down to a small, fixed-width tensor:

| File                           | Shape                                                                                                | Contents                 |
| ------------------------------ | ---------------------------------------------------------------------------------------------------- | ------------------------ |
| `x_y_trn_tst_cls_tensor.pkl` | `torch.Size([78657, 3, 1024]), torch.Size([40296, 3, 1024]), torch.Size([78657]), torch.Size([40296] | Training and Testing set |

The three channels, in order, are `[query ; bm25 ; dense]`:

1. **query** — the fine-tuned BGE-M3 embedding of `inst`.
2. **bm25** — top-5 BM25 neighbours, attention-pooled: `s = Σ αᵢeᵢ` with
   `α = softmax(BM25 scores)`.
3. **dense** — top-5 ANN neighbours, pooled the same way over cosine scores.

Ten retrieved neighbours never reach the classifier individually — each pipeline's top-5
collapses into a single 1024-d summary, so the head always sees exactly three tokens.
This is a **7.3× smaller input tensor** than encoding all 3.7M cases as training rows
(5.22 × 10⁸ vs. 3.79 × 10⁹ float positions).

An auxiliary scalar feature set is also used for the tree-based models:
`bm25_score`, `dense_score`, `cosine(query_emb, neighbor_query_emb)`,
`cosine(query_outcome_emb, neighbor_outcome_emb)`, and each neighbour's own severity
label, over all 10 retrievals.

### 2.5 Reproducing

```bash
# 1. Download and merge FAERS quarters      -> key_data/adr_{trn,oot}_new.pkl
#    FAERS_Prep.ipynb

# 2. Embed the 3.4M-case corpus             -> bge_m3_adr.lance
python adr_trn_embed.py

# 3. Build the BM25 index                   -> bm25s_adr_trn/
#    ADR_LanceDB_alpha.ipynb, BM25 section

# 4. Retrieve + attention-pool              -> x_y_trn_tst_cls_tensor.pkl
#    ADR_severity_binary.ipynb, "Prepare model input"

# 5. Train and evaluate
python xgboost_trn_a100_gpu.py
```

Raw FAERS quarters, the multi-GB intermediate pickles, and the `.lance` / BM25 stores are
**not committed** — they exceed GitHub limits. The scripts that regenerate them are.

---

## 3. Models

### 3.1 Retrieval encoder

Retrieval quality is the bottleneck, so the encoder is fine-tuned on FAERS
case↔outcome pairs rather than used off the shelf.

| Directory        | Base                      | Dim  | Role                                                       |
| ---------------- | ------------------------- | ---- | ---------------------------------------------------------- |
| `bge_m3_4adr/` | BAAI BGE-M3 (XLM-RoBERTa) | 1024 | **Primary encoder** for queries, corpus, and pooling |

Fine-tuning: [`finetune_embedding_large_new.ipynb`](finetune_embedding_large_new.ipynb).

### 3.2 Severity heads

| Artifact                                      | Model                                     | Input            |
| --------------------------------------------- | ----------------------------------------- | ---------------- |
| `ft_model/{llama,qwen}_unicode_classifier/` | LoRA adapters on Llama-3.2-1B / Qwen-0.8B | raw`inst` text |

Training entry points: `llama_ft.py` / `qwen_ft_2gpu.py`.

### 3.3 Baselines

| Family                          | What                                                                                       | Where                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------ |
| **Trivial**               | Majority class; unweighted majority vote over all 10 retrieved neighbours' severity labels | `ADR_severity_binary.ipynb`                          |
| **Retrieval scores only** | Classifier on the scalar features alone, no embeddings                                     | `ADR_severity_binary.ipynb`                          |
| **Sequence DL**           | CNN, LSTM, BiLSTM over`inst`                                                             | `*_baseline_test_pred.pkl`                           |
| **LLM zero/few-shot**     | Llama-3.3-70B-Instruct, Qwen2.5-72B-Instruct                                               | `llm_severity_binary_inference_batch.py`, `slurm/` |
| **Hosted LLM**            | `gpt-5-mini`                                                                             | `openai_severity_binary_inference_batch.py`          |

Base LLM weights are not vendored — point `--model` at a local snapshot or HF repo id.

---

## 4. Results

### 4.1 Channel ablation, 5-fold CV

From `ablationCV_GPU_5fold_results.pkl` (mean ± sd over 5 folds, threshold tuned per fold
to ≈ 0.43):

| Metric    | Fused`[query; bm25; dense]` |
| --------- | ----------------------------- |
| Accuracy  | 0.8455 ± 0.0033              |
| F1        | 0.5499 ± 0.0074              |
| Recall    | 0.5345 ± 0.0152              |
| Precision | 0.5667 ± 0.0120              |
| ROC-AUC   | 0.8234 ± 0.0044              |

Per-channel F1 and the learned fusion weights:

| Channel   | F1 alone | Fusion weight |
| --------- | -------- | ------------- |
| `query` | 0.5393   | 0.39          |
| `dense` | 0.5261   | 0.38          |
| `bm25`  | 0.3182   | 0.23          |

The model leans on query and dense roughly equally and discounts BM25 — consistent with
its weak standalone F1, but it is not dropped, because what it contributes is
complementary rather than redundant.

### 4.2 Out-of-time evaluation

On the held-out 2024q4–2025q2 reports:

- The **query-only** representation reaches the highest raw accuracy (**82.9%**) — which
  on a 17.5%-positive set is close to what predicting `NO` everywhere achieves.
- The **fused ensemble** wins on every metric that reflects catching severe cases:
  **F1 0.458, recall 0.469, ROC-AUC 0.773, PR-AUC 0.506.**
- **BM25 alone** has the highest precision (**0.588**) but collapses on recall
  (**0.076**). Lexical retrieval finds only a small subset of severe ADRs — but when it
  fires, it is usually right.

Tree-based heads on the same features (`ml_model_results.pkl`) trade off differently:
XGBoost reaches accuracy 0.863 / F1 0.480 / ROC-AUC 0.827 at a 0.5 threshold, and moving
the threshold pushes Random Forest to recall 0.657 at F1 0.487 — the operating point to
prefer when a missed severe case is the expensive error.

**Takeaway.** Combining semantic retrieval, lexical retrieval, and the query
representation captures complementary pharmacovigilance signals. Retrieval-enhanced
representations improve robustness on *future* reports specifically — the regime that
matters for surveillance, and the one a temporally-random split would hide.

## 5. Code

### 5.1 Pipeline

```
FAERS ASCII ──▶ FAERS_Prep.ipynb ──▶ key_data/adr_{trn,oot}_new.pkl
                                              │
                    ┌─────────────────────────┴──────────────────────────┐
                    ▼                                                    ▼
        adr_trn_embed.py                                    bm25s index build
        → bge_m3_adr.lance                                  → bm25s_adr_trn/
                    │                                                    │
                    │           top-5 dense            top-5 lexical     │
                    └────────────┬───────────────────────┬───────────────┘
                                 ▼                       ▼
                        attention pool            attention pool
                        α = softmax(cos)          α = softmax(BM25)
                                 └──────────┬────────────┘
                                            ▼
                              [ query ; bm25 ; dense ]  (N, 3, 1024)
                                            │
                          ┌─────────────────┼──────────────────┐
                          ▼                 ▼                  ▼
                 XGBoost / CatBoost / RF   BCE head      baselines
                          └─────────────────┼──────────────────┘
                                            ▼
                              threshold tuning · F1 / PR-AUC
```

### 5.2 Files

**Encoding and retrieval**

- `adr_trn_embed.py`, `embed_adr_pt.py`, `gpu1V100_embed.py` — encode the corpus.
- `compare_bm25_dense.py` — precision / recall / F1 of BM25 vs. dense retrieval on 2025q2.
- `utils/func.py` — `normalize_rows`,
- `top5_lbl_scr_hybrid.py`, `vect_top5_lbl_scr_hybrid.py`, `oot_lbl_scr_hybrid.py` —
  hybrid retrieval and neighbour-label transfer for the OOT set.

**Training**

- `xgboost_trn_a100_gpu.py`, `ml_trn.py` — GBDT heads with `query` / `bm25` / `dense`
  channel ablation and stratified 5-fold CV.
- `llama_ft.py`, `llama_ft_2gpu.py`, `qwen_ft_2gpu.py` — LoRA fine-tuning.

**Inference**

- `infer_ft.py`, `infer_val.py`.
- `llm_severity_binary_inference_batch.py`, `llm_severity_inference_batch.py`,
  `llm_context_inference.py`, `inf_llama3.5.py`.
- `openai_severity_binary_inference_batch.py`, `openai_zero_shot_inference_batch.py`,
  `openai_context_inference_batch.py`.

**Notebooks**

| Notebook                                         | Purpose                                                                        |
| ------------------------------------------------ | ------------------------------------------------------------------------------ |
| `ADR_severity_binary.ipynb`                    | **Main notebook** — feature construction, ablations, CV, OOT evaluation |
| `FAERS_Prep.ipynb`                             | Download, merge, curate FAERS quarters                                         |
| `ADR_LanceDB_alpha.ipynb`                      | LanceDB tables and BM25 index construction                                     |
| `finetune_embedding_large_new.ipynb`           | Encoder fine-tuning                                                            |
| `evaluation.ipynb`, `refit_evaluation.ipynb` | Metric aggregation                                                             |

## 6. Setup

```bash
conda create -n adr python=3.11 && conda activate adr
pip install torch transformers sentence-transformers accelerate peft datasets \
            lancedb pylance bm25s xgboost catboost scikit-learn \
            pandas pyarrow tqdm requests beautifulsoup4 wandb
```

Verified versions: `torch 2.9.1`, `transformers 4.57.3`, `sentence-transformers 5.5.1`,
`lancedb 0.33.0`, `bm25s 0.3.2`, `xgboost 3.1.3`, `catboost 1.2.10`, `peft 0.16.0`,
`pandas 2.3.3`, `pyarrow 20.0.0`, `scikit-learn 1.7.2`.

**Hardware.** Developed on 1–2× V100 (32 GB) and A100 nodes from UTSA ARC Cluster. Encoding the 3.4M-case
corpus is the expensive step and only needs redoing when the encoder is re-fine-tuned; batch sizes in the scripts (`BATCH_SIZE = 4096`, `ENCODE_BATCH_SIZE = 256`) are tuned for 32 GB VRAM. Training the severity head on the `(78657, 3, 1024)` tensor is cheap by comparison.

```bash
sbatch slurm/llama_zero_inf.slurm     # zero-shot Llama-3.3-70B
sbatch slurm/qwen_3_inf.slurm         # 3-shot Qwen2.5-72B
```

---

## 7. Repository layout

```
.
├── ADR_severity_binary.ipynb   # main experiment notebook
├── FAESR_Prep.ipynb            # main data sourcing notebook
├── Key_data/                   # curated train / OOT DataFrames
├── Utils/func.py               # normalization helpers, etc.
├── bge_m3_4adr/ ...            # fine-tuned embedding model (https://huggingface.co/nexusmed-ai/adr_bge_m3_embedding)
├── Submission/                 # paper source, figures 
```

---

## 8. Data use and licensing

FAERS is public, de-identified post-marketing surveillance data. The FDA is explicit that
reports are **voluntary and unverified** — a report does not establish causation, and
counts must not be read as incidence rates. This is research code for pharmacovigilance
signal modeling, **not** a clinical decision-support tool, and a severity prediction here
carries no clinical authority.

MedDRA terminology is licensed by MSSO and is not redistributed. Model weights inherit
their base-model licenses (BGE-M3 / MIT, Qwen / Apache-2.0, Llama / Llama Community
License, MedEmbed / Apache-2.0, DeBERTa / MIT).

<!-- TODO: add a LICENSE file for the code in this repo. -->

## 9. Citation

```bibtex
@misc{adr_severity_hybrid_rag,
  title  = {Attention-Weighted Retrieval-Augmented Data Engineering for ADR
Severity Classification},
  author = {Guo, David and Vishwamitra, Nishant and Choo, Kim-Kwang Raymond},
  note   = {University of Texas at San Antonio},
  publisher ={EMNLP},
  year   = {2026}
}
```
