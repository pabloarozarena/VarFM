# VarFM

Features of genetic variants from biological foundation models.

**Pilot (0.1.0):** two zero-shot log-likelihood features for the variants of a GRCh38 VCF.

| Model | Feature | Variants | Columns |
|---|---|---|---|
| ESM2 650M (`esm2`) | masked marginal: log-probability of the reference and of the mutant residue at the masked site | missense SNVs of MANE Select proteins | `esm2_lp_wt`, `esm2_lp_mut`, `esm2_llr` |
| Evo 2 7B (`evo2`) | log-likelihood of the 8,001 bp REF and ALT windows, summed from the first base where they differ | all (SNVs and indels) | `evo2_ll_ref`, `evo2_ll_alt`, `evo2_delta` |

`esm2_llr = lp_mut - lp_wt` and `evo2_delta = ll_alt - ll_ref`, in nats: negative values mean that the variant is
less likely than the reference.

## Colab

[notebooks/varfm_colab.ipynb](notebooks/varfm_colab.ipynb)
([open in Colab](https://colab.research.google.com/github/pabloarozarena/VarFM/blob/main/notebooks/varfm_colab.ipynb)):
upload a VCF and run all the cells. Evo 2 needs an A100, L4 or H100 runtime; on a T4 only ESM2 runs.

## Install

Python 3.11 or 3.12 (Evo 2 requires one of them):

```bash
python3 -m venv varfm-env && source varfm-env/bin/activate
pip install "varfm[esm2,evo2] @ git+https://github.com/pabloarozarena/VarFM.git"
```

- `varfm[esm2]` installs ESM2 only. Evo 2 needs a CUDA GPU with bfloat16 and about 20 GB of memory.
- flash-attn is not needed (see Evo 2 below).
- torch comes from PyPI, whose current builds need CUDA 13 (NVIDIA driver 580 or later). With an older driver,
  install torch first from the PyTorch index (e.g. `--index-url https://download.pytorch.org/whl/cu126`).
- On an NVIDIA GB10 / DGX Spark (aarch64), VarFM was tested with torch 2.12.1 for CUDA 13.0, installed before VarFM:
  `pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130`.

## Use

```bash
varfm variants.vcf --models esm2 evo2 --out varfm_out    # both models (the default)
varfm variants.vcf --models esm2 --limit 10              # one model, first 10 variants
```

The VCF comes first. `python -c "import varfm; print(varfm.example_vcf())"` prints the path of the example VCF
included in the package (100 ClinVar variants on chr1).

```python
import varfm

table = varfm.extract("variants.vcf", models=["esm2", "evo2"], out_dir="varfm_out")
varfm.extract(["1-930204-G-A"], models=["evo2"], out_dir="one_variant")
```

## Output

- `features.tsv`: one row per variant (multi-allelic records are split).
  - Variant and annotation: `variant` (`GRCh38:chrom:pos:ref:alt`), the VCF `id`, then `gene`, `transcript`,
    `consequence` and `protein_change` from MANE Select.
  - For each model: its values, `<model>_status` (`ok`, `not_applicable`, `failed`) and `<model>_reason`. Examples of
    reasons: `protein_unchanged`, `outside_cds`, `unsupported_kind`, `ref_mismatch`.
- `run.json`: versions, data sources, settings, timings and counts.
- `work/`: one checkpoint per model, written after every batch. A run that stops continues where it stopped when it
  is started again with the same `out_dir`.

## How the features are computed

- **Variants.**
  - Only CHROM, POS, ID, REF and ALT are read.
  - The REF allele must match GRCh38 (`ref_mismatch` otherwise). It is never corrected, and alleles are not
    re-normalized.
- **Annotation.**
  - MANE Select v1.5 (Ensembl IDs), one transcript per variant.
  - For a coding SNV, the coding sequence is rebuilt from GRCh38 and translated, and it must give MANE's protein.
- **ESM2.**
  - The protein is read in a window of at most 1,022 residues around the site. The site is masked, and the
    log-probabilities of the reference and the mutant residue are read there (log_softmax in float32).
  - Only missense SNVs are scored; other protein changes are `not_applicable`.
- **Evo 2.**
  - REF is the window of 8,001 bases centred on the variant; ALT is the same window carrying the alternative allele.
  - For an indel, the longer sequence is trimmed on the right. Log-probabilities are summed from the first position
    where the two differ, since the shared prefix adds the same to both.
  - The model runs in bfloat16, never in FP8.
  - Attention runs on PyTorch's `scaled_dot_product_attention` (`VARFM_EVO2_ATTENTION=flash` uses flash-attn instead).
    The two kernels round differently, and over 8,000 positions the log-likelihoods move by about 1 nat. The setting
    is part of the checkpoint's fingerprint, so values computed with different settings are never mixed.
  - A batch holds sequences of the same length only, because padding moves the results by a similar amount.

## Downloads

Downloads are cached in `~/.cache/varfm`, or in `$VARFM_HOME`:

| What | Source | Size |
|---|---|---|
| GRCh38 chromosomes, only those in the VCF | Ensembl release 115 | ~70 MB each compressed (chr1), ~250 MB on disk |
| MANE v1.5 transcripts and proteins | NCBI | 18 MB |
| ESM2 650M weights (MIT) | fair-esm, torch hub cache | 2.6 GB |
| Evo 2 7B weights (Apache-2.0) | Hugging Face `arcinstitute/evo2_7b`, HF cache; `HF_TOKEN` optional | 13.8 GB |

## Memory

Models run one at a time. Batches start from the memory left once the model is loaded, grow after calm batches and
halve when a batch does not fit or memory runs low.

On GPUs that share the system memory (NVIDIA GB10 / DGX Spark, Jetson), memory pressure can freeze the machine
instead of raising an error. There VarFM:
- keeps a reserve of memory free (24 GB of 130 GB on a GB10);
- caps PyTorch's allocator at 60% of the memory;
- loads checkpoints without memory-mapping;
- stops at once if free memory falls below a critical floor (8 GB); the run resumes from its checkpoint.

`VARFM_GPU_LOCK` names the lock file that keeps a single model on the GPU across runs (default
`/tmp/varfm_gpu.lock`).

## Tests

```bash
pip install -e ".[test]" && pytest
```

The tests run on the CPU in seconds: a toy genome and toy MANE files, and a stand-in model in place of the networks.
