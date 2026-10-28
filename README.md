# RR-9 imputation validation audit

Re-runs the validation from *A systematic imputation framework for sparse, multimodal space biology datasets: application to retinal imaging and omics from the RR9 mission* (bioRxiv 2026.06.09.730965) using the CSVs published in [GoJian/AI-ML_AWG](https://github.com/GoJian/AI-ML_AWG/tree/main/Manuscript_Code/rr9_imputation), and checks how the reported numbers change under leakage-free evaluation.

-- Sanmay Anand · NASA OSDR AI/ML AWG, digital twin subgroup

## Run

Easiest: open `rr9_audit_colab.ipynb` in [Google Colab](https://colab.research.google.com) and run the cells in order. Locally:

```bash
git clone https://github.com/GoJian/AI-ML_AWG.git
pip install -r requirements.txt
python audit.py --data AI-ML_AWG/Manuscript_Code/rr9_imputation
python audit.py --data AI-ML_AWG/Manuscript_Code/rr9_imputation --slow --seeds 5 --out audit_output_slow
```

The first command takes about a minute. `--slow` adds checks 4 and 5, which re-run MICE-Elastic ten times (about 12 minutes). Outputs go to `<out>/figures` and `<out>/results`. Reference outputs in `expected_output/` were produced by exactly these two commands.

## Checks

| # | Check | Result |
|---|---|---|
| 1 | Train on imputed, test on original | The 16 RNA-seq test animals are inside every 100-animal training set with identical values. Published RF/SVM accuracy of 1.00 (KNN, MICE-Elastic, Hybrid) falls to 0.75 / 0.78 / 0.65 (RF, balanced, 20 seeds) and SVM 0.50 when they are excluded. TUNEL largely holds (MICE-Elastic 0.94). |
| 2 | Masked value recovery | Published values reproduce exactly (KNN RMSE 357.73, MICE-Elastic 725.43). On the same mask, a gene-mean baseline scores RMSE 495.90 and pooled r = 0.998. In per-gene standard deviations: baseline 1.11, KNN 1.08, MICE-Elastic 1.42. |
| 3 | Co-observation | Proteomics shares 0 animals with RNA-seq and every IHC assay. microCT shares 1 animal with RNA-seq and 0 with IHC. |
| 4 | Group label as imputation input | Not the cause of the permutation result: p = 0.001 with and without it. Reproduced MICE-Elastic correlates 0.9995 with the published imputed values. |
| 5 | Nested re-imputation | Held-out RNA-seq animals masked before imputing (4 animal-disjoint folds x 2 repeats, 5 RF seeds). Accuracy on held-out real animals: real animals only 0.86, KNN-augmented 0.76, MICE-Elastic-augmented 0.78. |

## Caveats

- 16 RNA-seq animals: differences in check 5 are not statistically significant.
- The RNA-seq features were selected on the same animals, so all absolute accuracies are optimistic; compare settings, not absolute values.
- Check 1 classifiers use the published imputed files, whose imputers saw the test animals. Check 5 removes that.
- Prepared with AI assistance; every number was re-run and checked by the author.
