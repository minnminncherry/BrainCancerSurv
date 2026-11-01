# Research notebooks

Open Jupyter from this repository or its `notebook/` folder. Each notebook locates the repository and sets its working directory before loading relative paths. Run cells in order after restarting the kernel.

- `genomic_data_visualization.ipynb`: saved model comparisons, ResMLP confusion matrices, loss/C-index curves, and training times.
- `multimodal_analysis.ipynb`: multimodal result.
- `mri_data_visualization.ipynb`: saved MRI predictions, split previews, and reference-mask comparisons. Preview fold count comes from `script/mri.sh`.
- `wsi_data_visualization.ipynb`: read-only SVS/tissue-mask/patch previews; requires the configured local slide and OpenSlide.
- `genomic_data_clean.ipynb`: exploratory cleaning, MySQL operations, external-source exports, and optional XGBoost training. It is not a read-only analysis notebook. Configure local sources and database before running the relevant sections.
- `learning_biological.ipynb`: learning the relationship between the Genemutation and Genes.

Validation performed: all six notebook structures and code-cell syntax; end-to-end execution of the model-evaluation, multimodal, MRI, and WSI notebooks with local data. Genomic database writes, external exports, and XGBoost training were not executed. Structure/syntax checks do not verify external source schemas or database contents.

Plots are saved under `result/comparisons/`. Headless verification uses a noninteractive Matplotlib backend; Jupyter displays the plots normally.
