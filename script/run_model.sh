#!/bin/sh
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"

export PYTHONUNBUFFERED=1
printf '[%s] START run_model.sh\n' "$(date -Iseconds)"
trap 'status=$?; printf "[%s] END run_model.sh | exit=%s\n" "$(date -Iseconds)" "$status"' 0

# Commands from model_run.txt; run sequentially and stop on the first error.

# # GBM: nll_surv
# sh "$SCRIPT_DIR/genomic.sh" GBM resmlp nll_surv
# sh "$SCRIPT_DIR/genomic.sh" GBM mlp nll_surv
# sh "$SCRIPT_DIR/genomic.sh" GBM gen2vec nll_surv
# sh "$SCRIPT_DIR/genomic.sh" GBM snn nll_surv
# sh "$SCRIPT_DIR/wsi.sh" GBM mil nll_surv
# sh "$SCRIPT_DIR/wsi.sh" GBM transmil nll_surv
sh "$SCRIPT_DIR/mri.sh" GBM ViT nll_surv
sh "$SCRIPT_DIR/mri.sh" GBM 3dcnn nll_surv
# sh "$SCRIPT_DIR/fusion_genomic_wsi.sh" GBM multimodal_late_fusion nll_surv
# sh "$SCRIPT_DIR/fusion_genomic_wsi.sh" GBM multimodal_cross_attention nll_surv
# sh "$SCRIPT_DIR/fusion_genomic_mri.sh" GBM multimodal_mri_late_fusion nll_surv
# sh "$SCRIPT_DIR/fusion_genomic_mri.sh" GBM multimodal_mri_cross_attention nll_surv

# # GBM: cross_entropy
# sh "$SCRIPT_DIR/genomic.sh" GBM resmlp cross_entropy
# sh "$SCRIPT_DIR/genomic.sh" GBM mlp cross_entropy
# sh "$SCRIPT_DIR/genomic.sh" GBM gen2vec cross_entropy
# sh "$SCRIPT_DIR/genomic.sh" GBM snn cross_entropy
# sh "$SCRIPT_DIR/wsi.sh" GBM mil cross_entropy
# sh "$SCRIPT_DIR/wsi.sh" GBM transmil cross_entropy
# sh "$SCRIPT_DIR/mri.sh" GBM ViT cross_entropy
# sh "$SCRIPT_DIR/mri.sh" GBM 3dcnn cross_entropy
# sh "$SCRIPT_DIR/fusion_genomic_wsi.sh" GBM multimodal_late_fusion cross_entropy
# sh "$SCRIPT_DIR/fusion_genomic_wsi.sh" GBM multimodal_cross_attention cross_entropy
# sh "$SCRIPT_DIR/fusion_genomic_mri.sh" GBM multimodal_mri_late_fusion cross_entropy
# sh "$SCRIPT_DIR/fusion_genomic_mri.sh" GBM multimodal_mri_cross_attention cross_entropy

# # LGG: nll_surv
# sh "$SCRIPT_DIR/genomic.sh" LGG resmlp nll_surv
# sh "$SCRIPT_DIR/genomic.sh" LGG mlp nll_surv
# sh "$SCRIPT_DIR/genomic.sh" LGG gen2vec nll_surv
# sh "$SCRIPT_DIR/genomic.sh" LGG snn nll_surv
# sh "$SCRIPT_DIR/wsi.sh" LGG mil nll_surv
# sh "$SCRIPT_DIR/wsi.sh" LGG transmil nll_surv
# sh "$SCRIPT_DIR/mri.sh" LGG ViT nll_surv
# sh "$SCRIPT_DIR/mri.sh" LGG 3dcnn nll_surv
# sh "$SCRIPT_DIR/fusion_genomic_wsi.sh" LGG multimodal_late_fusion nll_surv
# sh "$SCRIPT_DIR/fusion_genomic_wsi.sh" LGG multimodal_cross_attention nll_surv
# sh "$SCRIPT_DIR/fusion_genomic_mri.sh" LGG multimodal_mri_late_fusion nll_surv
# sh "$SCRIPT_DIR/fusion_genomic_mri.sh" LGG multimodal_mri_cross_attention nll_surv

# # LGG: cross_entropy
# sh "$SCRIPT_DIR/genomic.sh" LGG resmlp cross_entropy
# sh "$SCRIPT_DIR/genomic.sh" LGG mlp cross_entropy
# sh "$SCRIPT_DIR/genomic.sh" LGG gen2vec cross_entropy
# sh "$SCRIPT_DIR/genomic.sh" LGG snn cross_entropy
# sh "$SCRIPT_DIR/wsi.sh" LGG mil cross_entropy
# sh "$SCRIPT_DIR/wsi.sh" LGG transmil cross_entropy
# sh "$SCRIPT_DIR/mri.sh" LGG ViT cross_entropy
# sh "$SCRIPT_DIR/mri.sh" LGG 3dcnn cross_entropy
# sh "$SCRIPT_DIR/fusion_genomic_wsi.sh" LGG multimodal_late_fusion cross_entropy
# sh "$SCRIPT_DIR/fusion_genomic_wsi.sh" LGG multimodal_cross_attention cross_entropy
# sh "$SCRIPT_DIR/fusion_genomic_mri.sh" LGG multimodal_mri_late_fusion cross_entropy
# sh "$SCRIPT_DIR/fusion_genomic_mri.sh" LGG multimodal_mri_cross_attention cross_entropy


# sh "$SCRIPT_DIR/mri.sh" GBM ViT nll_surv
# sh "$SCRIPT_DIR/mri.sh" GBM 3dcnn nll_surv
# sh "$SCRIPT_DIR/mri.sh" LGG ViT nll_surv
# sh "$SCRIPT_DIR/mri.sh" LGG 3dcnn nll_surv
# sh "$SCRIPT_DIR/fusion_genomic_mri.sh" GBM multimodal_mri_late_fusion nll_surv
# sh "$SCRIPT_DIR/fusion_genomic_mri.sh" GBM multimodal_mri_cross_attention nll_surv
# sh "$SCRIPT_DIR/fusion_genomic_mri.sh" LGG multimodal_mri_late_fusion nll_surv
# sh "$SCRIPT_DIR/fusion_genomic_mri.sh" LGG multimodal_mri_cross_attention nll_surv