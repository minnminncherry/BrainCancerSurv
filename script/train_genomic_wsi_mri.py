"""Run the diagram architecture using the existing fold loop; main.py stays unchanged."""
from pathlib import Path
from inspect import signature
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import main
from utils.process_args import _process_args
from dataset.SurvivalMultimodalMRIGeneDataset import SurvivalMultimodalMRIGeneDataset
from dataset.SurvivalMultimodalWSIMRIGeneDataset import SurvivalMultimodalWSIMRIGeneDataset


if __name__ == "__main__":
    args = _process_args()
    aliases = {"multimodal_wsi_mri_late_fusion": "mutation_guided_late_fusion",
               "multimodal_wsi_mri_cross_attention": "mutation_guided_cross_attention"}
    if args.mutation_file:
        args.modality = aliases.get(args.modality, args.modality)
    if args.modality not in {*aliases, *aliases.values()}:
        raise ValueError("Choose a three-modality late-fusion or cross-attention model.")
    if args.modality in aliases.values() and not args.mutation_file:
        raise ValueError("The diagram architecture requires --mutation_file.")
    options = {key: value for key, value in vars(args).items()
               if key in signature(SurvivalMultimodalMRIGeneDataset.__init__).parameters}
    dataset = SurvivalMultimodalWSIMRIGeneDataset(
        slide_file_name=args.slide_file_name, pt_dir=args.pt_dir,
        wsi_feature_dim=args.wsi_feature_dim, **options,
    )
    output = Path(os.getenv("RESULT_DIR", str(ROOT / "result"))) / args.modality / args.genomic_file_name
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    args.data_factory = dataset
    main(args, dataset)
