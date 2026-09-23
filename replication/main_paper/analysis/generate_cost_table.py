"""Generate manuscript assets from released aggregate results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
FINAL=ROOT/'aggregate_results'
DIAG=FINAL/'diagnostics'
FIGURES=ROOT/'figures'
TABLES=ROOT/'tables'
GENERATED=ROOT/'scalars'
for directory in (FIGURES,TABLES,GENERATED): directory.mkdir(parents=True,exist_ok=True)
def nn_metrics(): return pd.read_csv(FINAL/'neural_head_metrics.csv')
ENCODER_NAMES = {
    "resnet50": "ResNet50",
    "vgg16": "VGG16",
    "inception_v3": "InceptionV3",
    "mobilenet_v2": "MobileNetV2",
    "coco_deeplab": "COCO semantic",
    "ade20k_segformer": "ADE20K semantic",
    "dinov2_vitb14": "DINOv2",
    "siglip2_b16": "SigLIP 2",
    "convnext_b": "ConvNeXt",
    "vit_b16": "ViT",
}
def write_cost_table() -> None:
    records=json.loads((FINAL/'cost_records.json').read_text())
    rows = []
    for encoder in ENCODER_NAMES:
        values = records[encoder]
        first = values[0]
        parameter_count = next(
            (
                item.get("parameter_count") or item.get("cached_creation_parameter_count")
                for item in values
                if item.get("parameter_count") or item.get("cached_creation_parameter_count")
            ),
            None,
        )
        throughputs = [
            item.get("images_per_second") or item.get("cached_creation_images_per_second")
            for item in values
            if item.get("images_per_second") or item.get("cached_creation_images_per_second")
        ]
        parameter_text = "--" if parameter_count is None else f"{parameter_count / 1e6:.1f}"
        throughput_text = "--" if not throughputs else f"{np.median(throughputs):.1f}"
        rows.append(
            f"{ENCODER_NAMES[encoder]} & {parameter_text} & "
            f"{int(first['feature_dimension']):,} & {throughput_text}\\\\"
        )
    text = "\n".join(
        [
            r"\begin{tabular}{lrrr}", r"\toprule",
            r"Encoder & Parameters (M) & Dimension & Median images/s\\", r"\midrule",
            *rows, r"\bottomrule", r"\end{tabular}",
        ]
    )
    (TABLES / "generated_cost.tex").write_text(text + "\n", encoding="utf-8")
COST_NOTE = 'The seven records cover Toronto houses, OBS horses, BreaKHis, pneumonia, facial age, rice disease, and WILDS poverty; TensorFlow Flowers is reported separately. Throughput is the median of recorded per-encoder creation images/s, including historical external embeddings and package-created caches, rather than the zero extraction time of a cache-hit analysis. The horse ResNet50 record measures 151 newly extracted images, while other horse encoders measure 9,885 source images; other tasks also differ in sample size and extraction settings. Exact historical hardware equivalence is not established. Parameter counts use the first available Toronto record and its recorded counting convention; they are not medians. These values do not measure deployment latency.'
def write_cost_note():
    (GENERATED / 'cost_note.tex').write_text(COST_NOTE + '\n', encoding='utf-8')
if __name__=='__main__':
    write_cost_table()
    write_cost_note()
