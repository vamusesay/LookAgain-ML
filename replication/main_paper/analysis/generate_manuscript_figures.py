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
DATASETS = ["toronto_houses", "obs_horses", "breakhis", "pneumonia", "facial_age", "rice_disease"]
NAMES = {
    "toronto_houses": "Houses",
    "obs_horses": "Horses",
    "breakhis": "BreaKHis",
    "pneumonia": "Pneumonia",
    "facial_age": "Facial age",
    "rice_disease": "Rice disease",
}
ENCODERS = {
    "resnet50": "ResNet50",
    "vgg16": "VGG16",
    "inception_v3": "InceptionV3",
    "mobilenet_v2": "MobileNetV2",
    "coco_deeplab": "COCO semantic",
    "ade20k_segformer": "ADE20K semantic",
    "dinov2_vitb14": "DINOv2-B/14",
    "siglip2_b16": "SigLIP 2-B/16",
    "convnext_b": "ConvNeXt-B",
    "vit_b16": "ViT-B/16",
}
def metric_column(dataset: str) -> str:
    if dataset in {"toronto_houses", "obs_horses"}:
        return "test_r2"
    if dataset == "facial_age":
        return "test_mae"
    return "test_accuracy"

def main_figures() -> None:
    headline = pd.read_csv(FINAL / "fixed_headline.csv")
    fixed = pd.read_csv(FINAL / "fixed_all_methods.csv")
    nn = nn_metrics()
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))
    colors = ["#7f7f7f", "#1f77b4", "#d62728"]
    for panel, dataset in enumerate(("toronto_houses", "obs_horses")):
        axis = axes[panel]
        row = headline.loc[headline.dataset.eq(dataset)].iloc[0]

        methods = ["resnet50", "dinov2_vitb14", "siglip2_b16", row.validation_preferred_method]
        labels = ["ResNet50", "DINOv2-B/14", "SigLIP 2-B/16", "Linear stack" if row.validation_preferred_method == "linear_stack" else "Feature union"]
        intervals = json.loads((FINAL/"figure_intervals.json").read_text())[dataset]
        y_positions = np.arange(4)
        point_colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]
        for pos, (point, lower, upper), color in zip(y_positions, intervals, point_colors):
            axis.errorbar(point, pos, xerr=[[point - lower], [upper - point]], fmt="o", color=color, ecolor=color, capsize=3, lw=1.5, ms=6)
            axis.text(point + 0.012 * max(1.0, abs(point)), pos, f"{point:.3f}", va="center", fontsize=9)
        axis.set_yticks(y_positions, labels)
        axis.invert_yaxis()
        axis.set_xlabel("Untouched-test $R^2$")
        axis.set_title("Toronto houses" if panel == 0 else "OBS horses", fontweight="bold")
        axis.grid(axis="y", visible=False)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure1_motivating_contrast.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2))
    for axis, dataset in zip(axes.ravel(), DATASETS):
        row = headline.loc[headline.dataset.eq(dataset)].iloc[0]
        values = [row.resnet_test, row.selected_encoder_test, row.preferred_method_test]
        axis.bar([0, 1, 2], values, color=["#9ca3af", "#2563eb", "#f59e0b"])
        axis.set_xticks([0, 1, 2], ["ResNet", "single", "preferred"], rotation=20)
        axis.set_title(NAMES[dataset])
        axis.set_ylabel("MAE in years (lower better)" if dataset == "facial_age" else ("$R^2$" if dataset in {"toronto_houses", "obs_horses"} else "accuracy"))
    fig.tight_layout()
    fig.savefig(FIGURES / "figure2_cross_dataset.pdf", bbox_inches="tight")
    plt.close(fig)

    selected = []
    for dataset in ("toronto_houses", "obs_horses"):
        for encoder in ("resnet50", "dinov2_vitb14"):
            linear = fixed.loc[(fixed.dataset.eq(dataset)) & (fixed.method.eq(encoder)), "test_r2"].iloc[0]
            neural = nn.loc[(nn.dataset.eq(dataset)) & (nn.encoder.eq(encoder)), "test_r2"].iloc[0]
            selected.append((dataset, encoder, linear, neural))
    fig, axis = plt.subplots(figsize=(9.2, 3.25))
    x = np.arange(4)
    linear_values = [item[2] for item in selected]
    neural_values = [item[3] for item in selected]
    axis.scatter(x - 0.07, linear_values, s=55, color="#7f7f7f", label="Linear head", zorder=3)
    axis.scatter(x + 0.07, neural_values, s=55, color="#9467bd", label="NN head", zorder=3)
    for index, (linear, neural) in enumerate(zip(linear_values, neural_values)):
        axis.plot([index - 0.07, index + 0.07], [linear, neural], color="#c7c7c7", lw=1.2)
        axis.text(index - 0.09, linear + 0.014, f"{linear:.3f}", ha="right", fontsize=8)
        axis.text(index + 0.09, neural + 0.014, f"{neural:.3f}", ha="left", fontsize=8)
    axis.set_xticks(x, ["Houses\nResNet50", "Houses\nDINOv2", "Horses\nResNet50", "Horses\nDINOv2"])
    axis.set_ylabel("Untouched-test $R^2$")
    axis.legend(frameon=False, loc="upper left", ncol=2)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure3_nn_robustness.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = ["% Generated by analysis/generate_manuscript_figures.py. Do not edit."]
    house_gap = selected[1][3] - selected[0][3]
    horse_gap = selected[3][3] - selected[2][3]
    lines.append(rf"\newcommand{{\HouseNNGap}}{{{house_gap:.3f}}}")
    lines.append(rf"\newcommand{{\HorseNNGap}}{{{horse_gap:.3f}}}")
    (GENERATED / "nn_scalars.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

def correlation_figure(dataset: str, output_name: str) -> None:
    keys=list(ENCODERS)
    labels=[ENCODERS[key].replace(" semantic", "").replace("-B/14", "").replace("-B/16", "") for key in keys]
    record=json.loads((FINAL/'correlations.json').read_text())[dataset]
    matrices=[np.array(record['prediction']),np.array(record['residual'])]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))
    for axis, matrix, title in zip(axes, matrices, ("Prediction correlation", "Residual correlation")):
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
        axis.set_xticks(range(len(keys)), labels, rotation=55, ha="right", fontsize=6.5)
        axis.set_yticks(range(len(keys)), labels, fontsize=6.5)
        axis.set_title(title)
    fig.colorbar(image, ax=axes, fraction=.026, pad=.03)
    fig.subplots_adjust(left=.13, right=.93, bottom=.24, top=.9, wspace=.35)
    fig.savefig(FIGURES / output_name, bbox_inches="tight")
    plt.close(fig)
if __name__=='__main__':
    main_figures()
    correlation_figure('toronto_houses','appendix_house_correlations.pdf')
    correlation_figure('obs_horses','appendix_horse_correlations.pdf')
