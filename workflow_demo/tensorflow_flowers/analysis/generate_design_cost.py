"""Create design and cost tables from the released Flowers records."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tables';OUT.mkdir(parents=True,exist_ok=True)
prep=json.loads((ROOT/'reference_results/preparation_summary.json').read_text())
cost=json.loads((ROOT/'reference_results/extraction_costs.json').read_text())
NAMES={'resnet50':'ResNet50','vgg16':'VGG16','inception_v3':'InceptionV3','mobilenet_v2':'MobileNetV2','coco_deeplab':'COCO DeepLab','ade20k_segformer':'ADE20K SegFormer','dinov2_vitb14':'DINOv2-B/14','siglip2_b16':'SigLIP 2-B/16','convnext_b':'ConvNeXt-B','vit_b16':'ViT-B/16'}
rows=[]
for k,label in NAMES.items():
    c=cost[k];peak=c.get('peak_gpu_memory_bytes')
    peak='--' if peak is None else f'{peak/2**30:.2f}'
    rows.append(label+' & '+f"{c['extraction_seconds']:.1f} & {c['images_per_second']:.1f} & "+peak+r'\\')
(OUT/'computational_cost.tex').write_text('\n'.join([r'\begin{tabular}{lrrr}',r'\toprule',r'Encoder & Seconds & Images/s & Peak GPU GiB\\',r'\midrule',*rows,r'\bottomrule',r'\end{tabular}'])+'\n')

import pandas as pd
repetitions=pd.read_csv(ROOT/'reference_results/repeated_metrics.csv').split_index.nunique()
audit=prep['audit'];split=prep['split']['counts']
values=[('JPEG images in the source archive',f"{audit['source_images']:,}"),
('Excluded cross-class duplicate rows',str(audit['excluded_cross_class_duplicate_rows'])),
('Analytical observations',f"{audit['analysis_rows']:,}"),
('Duplicate-safe groups',f"{audit['unique_analysis_groups']:,}"),
('Train / validation / locked test',' / '.join(f'{split[k]:,}' for k in ['train','validation','test'])),
('Supplementary group-safe splits',str(repetitions)),
('Paired locked-test bootstrap draws','2,000')]
(OUT/'study_design.tex').write_text('\n'.join([r'\begin{tabular}{lr}',r'\toprule',r'Quantity & Value\\',r'\midrule',*[a+' & '+b+r'\\' for a,b in values],r'\bottomrule',r'\end{tabular}'])+'\n')
