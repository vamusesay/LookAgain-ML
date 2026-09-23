"""Verify headline values and split counts from released aggregate records."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
fixed=pd.read_csv(ROOT/'reference_results/fixed_metrics.csv').set_index('method')
for method,metric,value in [('siglip2_b16','test_accuracy',0.9836),('siglip2_b16','test_log_loss',0.0597),('linear_stack','test_accuracy',0.9836),('linear_stack','test_log_loss',0.0777)]:
    assert f'{fixed.loc[method,metric]:.4f}'==f'{value:.4f}'
rep=pd.read_csv(ROOT/'reference_results/repeated_metrics.csv')
assert rep.split_index.nunique()==12
prep=json.loads((ROOT/'reference_results/preparation_summary.json').read_text())
assert prep['audit']['source_images']==3670 and prep['audit']['excluded_cross_class_duplicate_rows']==2
assert prep['audit']['analysis_rows']==3668 and prep['audit']['unique_analysis_groups']==3666
assert prep['split']['counts']=={'train':2568,'validation':550,'test':550}
print('PASS: released aggregate reference values; model execution NOT RUN')
