"""Fit the neural-head workflow using its fixed configuration and authorized local inputs."""
import os, sys, json, hashlib, pickle
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(os.environ['LOOKAGAIN_ANALYSIS_ROOT'])
WORK=Path(os.environ['LOOKAGAIN_OUTPUT_ROOT'])
PACKAGE=Path(__file__).resolve().parents[3]
from _common import load_config, load_manifest
from _reference_features import known_reference_specifications, load_reference_features
from run_nested_protocol import ENCODERS, load_toronto_features
from lookagain_ml.splitting import validate_partition_safety
from lookagain_ml.metrics import regression_metrics, classification_metrics
from lookagain_ml.inner_selected_neural_heads import fit_inner_selected_neural_heads, refit_locked_neural_heads, array_digest
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str,allow_nan=False),encoding='utf8')

def digest_list(x):return hashlib.sha256('\n'.join(map(str,x)).encode()).hexdigest()

def run(dataset):
    config=load_config(ROOT/'configs/representation_risk.full.json')
    spec=next(s for s in config['datasets'] if s['name']==dataset)
    f=load_manifest(Path(spec['manifest']))
    y=f.outcome.to_numpy(float) if spec['task']=='regression' else None
    if spec['task']=='classification':
        classes=sorted(f.outcome.astype(str).unique())
        y=f.outcome.astype(str).map({c:i for i,c in enumerate(classes)}).to_numpy(int)
    if dataset=='toronto_houses':
        features,verification=load_toronto_features(f,Path(os.environ['LOOKAGAIN_TORONTO_CACHE']))
    else:
        refs=known_reference_specifications(spec['reference_embedding_layout'],config['research_root'],dataset=spec.get('reference_dataset'))
        features,_,verification=load_reference_features(f,{e:refs[e] for e in ENCODERS})
    g=f.group_id.astype(str).to_numpy(); labels=f.split.to_numpy()
    validate_partition_safety(labels,g,f.image_sha256.astype(str).to_numpy())
    train=labels=='train'; val=labels=='validation'; test=labels=='test'; refit=train|val
    output=WORK/'outputs'/dataset;output.mkdir(exist_ok=True,parents=True)
    f.to_csv(output/'manifest.private.csv',index=False)
    source_identity={p.name:sha(p) for p in (PACKAGE/'src/lookagain_ml').glob('*.py')}
    common=dict(dataset=dataset,package_base_commit='877908c916573109b2106cd5fec82ec83a6153de',
                source_module_hashes=source_identity,configuration_sha256=sha(ROOT/'configs/representation_risk.full.json'),
                manifest_sha256=sha(output/'manifest.private.csv'),ids_sha256=digest_list(f.observation_id),
                outcome_sha256=array_digest(y),group_sha256=digest_list(g),split_sha256=digest_list(labels),
                groups=int(len(np.unique(g))),rows=len(f),feature_verification=verification,
                checkpoint_source='verified frozen embeddings')
    print(json.dumps(dict(dataset=dataset,stage='verified features; inner tuning')),flush=True)
    fitted=fit_inner_selected_neural_heads({e:x[train] for e,x in features.items()},y[train],g[train],task=spec['task'],
                                   regression_selection_metric='mae' if dataset=='facial_age' else 'mse')
    validation=fitted.predict_members({e:x[val] for e,x in features.items()})
    write(output/'outer_training_audit.json',{**common,**fitted.audit(),
          'fit_observation_ids':f.loc[train,'observation_id'].tolist(),
          'outer_validation_observation_ids':f.loc[val,'observation_id'].tolist()})
    np.savez_compressed(output/'validation_member_predictions.npz',**validation)
    with open(output/'outer_training_models.private.pkl','wb') as out:pickle.dump(fitted,out)
    print(json.dumps(dict(dataset=dataset,stage='locked train+validation refit',architecture=fitted.decisions['configuration_name'])),flush=True)
    final=refit_locked_neural_heads({e:x[refit] for e,x in features.items()},y[refit],fitted.decisions,fit_partition='train_plus_validation')
    members=final.predict_members({e:x[test] for e,x in features.items()})
    np.savez_compressed(output/'test_member_predictions.npz',**members)
    with open(output/'final_models.private.pkl','wb') as out:pickle.dump(final,out)
    averaged={e:p.mean(axis=0) for e,p in members.items()}
    np.savez_compressed(output/'neural_head_predictions.npz',**averaged)
    compute=regression_metrics if spec['task']=='regression' else classification_metrics
    records=[]
    for enc,pp in averaged.items():
        metric=compute(y[test],pp)
        seedmetrics=[compute(y[test],p) for p in members[enc]]
        records.append(dict(dataset=dataset,encoder=enc,**{'test_'+k:v for k,v in metric.items()},
            within_five_years=float(np.mean(np.abs(y[test]-pp)<=5)) if dataset=='facial_age' else None,
            seed_mae_sd=float(np.std([m['mae'] for m in seedmetrics],ddof=1)) if spec['task']=='regression' else None,
            seed_count=5,seed_r2_sd=float(np.std([m['r2'] for m in seedmetrics],ddof=1)) if spec['task']=='regression' else None))
    pd.DataFrame(records).to_csv(output/'neural_head_metrics.csv',index=False)
    write(output/'final_audit.json',{**common,**final.audit(),
          'fit_observation_ids':f.loc[refit,'observation_id'].tolist(),
          'test_observation_ids':f.loc[test,'observation_id'].tolist(),
          'prediction_file_sha256':sha(output/'test_member_predictions.npz'),
          'model_file_sha256':sha(output/'final_models.private.pkl'),
          'seed_aggregation':'metric from mean predictions; sample SD of seed-specific metrics separate'})
    print(json.dumps(dict(dataset=dataset,stage='complete',metrics=records)),flush=True)
if __name__=='__main__':run(sys.argv[1])
