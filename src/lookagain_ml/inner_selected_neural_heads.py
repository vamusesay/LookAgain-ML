"""Inner-only architecture/epoch selection and locked neural-head refitting.

The existing paper grid, AdamW fitting objectives and five initialization seeds
are retained. The caller assesses the frozen outer-training fit on outer
validation before optionally refitting the same decisions on train+validation.
No test outcomes are accepted by this API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import hashlib
import numpy as np

from .exceptions import InputValidationError
from .heads import _fit_preprocessor
from .validation_selection import _inner_holdout
from .neural_heads import (
    PAPER_NEURAL_CONFIGS, PAPER_NEURAL_SEEDS, PAPER_TUNING_SEED,
    _fit_one, _digest, FittedSeedAveragedGeluHead,
)


def array_digest(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.shape).encode() + str(a.dtype).encode() + a.tobytes()).hexdigest()


@dataclass
class InnerSelectedNeuralHeads:
    heads: dict
    transforms: dict
    decisions: dict
    audit_record: dict

    def transform(self, name, x):
        scaler, pca = self.transforms[name]
        z = scaler.transform(np.asarray(x, dtype=float))
        return pca.transform(z) if pca is not None else z

    def predict_members(self, features: Mapping[str, np.ndarray]) -> dict:
        return {name: np.stack([h.predict(self.transform(name, features[name])) for h in head.heads])
                for name, head in self.heads.items()}

    def predict(self, features):
        return {name: value.mean(axis=0) for name, value in self.predict_members(features).items()}

    def audit(self):
        return self.audit_record


def refit_locked_neural_heads(features, outcomes, decisions, *, fit_partition):
    """Refit locked architecture and per-encoder/per-seed epoch budgets.

    No validation or test outcomes are accepted separately, and no new tuning
    decisions are made. All preprocessing is refitted only on these fit rows.
    """
    heads, transforms, records = {}, {}, {}
    y = np.asarray(outcomes)
    for name, x in features.items():
        z, _, scaler, pca = _fit_preprocessor(
            x, x, max_components=decisions['max_components'], pca_iterated_power=3,
            random_state=decisions['random_state'], working_dtype='float64')
        transforms[name] = (scaler, pca)
        members = []
        for seed, epoch in zip(decisions['seeds'], decisions['epochs'][name]):
            member = _fit_one(
                decisions['task'], z, y, z[:0], y[:0],
                configuration_name=decisions['configuration_name'],
                configuration=decisions['configuration'], seed=seed,
                max_epochs=epoch, patience=decisions['patience'],
                batch_size=decisions['batch_size'],
                regression_selection_metric=decisions['selection_metric'] if decisions['task']=='regression' else 'mse',
                fixed_epochs=True, stable_scaling=True)
            members.append(member)
        heads[name] = FittedSeedAveragedGeluHead(decisions['task'], members, decisions)
        member_records = []
        for member in members:
            record = member.audit()
            record.update(early_stopping_partition='none; locked inner-selected epoch budget',
                          preprocessing_fit_partition=fit_partition)
            member_records.append(record)
        records[name] = dict(input_dimension=int(x.shape[1]), transformed_dimension=int(z.shape[1]),
                            transformed_fit_sha256=array_digest(z),
                            scaler_mean_sha256=array_digest(scaler.mean_),
                            scaler_scale_sha256=array_digest(scaler.scale_),
                            pca_components_sha256=array_digest(pca.components_) if pca is not None else None,
                            members=member_records)
    audit = dict(protocol='honest_inner_selected_neural_v1', decisions=decisions,
                 fit_partition=fit_partition, fit_rows=len(y), outcome_sha256=array_digest(y),
                 fitting_objective='standardized squared error' if decisions['task']=='regression' else 'weighted cross entropy',
                 selection_metric=decisions['selection_metric'], encoders=records,
                 test_outcomes_used=False)
    return InnerSelectedNeuralHeads(heads, transforms, decisions, audit)


def fit_inner_selected_neural_heads(features, outcomes, groups, *, task,
                           regression_selection_metric='mse', random_state=20260818,
                           inner_folds=3, max_components=128,
                           configurations=PAPER_NEURAL_CONFIGS,
                           initialization_seeds=PAPER_NEURAL_SEEDS,
                           max_epochs=70, patience=7, batch_size=1024):
    """Tune only inside outer training, then fit frozen outer-training heads."""
    y, g = np.asarray(outcomes), np.asarray(groups).astype(str)
    if task not in {'regression','classification'}:
        raise InputValidationError('Unsupported task')
    if len(g)!=len(y) or any(len(x)!=len(y) for x in features.values()):
        raise InputValidationError('Features, outcomes and groups must align')
    if regression_selection_metric not in {'mae','mse'}:
        raise InputValidationError('Unsupported regression selection metric')
    fit, tune = _inner_holdout(g, inner_folds, random_state + 8001)
    if set(g[fit]) & set(g[tune]):
        raise InputValidationError('Inner groups overlap')
    transformed = {}
    for name,x in features.items():
        a,b,_,_ = _fit_preprocessor(x[fit],x[tune],max_components=max_components,
                                   pca_iterated_power=3,random_state=random_state+8001,
                                   working_dtype='float64')
        transformed[name]=(a,b)
    rows=[]
    def train(name,cfg,seed):
        a,b=transformed[name]
        return _fit_one(task,a,y[fit],b,y[tune],configuration_name=cfg,
                        configuration=configurations[cfg],seed=seed,max_epochs=max_epochs,
                        patience=patience,batch_size=batch_size,
                        regression_selection_metric=regression_selection_metric,stable_scaling=True)
    for cfg in configurations:
        for anchor in ('resnet50','dinov2_vitb14'):
            h=train(anchor,cfg,PAPER_TUNING_SEED)
            rows.append(dict(configuration=cfg,anchor=anchor,inner_loss=h.validation_loss,best_epoch=h.best_epoch))
    means={cfg:float(np.mean([r['inner_loss'] for r in rows if r['configuration']==cfg])) for cfg in configurations}
    selected=min(means,key=means.get)
    epochs={name:[train(name,selected,int(seed)).best_epoch for seed in initialization_seeds] for name in features}
    decisions=dict(task=task,selection_metric=regression_selection_metric if task=='regression' else 'log_loss',
                   configuration_name=selected,configuration=dict(configurations[selected]),
                   seeds=list(map(int,initialization_seeds)),epochs=epochs,
                   max_components=max_components,random_state=random_state,
                   patience=patience,batch_size=batch_size,max_epochs=max_epochs,
                   architecture_grid=rows,mean_inner_loss=means,tuning_seed=PAPER_TUNING_SEED,
                   inner_fit_indices=np.flatnonzero(fit).tolist(),inner_tune_indices=np.flatnonzero(tune).tolist(),
                   inner_group_overlap=0,selection_partition='outer-training group-safe inner holdout',
                   outer_validation_outcomes_used_for_tuning=False)
    return refit_locked_neural_heads(features,y,decisions,fit_partition='outer_train')
