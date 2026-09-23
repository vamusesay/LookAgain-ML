import numpy as np
import pytest
from lookagain_ml.inner_selected_neural_heads import fit_inner_selected_neural_heads, refit_locked_neural_heads
from lookagain_ml.neural_heads import _fit_one


def test_group_inner_selection_and_locked_refit():
    rng=np.random.default_rng(11)
    features={name:rng.normal(size=(36,7)) for name in ('resnet50','dinov2_vitb14')}
    y=features['resnet50'][:,0]+20
    groups=np.repeat(np.arange(18),2)
    configs={'tiny':dict(hidden=(4,),dropout=.2,lr=.001,weight_decay=.0001)}
    model=fit_inner_selected_neural_heads(features,y,groups,task='regression',regression_selection_metric='mae',
                                 max_components=3,configurations=configs,initialization_seeds=(101,202),max_epochs=2)
    d=model.decisions
    assert not set(groups[d['inner_fit_indices']]) & set(groups[d['inner_tune_indices']])
    assert d['selection_metric']=='mae'
    assert not d['outer_validation_outcomes_used_for_tuning']
    assert model.audit()['fit_rows']==36
    for record in model.audit()['encoders'].values():
        assert record['transformed_dimension']==3
        assert all(r['early_stopping_partition'].startswith('none; locked') for r in record['members'])
    extended={k:np.vstack([x,rng.normal(size=(8,7))]) for k,x in features.items()}
    refitted=refit_locked_neural_heads(extended,np.r_[y,np.arange(8)+20],d,fit_partition='train_plus_validation')
    assert refitted.decisions==d
    assert refitted.audit()['fit_rows']==44
    for k,epochs in d['epochs'].items():
        assert [h.best_epoch for h in refitted.heads[k].heads]==epochs
    members=refitted.predict_members(features)
    for key,pred in refitted.predict(features).items():
        assert np.allclose(pred,members[key].mean(axis=0))


def test_mae_checkpoint_loss_is_absolute_not_squared():
    rng=np.random.default_rng(3)
    a=rng.normal(size=(20,3)); b=rng.normal(size=(6,3))
    y=np.linspace(1,30,20); hold=np.array([2,4,8,16,32,64.])
    fitted=_fit_one('regression',a,y,b,hold,configuration_name='tiny',
                    configuration=dict(hidden=(4,),dropout=.2,lr=.001,weight_decay=.0001),
                    seed=101,max_epochs=1,patience=7,batch_size=1024,regression_selection_metric='mae')
    expected=np.mean(np.abs(hold-fitted.predict(b)))
    assert fitted.validation_loss==pytest.approx(expected,rel=1e-5)
    assert not np.isclose(fitted.validation_loss,np.mean((hold-fitted.predict(b))**2))


def test_locked_fit_has_no_validation_outcome_dependency():
    rng=np.random.default_rng(9); a=rng.normal(size=(12,3)); y=rng.normal(size=12)
    args=dict(configuration_name='tiny',configuration=dict(hidden=(4,),dropout=.2,lr=.001,weight_decay=.0001),
              seed=101,max_epochs=3,patience=1,batch_size=1024,fixed_epochs=True,stable_scaling=True)
    left=_fit_one('regression',a,y,a[:2],np.array([1.,2.]),**args)
    right=_fit_one('regression',a,y,a[:2],np.array([1e10,-1e10]),**args)
    assert left.best_epoch==right.best_epoch==3
    assert np.array_equal(left.predict(a),right.predict(a))


def test_classification_locked_fit_ignores_assessment_labels():
    rng=np.random.default_rng(9); a=rng.normal(size=(12,3)); y=np.tile([0,1],6)
    args=dict(configuration_name='tiny',configuration=dict(hidden=(4,),dropout=.2,lr=.001,weight_decay=.0001),
              seed=101,max_epochs=2,patience=1,batch_size=1024,fixed_epochs=True,stable_scaling=True)
    left=_fit_one('classification',a,y,a[:0],y[:0],**args)
    right=_fit_one('classification',a,y,a[:2],np.array([10,11]),**args)
    assert np.array_equal(left.predict(a),right.predict(a))
    assert np.allclose(left.predict(a).sum(axis=1),1)
