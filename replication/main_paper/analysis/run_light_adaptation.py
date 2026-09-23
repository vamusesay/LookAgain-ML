"""Reviewer-requested light adaptation on the locked Toronto house split.

Both models use the same geometry augmentations, optimizer, early-stopping rule,
and three seeds fixed in the implemented workflow.  The final ResNet stage, or final DINOv2 block and final layer normalization,
and a scalar head are trainable. Training mode permits batch-normalization buffers to update.
"""

from __future__ import annotations

import os
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from transformers import AutoModel


ROOT=Path(os.environ['LOOKAGAIN_ADAPTATION_OUTPUT'])
HOUSE=Path(os.environ['LOOKAGAIN_HOUSE_INPUT'])
DINO_CHECKPOINT=Path(os.environ['LOOKAGAIN_DINO_CHECKPOINT'])
SEEDS=[101,202,303]
MEAN=[.485,.456,.406]; STD=[.229,.224,.225]
TRAIN_T=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.85,1.0),interpolation=transforms.InterpolationMode.BICUBIC),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
EVAL_T=transforms.Compose([transforms.Resize(256,interpolation=transforms.InterpolationMode.BICUBIC),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])


class Houses(Dataset):
    def __init__(self,frame,tfm,ymean,ysd): self.frame=frame.reset_index(drop=True);self.tfm=tfm;self.ymean=ymean;self.ysd=ysd
    def __len__(self): return len(self.frame)
    def __getitem__(self,i):
        r=self.frame.iloc[i]
        with Image.open(r.image_path) as im: x=self.tfm(im.convert("RGB"))
        return x,torch.tensor((float(r.actual_log_price)-self.ymean)/self.ysd,dtype=torch.float32)


class ResNetRegressor(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2);dim=self.encoder.fc.in_features;self.encoder.fc=nn.Identity();self.head=nn.Linear(dim,1)
        for p in self.encoder.parameters(): p.requires_grad=False
        for p in self.encoder.layer4.parameters(): p.requires_grad=True
    def forward(self,x): return self.head(self.encoder(x)).squeeze(1)


class DinoRegressor(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=AutoModel.from_pretrained(str(DINO_CHECKPOINT),local_files_only=True);dim=self.encoder.config.hidden_size;self.head=nn.Linear(dim,1)
        for p in self.encoder.parameters(): p.requires_grad=False
        for p in self.encoder.encoder.layer[-1].parameters(): p.requires_grad=True
        if hasattr(self.encoder,"layernorm"):
            for p in self.encoder.layernorm.parameters():p.requires_grad=True
    def forward(self,x): return self.head(self.encoder(pixel_values=x).last_hidden_state[:,0]).squeeze(1)


@torch.no_grad()
def predict(model,loader,device,ymean,ysd):
    model.eval();ys=[];ps=[]
    for x,y in loader:
        p=model(x.to(device)).cpu().numpy()*ysd+ymean;ys.append(y.numpy()*ysd+ymean);ps.append(p)
    return np.concatenate(ys),np.concatenate(ps)


def run_one(model_name,seed,frame,device):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    train=frame[frame.split.eq("train")];val=frame[frame.split.eq("validation")];test=frame[frame.split.eq("test")]
    ymean=float(train.actual_log_price.mean());ysd=float(train.actual_log_price.std())
    gen=torch.Generator().manual_seed(seed)
    dltr=DataLoader(Houses(train,TRAIN_T,ymean,ysd),batch_size=64,shuffle=True,num_workers=4,pin_memory=True,generator=gen,persistent_workers=True)
    dlv=DataLoader(Houses(val,EVAL_T,ymean,ysd),batch_size=128,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True)
    dlt=DataLoader(Houses(test,EVAL_T,ymean,ysd),batch_size=128,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True)
    model=(ResNetRegressor() if model_name=="resnet50" else DinoRegressor()).to(device)
    enc=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith("head")];head=list(model.head.parameters())
    opt=torch.optim.AdamW([{"params":enc,"lr":1e-5},{"params":head,"lr":1e-4}],weight_decay=1e-4)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda");best=None;best_loss=np.inf;patience=0;history=[];start=time.time()
    for epoch in range(1,7):
        model.train();losses=[]
        for x,y in dltr:
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=device.type=="cuda"): loss=nn.functional.mse_loss(model(x),y)
            scaler.scale(loss).backward();scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
        vy,vp=predict(model,dlv,device,ymean,ysd);vl=float(mean_squared_error(vy,vp));history.append({"epoch":epoch,"train_standardized_mse":float(np.mean(losses)),"validation_mse":vl})
        if vl<best_loss-1e-5: best_loss=vl;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};patience=0
        else: patience+=1
        print(model_name,seed,"epoch",epoch,"validation_mse",round(vl,6),flush=True)
        if patience>=2:break
    model.load_state_dict(best);ty,tp=predict(model,dlt,device,ymean,ysd)
    metrics={"model":model_name,"seed":seed,"test_r2":r2_score(ty,tp),"test_mse":mean_squared_error(ty,tp),"test_mae":mean_absolute_error(ty,tp),"best_validation_mse":best_loss,"epochs":len(history),"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"total_parameters":sum(p.numel() for p in model.parameters()),"elapsed_seconds":time.time()-start,"peak_gpu_memory_gb":torch.cuda.max_memory_allocated()/1e9 if device.type=="cuda" else 0}
    pd.DataFrame({"actual":ty,"predicted":tp}).to_csv(ROOT/"fine_tuning"/f"houses_{model_name}_seed{seed}_predictions.csv",index=False)
    (ROOT/"fine_tuning"/f"houses_{model_name}_seed{seed}_history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    return metrics


def recover_one(model_name,seed,device):
    pred_path=ROOT/"fine_tuning"/f"houses_{model_name}_seed{seed}_predictions.csv"
    history_path=ROOT/"fine_tuning"/f"houses_{model_name}_seed{seed}_history.json"
    if not (pred_path.exists() and history_path.exists()):return None
    p=pd.read_csv(pred_path);hist=json.loads(history_path.read_text(encoding="utf-8"));model=(ResNetRegressor() if model_name=="resnet50" else DinoRegressor())
    return {"model":model_name,"seed":seed,"test_r2":r2_score(p.actual,p.predicted),"test_mse":mean_squared_error(p.actual,p.predicted),"test_mae":mean_absolute_error(p.actual,p.predicted),"best_validation_mse":min(x["validation_mse"] for x in hist),"epochs":len(hist),"trainable_parameters":sum(q.numel() for q in model.parameters() if q.requires_grad),"total_parameters":sum(q.numel() for q in model.parameters()),"elapsed_seconds":np.nan,"peak_gpu_memory_gb":np.nan}


if __name__=="__main__":
    frame=pd.read_csv(HOUSE/"results"/"master_sample_and_split.csv",low_memory=False);device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows=[]
    for model in ["resnet50","dinov2_vitb14"]:
        for seed in SEEDS:
            if device.type=="cuda":torch.cuda.reset_peak_memory_stats()
            print(model,seed,flush=True);recovered=recover_one(model,seed,device);rows.append(recovered if recovered is not None else run_one(model,seed,frame,device));pd.DataFrame(rows).to_csv(ROOT/"fine_tuning"/"house_light_adaptation_seed_results.csv",index=False)
    summary=pd.DataFrame(rows).groupby("model").agg({"test_r2":["mean","std"],"test_mse":["mean","std"],"test_mae":["mean","std"],"elapsed_seconds":"mean","peak_gpu_memory_gb":"max","trainable_parameters":"first","total_parameters":"first"})
    summary.to_csv(ROOT/"fine_tuning"/"house_light_adaptation_summary.csv")
    (ROOT/"fine_tuning"/"protocol.json").write_text(json.dumps({"device":str(device),"seeds":SEEDS,"epochs_max":6,"early_stopping_patience":2,"batch_size":64,"optimizer":"AdamW","encoder_lr":1e-5,"head_lr":1e-4,"weight_decay":1e-4,"augmentation":"random resized crop 0.85-1.0 plus horizontal flip; shared geometry; ImageNet normalization","trainable":"ResNet50 final residual stage and scalar head; DINOv2 final transformer block, final layer normalization, and scalar head; ResNet50 batch-normalization running statistics update-enabled in training mode","selection":"validation MSE only"},indent=2),encoding="utf-8")
