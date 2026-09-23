"""Extract ten frozen encoders without fitting any prediction head."""
import argparse, gc, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from lookagain_ml.cache import EmbeddingCache, build_cache_identity
from lookagain_ml.encoders import create_encoder, encoder_metadata
from run_flowers import ENCODERS, digest_strings

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--working-root',type=Path,required=True)
    ap.add_argument('--cache-root',type=Path,required=True)
    ap.add_argument('--model-cache-root',type=Path,required=True)
    ap.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    ap.add_argument('--batch-size',type=int,default=8)
    a=ap.parse_args()
    frame=pd.read_csv(a.working_root/'splits/tf_flowers_locked_split_v1.csv',dtype={'observation_id':str})
    paths=[a.working_root/'staged/flower_photos'/p for p in frame.relative_image_path]
    for path,expected in zip(paths,frame.image_sha256):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=str(expected).lower():raise RuntimeError('Image digest mismatch')
    cache=EmbeddingCache(a.cache_root)
    for name in ENCODERS:
        metadata=encoder_metadata(name)
        identity=build_cache_identity(metadata,row_count=len(frame),
            ordered_observation_id_sha256=digest_strings(frame.observation_id.tolist()),
            ordered_image_hash_sha256=digest_strings(frame.image_sha256.str.lower().tolist()))
        hit=cache.load(identity)
        if hit is not None:
            print(name,'verified cache reused',flush=True);continue
        encoder=create_encoder(name,pretrained=True,device=a.device,model_cache_dir=a.model_cache_root)
        if encoder.metadata.to_dict()!=metadata.to_dict():raise RuntimeError('Checkpoint identity changed')
        matrix=encoder.encode([str(p) for p in paths],batch_size=a.batch_size)
        if matrix.shape[0]!=len(frame) or not np.isfinite(matrix).all():raise RuntimeError('Invalid representation matrix')
        cache.save(identity,matrix,creation_costs=encoder.costs)
        print(name,'complete',flush=True)
        del encoder,matrix;gc.collect()
if __name__=='__main__':main()
