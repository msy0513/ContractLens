#!/usr/bin/env python3
"""Surface-invariance and local counterfactual validation of locked units."""
import argparse, csv, json, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def load_jsonl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def atomic_json(p, v):
    p.parent.mkdir(parents=True, exist_ok=True); q=p.with_suffix(p.suffix+'.tmp')
    with open(q,'w',encoding='utf-8') as f: json.dump(v,f,indent=2)
    os.replace(q,p)


def write_csv(p, rows):
    if not rows:return
    p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def windows(tok,text,max_length,overlap):
    ids=tok.encode(text,add_special_tokens=False,verbose=False) or [tok.unk_token_id or tok.eos_token_id]
    pre=[tok.bos_token_id] if tok.bos_token_id is not None else []
    suf=[tok.eos_token_id] if tok.eos_token_id is not None else []
    budget=max_length-len(pre)-len(suf); step=budget-overlap; out=[]
    for start in range(0,len(ids),step):
        content=ids[start:start+budget]
        out.append((pre+content+suf,len(pre),len(content),0 if start==0 else min(overlap,len(content))))
        if start+budget>=len(ids):break
    return len(ids),out


def eligible_units(stage10):
    d=json.load(open(stage10,encoding='utf-8')); out={}
    for r in d['results']:
        if r['passes_sufficiency'] and r['passes_necessity_validation'] and r['stable_unit_count']:
            out[int(r['layer'])]=np.asarray(r['stable_units'],dtype=np.int64)
    return out


def extract(model,tok,rows,field,units,out_dir,max_length,overlap,device):
    out_dir.mkdir(parents=True,exist_ok=True); stores={}
    for layer,u in units.items():
        stores[layer]=np.lib.format.open_memmap(out_dir/f'layer_{layer:02d}.npy',mode='w+',dtype=np.float32,shape=(len(rows),len(u)))
    cache={}; hooks=[]
    for layer,u in units.items():
        idx=torch.tensor(u,dtype=torch.long,device=device)
        def mk(l,ii):
            def hook(_m,inp):cache[l]=inp[0].detach().index_select(-1,ii)
            return hook
        hooks.append(model.layers[layer-1].mlp.down_proj.register_forward_pre_hook(mk(layer,idx)))
    try:
        with torch.inference_mode():
            for ri,row in enumerate(rows):
                expected,chunks=windows(tok,str(row[field]),max_length,overlap)
                sums={l:np.zeros(len(u),dtype=np.float64) for l,u in units.items()}; count=0
                for ids,off,n,unique in chunks:
                    x=torch.tensor(ids,dtype=torch.long,device=device)[None,:]; mask=torch.ones_like(x)
                    cache.clear(); model(input_ids=x,attention_mask=mask)
                    begin=off+unique;end=off+n;count+=end-begin
                    for l in units:sums[l]+=cache[l][0,begin:end].float().sum(0).cpu().numpy()
                if count!=expected:raise RuntimeError(f'token mismatch {ri}: {count}!={expected}')
                for l in units:stores[l][ri]=sums[l]/count;stores[l].flush()
                if (ri+1)%20==0 or ri+1==len(rows):
                    atomic_json(out_dir/'checkpoint.json',{'completed':ri+1,'total':len(rows),'field':field})
                    print(json.dumps({'phase':'extract','field':field,'processed':ri+1,'total':len(rows)}),flush=True)
    finally:
        for h in hooks:h.remove()


def cosine_rows(a,b):
    return np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-12)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage10',default='runs/stage10_codegemma_seed0_stable_units/summary.json')
    ap.add_argument('--model',default='base/CodeGemma-7B')
    ap.add_argument('--stage9',default='runs/stage9_codegemma_seed0_dna_mlp')
    ap.add_argument('--binary',default='runs/stage3_smartintent_sliding/sliding_tasks/binary.jsonl')
    ap.add_argument('--surface',default='runs/stage8_codegemma_seed0_binary_cf_inv_v3/data_inv/surface_sources_selected.jsonl')
    ap.add_argument('--surface-manifest',default='runs/stage8_codegemma_seed0_binary_cf_inv_v3/data_inv/surface_manifest_selected.jsonl')
    ap.add_argument('--cf',default='runs/stage8_codegemma_seed0_binary_cf_inv_v3/data_cf/paired_sources.jsonl')
    ap.add_argument('--cf-manifest',default='runs/stage8_codegemma_seed0_binary_cf_inv_v3/data_cf/pair_manifest.jsonl')
    ap.add_argument('--output',default='runs/stage11_codegemma_seed0_unit_semantics')
    ap.add_argument('--max-length',type=int,default=2048);ap.add_argument('--overlap',type=int,default=512)
    ap.add_argument('--device',default='cuda');args=ap.parse_args();out=Path(args.output)
    units=eligible_units(args.stage10)
    if not units:
        atomic_json(out/'summary.json',{'status':'no_validation_eligible_units','stage10':args.stage10});print('no eligible units');return
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True,use_fast=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModel.from_pretrained(args.model,local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True)
    model.requires_grad_(False);model.eval();model.config.use_cache=False;model.to(args.device)
    if sum(p.numel() for p in model.parameters() if p.requires_grad) or model.training:raise RuntimeError('freeze failed')
    surface=load_jsonl(args.surface); cf=load_jsonl(args.cf)
    extract(model,tok,surface,'text',units,out/'surface_features',args.max_length,args.overlap,args.device)
    extract(model,tok,cf,'focus_text',units,out/'cf_focus_features',args.max_length,args.overlap,args.device)
    del model;torch.cuda.empty_cache()
    binary=load_jsonl(args.binary);pos={int(r['feature_index']):i for i,r in enumerate(binary)}
    sm=load_jsonl(args.surface_manifest);cm=load_jsonl(args.cf_manifest)
    results=[]
    for layer,u in units.items():
        original=np.load(Path(args.stage9)/'mlp_features'/f'layer_{layer:02d}.npy',mmap_mode='r')
        surf=np.load(out/'surface_features'/f'layer_{layer:02d}.npy',mmap_mode='r')
        orig_s=np.asarray([original[pos[int(m['original_feature_index'])],u] for m in sm],dtype=np.float32)
        surf_s=np.asarray(surf,dtype=np.float32)
        cos=cosine_rows(orig_s,surf_s); surface_l2=np.linalg.norm(orig_s-surf_s,axis=1)
        focus=np.load(out/'cf_focus_features'/f'layer_{layer:02d}.npy',mmap_mode='r')
        pair_rows=[];intent_values=defaultdict(list);unit_cf=[];unit_surface=[]
        for m in cm:
            a=np.asarray(focus[int(m['original_source_index'])]);b=np.asarray(focus[int(m['counterfactual_source_index'])])
            ds=[]
            for v in m['local_surface_variants']:
                ds.append(np.abs(a-np.asarray(focus[int(v['source_index'])])))
            cf_abs=np.abs(a-b);sf_abs=np.mean(ds,axis=0) if ds else np.zeros_like(cf_abs)
            unit_cf.append(cf_abs);unit_surface.append(sf_abs)
            ratio=float(np.linalg.norm(cf_abs)/(np.mean([np.linalg.norm(x) for x in ds])+1e-8)) if ds else None
            pair_rows.append({'pair_id':m['pair_id'],'split':m['split'],'active_intents':'|'.join(m['active_intents']),
                              'cf_l2':float(np.linalg.norm(cf_abs)),'surface_l2_mean':float(np.mean([np.linalg.norm(x) for x in ds])) if ds else 0.,'ratio':ratio})
            for intent in m['active_intents']:intent_values[intent].append(ratio)
        unit_cf=np.mean(unit_cf,axis=0);unit_surface=np.mean(unit_surface,axis=0);unit_ratio=unit_cf/(unit_surface+1e-8)
        unit_rows=[{'layer':layer,'unit':int(unit),'mean_cf_abs_change':float(unit_cf[j]),
                    'mean_surface_abs_change':float(unit_surface[j]),'cf_surface_ratio':float(unit_ratio[j])}
                   for j,unit in enumerate(u)]
        write_csv(out/f'layer_{layer:02d}_unit_semantics.csv',sorted(unit_rows,key=lambda r:-r['cf_surface_ratio']))
        write_csv(out/f'layer_{layer:02d}_pair_semantics.csv',pair_rows)
        results.append({'layer':layer,'stable_units':len(u),'surface_cosine_mean':float(np.mean(cos)),
                        'surface_cosine_p05':float(np.percentile(cos,5)),'surface_l2_mean':float(np.mean(surface_l2)),
                        'cf_surface_ratio_median':float(np.median([r['ratio'] for r in pair_rows if r['ratio'] is not None])),
                        'intent_ratio_mean':{k:float(np.mean(v)) for k,v in intent_values.items()},
                        'units_cf_surface_ratio_gt_1':int(np.sum(unit_ratio>1.0)),
                        'units_cf_surface_ratio_gt_1p5':int(np.sum(unit_ratio>1.5))})
    atomic_json(out/'summary.json',{'status':'complete','frozen_backbone':True,'eligible_layers':sorted(units),
        'surface_variants':len(surface),'counterfactual_source_variants':len(cf),'counterfactual_pairs':len(cm),
        'selection_uses_semantic_data':False,'results':results})
    print(json.dumps({'status':'complete','output':str(out)}),flush=True)


if __name__=='__main__':main()
