#!/usr/bin/env python3
"""Token-level causal intervention on validation-locked CodeGemma MLP units."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,f1_score
from transformers import AutoModel,AutoTokenizer


def load_jsonl(p):return [json.loads(x) for x in open(p,encoding='utf-8') if x.strip()]
def atomic_json(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
 with open(q,'w',encoding='utf-8') as f:json.dump(v,f,indent=2)
 os.replace(q,p)
def metrics(y,p):
 pred=p>=.5;return {'accuracy':float(accuracy_score(y,pred)),'auroc':float(roc_auc_score(y,p)),
 'auprc':float(average_precision_score(y,p)),'f1':float(f1_score(y,pred))}
def windows(tok,text,max_length,overlap):
 ids=tok.encode(text,add_special_tokens=False,verbose=False) or [tok.unk_token_id or tok.eos_token_id]
 pre=[tok.bos_token_id] if tok.bos_token_id is not None else [];suf=[tok.eos_token_id] if tok.eos_token_id is not None else []
 budget=max_length-len(pre)-len(suf);step=budget-overlap;out=[]
 for start in range(0,len(ids),step):
  c=ids[start:start+budget];out.append((pre+c+suf,len(pre),len(c),0 if start==0 else min(overlap,len(c))))
  if start+budget>=len(ids):break
 return len(ids),out
def specificity_ci(y,base,top,randoms,seed,repeats=1000):
 rng=np.random.default_rng(seed);vals=[];n=len(y);randoms=np.asarray(randoms)
 for _ in range(repeats):
  ix=rng.integers(0,n,n);yy=y[ix]
  if np.unique(yy).size<2:continue
  rp=randoms[int(rng.integers(0,len(randoms)))]
  vals.append((roc_auc_score(yy,base[ix])-roc_auc_score(yy,top[ix]))-(roc_auc_score(yy,base[ix])-roc_auc_score(yy,rp[ix])))
 a=np.asarray(vals);return {'mean':float(a.mean()),'lower':float(np.percentile(a,2.5)),'upper':float(np.percentile(a,97.5))}


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage10',default='runs/stage10_codegemma_seed0_stable_units/summary.json')
 ap.add_argument('--model',default='base/CodeGemma-7B');ap.add_argument('--stage9',default='runs/stage9_codegemma_seed0_dna_mlp')
 ap.add_argument('--hidden',default='runs/stage5_smartintent_gemma_eos_overlap_corrected/hidden/features_token_exact_mean')
 ap.add_argument('--union',default='runs/stage3_smartintent_sliding/sliding_union.jsonl');ap.add_argument('--binary',default='runs/stage3_smartintent_sliding/sliding_tasks/binary.jsonl')
 ap.add_argument('--manifest',default='runs/stage4_smartintent_full_trials/manifests/trial_seed_0.json')
 ap.add_argument('--output',default='runs/stage12_codegemma_seed0_true_intervention');ap.add_argument('--random-repeats',type=int,default=10)
 ap.add_argument('--max-length',type=int,default=2048);ap.add_argument('--overlap',type=int,default=512);ap.add_argument('--device',default='cuda');args=ap.parse_args();out=Path(args.output)
 s10=json.load(open(args.stage10));eligible={int(r['layer']):np.asarray(r['stable_units'],dtype=np.int64) for r in s10['results'] if r['passes_sufficiency'] and r['passes_necessity_validation'] and r['stable_unit_count']}
 if not eligible:atomic_json(out/'summary.json',{'status':'no_validation_eligible_units'});print('no eligible units');return
 union=load_jsonl(args.union);binary=load_jsonl(args.binary);pos={int(r['feature_index']):i for i,r in enumerate(binary)}
 records=json.load(open(args.manifest))['records'];ix={};y={}
 for split in ['train','val','test']:
  rr=[r for r in records if r['split']==split];ix[split]=np.asarray([int(r['feature_index']) for r in rr]);y[split]=np.asarray([int(r['label']) for r in rr])
 final=np.load(Path(args.hidden)/'layer_28.npy',mmap_mode='r')
 clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced',random_state=0))
 clf.fit(np.asarray(final[ix['train']],dtype=np.float32),y['train'])
 base_val_p=clf.predict_proba(np.asarray(final[ix['val']],dtype=np.float32))[:,1];base_test_p=clf.predict_proba(np.asarray(final[ix['test']],dtype=np.float32))[:,1]
 tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True,use_fast=True)
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 model=AutoModel.from_pretrained(args.model,local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True)
 model.requires_grad_(False);model.eval();model.config.use_cache=False;model.to(args.device)
 if sum(p.numel() for p in model.parameters() if p.requires_grad) or model.training:raise RuntimeError('freeze failed')
 state={'units':None,'replacement':None,'content':None}
 def hook(_m,inp):
  x=inp[0];z=x.clone();mask=state['content'];u=state['units'];rep=state['replacement']
  if u is not None and len(u):
   content=z[:,mask,:].clone();values=rep.view(1,1,-1).expand(content.shape[0],content.shape[1],-1)
   content.index_copy_(2,u,values);z[:,mask,:]=content
  return (z,)
 results=[]
 for layer,units in eligible.items():
  handle=model.layers[layer-1].mlp.down_proj.register_forward_pre_hook(hook)
  mean=np.load(Path(args.stage9)/f'layer_{layer:02d}'/'mean_activation.npy')
  def run_rows(feature_indices,chosen):
   state['units']=torch.tensor(chosen,dtype=torch.long,device=args.device);state['replacement']=torch.tensor(mean[chosen],dtype=torch.bfloat16,device=args.device)
   features=[]
   with torch.inference_mode():
    for ri,fi in enumerate(feature_indices):
     expected,chunks=windows(tok,str(union[int(fi)]['text']),args.max_length,args.overlap);total=np.zeros(3072,dtype=np.float64);count=0
     for ids,off,n,unique in chunks:
      x=torch.tensor(ids,dtype=torch.long,device=args.device)[None,:];mask=torch.ones_like(x);content=torch.zeros(len(ids),dtype=torch.bool,device=args.device);content[off:off+n]=True;state['content']=content
      h=model(input_ids=x,attention_mask=mask).last_hidden_state[0];begin=off+unique;end=off+n;total+=h[begin:end].float().sum(0).cpu().numpy();count+=end-begin
     if count!=expected:raise RuntimeError('token accounting')
     features.append(total/count)
     if (ri+1)%20==0:print(json.dumps({'phase':'true_intervention','layer':layer,'processed':ri+1,'total':len(feature_indices)}),flush=True)
   return clf.predict_proba(np.asarray(features,dtype=np.float32))[:,1]
  top_val=run_rows(ix['val'],units);rng=np.random.default_rng(30000+layer);rin=[];rex=[];non=np.setdiff1d(np.arange(len(mean)),units)
  for r in range(args.random_repeats):
   ru=rng.choice(len(mean),size=len(units),replace=False);ex=rng.choice(non,size=len(units),replace=False)
   rin.append(run_rows(ix['val'],ru));rex.append(run_rows(ix['val'],ex));atomic_json(out/f'layer_{layer:02d}_checkpoint.json',{'random_completed':r+1,'random_total':args.random_repeats})
  ci=specificity_ci(y['val'],base_val_p,top_val,rin,layer)
  layer_result={'layer':layer,'stable_units':len(units),'baseline_validation':metrics(y['val'],base_val_p),'top_intervention_validation':metrics(y['val'],top_val),
   'random_in_validation_auroc_mean':float(np.mean([roc_auc_score(y['val'],p) for p in rin])),'random_ex_validation_auroc_mean':float(np.mean([roc_auc_score(y['val'],p) for p in rex])),
   'paired_specificity_ci_validation':ci,'passes_true_intervention':bool(ci['lower']>0)}
  if ci['lower']>0:
   top_test=run_rows(ix['test'],units);layer_result['baseline_test_development_only']=metrics(y['test'],base_test_p);layer_result['top_intervention_test_development_only']=metrics(y['test'],top_test)
  results.append(layer_result);atomic_json(out/f'layer_{layer:02d}_result.json',layer_result);handle.remove()
 model.cpu();torch.cuda.empty_cache();atomic_json(out/'summary.json',{'status':'complete','frozen_backbone':True,'downstream_probe':'Layer 28 exact-token mean hidden state','eligible_layers':sorted(eligible),'results':results})
 print(json.dumps({'status':'complete','output':str(out)}),flush=True)


if __name__=='__main__':main()
