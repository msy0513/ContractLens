#!/usr/bin/env python3
"""Leave-one-intent-out cross-type comparison on the complete-case union.

All methods use identical held-out-only positives versus normal negatives.
The frozen CodeGemma sequence cache and pre-registered 1,168-unit MLP cache
are reused; held-out labels are never used for layer/unit selection.
"""
import csv,json,importlib.util,os
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,average_precision_score,f1_score,roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from intent_comparison_common import INTENTS,OUT,atomic_json,seed_all
ROOT=Path('/root/autodl-tmp/models')
DATA_DIR=OUT / os.environ.get('CROSS_TYPE_DATA_DIR', 'cross_type')
OUTDIR=OUT / os.environ.get('CROSS_TYPE_RESULTS_DIR', 'cross_type_results')
OUTDIR.mkdir(parents=True,exist_ok=True)
BASE=ROOT/'runs/train_new_intent_baselines_seed0.py'; sp=importlib.util.spec_from_file_location('nb',BASE); nb=importlib.util.module_from_spec(sp); sp.loader.exec_module(nb)
AdaptedDBESN,AdaptedSCMDetector=nb.AdaptedDBESN,nb.AdaptedSCMDetector
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LAYERS=(16,18,19,21); METHODS=('smartintentnn','smartintentv2','dbesn','scmdetector','smartdna')

def met(y,p):
 y=np.asarray(y);p=np.asarray(p);q=p>=.5
 return {'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'accuracy':float(accuracy_score(y,q)),'f1':float(f1_score(y,q,zero_division=0))}
def probe(X,y):
 c=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=3000,class_weight='balanced',random_state=0));c.fit(np.asarray(X,dtype=np.float32),y);return c
def predict(m,feat,mask,ix):
 m.eval();o=[]
 with torch.inference_mode():
  for st in range(0,len(ix),32):
   x=torch.from_numpy(np.asarray(feat[ix[st:st+32]],dtype=np.float32)).to(DEVICE); mm=torch.from_numpy(np.asarray(mask[ix[st:st+32]],dtype=np.bool_)).to(DEVICE)
   with torch.autocast(device_type=DEVICE.type,dtype=torch.float16,enabled=DEVICE.type=='cuda'):
    z=m(x,mm);z=z[0] if isinstance(z,tuple) else z
   o.extend(torch.sigmoid(z).float().cpu().numpy())
 return np.asarray(o)
class Seq(torch.nn.Module):
 def __init__(self,v2=False,centers=None):
  super().__init__();d=768 if v2 else 3072;self.p=torch.nn.Linear(3072,d) if v2 else torch.nn.Identity();self.l=torch.nn.LSTM(d,128,batch_first=True,bidirectional=True);self.o=torch.nn.Linear(256,1);self.c=centers;self.v2=v2
 def forward(self,x,m):
  x=x.float()
  if self.c is not None:
   xn=torch.nn.functional.normalize(x,dim=-1);cn=torch.nn.functional.normalize(self.c,dim=-1);dist=1-torch.einsum('bsd,kd->bsk',xn,cn).max(-1).values;x=torch.where((dist<.21).unsqueeze(-1),x*2.,x)
  _,(h,_)=self.l(self.p(x)*m.unsqueeze(-1).float());return self.o(torch.cat([h[-2],h[-1]],-1)).squeeze(-1)
def train_seq(v2,highlight,feat,mask,tr,yt,va,yv,te):
 centers=None
 if highlight:
  a=np.asarray(feat[tr],dtype=np.float32).reshape(-1,3072); centers=torch.tensor(MiniBatchKMeans(10,random_state=0,n_init=2,batch_size=256,max_iter=40).fit(a[np.linspace(0,len(a)-1,min(len(a),10000),dtype=int)]).cluster_centers_,device=DEVICE)
 m=Seq(v2,centers).to(DEVICE);opt=torch.optim.Adam(m.parameters(),lr=5e-4 if v2 else 1e-3);best=None;ba=-1;stale=0
 for ep in range(12):
  m.train();order=np.random.default_rng(ep).permutation(len(tr))
  for s in range(0,len(order),8):
   ids=tr[order[s:s+8]];x=torch.from_numpy(np.asarray(feat[ids],dtype=np.float32)).to(DEVICE);mm=torch.from_numpy(np.asarray(mask[ids],dtype=np.bool_)).to(DEVICE);t=torch.tensor(y_global[ids],dtype=torch.float32,device=DEVICE);opt.zero_grad();z=m(x,mm);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,t);loss.backward();opt.step()
  p=predict(m,feat,mask,va);a=roc_auc_score(y_global[va],p)
  if a>ba:ba=a;best={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};stale=0
  else:stale+=1
  if stale>=3:break
 m.load_state_dict(best);return met(y_global[va],predict(m,feat,mask,va)),met(y_global[te],predict(m,feat,mask,te))
def kneedle(s):
 o=np.argsort(-s,kind='stable');v=s[o];x=np.linspace(0,1,len(v));y=(v[0]-v)/(v[0]-v[-1]+1e-12);ii=np.arange(1,len(v)-1);return o[:int(ii[np.argmax((y-x)[ii])])+1]
def smartdna(unit,layer,tr,va,te):
 c=probe(unit[tr],y_global[tr]);sc=c.named_steps['logisticregression'].coef_[0]/c.named_steps['standardscaler'].scale_;p=c.predict_proba(unit[tr])[:,1];err=np.abs(p-y_global[tr]);activation=np.mean(np.abs(np.asarray(unit[tr],dtype=np.float32)),axis=0);grad=np.mean(err[:,None]*np.abs(sc)[None,:],axis=0);scores=np.abs(activation*grad*sc);units=kneedle(scores);r=probe(unit[tr][:,units],y_global[tr]);return met(y_global[va],r.predict_proba(unit[va][:,units])[:,1]),met(y_global[te],r.predict_proba(unit[te][:,units])[:,1]),len(units)

def main():
 global y_global
 seed_all(0);rows=[json.loads(x) for x in (OUT/'selected_union.jsonl').open() if x.strip()];y_global=np.asarray([r['label'] for r in rows],dtype=np.int64);seq=np.load(OUT/'features/last_layer_segments.f16.npy',mmap_mode='r');mask=np.load(OUT/'features/segment_mask.bool.npy',mmap_mode='r');units=np.load(OUT/'features/stable_mlp_units.f16.npy',mmap_mode='r');meta=json.loads((OUT/'features/metadata.json').read_text());offs=meta['stable_unit_offsets'];uf={l:units[:,int(offs[str(l)]['start']):int(offs[str(l)]['stop'])] for l in LAYERS}; lf={l:np.asarray(np.load(OUT/'features/all_layer_means.f16.npy',mmap_mode='r')[:,l],dtype=np.float16) for l in LAYERS}
 results=[]
 for held in INTENTS:
  p=OUTDIR/f'{held}.json'
  if p.exists(): results.append(json.loads(p.read_text()));continue
  d=json.loads((DATA_DIR/f'{held}.json').read_text()); idx={s:np.asarray([x['union_index'] for x in d['records'][s]],dtype=np.int64) for s in ('train','val','test')}
  counts={s:{'n':len(d['records'][s]),'positive':sum(int(x['label']) for x in d['records'][s]),'negative':sum(int(x['label'])==0 for x in d['records'][s])} for s in ('train','val','test')}
  if any(counts[s]['positive']==0 or counts[s]['negative']==0 for s in ('train','val','test')):
   atomic_json(p,{'held_out_intent':held,'status':'N/A','reason':'test or training split has only one class','counts':counts});continue
  tr,va,te=idx['train'],idx['val'],idx['test']; rec={'held_out_intent':held,'train_n':len(tr),'validation_n':len(va),'test_n':len(te),'counts':counts,'low_test_positive_count':counts['test']['positive']<10,'methods':{}}
  # ordinary critical-layer selection
  scores=[]
  for l in LAYERS:
   c=probe(lf[l][tr],y_global[tr]);scores.append((roc_auc_score(y_global[va],c.predict_proba(lf[l][va])[:,1]),l))
  # A layer-wise probe is used only as an internal validation selector for
  # SmartDNA; it is not reported as a comparison method.
  bl=max(scores)
  for name,v2,hi in [('smartintentnn',False,True),('smartintentv2',True,False)]:
   v,t=train_seq(v2,hi,seq,mask,tr,y_global[tr],va,y_global[va],te);rec['methods'][name]={'validation':v,'test':t}
  for name,cls in [('dbesn',AdaptedDBESN),('scmdetector',AdaptedSCMDetector)]:
   m=cls().to(DEVICE);opt=torch.optim.Adam(m.parameters(),lr=1e-5 if name=='dbesn' else 1e-4);best=None;ba=-1;stale=0
   for ep in range(12):
    m.train();ord=np.random.default_rng(ep).permutation(tr)
    for s in range(0,len(ord),8):
     ids=ord[s:s+8];x=torch.from_numpy(np.asarray(seq[ids],dtype=np.float32)).to(DEVICE);mm=torch.from_numpy(np.asarray(mask[ids],dtype=np.bool_)).to(DEVICE);target=torch.tensor(y_global[ids],dtype=torch.float32,device=DEVICE);opt.zero_grad();z=m(x,mm);z=z[0] if isinstance(z,tuple) else z;loss=torch.nn.functional.binary_cross_entropy_with_logits(z,target);loss.backward();opt.step()
    pp=predict(m,seq,mask,va);a=roc_auc_score(y_global[va],pp)
    if a>ba:ba=a;best={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};stale=0
    else:stale+=1
    if stale>=3:break
   m.load_state_dict(best);rec['methods'][name]={'validation':met(y_global[va],predict(m,seq,mask,va)),'test':met(y_global[te],predict(m,seq,mask,te))}
  dv,dt,n=smartdna(uf[bl[1]],bl[1],tr,va,te);rec['methods']['smartdna']={'selected_layer':bl[1],'kneedle_count':n,'validation':dv,'test':dt};atomic_json(p,rec);results.append(rec);print(json.dumps({'held_out':held,'layer':bl[1],'test':{k:round(v['test']['auroc'],3) for k,v in rec['methods'].items()}}),flush=True)
 rowsout=[]
 for h in INTENTS:
  d=json.loads((OUTDIR/f'{h}.json').read_text());
  if d.get('status')=='N/A':continue
  for m,v in d['methods'].items():
   if m not in METHODS:
    continue
   rowsout.append({'held_out_intent':h,'method':m,'auroc':v['test']['auroc'],'auprc':v['test']['auprc'],'accuracy':v['test']['accuracy'],'f1':v['test']['f1'],'test_n':d['test_n']})
 with (OUTDIR/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rowsout[0]));w.writeheader();w.writerows(rowsout)
 protocol = os.environ.get(
  'CROSS_TYPE_PROTOCOL',
  'leave-one-intent-out held-out-only positive vs normal',
 )
 atomic_json(OUTDIR/'summary.json',{'protocol':protocol,'results':rowsout,'maxsell_status':'N/A: no held-out-only MaxSell contracts in global union'})
 # macro plot
 fig,ax=plt.subplots(figsize=(14,6));
 for m in METHODS:
  rr=[r for r in rowsout if r['method']==m]; ax.bar(m,np.mean([r['auroc'] for r in rr])*100,label=m)
 ax.set_ylabel('Macro Test AUROC (%)');ax.set_ylim(0,100);ax.tick_params(axis='x',rotation=35);ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(OUTDIR/'cross_type_macro_auroc.png',dpi=180);plt.close(fig)
 print(json.dumps({'status':'complete','output':str(OUTDIR)}),flush=True)
if __name__=='__main__':main()
