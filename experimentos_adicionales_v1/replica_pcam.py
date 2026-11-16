"""Réplica parcial descrita por artículo: InceptionV3 desde cero, PCam oficial.
No equivale a reproducir bit a bit histobenchmark. No usa NuCLS ni su reserva.
"""
import argparse,json,os,random,time,platform,hashlib
from pathlib import Path
import numpy as np
import torch,torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import PCAM
from torchvision import transforms as T
from sklearn.metrics import roc_auc_score,accuracy_score
from scipy.stats import t
import estadistica_cierre as st

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(path,data):
 tmp=path.with_suffix('.tmp');torch.save(data,tmp);tmp.replace(path)
def evaluate(model,loader,device):
 ys=[];ps=[];model.eval()
 with torch.no_grad():
  for x,y in loader:ys.append(y.numpy());ps.append(model(x.to(device)).softmax(1)[:,1].cpu().numpy())
 y=np.concatenate(ys);p=np.concatenate(ps);return y,p,float(roc_auc_score(y,p))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--datos',type=Path,required=True);ap.add_argument('--salida',type=Path,required=True);a=ap.parse_args();out=a.salida
 if out.is_symlink():raise ValueError('Salida enlazada')
 out.mkdir(parents=True,exist_ok=True);lock=out/'EJECUCION_ACTIVA.json'
 if lock.exists():
  q=json.loads(lock.read_text())
  if q['host']!=platform.node():raise RuntimeError('Marca de otra sesión; verificar antes de reanudar')
  try:os.kill(q['pid'],0)
  except ProcessLookupError:lock.rename(out/('bloqueo_'+str(time.time_ns())+'.json'))
  else:raise RuntimeError('Proceso activo')
 with lock.open('x') as f:json.dump(dict(host=platform.node(),pid=os.getpid()),f)
 try:
  assert torch.cuda.is_available(),'Requiere GPU';device=torch.device('cuda');torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  config=dict(arquitectura='torchvision.inception_v3',pesos='desde_cero',semillas=list(range(8301,8306)),epocas=20,batch=64,lr=.0015,weight_decay=.001,criterio='abs(media AUC_test - 0.9488)<=0.01; IC95 t entre cinco semillas descriptivo',referencia_auc=.9488,delta=.01,torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,codigo_sha256=sha(Path(__file__)),limites=['Implementación torchvision; no equivalencia bit a bit con repositorio del paper','Augmentación según texto: giro continuo180, reflejos y hue0.5; código público contiene discrepancias','Interpolación bilineal; sin estandarización ImageNet; aux_logits=False. Estas decisiones no están totalmente especificadas en artículo','IC entre semillas condicionado a partición; no sustituye bootstrap por WSI'])
  p=out/'protocolo.json'
  if p.exists():assert json.loads(p.read_text())==config,'Protocolo/entorno cambió'
  else:st.save(p,config)
  train_tf=T.Compose([T.RandomRotation(180),T.RandomHorizontalFlip(),T.RandomVerticalFlip(),T.ColorJitter(hue=.5),T.Resize((299,299),antialias=True),T.ToTensor()]);val_tf=T.Compose([T.Resize((299,299),antialias=True),T.ToTensor()])
  train=PCAM(str(a.datos),split='train',transform=train_tf,download=True);val=PCAM(str(a.datos),split='val',transform=val_tf,download=True)
  assert len(train)==262144 and len(val)==32768
  hashes={str(p.relative_to(a.datos)):sha(p) for p in a.datos.rglob('*.h5') if 'test' not in p.name};st.save(out/'datos_desarrollo_sha256.json',hashes)
  vl=DataLoader(val,batch_size=64,num_workers=0);frozen=[]
  for seed in config['semillas']:
   folder=out/str(seed);folder.mkdir(exist_ok=True);done=folder/'entrenamiento_completo.json'
   if done.exists():
    q=json.loads(done.read_text());assert sha(folder/'mejor.pt')==q['checkpoint_sha256'];frozen.append(q);continue
   random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);gen=torch.Generator().manual_seed(seed)
   model=torchvision.models.inception_v3(weights=None,num_classes=2,aux_logits=False,init_weights=False).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.001);start=1;best=-1.;history=[];last=folder/'ultimo.pt'
   if last.exists():
    ck=torch.load(last,map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['opt'])
    for state in opt.state.values():
     for k in ['exp_avg','exp_avg_sq']:
      if k in state:state[k]=state[k].to(device)
    start=ck['epoch']+1;best=ck['best'];history=ck['history'];random.setstate(ck['random']);np.random.set_state(ck['numpy']);torch.set_rng_state(ck['torch']);torch.cuda.set_rng_state_all(ck['cuda']);gen.set_state(ck['loader']);atomic(folder/'mejor.pt',ck['best_model'])
   tr=DataLoader(train,batch_size=64,shuffle=True,generator=gen,num_workers=0)
   for epoch in range(start,21):
    tic=time.perf_counter();model.train();total=0.;count=0
    for j,(x,y) in enumerate(tr):
     opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(x.to(device)),y.to(device));assert torch.isfinite(loss);loss.backward();opt.step();total+=float(loss.detach())*len(y);count+=len(y)
     if j%500==0:print('PCam',seed,epoch,j,'/',len(tr),flush=True)
    _,_,auc=evaluate(model,vl,device)
    if auc>best:best=auc;atomic(folder/'mejor.pt',model.state_dict())
    history.append(dict(epoca=epoch,loss=total/count,val_auc=auc,segundos=time.perf_counter()-tic));st.save(folder/'historia.json',history)
    atomic(last,dict(epoch=epoch,model=model.state_dict(),opt=opt.state_dict(),best=best,best_model=torch.load(folder/'mejor.pt',map_location='cpu',weights_only=True),history=history,random=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),loader=gen.get_state()));print('PCam',seed,epoch,auc,flush=True)
   q=dict(semilla=seed,val_auc=best,checkpoint_sha256=sha(folder/'mejor.pt'));st.save(done,q);frozen.append(q);del model,opt;torch.cuda.empty_cache()
  st.save(out/'congelados_antes_test.json',frozen)
  test=PCAM(str(a.datos),split='test',transform=val_tf,download=True);assert len(test)==32768;tl=DataLoader(test,batch_size=64,num_workers=0);results=[]
  for q in frozen:
   folder=out/str(q['semilla']);result=folder/'test.json'
   if result.exists():
    r=json.loads(result.read_text());assert r['checkpoint_sha256']==q['checkpoint_sha256'] and sha(folder/'test.npz')==r['pred_sha256'];results.append(r);continue
   model=torchvision.models.inception_v3(weights=None,num_classes=2,aux_logits=False,init_weights=False).to(device);model.load_state_dict(torch.load(folder/'mejor.pt',map_location='cpu',weights_only=True));y,p,auc=evaluate(model,tl,device)
   with (folder/'test.tmp').open('wb') as f:np.savez_compressed(f,y=y,p=p)
   (folder/'test.tmp').replace(folder/'test.npz');r=dict(**q,auc_test=auc,accuracy_test=accuracy_score(y,p>=.5),pred_sha256=sha(folder/'test.npz'));st.save(result,r);results.append(r);del model;torch.cuda.empty_cache()
  aucs=np.array([r['auc_test'] for r in results]);half=t.ppf(.975,4)*aucs.std(ddof=1)/np.sqrt(5)
  st.save(out/'resumen.json',dict(auc_media=aucs.mean(),IC95_semillas=[aucs.mean()-half,aucs.mean()+half],diferencia_referencia=aucs.mean()-.9488,cumple_criterio_numerico=bool(abs(aucs.mean()-.9488)<=.01),alcance='Réplica parcial con diferencias técnicas declaradas; criterio numérico no demuestra réplica exacta'))
 finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
