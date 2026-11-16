"""Experimentos nuevos E2, separados de fuentes y del cierre de reserva."""
import argparse,csv,json,os,platform,time,random,hashlib,zipfile,types
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import core_v2 as core
import cierre_unico as frozen
import estadistica_cierre as stats

MEAN=[.6392034726251884,.45917053999289786,.6277451963521469]
SD=[.2139743515891908,.22149669223009963,.1742893379690752]
class AblationCrops(torch.utils.data.Dataset):
 condition='control'
 def __init__(self,root,rows,augment=False):self.root,self.rows,self.augment=root,rows,augment
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(self.root/r['archivo']) as im:x=np.asarray(im).copy()
  if self.augment:
   # Draw both random numbers even if an operation is ablated.
   rot=random.randrange(4);flip=random.random()<.5
   if self.condition!='sin_rotacion':x=np.rot90(x,rot)
   if self.condition!='sin_reflejo' and flip:x=x[:,::-1]
  t=torch.from_numpy(x.copy()).permute(2,0,1).float()/255
  if self.condition!='sin_estandarizacion':t=(t-torch.tensor(MEAN)[:,None,None])/torch.tensor(SD)[:,None,None]
  return t,core.CLASSES.index(r['clase'])

def gradcam(model,x,target):
 captures=[];hooks=[]
 def hook(name):
  def capture(module,inp,out):
   if isinstance(out,torch.Tensor) and out.ndim==4 and min(out.shape[-2:])>=2:captures.append((name,out))
  return capture
 for name,module in model.named_modules():
  if isinstance(module,torch.nn.Conv2d):hooks.append(module.register_forward_hook(hook(name)))
 try:
  logits=model(x);chosen=None
  for name,act in reversed(captures):
   g=torch.autograd.grad(logits[0,target],act,retain_graph=True,allow_unused=True)[0]
   if g is not None:
    cam=torch.relu((g.mean((2,3),keepdim=True)*act).sum(1,keepdim=True));chosen=(name,cam);break
  if chosen is None:raise ValueError('No hay activación convolucional espacial conectada; Grad-CAM no aplicable')
  name,cam=chosen;cam=torch.nn.functional.interpolate(cam,size=x.shape[-2:],mode='bilinear',align_corners=False)[0,0];span=float((cam.max()-cam.min()).detach());cam=(cam-cam.min())/(span+1e-12)
  return cam.detach(),name,span
 finally:
  for h in hooks:h.remove()

def sanity(model_name,data,groups,out):
 path=out/'sanidad.json'
 if path.exists():print('OMITIDA SANIDAD COMPLETA');return
 chosen=[]
 for c in core.CLASSES:chosen.extend([r for r in groups['train'] if r['clase']==c][:11])
 core.seed_all(8201);dataset=AblationCrops(data,chosen);x=torch.stack([dataset[i][0] for i in range(len(dataset))]).cuda();y=torch.tensor([dataset[i][1] for i in range(len(dataset))]).cuda()
 model=frozen.build(model_name).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0);history=[];streak=0
 for step in range(1,301):
  model.train();opt.zero_grad(set_to_none=True);logits=model(x);loss=torch.nn.functional.cross_entropy(logits,y);loss.backward();opt.step()
  if step%10==0:
   model.eval()
   with torch.no_grad():acc=float((model(x).argmax(1)==y).float().mean())
   history.append(dict(paso=step,loss=float(loss.detach()),accuracy_lote=acc));print('SANIDAD',step,acc,flush=True);streak=streak+1 if acc>=.99 else 0
   if streak>=3:break
 stats.save(path,dict(modelo=model_name,n=len(chosen),archivos=[r['archivo'] for r in chosen],semilla=8201,criterio='accuracy>=0.99 tres controles consecutivos; máximo300 pasos',superado=streak>=3,historia=history,nota='Control positivo de capacidad/pipeline; no mide generalización ni ruido de etiquetas.'))
 del model,opt;torch.cuda.empty_cache()

def attribution(top,data,groups,runs,out):
 indices=[]
 for cls in core.CLASSES:
  candidates=[i for i,r in enumerate(groups['validation']) if r['clase']==cls];seen=set()
  for i in candidates:
   pt=groups['validation'][i]['paciente']
   if pt not in seen:indices.append(i);seen.add(pt)
   if len(seen)==8:break
 for name in top:
  previous=None;target_classes={}
  for seed in [6101,6102]:
   folder=out/'gradcam'/f'{name}_{seed}';folder.mkdir(parents=True,exist_ok=True);marker=folder/'completo.json'
   if marker.exists():
    q=json.loads(marker.read_text())
    for f,h in q['archivos'].items():assert frozen.sha(folder/f)==h
    previous=np.load(folder/'mapas.npz')['mapas'];target_classes={int(k):v for k,v in q['objetivos'].items()};continue
   run=next(r for r in runs if r['modelo']==name and r['semilla']==seed);model=frozen.build(name).cuda().eval();model.load_state_dict(torch.load(Path(run['ruta'])/'mejor.pt',weights_only=True,map_location='cpu'));maps=[];records=[]
   try:
    for i in indices:
     x=AblationCrops(data,groups['validation'])[i][0].unsqueeze(0).cuda();x.requires_grad_(True)
     if i not in target_classes:
      with torch.no_grad():target_classes[i]=int(model(x).argmax(1))
     target=target_classes[i];cam,layer,span=gradcam(model,x,target);maps.append(cam.cpu().numpy());rank=cam.flatten().argsort(descending=True);curve=[];random_rank=torch.randperm(cam.numel(),generator=torch.Generator().manual_seed(8201+i)).cuda()
     for frac in [0,.1,.25,.5,.75,1.]:
      vals=[]
      for order in [rank,random_rank]:
       mask=torch.ones(cam.numel(),device='cuda');mask[order[:int(frac*cam.numel())]]=0;mask=mask.reshape(1,1,*cam.shape)
       with torch.no_grad():vals.append(float(model(x.detach()*mask).softmax(1)[0,target]))
      curve.append(dict(fraccion=frac,prob_borrado_cam=vals[0],prob_borrado_aleatorio=vals[1]))
     Image.fromarray(np.uint8(cam.cpu().numpy()*255)).save(folder/f'cam_{i}.png')
     r=groups['validation'][i];records.append(dict(archivo=r['archivo'],paciente=r['paciente'],objetivo=target,capa=layer,rango_original=span,mapa_constante=span<=1e-12,borrado=curve))
    arr=np.stack(maps);np.savez_compressed(folder/'mapas.npz',mapas=arr);stats.save(folder/'atribuciones.json',records)
    if previous is not None:stats.save(folder/'estabilidad.json',dict(correlacion=[float(np.corrcoef(a.ravel(),b.ravel())[0,1]) if np.std(a)>0 and np.std(b)>0 else None for a,b in zip(previous,arr)],objetivo='Misma clase objetivo fijada con semilla6101',nota='Correlación descriptiva; mapas constantes no cuentan como estables'))
    stats.save(marker,dict(objetivos=target_classes,archivos={p.name:frozen.sha(p) for p in folder.iterdir() if p.is_file()},checkpoint_sha256=run['huellas']['mejor.pt']))
   except ValueError as e:stats.save(folder/'no_aplicable.json',dict(razon=str(e),modelo=name));break
   finally:del model;torch.cuda.empty_cache()
   previous=arr

def ablations(name,data,groups,runs,out):
 controls=[r for r in runs if r['modelo']==name];core.build=frozen.build;core.Crops=AblationCrops;core.summary=core.archive
 contrasts=[]
 for condition in ['sin_rotacion','sin_reflejo','sin_estandarizacion']:
  AblationCrops.condition=condition;folder=out/'ablaciones'/condition;folder.mkdir(parents=True,exist_ok=True);results=[]
  for original in controls:
   args=types.SimpleNamespace(salida=folder,datos=data,batch=32,lr=original['lr'],weight_decay=original['weight_decay'],epocas=25,min_epocas=5,paciencia=5)
   run=folder/f"{name}_{original['semilla']}";marker=run/'verificado.json'
   if marker.exists():
    for file,h in json.loads(marker.read_text()).items():assert frozen.sha(run/file)==h
   else:
    # Existing completed result without verification is checked, never silently trusted.
    core.run_one(name,original['semilla'],groups,args,torch.device('cuda'))
    rr=json.loads((run/'resultado.json').read_text());pp=frozen.read_prediction(run/'predicciones_validacion.csv',groups['validation']);yy=np.array([core.CLASSES.index(r['clase']) for r in groups['validation']]);assert abs(core.scores(yy,pp.argmax(1))['macro_f1']-rr['validacion']['macro_f1'])<1e-9
    stats.save(marker,{f:frozen.sha(run/f) for f in ['mejor.pt','resultado.json','historia.json','predicciones_validacion.csv']})
   results.append(json.loads((run/'resultado.json').read_text())['validacion']['macro_f1'])
  contrasts.append(dict(ablacion=condition,**stats.paired(np.array(results),np.array([r['validacion'] for r in controls]))))
 for key in ['t','wilcoxon','permutacion']:
  for row,p in zip(contrasts,stats.holm([r['p_'+key] for r in contrasts])):row['p_'+key+'_Holm3']=p
 
 for r in contrasts:
  for k in ['potencia_delta002_alpha0005','R_aproximado_potencia80','MDE_alpha0005']:r.pop(k,None)
 stats.table(out/'ablaciones_contrastes.csv',contrasts);AblationCrops.condition='control'


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--drive',type=Path,default=Path('/content/drive/MyDrive'));parser.add_argument('--bloques',nargs='+',choices=['sanidad','gradcam','ablaciones'],default=['sanidad','gradcam','ablaciones']);a=parser.parse_args()
 frozen.check_idle(a.drive)
 if (a.drive/'NuCLS_E2_CIERRE_RESERVA_V1/EJECUCION_ACTIVA.json').exists():raise RuntimeError('El cierre está activo; esperar')
 out=a.drive/'NuCLS_E2_EXPERIMENTOS_POST_V1'
 if out.is_symlink():raise ValueError('Salida enlazada no permitida')
 out.mkdir(exist_ok=True);lock=out/'EJECUCION_ACTIVA.json'
 if lock.exists():
  q=json.loads(lock.read_text())
  if q.get('host')!=platform.node():raise RuntimeError('Marca de otra sesión; verificar sin borrarla')
  try:os.kill(q['pid'],0)
  except ProcessLookupError:lock.rename(out/('bloqueo_previo_'+str(time.time_ns())+'.json'))
  else:raise RuntimeError('Experimento activo')
 with lock.open('x') as f:json.dump(dict(pid=os.getpid(),host=platform.node()),f)
 try:
  assert torch.cuda.is_available(),'Requiere GPU';torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  with zipfile.ZipFile(a.drive/'NuCLS_V2_DATOS.zip') as z:
   raw=z.read('manifiesto.csv');assert hashlib.sha256(raw).hexdigest()==core.EXPECTED
   rr=list(csv.DictReader(raw.decode().splitlines()));groups={s:[r for r in rr if r['particion']==s and r['estado']=='incluida'] for s in ['train','validation']};data=Path('/content/nucls_e2_post_datos');data.mkdir(exist_ok=True);frozen.extract(z,data,groups['train']+groups['validation'])
  sample_path=out/'revision_ruido_aleatoria_train.csv'
  if not sample_path.exists():
   chosen=np.random.default_rng(15112026).choice(len(groups['train']),min(300,len(groups['train'])),replace=False)
   stats.table(sample_path,[dict(archivo=groups['train'][i]['archivo'],paciente=groups['train'][i]['paciente'],etiqueta_original=groups['train'][i]['clase'],probabilidad_inclusion=len(chosen)/len(groups['train']),etiqueta_revisada='',revisor='',motivo='') for i in sorted(chosen)])
  runs,_=frozen.inventory(a.drive,groups)
  top=sorted(frozen.NAMES,key=lambda n:(-np.mean([r['validacion'] for r in runs if r['modelo']==n]),n))[:3]
  plan=dict(top3=top,mejor=top[0],semillas=frozen.SEEDS,corridas=runs,manifest=core.EXPECTED,torch=torch.__version__,timm=frozen.timm.__version__,numpy=np.__version__,codigo={p.name:frozen.sha(p) for p in Path(__file__).parent.glob('*.py')},intervenciones=['sin_rotacion','sin_reflejo','sin_estandarizacion'],nota='Selección solo validación; exploratorio posterior al desarrollo; no evalúa reserva de modelos modificados')
  p=out/'plan.json'
  if p.exists():assert json.loads(p.read_text())==plan,'Cambió plan/entorno; se conservó el resultado anterior'
  else:stats.save(p,plan)
  # Verify control predictions with the exact inference recipe before paired ablations.
  if 'ablaciones' in a.bloques:
   for r in [r for r in runs if r['modelo']==top[0]]:
    model=frozen.build(r['modelo']).cuda().eval();model.load_state_dict(torch.load(Path(r['ruta'])/'mejor.pt',weights_only=True,map_location='cpu'));y,pred=frozen.infer(model,data,groups['validation']);original=frozen.read_prediction(Path(r['ruta'])/'predicciones_validacion.csv',groups['validation']);assert np.allclose(pred,original,atol=2e-5) and np.array_equal(pred.argmax(1),original.argmax(1)),'Control no reproducido';del model;torch.cuda.empty_cache()
  if 'sanidad' in a.bloques:sanity(top[0],data,groups,out)
  if 'gradcam' in a.bloques:attribution(top,data,groups,runs,out)
  if 'ablaciones' in a.bloques:ablations(top[0],data,groups,runs,out)
  stats.save(out/'estado.json',dict(estado='BLOQUES_SOLICITADOS_COMPLETADOS',bloques=a.bloques,reserva_evaluada=False,nota='Revisar resultados de sanidad y aplicabilidad de Grad-CAM; no acredita PCam, techo o revisión humana'))
 finally:
  try:frozen.archive(out)
  finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
