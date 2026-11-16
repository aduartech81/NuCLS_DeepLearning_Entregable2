"""Robustez, latencia y atribución exploratorias de top3 en desarrollo; no reserva."""
import argparse,sys,json,hashlib,time,importlib.util
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageFilter

def main():
 p=argparse.ArgumentParser();p.add_argument('--codigo',type=Path,required=True);p.add_argument('--nuevos',type=Path,required=True);p.add_argument('--datos',type=Path,required=True);p.add_argument('--drive',type=Path,required=True);p.add_argument('--salida',type=Path,required=True);a=p.parse_args()
 sys.path.insert(0,str(a.codigo/'modelos'));import core_v2 as core,adicionales_e2 as old
 spec=importlib.util.spec_from_file_location('modelos_nuevos',a.nuevos/'modelos/adicionales_e2.py');new=importlib.util.module_from_spec(spec);spec.loader.exec_module(new)
 roots=[a.drive/'NuCLS_E2_ADICIONALES_EXPLORATORIO/piloto',a.drive/'NuCLS_E2_TRES_NUEVOS_V1/piloto']
 names=['resnet18','mobilenetv3_small_100','efficientnet_b0','deit_tiny_patch16_224','swin_tiny_patch4_window7_224','mobilevit_xxs','mixer_s16_224','convnext_tiny','inception_v3','sequencer2d_s'];runs={}
 for name in names:
  root=roots[1] if name in new.MODELS else roots[0];rr=[]
  for seed in range(6101,6106):
   path=root/f'{name}_{seed}';r=json.loads((path/'resultado.json').read_text());rr.append((path,r))
  runs[name]=rr
 top=sorted(names,key=lambda n:(-np.mean([r['validacion']['macro_f1'] for _,r in runs[n]]),n))[:3]
 a.salida.mkdir(parents=True,exist_ok=True);groups=core.read_data(a.datos);rows=groups['validation'];mean=torch.tensor([.6392034726251884,.45917053999289786,.6277451963521469])[:,None,None];sd=torch.tensor([.2139743515891908,.22149669223009963,.1742893379690752])[:,None,None]
 recipe={'seleccion':top,'semillas_atribucion':[6101,6102],'semilla_robustez':6101,'brillo':[.8,1.2,1.4],'desenfoque_sigma':[.5,1.,2.],'ruido_sd':[.01,.03,.05],'interpretacion':'Exploratorio en validación usada para ajuste; saliencia gradiente×entrada, no Grad-CAM ni evidencia causal','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 f=a.salida/'protocolo.json'
 if f.exists() and json.loads(f.read_text())!=recipe:raise ValueError('Cambió evaluación; usa salida versionada')
 core.save_json(f,recipe)
 samples=[];seen=set()
 for i,r in enumerate(rows):
  key=(r['paciente'],r['clase'])
  if key not in seen and sum(rows[j]['clase']==r['clase'] for j in samples)<8:samples.append(i);seen.add(key)
 def tensor(i,kind='limpio',level=0):
  with Image.open(a.datos/rows[i]['archivo']) as im:
   im=im.convert('RGB')
   if kind=='desenfoque':im=im.filter(ImageFilter.GaussianBlur(level))
   ar=np.asarray(im,dtype=np.float32)/255
  if kind=='brillo':ar=ar*level
  if kind=='ruido':ar=ar+np.random.default_rng(9100+i).normal(0,level,ar.shape)
  return (torch.from_numpy(np.clip(ar,0,1).astype('float32')).permute(2,0,1)-mean)/sd
 for name in top:
  previous=None
  for seed in [6101,6102]:
   dest=a.salida/f'{name}_{seed}';dest.mkdir(exist_ok=True);checkpoint=runs[name][seed-6101][0]/'mejor.pt';h=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
   if (dest/'completo.json').exists():
    if json.loads((dest/'completo.json').read_text())['checkpoint_sha256']!=h:raise ValueError('Checkpoint cambiado')
    previous=np.load(dest/'saliencias.npy');print('EVALUACIÓN OMITIDA',name,seed,flush=True);continue
   core.seed_all(9100);model=(new.build(name) if name in new.MODELS else old.build(name)).cuda().eval();model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True));maps=[];attrib=[]
   for i in samples:
    x=tensor(i).unsqueeze(0).cuda().requires_grad_(True);logits=model(x);target=int(logits.argmax(1));grad=torch.autograd.grad(logits[0,target],x)[0];sal=(grad*x).abs().sum(1)[0];maps.append(sal.detach().cpu().numpy());order=sal.flatten().argsort(descending=True);scores=[]
    for frac in [0.,.25,.5,.75,1.]:
     mask=torch.ones(128*128,device='cuda');mask[order[:int(frac*128*128)]]=0;mask=mask.reshape(1,1,128,128)
     with torch.no_grad():scores.append([frac,float(model(x.detach()*mask).softmax(1)[0,target]),float(model(x.detach()*(1-mask)).softmax(1)[0,target])])
    v=maps[-1];heat=np.uint8(255*(v-v.min())/(np.ptp(v)+1e-12));Image.fromarray(heat).save(dest/f'saliencia_{i}.png')
    attrib.append(dict(archivo=rows[i]['archivo'],paciente=rows[i]['paciente'],real=rows[i]['clase'],predicha=core.CLASSES[target],curvas_eliminacion_insercion=scores))
   arr=np.stack(maps);np.save(dest/'saliencias.npy',arr);core.save_json(dest/'atribuciones.json',attrib)
   if previous is not None:
    correlations=[float(np.corrcoef(x.ravel(),y.ravel())[0,1]) if np.std(x)>0 and np.std(y)>0 else None for x,y in zip(previous,arr)];core.save_json(dest/'estabilidad.json',{'correlacion_pearson_mapas':correlations,'nota':'Clases predichas pueden diferir; revisar atribuciones antes de interpretar estabilidad'})
   previous=arr
   if seed==6101:
    x=tensor(0).unsqueeze(0).cuda()
    with torch.no_grad():
     for _ in range(10):model(x)
     timing=[]
     for _ in range(30):
      torch.cuda.synchronize();start=time.perf_counter();model(x);torch.cuda.synchronize();timing.append(time.perf_counter()-start)
    core.save_json(dest/'latencia.json',dict(lote=1,repeticiones=30,mediana_ms=float(np.median(timing)*1000),p95_ms=float(np.percentile(timing,95)*1000),gpu=torch.cuda.get_device_name(0),nota='Sin carga de imagen ni transferencia; no latencia clínica'))
    for kind,levels in [('limpio',[0]),('brillo',[.8,1.2,1.4]),('desenfoque',[.5,1.,2.]),('ruido',[.01,.03,.05])]:
     for level in levels:
      path=dest/f'{kind}_{level}.csv'
      if path.exists():continue
      result=[]
      for start in range(0,len(rows),32):
       with torch.no_grad():prob=model(torch.stack([tensor(i,kind,level) for i in range(start,min(start+32,len(rows)))]).cuda()).softmax(1).cpu().numpy()
       for offset,pr in enumerate(prob):
        r=rows[start+offset];result.append(dict(paciente=r['paciente'],tss=r['tss'],archivo=r['archivo'],clase_real=r['clase'],p_tumor=float(pr[0]),p_linfocito=float(pr[1]),p_estroma=float(pr[2])))
      tmp=path.with_suffix('.tmp');core.csv_write(tmp,result);tmp.replace(path);print('ROBUSTEZ',name,kind,level,flush=True)
   core.save_json(dest/'completo.json',{'checkpoint_sha256':h});del model;torch.cuda.empty_cache()
if __name__=='__main__':main()
