"""Bloque exploratorio E2. Perfil -> 4 configuraciones/modelo -> piloto de 5 semillas.
No abre reserva. No modifica E1. No presenta el piloto como benchmark confirmatorio.
"""
import argparse,hashlib,json,os,platform,sys,time,zipfile
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'modelos'))
GRID=[{'lr':lr,'weight_decay':wd} for lr in (1e-4,1e-3) for wd in (0.,1e-4)]
SEEDS=list(range(6101,6106))
def atomic(path,data):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(path)
def frozen(path,data):
 if path.exists():
  if json.loads(path.read_text())!=data:raise ValueError('Configuración distinta: '+str(path))
 else:atomic(path,data)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pack(root):
 target=root.parent/(root.name+'_RESULTADOS.zip');tmp=target.with_suffix('.tmp')
 with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(root.rglob('*')):
   if p.is_file() and p.suffix in ('.csv','.json','.txt','.html') and p.name!='EJECUCION_ACTIVA.json':z.write(p,p.relative_to(root))
 tmp.replace(target);return target

def validate_complete(run,name,seed):
 p=run/'resultado.json'
 if not p.exists():return False
 r=json.loads(p.read_text())
 if (r['modelo'],r['semilla'],r['estado'])!=(name,seed,'FASE_COMPLETADA_VALIDACION'):raise ValueError('Resultado inconsistente: '+str(run))
 for f in ('mejor.pt','ultimo.pt','historia.json','curvas.csv','predicciones_validacion.csv','metricas_por_paciente.csv'):
  if not (run/f).is_file() or not (run/f).stat().st_size:raise ValueError('Falta evidencia: '+str(run/f))
 h=json.loads((run/'historia.json').read_text())
 if len(h)!=r['epocas_ejecutadas'] or abs(max(x['val_macro_f1'] for x in h)-r['validacion']['macro_f1'])>1e-9:raise ValueError('Historia inconsistente')
 return True

def main():
 p=argparse.ArgumentParser();p.add_argument('--datos',type=Path,required=True);p.add_argument('--salida',type=Path,required=True)
 p.add_argument('--fase',choices=['perfil','busqueda','piloto'],default='perfil')
 p.add_argument('--autorizar',default='')
 a=p.parse_args()
 if a.fase!='perfil' and a.autorizar!='EJECUTAR_TRES_NUEVOS_V1':raise ValueError('Entrenamiento bloqueado: revisar perfil y autorizar expresamente.')
 if a.salida.name!='NuCLS_E2_TRES_NUEVOS_V1':raise ValueError('Usa la carpeta exclusiva NuCLS_E2_TRES_NUEVOS_V1')
 if a.salida.is_symlink():raise ValueError('Salida enlazada no permitida')
 a.salida=a.salida.resolve();a.datos=a.datos.resolve()
 if a.salida==a.datos or a.salida in a.datos.parents or a.datos in a.salida.parents:raise ValueError('Datos/salida se solapan')
 a.salida.mkdir(parents=True,exist_ok=True)
 previo=a.salida.parent/'NuCLS_E2_ADICIONALES_EXPLORATORIO'/'EJECUCION_ACTIVA.json'
 if previo.exists():raise RuntimeError('La búsqueda anterior tiene una marca activa. No se iniciará otra ejecución ni se retirará su bloqueo.')
 # No aceptar subcarpetas enlazadas que puedan redirigir escrituras a E1.
 if any(x.is_symlink() for x in a.salida.rglob('*')):raise ValueError('Salida con enlaces simbólicos')
 lock=a.salida/'EJECUCION_ACTIVA.json'
 try:
  with lock.open('x') as f:json.dump({'pid':os.getpid(),'host':platform.node(),'hora':time.time(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip() if Path('/proc/sys/kernel/random/boot_id').exists() else None},f)
 except FileExistsError:raise RuntimeError('Ya hay un bloqueo. No lances otra copia; verifica si la ejecución anterior sigue activa.')
 try:execute(a)
 finally:lock.unlink(missing_ok=True)

def execute(a):
 import numpy as np,torch,torchvision,timm
 import core_v2 as core
 from adicionales_e2 import MODELS,build
 from torch.utils.data import DataLoader
 if not torch.cuda.is_available():raise RuntimeError('Este cuaderno requiere GPU; no inicia entrenamiento en CPU.')
 if timm.__version__!='1.0.22':raise RuntimeError('Se requiere timm==1.0.22')
 torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 device=torch.device('cuda')
 files=[Path(__file__),Path(core.__file__),Path(__file__).resolve().parents[1]/'modelos/candidatos_v2.py',Path(__file__).resolve().parents[1]/'modelos/adicionales_e2.py']
 protocol=dict(tipo='COMPLEMENTO_TRES_MODELOS_EXPLORATORIO_NO_CONFIRMATORIO',modelos=list(MODELS),semillas_piloto=SEEDS,grilla=GRID,batch=32,epocas=25,paciencia=5,min_epocas=5,pretrained=False,precision='float32',resolucion=128,manifest=core.EXPECTED,datos=str(a.datos),motor={f.name:digest(f) for f in files},reserva='NO_LEER',baseline='resnet50',delta=.02)
 frozen(a.salida/'protocolo_fijado.json',protocol)
 env=dict(torch=torch.__version__,torchvision=torchvision.__version__,timm=timm.__version__,numpy=np.__version__,python=platform.python_version(),gpu=torch.cuda.get_device_name(0))
 frozen(a.salida/'entorno_fijado.json',env)
 groups=core.read_data(a.datos)
 core.build=build
 # Evitar comprimir todo el historial tras cada época; los checkpoints aún se guardan.
 core.archive=lambda out:out
 core.summary=lambda out:out
 def run(folder,name,seed,hp):
  folder.mkdir(parents=True,exist_ok=True);dest=folder/(name+'_'+str(seed))
  if validate_complete(dest,name,seed):print('OMITIDA COMPLETA',name,seed,flush=True);return json.loads((dest/'resultado.json').read_text())
  args=argparse.Namespace(salida=folder,datos=a.datos,batch=32,epocas=25,paciencia=5,min_epocas=5,**hp)
  torch.cuda.reset_peak_memory_stats();core.run_one(name,seed,groups,args,device)
  assert validate_complete(dest,name,seed)
  atomic(dest/'memoria.json',{'max_allocated_bytes':torch.cuda.max_memory_allocated(),'max_reserved_bytes':torch.cuda.max_memory_reserved()})
  print('Respaldo:',pack(a.salida),flush=True)
  return json.loads((dest/'resultado.json').read_text())
 profile=a.salida/'perfil.json'
 if a.fase=='perfil':
  results=[];dataset=core.Crops(a.datos,groups['train'],False)
  # Una sola tanda real, igual en todos; perfil no aporta métricas para seleccionar modelos.
  x,y=next(iter(DataLoader(dataset,batch_size=32,shuffle=False)));x=x.to(device);y=y.to(device)
  for name in MODELS:
   core.seed_all(7000);model=build(name).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001)
   torch.cuda.reset_peak_memory_stats();times=[];losses=[]
   for i in range(12):
    torch.cuda.synchronize();tic=time.perf_counter();model.train();opt.zero_grad(set_to_none=True)
    logits=model(x);loss=torch.nn.functional.cross_entropy(logits,y)
    if logits.shape!=(32,3) or not torch.isfinite(loss):raise ValueError('Sanidad fallida: '+name)
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
    torch.cuda.synchronize();times.append(time.perf_counter()-tic);losses.append(float(loss.detach()))
   sec=float(np.median(times[2:]));result=dict(modelo=name,familia=MODELS[name][0],parametros=sum(p.numel() for p in model.parameters()),segundos_lote_train=sec,estimacion_segundos_epoca_solo_train=sec*455,memoria_pico_bytes=torch.cuda.max_memory_allocated(),loss_inicial=losses[0],loss_final=losses[-1],nota='12 pasos: compatibilidad; no demuestra sobreajuste de un lote ni rendimiento final.')
   results.append(result);atomic(profile,results);print(json.dumps(result),flush=True)
   del model,opt;torch.cuda.empty_cache()
  estimate=sum(r['estimacion_segundos_epoca_solo_train'] for r in results)
  atomic(a.salida/'presupuesto.json',dict(horas_solo_train_si_10_epocas=estimate*10*9/3600,horas_solo_train_max25=estimate*25*9/3600,nota='4 ajustes + 5 semillas por modelo. Añadir validación, I/O y latencia. No incluye benchmark final, robustez ni ablación.'))
  print('PERFIL COMPLETO',pack(a.salida),flush=True)
 if a.fase=='perfil':return
 if not profile.exists() or len(json.loads(profile.read_text()))!=len(MODELS):raise RuntimeError('Ejecuta primero perfil de los tres modelos.')
 selection=a.salida/'seleccion_hiperparametros.json'
 if a.fase=='busqueda':
  choices={}
  for name in MODELS:
   trials=[]
   for i,hp in enumerate(GRID):
    r=run(a.salida/'busqueda'/f'config_{i}',name,7001,hp);trials.append(r['validacion']['macro_f1'])
   # Empate: primera configuración en la grilla (lr menor y wd menor).
   i=max(range(len(trials)),key=lambda j:trials[j]);choices[name]=dict(configuracion=i,**GRID[i],valores=trials)
  frozen(selection,choices)
  print('BUSQUEDA COMPLETA',pack(a.salida),flush=True)
 if a.fase=='piloto':
  choices=json.loads(selection.read_text())
  if set(choices)!=set(MODELS):raise ValueError('Selección incompleta')
  # Semilla primero: cobertura equilibrada ante un corte de sesión.
  for seed in SEEDS:
   for name in MODELS:
    hp={k:choices[name][k] for k in ('lr','weight_decay')};run(a.salida/'piloto',name,seed,hp)
  atomic(a.salida/'estado.json',dict(estado='PILOTO_5_COMPLETADO',modelos=len(MODELS),corridas=len(MODELS)*len(SEEDS),reserva_evaluada=False,pendiente='Dimensionar R con potencia; referencias simples, replicación, robustez, interpretabilidad y ablación.'))
  print('PILOTO COMPLETO',pack(a.salida),flush=True)
if __name__=='__main__':main()
