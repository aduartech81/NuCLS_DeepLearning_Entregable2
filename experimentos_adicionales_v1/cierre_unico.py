"""Cierre E2 por inferencia. Ningún optimizador, fit ni entrenamiento profundo."""
import os,sys,json,csv,hashlib,zipfile,shutil,time,platform,datetime,importlib.util,html,traceback
from pathlib import Path
import numpy as np
import torch,torchvision,timm
from PIL import Image
from torch.utils.data import DataLoader
import core_v2 as core
import candidatos_v2 as bases
import modelos_siete as siete
import modelos_tres as nuevos
import estadistica_cierre as st
BASES=['softmaxcnn_bn','densenet121','resnet50']
ADICIONALES=['resnet18','mobilenetv3_small_100','efficientnet_b0','deit_tiny_patch16_224','swin_tiny_patch4_window7_224','mobilevit_xxs','mixer_s16_224','convnext_tiny','inception_v3','sequencer2d_s']
NAMES=BASES+ADICIONALES
SEEDS=list(range(6101,6106))
EXPECTED=core.EXPECTED

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_npz(path,**a):
 tmp=path.with_suffix('.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**a)
 tmp.replace(path)
def archive(out):
 target=out.parent/(out.name+'_RESULTADOS.zip');tmp=target.with_suffix('.tmp')
 with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(out.rglob('*')):
   if p.is_file() and p.name!='EJECUCION_ACTIVA.json' and p.suffix in ['.json','.csv','.npz','.html','.png','.txt','.py']:
    z.write(p,str(p.relative_to(out)))
 tmp.replace(target);print('RESPALDO:',target,flush=True)
def check_idle(drive):
 for folder in ['NuCLS_E2_ADICIONALES_EXPLORATORIO','NuCLS_E2_TRES_NUEVOS_V1','NuCLS_E2_COLA_UNICA_V1']:
  p=drive/folder/'EJECUCION_ACTIVA.json'
  if p.exists():raise RuntimeError('Marca de ejecución presente: '+str(p)+'. No se borró. Ejecutar cierre solo tras terminar la cola.')
 proc=Path('/proc')
 if not proc.exists():raise RuntimeError('Este cierre de producción exige Colab/Linux.')
 for p in proc.iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:parts=(p/'cmdline').read_bytes().split(b'\0')
  except OSError:continue
  if any(Path(x.decode(errors='replace')).name in ['adicionales_e2.py','piloto_siete.py','continuar_semillas.py','evaluacion_desarrollo.py'] for x in parts):raise RuntimeError('Proceso de desarrollo todavía activo; no se interviene.')

def read_prediction(path,rows):
 with Path(path).open() as f:rr=list(csv.DictReader(f))
 mapping={r['archivo']:r for r in rr}
 if len(mapping)!=len(rr) or set(mapping)!={r['archivo'] for r in rows}:raise ValueError('Identidad de predicciones incompatible: '+str(path))
 probs=[]
 for row in rows:
  q=mapping[row['archivo']]
  if q['paciente']!=row['paciente'] or q['clase_real']!=row['clase']:raise ValueError('Paciente/etiqueta incompatible')
  probs.append([float(q[k]) for k in ['p_tumor','p_linfocito','p_estroma']])
 p=np.asarray(probs)
 if not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('Probabilidades inválidas')
 return p

def inventory(drive,groups):
 root=drive/'NuCLS_E2_CONTINUACION_BASES_22';idx=root/'indice_validacion.csv'
 if not idx.is_file():idx=root/'final/indice_validacion.csv'
 if not idx.is_file():raise FileNotFoundError('No se encontró índice de las bases en este Drive. No se monta otra cuenta automáticamente.')
 with idx.open() as f:entries=list(csv.DictReader(f))
 mapping={}
 for q in entries:
  k=(q['modelo'],int(q['semilla']))
  if k in mapping:raise ValueError('Índice de bases duplicado')
  mapping[k]=q
 result=[];missing=[]
 for name in NAMES:
  for seed in SEEDS:
   if name in BASES:
    item=mapping.get((name,seed));folder=Path(item['ruta']) if item else None
   else:folder=drive/('NuCLS_E2_TRES_NUEVOS_V1' if name in nuevos.MODELS else 'NuCLS_E2_ADICIONALES_EXPLORATORIO')/'piloto'/f'{name}_{seed}'
   files=['resultado.json','historia.json','predicciones_validacion.csv','mejor.pt']
   if folder is None or any(not (folder/f).is_file() for f in files):missing.append(f'{name}_{seed}');continue
   q=json.loads((folder/'resultado.json').read_text());history=json.loads((folder/'historia.json').read_text())
   if name in BASES and item.get('resultado_sha256') and sha(folder/'resultado.json')!=item['resultado_sha256']:raise ValueError('Cambió resultado del índice de bases')
   if (q['modelo'],q['semilla'],q['estado'])!=(name,seed,'FASE_COMPLETADA_VALIDACION'):raise ValueError('Identidad incompatible '+str(folder))
   if len(history)!=q['epocas_ejecutadas'] or abs(max(h['val_macro_f1'] for h in history)-q['validacion']['macro_f1'])>1e-9:raise ValueError('Historia incompatible')
   p=read_prediction(folder/'predicciones_validacion.csv',groups['validation']);y=np.array([core.CLASSES.index(r['clase']) for r in groups['validation']])
   if abs(core.scores(y,p.argmax(1))['macro_f1']-q['validacion']['macro_f1'])>1e-8:raise ValueError('Métricas de validación incoherentes')
   result.append(dict(modelo=name,semilla=seed,ruta=str(folder),huellas={f:sha(folder/f) for f in files},validacion=q['validacion']['macro_f1'],lr=q['lr'],weight_decay=q['weight_decay'],mejor_epoca=q['mejor_epoca']))
 if missing:raise RuntimeError('No se abre reserva. Faltan corridas/checkpoints: '+', '.join(missing))
 return result,idx

def build(name):
 if name in BASES:return bases.build(name)
 if name in nuevos.MODELS:return nuevos.build(name)
 return siete.build(name)

def extract(z,root,rows):
 for i,r in enumerate(rows):
  path=(root/r['archivo']).resolve()
  if not path.is_relative_to(root.resolve()):raise ValueError('Ruta de imagen inválida')
  if path.is_file() and sha(path)==r['sha_archivo']:continue
  data=z.read(r['archivo'])
  if hashlib.sha256(data).hexdigest()!=r['sha_archivo']:raise ValueError('Archivo de datos modificado')
  path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
  if i%3000==0:print('Preparando datos',i,'/',len(rows),flush=True)

def infer(model,root,rows):
 loader=DataLoader(core.Crops(root,rows),batch_size=32,num_workers=0,shuffle=False)
 _,y,p=core.evaluate(model,loader,torch.device('cuda'));return y,p

def cache_infer(model,root,rows,target,identity):
 marker=target.with_suffix('.json')
 if marker.exists():
  meta=json.loads(marker.read_text())
  if meta['identidad']!=identity or sha(target)!=meta['predicciones_sha256']:raise ValueError('Caché modificada/incompatible: '+str(target))
  with np.load(target,allow_pickle=False) as z:y=z['y'];p=z['p']
  print('OMITIDA INFERENCIA VERIFICADA',target.name,flush=True);return y,p
 y,p=infer(model,root,rows);write_npz(target,y=y,p=p);st.save(marker,{'identidad':identity,'predicciones_sha256':sha(target)});return y,p

def write_predictions(path,rows,p):
 rr=[dict(paciente=r['paciente'],tss=r['tss'],archivo=r['archivo'],clase_real=r['clase'],clase_predicha=core.CLASSES[int(p[i].argmax())],p_tumor=p[i,0],p_linfocito=p[i,1],p_estroma=p[i,2]) for i,r in enumerate(rows)];st.table(path,rr)

def historical(drive,run,rows,out):
 # Solo resultados propios anteriores y solo si coincide el checkpoint exacto.
 candidates=list(drive.glob('*CIERRE_11/reserva/'+f"{run['modelo']}_{run['semilla']}.json"))
 for f in candidates:
  q=json.loads(f.read_text())
  if q.get('checkpoint_sha256')!=run['huellas']['mejor.pt']:continue
  frozen=f.parent.parent/'evaluacion_reserva_fijada.json'
  if not frozen.is_file() or json.loads(frozen.read_text()).get('manifest_sha256')!=EXPECTED:continue
  csvpath=f.with_suffix('.csv')
  if not csvpath.exists():continue
  p=read_prediction(csvpath,rows);st.save(out/'reutilizacion_E1.json',dict(origen=str(f),sha_resultado=sha(f),sha_predicciones=sha(csvpath),checkpoint=run['huellas']['mejor.pt']));return p
 return None

def cost(model,x):
 model.eval();times=[]
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 with torch.no_grad():
  for _ in range(10):model(x)
  torch.cuda.synchronize()
  for _ in range(30):
   t=time.perf_counter();model(x);torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000/len(x))
 peak=torch.cuda.max_memory_allocated();flops=None;error=None
 try:
  with torch.no_grad(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],with_flops=True) as prof:model(x)
  flops=sum(e.flops for e in prof.key_averages())/len(x)
 except Exception as e:error=str(e)
 return dict(parametros=sum(p.numel() for p in model.parameters()),latencia_ms_por_recorte_batch32=float(np.median(times)),batch=len(x),memoria_pico_inferencia_bytes=peak,FLOPs_operaciones_cubiertas_por_profiler=flops,nota='FLOPs parciales: operadores no cubiertos no equivalen a cero; memoria de inferencia, no pico de entrenamiento.',incidencia=error)

def main(drive):
 drive=Path(drive);out=drive/'NuCLS_E2_CIERRE_RESERVA_V1';
 if out.is_symlink():raise RuntimeError('La salida no puede ser un enlace a otra carpeta')
 out.mkdir(exist_ok=True);check_idle(drive)
 lock=out/'EJECUCION_ACTIVA.json'
 # Recuperación automática solo de un PID muerto en el mismo host; otra sesión se conserva.
 if lock.exists():
  info=json.loads(lock.read_text())
  if info.get('host')!=platform.node():raise RuntimeError('Cierre de otra sesión marcado activo; verificar antes de reanudar. No se borró.')
  try:os.kill(int(info['pid']),0)
  except ProcessLookupError:lock.rename(lock.with_name('bloqueo_previo_'+str(time.time_ns())+'.json'))
  else:raise RuntimeError('Ya existe un cierre activo.')
 with lock.open('x') as f:json.dump(dict(pid=os.getpid(),host=platform.node()),f)
 status={'estado':'PREPARANDO_SIN_ABRIR_RESERVA','fase':'preflight'}
 try:
  if not torch.cuda.is_available():raise RuntimeError('Seleccione GPU antes de ejecutar; no se reentrena.')
  torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(2)
  archive_path=drive/'NuCLS_V2_DATOS.zip'
  if not archive_path.is_file():raise FileNotFoundError('Falta NuCLS_V2_DATOS.zip en el Drive conectado; no usar un ZIP de métricas como sustituto.')
  local=Path('/content/nucls_e2_cierre_datos');local.mkdir(exist_ok=True)
  with zipfile.ZipFile(archive_path) as z:raw=z.read('manifiesto.csv')
  if hashlib.sha256(raw).hexdigest()!=EXPECTED:raise ValueError('Manifiesto distinto de la partición heredada')
  allrows=list(csv.DictReader(raw.decode().splitlines()));groups={s:[r for r in allrows if r['particion']==s and r['estado']=='incluida'] for s in ['train','validation']};reserve=[r for r in allrows if r['particion']=='reserva_candidata']
  assert [len(groups['train']),len(groups['validation']),len(reserve)]==[14539,2049,969]
  assert [len({r['paciente'] for r in rr}) for rr in [groups['train'],groups['validation'],reserve]]==[83,21,6]
  for key in ['paciente','tss','sha_pixeles']:
   sets=[{r[key] for r in rr} for rr in [groups['train'],groups['validation'],reserve]]
   assert all(not (sets[i]&sets[j]) for i in range(3) for j in range(i+1,3)),key
  runs,idx=inventory(drive,groups)
  env=dict(torch=torch.__version__,torchvision=torchvision.__version__,timm=timm.__version__,numpy=np.__version__,python=platform.python_version(),gpu=torch.cuda.get_device_name())
  import scipy,sklearn,matplotlib
  env.update(scipy=scipy.__version__,sklearn=sklearn.__version__,matplotlib=matplotlib.__version__)
  code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
  baseline_root=drive/'NuCLS_E2_REFERENCIAS_SIMPLES';baseline_files={p.name:sha(p) for p in [baseline_root/'resultado.json',baseline_root/'logistica.joblib',baseline_root/'logistica_predicciones.csv'] if p.is_file()}
  plan=dict(modelos=NAMES,adicionales=ADICIONALES,semillas=SEEDS,corridas=runs,manifest_sha256=EXPECTED,codigo=code,entorno=env,baseline_archivos=baseline_files,bootstrap_pacientes=st.B,semilla_bootstrap=st.SEED,delta=.02,delta_g=.02,delta_g_nota='Margen operativo prospectivo del cierre sobre macro-F1; no validación clínica ni preregistro previo al entrenamiento.',reserva_expuesta_E1=True,criterio='Checkpoint elegido en validación; cinco primeras semillas comunes; ningún reajuste tras reserva')
  inputfile=out/'entradas_fijadas.json'
  if inputfile.exists():
   if json.loads(inputfile.read_text())!=plan:raise ValueError('Entradas/entorno cambiaron: no se sobrescribe ni se crea otra evaluación automáticamente.')
  else:st.save(inputfile,plan)
  pid=sha(inputfile)
  evidence_code=out/'codigo';evidence_code.mkdir(exist_ok=True)
  for source in Path(__file__).parent.glob('*.py'):shutil.copyfile(source,evidence_code/source.name)
  # Extraer desarrollo y verificar compatibilidad de los 65 checkpoints ANTES de abrir imágenes de reserva.
  with zipfile.ZipFile(archive_path) as z:extract(z,local,groups['train']+groups['validation'])
  (local/'manifiesto.csv').write_bytes(raw)
  baseline_preflight(drive,local,groups,plan)
  for run in runs:
   dest=out/'corridas'/f"{run['modelo']}_{run['semilla']}";dest.mkdir(parents=True,exist_ok=True);target=dest/'validation.npz';identity=dict(plan=pid,checkpoint=run['huellas']['mejor.pt'],particion='validation')
   if target.with_suffix('.json').exists():cache_infer(None,local,groups['validation'],target,identity);continue
   model=build(run['modelo']).cuda().eval();model.load_state_dict(torch.load(Path(run['ruta'])/'mejor.pt',weights_only=True,map_location='cpu'))
   y,p=infer(model,local,groups['validation']);original=read_prediction(Path(run['ruta'])/'predicciones_validacion.csv',groups['validation'])
   error=float(np.max(abs(p-original)))
   if error>2e-5 or not np.array_equal(p.argmax(1),original.argmax(1)):raise ValueError(f"Checkpoint/implementación no reproduce validación {run['modelo']} {run['semilla']}; error={error}. Reserva no abierta.")
   write_npz(target,y=y,p=p);st.save(target.with_suffix('.json'),dict(identidad=identity,predicciones_sha256=sha(target),discrepancia_max=error));del model;torch.cuda.empty_cache();print('VALIDACIÓN COMPATIBLE',run['modelo'],run['semilla'],flush=True)
  check_idle(drive)
  for run in runs:
   if sha(Path(run['ruta'])/'mejor.pt')!=run['huellas']['mejor.pt']:raise ValueError('Checkpoint cambió antes de congelar')
  freeze=out/'congelamiento_previo_reserva.json'
  if not freeze.exists():st.save(freeze,dict(plan_sha256=pid,fecha_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),estado='CONGELADO_ANTES_DE_ABRIR_RESERVA'))
  elif json.loads(freeze.read_text())['plan_sha256']!=pid:raise ValueError('Congelamiento incompatible')
  status.update(estado='EVALUANDO_RESERVA_CONGELADA',fase='inferencia');st.save(out/'estado.json',status);archive(out)
  with zipfile.ZipFile(archive_path) as z:extract(z,local,reserve)
  predictions={};gaps=[];costs=[]
  for run in runs:
   name=run['modelo'];seed=run['semilla'];dest=out/'corridas'/f'{name}_{seed}';marker=dest/'completo.json'
   identity=dict(plan=pid,checkpoint=run['huellas']['mejor.pt'],particion='reserva')
   if marker.exists():
    m=json.loads(marker.read_text());assert m['plan']==pid
    for file,h in m['archivos'].items():assert sha(dest/file)==h
    with np.load(dest/'reserva.npz',allow_pickle=False) as z:predictions[(name,seed)]=z['p']
    gaps.append(json.loads((dest/'brecha.json').read_text()));costs.append(json.loads((dest/'costo.json').read_text()));print('OMITIDO CIERRE COMPLETO',name,seed,flush=True);continue
   model=build(name).cuda().eval();model.load_state_dict(torch.load(Path(run['ruta'])/'mejor.pt',weights_only=True,map_location='cpu'))
   target=dest/'reserva.npz'
   if not target.with_suffix('.json').exists():
    old=historical(drive,run,reserve,dest) if name in BASES else None
    if old is not None:
     write_npz(target,y=np.array([core.CLASSES.index(r['clase']) for r in reserve]),p=old);st.save(target.with_suffix('.json'),dict(identidad=identity,predicciones_sha256=sha(target)));print('REUTILIZADA RESERVA E1',name,seed,flush=True)
   y,p=cache_infer(model,local,reserve,target,identity);predictions[(name,seed)]=p;write_predictions(dest/'predicciones_reserva.csv',reserve,p)
   ty,tp=cache_infer(model,local,groups['train'],dest/'train_limpio.npz',dict(plan=pid,checkpoint=run['huellas']['mejor.pt'],particion='train_limpio'))
   gap=dict(modelo=name,semilla=seed,macro_f1_train_limpio=core.scores(ty,tp.argmax(1))['macro_f1'],macro_f1_val=run['validacion'],macro_f1_reserva=core.scores(y,p.argmax(1))['macro_f1'])
   gap['train_menos_val']=gap['macro_f1_train_limpio']-gap['macro_f1_val'];gap['val_menos_reserva']=gap['macro_f1_val']-gap['macro_f1_reserva'];gaps.append(gap);st.save(dest/'brecha.json',gap)
   c=dict(modelo=name,semilla=seed,**cost(model,torch.stack([core.Crops(local,groups['validation'])[i][0] for i in range(32)]).cuda()));costs.append(c);st.save(dest/'costo.json',c)
   st.save(marker,dict(plan=pid,archivos={f:sha(dest/f) for f in ['reserva.npz','train_limpio.npz','predicciones_reserva.csv','brecha.json','costo.json']}));del model;torch.cuda.empty_cache();archive(out)
  status.update(fase='estadistica');st.save(out/'estado.json',status)
  y=np.array([core.CLASSES.index(r['clase']) for r in reserve]);patients=np.array([r['paciente'] for r in reserve]);summary=st.analyze(out,plan,predictions,y,patients,reserve)
  st.table(out/'brechas_por_semilla.csv',gaps);st.table(out/'costos_por_semilla.csv',costs)
  # Brecha: contraste exploratorio de margen; no declaración de ausencia de sobreajuste por p>0,05.
  bg=[]
  for name in NAMES:
   d=np.array([r['train_menos_val'] for r in gaps if r['modelo']==name]);sd=d.std(ddof=1);upper=d.mean()+st.stats.t.ppf(.95,len(d)-1)*sd/np.sqrt(len(d));bg.append(dict(modelo=name,brecha_media=d.mean(),limite_superior95_unilateral=upper,delta_g=.02,p_inferior_margen=st.stats.ttest_1samp(d,.02,alternative='less').pvalue if sd else None,nota='Margen operativo definido para cierre; selección en validación. Ausencia de significación no prueba ausencia de sobreajuste.'))
  
  for r,pv in zip(bg,st.holm([r['p_inferior_margen'] if r['p_inferior_margen'] is not None else 1 for r in bg])):r['p_Holm13']=pv
  st.table(out/'brecha_margen.csv',bg)
  baseline_status=baselines_saved(drive,out,local,groups,reserve,y,patients,plan)
  st.save(out/'limitaciones.json',dict(pacientes_reserva=6,prueba_expuesta_E1=True,Wilcoxon_cinco_pares='p mínimo bilateral exacto 0,0625',inferencia='Semillas condicionadas a partición; diagnósticos iid no sustituyen contraste agrupado',no_incluido=['Réplicas nuevas de entrenamiento','Ablaciones','Réplica PCam','Techo multi-rater','Cobertura conforme independiente','Nueva ejecución de robustez/Grad-CAM'],baselines=baseline_status))
  development_statistics(out,plan,gaps,costs)
  report(out,summary,bg,baseline_status)
  status.update(estado='CIERRE_RESERVA_Y_ESTADISTICA_COMPLETADO',fase='final',corridas=65,baselines=baseline_status);st.save(out/'estado.json',status);archive(out);print(json.dumps(status,ensure_ascii=False,indent=2))
 except BaseException as e:
  status.update(estado='DETENIDO_CON_EVIDENCIA_CONSERVADA',error=str(e));st.save(out/'estado.json',status);archive(out);raise
 finally:lock.unlink(missing_ok=True)


def features(root,rows):
 xs=[]
 for r in rows:
  with Image.open(root/r['archivo']) as im:a=np.asarray(im.convert('RGB'),dtype=np.float32)/255
  pooled=a.reshape(16,8,16,8,3).mean((1,3)).ravel()
  hist=np.concatenate([np.histogram(a[:,:,c],bins=16,range=(0,1),density=False)[0]/(128*128) for c in range(3)])
  xs.append(np.r_[pooled,hist])
 return np.asarray(xs)

def baseline_preflight(drive,local,groups,plan):
 root=drive/'NuCLS_E2_REFERENCIAS_SIMPLES';required=['resultado.json','logistica.joblib','logistica_predicciones.csv']
 if not all(f in plan['baseline_archivos'] for f in required):return
 import joblib,sklearn
 for f,h in plan['baseline_archivos'].items():
  if sha(root/f)!=h:raise ValueError('Cambió referencia clásica congelada')
 metadata=json.loads((root/'resultado.json').read_text())
 if metadata.get('sklearn')!=sklearn.__version__:raise RuntimeError('La referencia clásica exige scikit-learn '+str(metadata.get('sklearn'))+'. Instale esa versión antes de ejecutar el cierre; no se abre reserva.')
 bundle=joblib.load(root/'logistica.joblib')
 if bundle['clases']!=core.CLASSES or list(bundle['model'].classes_)!=[0,1,2]:raise ValueError('Orden de clases de referencia incompatible')
 p=bundle['model'].predict_proba(bundle['scaler'].transform(features(local,groups['validation'])))
 if not np.allclose(p,read_prediction(root/'logistica_predicciones.csv',groups['validation']),atol=1e-8,rtol=1e-7):raise ValueError('Referencia clásica no reproduce validación; no se abre reserva')

def baselines_saved(drive,out,local,groups,reserve,y,patients,plan):
 root=drive/'NuCLS_E2_REFERENCIAS_SIMPLES';target=out/'referencias';target.mkdir(exist_ok=True)
 train_y=np.array([core.CLASSES.index(r['clase']) for r in groups['train']]);majority=int(np.bincount(train_y,minlength=3).argmax());prob=np.tile(np.eye(3)[majority],(len(y),1));predictions={'mayoritaria':prob}
 status={'mayoritaria':'EVALUADA_REGLA_TRAIN','logistica':'PENDIENTE_NO_HAY_MODELO_GUARDADO'}
 required=['resultado.json','logistica.joblib','logistica_predicciones.csv']
 if all(f in plan['baseline_archivos'] for f in required):
  import joblib
  for f,h in plan['baseline_archivos'].items():
   if sha(root/f)!=h:raise ValueError('Referencia clásica modificada después de congelar')
  cache=target/'logistica.npz';marker=cache.with_suffix('.json')
  if marker.exists():
   assert json.loads(marker.read_text())=={'modelo_sha256':plan['baseline_archivos']['logistica.joblib'],'npz_sha256':sha(cache)}
   with np.load(cache,allow_pickle=False) as z:p=z['p']
  else:
   bundle=joblib.load(root/'logistica.joblib');p=bundle['model'].predict_proba(bundle['scaler'].transform(features(local,reserve)));write_npz(cache,p=p)
   st.save(marker,dict(modelo_sha256=plan['baseline_archivos']['logistica.joblib'],npz_sha256=sha(cache)))
  predictions['logistica']=p;status['logistica']='EVALUADA_MODELO_GUARDADO_SIN_FIT'
 boot,_=st.patient_boot(y,np.stack([p[None,...] for p in predictions.values()]),patients)
 for i,(name,p) in enumerate(predictions.items()):
  m=st.metrics(y,p);m['IC95_macro_f1_paciente']=np.quantile(boot[:,i],[.025,.975]).tolist();st.save(target/(name+'_metricas.json'),m);write_predictions(target/(name+'_reserva.csv'),reserve,p)
 # Diagnóstico binomial con p0 estimado SOLO en train; no prueba primaria (dependencia intrapaciente).
 p0=float(np.mean(train_y==majority));diag=[]
 for name in NAMES:
  with np.load(out/'corridas'/f'{name}_{SEEDS[0]}'/'reserva.npz',allow_pickle=False) as z:p=z['p']
  diag.append(dict(modelo=name,semilla=SEEDS[0],p0_train=p0,aciertos=int((p.argmax(1)==y).sum()),n=len(y),p_binomial_iid=st.stats.binomtest(int((p.argmax(1)==y).sum()),len(y),p0,alternative='greater').pvalue,nota='Diagnóstico iid condicionado a prevalencia train; pacientes no independientes por recorte'))
 for r,pv in zip(diag,st.holm([r['p_binomial_iid'] for r in diag])):r['p_Holm13_iid']=pv
 st.table(out/'diagnostico_binomial_iid.csv',diag)
 deep=[]
 for name in NAMES:
  pp=[]
  for seed in SEEDS:
   with np.load(out/'corridas'/f'{name}_{seed}'/'reserva.npz',allow_pickle=False) as z:pp.append(z['p'])
  deep.append(pp)
 deep_boot,_=st.patient_boot(y,np.asarray(deep),patients);comparisons=[]
 for i,name in enumerate(NAMES):
  for j,reference in enumerate(predictions):
   delta=deep_boot[:,i]-boot[:,j];ci=np.quantile(delta,[.025,.975]);comparisons.append(dict(modelo=name,referencia=reference,IC95_delta_macro_f1_paciente_inf=ci[0],IC95_delta_macro_f1_paciente_sup=ci[1],nota='Remuestreo emparejado de pacientes; referencia determinista única, no cinco entrenamientos'))
 st.table(out/'contrastes_referencias_paciente.csv',comparisons)
 return status

def development_statistics(out,plan,gaps,costs):
 contrasts=[]
 for name in ADICIONALES:
  a=[next(r['macro_f1_val'] for r in gaps if r['modelo']==name and r['semilla']==s) for s in SEEDS]
  b=[next(r['macro_f1_val'] for r in gaps if r['modelo']=='resnet50' and r['semilla']==s) for s in SEEDS]
  contrasts.append(dict(modelo=name,**st.paired(a,b)))
 for test in ['t','wilcoxon','permutacion']:
  for r,pv in zip(contrasts,st.holm([r['p_'+test] for r in contrasts])):r['p_'+test+'_Holm10']=pv
 st.table(out/'contrastes_validacion_vs_resnet50.csv',contrasts)
 values=np.array([[next(r['macro_f1_val'] for r in gaps if r['modelo']==name and r['semilla']==seed) for name in NAMES] for seed in SEEDS])
 ranks=np.apply_along_axis(lambda x:st.stats.rankdata(-x),1,values).mean(0);n,k=values.shape;se=np.sqrt(k*(k+1)/(6*n));f=st.stats.friedmanchisquare(*values.T)
 st.save(out/'omnibus_validacion.json',dict(estadistico=f.statistic,p=f.pvalue,rangos=dict(zip(NAMES,ranks)),nota='Exploratorio: checkpoint y configuración seleccionados con validación'))
 st.table(out/'nemenyi_validacion.csv',[dict(modelo_a=NAMES[i],modelo_b=NAMES[j],p_nemenyi=st.stats.studentized_range.sf(abs(ranks[i]-ranks[j])/se*np.sqrt(2),k,np.inf)) for i in range(k) for j in range(i+1,k)])
 import matplotlib.pyplot as plt
 data=[]
 for name in NAMES:
  rr=[r for r in costs if r['modelo']==name];gg=[r for r in gaps if r['modelo']==name]
  data.append(dict(modelo=name,latencia_ms=float(np.median([r['latencia_ms_por_recorte_batch32'] for r in rr])),macro_f1_val=float(np.mean([r['macro_f1_val'] for r in gg])),macro_f1_reserva=float(np.mean([r['macro_f1_reserva'] for r in gg]))))
 for r in data:r['pareto_validacion']=not any(q['latencia_ms']<=r['latencia_ms'] and q['macro_f1_val']>=r['macro_f1_val'] and (q['latencia_ms']<r['latencia_ms'] or q['macro_f1_val']>r['macro_f1_val']) for q in data)
 st.table(out/'pareto_costo_validacion.csv',data)
 fig,ax=plt.subplots(figsize=(11,6))
 for r in data:
  ax.scatter(r['latencia_ms'],r['macro_f1_val'],color='#d88828' if r['pareto_validacion'] else '#087c85');ax.annotate(r['modelo'],(r['latencia_ms'],r['macro_f1_val']),fontsize=7,xytext=(3,3),textcoords='offset points')
 ax.set_xlabel('Latencia de inferencia ms/recorte · batch 32');ax.set_ylabel('Macro-F1 de validación (5 semillas)');ax.set_title('Costo y desempeño · sin selección por reserva');fig.tight_layout();fig.savefig(out/'pareto.png',dpi=170);plt.close(fig)
 # Curvas existentes; lectura de evidencia, sin entrenamiento.
 fig,axes=plt.subplots(4,4,figsize=(14,12));axes=axes.ravel()
 for i,name in enumerate(NAMES):
  for run in plan['corridas']:
   if run['modelo']!=name:continue
   h=json.loads((Path(run['ruta'])/'historia.json').read_text());axes[i].plot(range(1,len(h)+1),[v['val_macro_f1'] for v in h],alpha=.65,label=str(run['semilla']))
  axes[i].set_title(name,fontsize=9);axes[i].set_ylim(0,1);axes[i].set_xlabel('Época');axes[i].set_ylabel('F1 validación')
 for ax in axes[len(NAMES):]:ax.axis('off')
 fig.tight_layout();fig.savefig(out/'curvas_validacion.png',dpi=150);plt.close(fig)

def report(out,summary,gaps,baseline_status):
 def tab(rows):
  if not rows:return ''
  keys=list(rows[0]);return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(f'{r[k]:.4f}' if isinstance(r[k],(float,np.floating)) else str(r[k]))+'</td>' for k in keys)+'</tr>' for r in rows)+'</table></div>'
 links=''.join('<li><a href="'+html.escape(p.name)+'">'+html.escape(p.name)+'</a></li>' for p in sorted(out.glob('*.csv')))
 body='<h1>NuCLS · Evaluación E2</h1><p>13 modelos · 5 semillas comunes · reserva heredada de 6 pacientes</p><p>Checkpoint elegido en validación y congelado antes de esta evaluación. La reserva ya fue observada en E1. Los intervalos por paciente y los contrastes entre semillas representan fuentes distintas de incertidumbre.</p><h2>Desempeño en reserva</h2>'+tab(summary)+'<img src="benchmark_reserva.png"><h2>Comparación de modelos</h2><img src="diferencia_critica.png"><p>Holm se aplica a los diez contrastes frente a ResNet50. Con cinco pares, Wilcoxon bilateral exacto no puede alcanzar p &lt; 0,05. La ausencia de significación no demuestra equivalencia. DeLong, McNemar y binomial por recorte son diagnósticos bajo independencia; la permutación agrupada y el bootstrap por paciente se informan por separado.</p><h2>Validación, costo y ajuste</h2><img src="pareto.png"><img src="curvas_validacion.png">'+tab(gaps)+'<h2>Referencias simples</h2><pre>'+html.escape(json.dumps(baseline_status,ensure_ascii=False,indent=2))+'</pre><h2>Evidencias</h2><ul>'+links+'</ul><h2>Alcance</h2><p>Este cierre calcula inferencia y estadística; no completa entrenamientos, ablaciones, réplica PCam ni una garantía de cobertura conforme. Los FLOPs del perfilador cubren solo operadores compatibles. Los IC por seis pacientes requieren cautela. La diferencia mínima detectable se calcula con aproximación t y α=0,005; no acredita retrospectivamente potencia suficiente.</p>'
 (out/'informe.html').write_text('<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NuCLS E2 · Cierre</title><style>body{font:16px system-ui;background:#f3f6f8;color:#152d3a;max-width:1200px;margin:auto;padding:40px}h1{font-size:42px}h2{margin-top:42px;color:#087c85}p{line-height:1.65}table{border-collapse:collapse;background:white;font-size:13px;width:100%}td,th{padding:10px;border-bottom:1px solid #d8e1e6;text-align:left}th{background:#163d4b;color:white}.scroll{overflow:auto}img{width:100%;max-width:1100px;margin:20px 0;background:white}a{color:#087c85}</style>'+body+'</html>')

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--drive',type=Path,default=Path('/content/drive/MyDrive'));args=parser.parse_args();main(args.drive)
