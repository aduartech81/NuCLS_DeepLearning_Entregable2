"""Continúa únicamente desarrollo del plan V2; origen de E1 siempre de solo lectura.
Inventario usa biblioteca estándar. Entrenamiento importa el motor original al final.
"""
import argparse,csv,hashlib,json,math,os,platform,shutil,socket,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
MODELS=('softmaxcnn_bn','densenet121','resnet50')
SEEDS=range(6101,6123)
EXPECTED='623adf36520751addc324e44cd1b89d95e47842e53cda494914c0248a8be86d4'
SOURCE_CODE='ded095f538408ede724b3309111e9f3e39a0a6a0de87528035e70ffcd6b5850d'
CLASSES=['tumor_nonMitotic','lymphocyte','nonTILnonMQ_stromal']

def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def require(ok,msg):
 if not ok:raise ValueError(msg)
def separate(origin,out,data):
 paths=[origin.resolve(),out.resolve(),data.resolve()]
 for i,p in enumerate(paths):
  for q in paths[i+1:]:require(not(p.is_relative_to(q) or q.is_relative_to(p)),'Origen, salida nueva y datos deben ser carpetas separadas, no anidadas.')
def setup(origin):
 cfg=read(origin/'configuracion.json');hp=read(origin/'hiperparametros_seleccionados.json');plan=read(origin/'plan_corridas.json')
 expected=dict(batch=32,epocas_maximas=25,paciencia=5,min_epocas=5,manifest_sha256=EXPECTED,preentrenamiento=False,codigo_sha256=SOURCE_CODE)
 for k,v in expected.items():require(cfg.get(k)==v,f'Configuración original no compatible: {k}')
 require(cfg.get('modelos')==list(MODELS),'Modelos originales distintos');require(plan['R_por_modelo']==22,'El plan original no es de 22 semillas; revisar antes de continuar.')
 for m in MODELS:
  require(hp[m]['lr']==.001 and hp[m]['weight_decay']==(1e-4 if m=='softmaxcnn_bn' else 0.),'Hiperparámetros originales inesperados: '+m)
 return cfg,hp,plan

def completed(run,model,seed,hp,frozen=None):
 """No basta con que exista resultado.json: se verifica la evidencia de validación."""
 for file in ('resultado.json','historia.json','curvas.csv','predicciones_validacion.csv','metricas_por_paciente.csv','mejor.pt'):
  require((run/file).is_file() and (run/file).stat().st_size>0,f'Corrida declarada completa sin {file}: {run}')
 r=read(run/'resultado.json');hist=read(run/'historia.json')
 require(r.get('estado')=='FASE_COMPLETADA_VALIDACION' and r.get('modelo')==model and r.get('semilla')==seed,'Identidad/estado incompatible: '+str(run))
 for k in ('lr','weight_decay'):require(r[k]==hp[model][k],f'{k} incompatible: {run}')
 require(1<=len(hist)<=25 and r['epocas_ejecutadas']==len(hist),'Historia incompleta: '+str(run))
 require([h['epoca'] for h in hist]==list(range(1,len(hist)+1)),'Épocas no consecutivas: '+str(run))
 best=max(h['val_macro_f1'] for h in hist)
 require(all(math.isfinite(h['val_macro_f1']) and 0<=h['val_macro_f1']<=1 for h in hist),'Métricas no finitas')
 require(math.isclose(best,r['validacion']['macro_f1'],abs_tol=1e-10),'Mejor F1 no coincide con historia: '+str(run))
 require(1<=r['mejor_epoca']<=len(hist) and math.isclose(hist[r['mejor_epoca']-1]['val_macro_f1'],best,abs_tol=1e-10),'Mejor época incompatible')
 with (run/'predicciones_validacion.csv').open() as f:rows=list(csv.DictReader(f))
 require(len(rows)==2049 and len({x['paciente'] for x in rows})==21,'Conteos de validación no coinciden: '+str(run))
 require(len({(x['paciente'],x['archivo']) for x in rows})==2049,'Predicciones duplicadas')
 cm=[[0]*3 for _ in range(3)]
 for x in rows:cm[CLASSES.index(x['clase_real'])][CLASSES.index(x['clase_predicha'])]+=1
 f1=sum(2*cm[i][i]/max(sum(cm[i])+sum(row[i] for row in cm),1) for i in range(3))/3
 require(math.isclose(f1,r['validacion']['macro_f1'],abs_tol=1e-10),'F1 no reproducible desde predicciones')
 weight_hash=None
 if frozen:
  weight_hash=sha(run/'mejor.pt');require(weight_hash==frozen,'Checkpoint cambió respecto al cierre de E1: '+str(run))
 return dict(estado='completa',ruta=str(run),modelo=model,semilla=seed,epocas=len(hist),macro_f1_validacion=r['validacion']['macro_f1'],resultado_sha256=sha(run/'resultado.json'),checkpoint_sha256=weight_hash)

def inventory(origin,out,hp):
 frozen={};freeze=origin/'evaluacion_reserva_fijada.json'
 if freeze.exists():frozen=read(freeze).get('sha256_checkpoints',{})
 allrows=[]
 for seed in SEEDS:
  for model in MODELS:
   key=f'{model}_{seed}';old=origin/'final'/key;new=out/'final'/key
   print('Inventario:',key,flush=True)
   require(new.resolve().is_relative_to(out.resolve()),'Una ruta de salida apunta fuera de E2: '+str(new))
   if (new/'resultado.json').exists():row=completed(new,model,seed,hp)
   elif (old/'resultado.json').exists():row=completed(old,model,seed,hp,frozen.get(key))
   else:
    require(seed>6111,f'Falta evidencia completa del cierre E1: {key}. No se reentrena automáticamente.')
    source=new if (new/'ultimo.pt').is_file() else old if (old/'ultimo.pt').is_file() else new if new.exists() and any(new.iterdir()) else old
    if (source/'ultimo.pt').is_file():row=dict(estado='reanudar',ruta=str(source),modelo=model,semilla=seed)
    else:
     require(not source.exists() or not any(p.is_file() and p.name!='procedencia_reanudacion.json' for p in source.iterdir()),f'Archivos parciales sin ultimo.pt: {source}. Se detiene para no repetir una corrida sin diagnóstico.')
     row=dict(estado='nueva',ruta=str(new),modelo=model,semilla=seed)
   allrows.append(row)
 return allrows

def copy_resume(source,target):
 """Copia atómica individual. ultimo.pt se publica al final; nunca escribe en origen."""
 require(not target.is_symlink(),'No se copia sobre un enlace simbólico')
 if source.resolve()==target.resolve():return
 target.mkdir(parents=True,exist_ok=True)
 for name in ('mejor.pt','historia.json','curvas.csv','ultimo.pt'):
  p=source/name
  if p.exists():
   tmp=target/(name+'.copia_tmp');shutil.copyfile(p,tmp);require(sha(tmp)==sha(p),'Copia de checkpoint no íntegra');tmp.replace(target/name)
 save(target/'procedencia_reanudacion.json',dict(origen=str(source),ultimo_sha256=sha(target/'ultimo.pt'),fecha_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))

def validate_checkpoint(path,torch):
 ck=torch.load(path,map_location='cpu',weights_only=False) # Checkpoint propio del usuario, no de terceros.
 keys={'model','optimizer','epoch','history','best_model','best','best_epoch','bad','python_rng','numpy_rng','torch_rng','loader_rng','device_rng'}
 require(keys<=ck.keys(),'Checkpoint sin estado completo de entrenamiento/RNG')
 require(1<=ck['epoch']<=25 and len(ck['history'])==ck['epoch'],'Checkpoint: época/historia incompatible')
 require([h['epoca'] for h in ck['history']]==list(range(1,ck['epoch']+1)),'Checkpoint: historia no consecutiva')
 require(math.isclose(max(h['val_macro_f1'] for h in ck['history']),ck['best'],abs_tol=1e-12),'Checkpoint: mejor F1 incompatible')
 epoch=ck['epoch'];del ck;return epoch

def persist_index(out,rows):
 save(out/'inventario_corridas.json',rows)
 with (out/'indice_validacion.csv').open('w',newline='') as f:
  cols=['modelo','semilla','estado','ruta','epocas','macro_f1_validacion','resultado_sha256','checkpoint_sha256'];w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main(argv=None):
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--origen',type=Path,required=True);ap.add_argument('--salida',type=Path,required=True);ap.add_argument('--datos',type=Path,required=True);ap.add_argument('--ejecutar',action='store_true');ap.add_argument('--liberar-bloqueo',action='store_true');a=ap.parse_args(argv)
 a.origen=a.origen.resolve();a.salida=a.salida.resolve();a.datos=a.datos.resolve();separate(a.origen,a.salida,a.datos)
 cfg,hp,plan=setup(a.origen)
 # Un manifiesto interno también fija el código, incluso al cargar desde un ZIP local.
 for rel,digest in read(ROOT/'protocolo/integridad_motor.json').items():require(sha(ROOT/rel)==digest,'Código del motor modificado: '+rel)
 require(sha(a.datos/'manifiesto.csv')==EXPECTED,'Manifiesto distinto del dataset original')
 a.salida.mkdir(parents=True,exist_ok=True);lock=a.salida/'EJECUCION_ACTIVA.json'
 if a.liberar_bloqueo:
  require(not a.ejecutar,'Liberar bloqueo y ejecutar son pasos separados')
  if lock.exists():
   l=read(lock)
   if l.get('host')==socket.gethostname():
    try:os.kill(l['pid'],0)
    except ProcessLookupError:pass
    else:raise ValueError('El proceso registrado sigue activo. No se libera el bloqueo.')
   lock.rename(a.salida/f'bloqueo_retirado_{time.time_ns()}.json')
  print('Bloqueo retirado. Solo volver a ejecutar si no hay otro Colab activo sobre esta salida.');return
 require(not lock.exists(),'Hay una ejecución activa o interrumpida. No lance dos sesiones sobre esta salida. Tras verificar que terminó, use --liberar-bloqueo.')
 fixed=dict(origen=str(a.origen),datos=str(a.datos),manifest_sha256=EXPECTED,semillas=list(SEEDS),modelos=list(MODELS),hiperparametros=hp,baseline_heredado='resnet50',evaluar_reserva=False,plan_original_sha256=sha(a.origen/'plan_corridas.json'),configuracion_original_sha256=sha(a.origen/'configuracion.json'))
 configpath=a.salida/'continuacion_fijada.json'
 if configpath.exists():require(read(configpath)==fixed,'Cambió el protocolo o las rutas de la continuación')
 else:save(configpath,fixed)
 print('Comprobando resultados guardados. El origen solo se lee; la reserva no se abre. Leer hashes de Drive puede tardar varios minutos.',flush=True)
 rows=inventory(a.origen,a.salida,hp);persist_index(a.salida,rows)
 counts={s:sum(r['estado']==s for r in rows) for s in ['completa','reanudar','nueva']};print(json.dumps(counts,ensure_ascii=False),flush=True)
 for r in rows:
  if r['estado']!='completa':print(r['estado'].upper(),r['modelo'],r['semilla'],flush=True)
 if not a.ejecutar:print('INVENTARIO LISTO. No se ha entrenado nada. Añadir --ejecutar para continuar.');return
 if counts['completa']==66:print('Las 66 corridas ya están completas. No hay trabajo pendiente.');return
 os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.path.insert(0,str(ROOT/'modelos'))
 import core_v2 as core
 torch=core.torch;np=core.np
 require(torch.cuda.is_available(),'Se requiere GPU CUDA para mantener el protocolo; seleccione GPU en el NUEVO notebook.')
 for k,v in [('torch',torch.__version__),('torchvision',core.torchvision.__version__),('numpy',np.__version__)]:require(str(v)==str(cfg[k]),f'Entorno distinto: {k} actual={v}, original={cfg[k]}. No se cambia ni instala automáticamente.')
 require(platform.python_version_tuple()[:2]==tuple(cfg['python'].split()[0].split('.')[:2]),'Versión mayor/menor de Python distinta del original')
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
 fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w') as f:json.dump(dict(pid=os.getpid(),host=socket.gethostname(),inicio=time.time()),f)
 try:
  save(a.salida/f'entorno_{time.time_ns()}.json',dict(torch=str(torch.__version__),torchvision=str(core.torchvision.__version__),numpy=np.__version__,python=sys.version,gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,determinismo=True))
  groups=core.read_data(a.datos);device=torch.device('cuda');final=a.salida/'final';final.mkdir(exist_ok=True)
  # Los checkpoints siguen guardándose cada época. Se evita recomprimir todo cada época.
  archive_original=core.archive;core.archive=lambda out: out/'indice_validacion.csv'
  args=SimpleNamespace(datos=a.datos,salida=final,batch=32,epocas=25,paciencia=5,min_epocas=5)
  completed_count=counts['completa']
  for i,row in enumerate(rows):
   if row['estado']=='completa':continue
   # Revalidar por si se creó una salida antes de continuar; nunca ignorar corrupción.
   model,seed=row['modelo'],row['semilla'];target=final/f'{model}_{seed}'
   if (target/'resultado.json').exists():rows[i]=completed(target,model,seed,hp);persist_index(a.salida,rows);continue
   if row['estado']=='reanudar':
    source=Path(row['ruta']);epoch=validate_checkpoint(source/'ultimo.pt',torch);print('Checkpoint válido:',model,seed,'última época guardada',epoch,flush=True);copy_resume(source,target)
   args.lr=hp[model]['lr'];args.weight_decay=hp[model]['weight_decay']
   print(f'CONTINUACIÓN E2: {completed_count}/66 completas. Siguiente: {model} {seed}',flush=True)
   core.run_one(model,seed,groups,args,device)
   rows[i]=completed(target,model,seed,hp);completed_count+=1;persist_index(a.salida,rows)
  require(all(r['estado']=='completa' for r in rows),'Quedaron corridas pendientes')
  save(a.salida/'estado_continuacion.json',dict(estado='DESARROLLO_22_COMPLETADO_SIN_ABRIR_RESERVA',corridas=66,baseline_heredado='resnet50',nota='Completa el plan anterior; no demuestra potencia suficiente para el nuevo Top 10.'))
  print('COMPLETADO: 22 semillas × 3 modelos. Reserva NO evaluada. Evidencia:',archive_original(a.salida),flush=True)
 finally:
  lock.unlink(missing_ok=True)

if __name__=='__main__':
 try:main()
 except (ValueError,FileNotFoundError,KeyError,json.JSONDecodeError) as e:raise SystemExit('DETENIDO SIN REENTRENAR AUTOMÁTICAMENTE: '+str(e))
