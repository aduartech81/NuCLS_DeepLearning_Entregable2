"""Inventario y congelamiento previo a evaluación. No entrena ni lee imágenes de reserva."""
import argparse,csv,hashlib,json,datetime
from pathlib import Path
BASES=['softmaxcnn_bn','densenet121','resnet50']
SIETE=['resnet18','mobilenetv3_small_100','efficientnet_b0','deit_tiny_patch16_224','swin_tiny_patch4_window7_224','mobilevit_xxs','mixer_s16_224']
NUEVOS=['convnext_tiny','inception_v3','sequencer2d_s']
SEEDS=range(6101,6106)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def inspect_run(path,model,seed):
 result=path/'resultado.json';weights=path/'mejor.pt';history=path/'historia.json';pred=path/'predicciones_validacion.csv'
 missing=[x.name for x in (result,weights,history,pred) if not x.is_file()]
 if missing:return None,{'modelo':model,'semilla':seed,'ruta':str(path),'faltantes':missing}
 q=json.loads(result.read_text());hist=json.loads(history.read_text())
 if (q['modelo'],q['semilla'],q['estado'])!=(model,seed,'FASE_COMPLETADA_VALIDACION'):raise ValueError('Identidad/estado incompatible: '+str(result))
 if len(hist)!=q['epocas_ejecutadas'] or abs(max(x['val_macro_f1'] for x in hist)-q['validacion']['macro_f1'])>1e-9:raise ValueError('Historia incompatible: '+str(path))
 with pred.open() as f: rr=list(csv.DictReader(f))
 if len(rr)!=2049 or len({x['archivo'] for x in rr})!=2049:raise ValueError('Validación incompleta o duplicada: '+str(pred))
 return {'modelo':model,'semilla':seed,'ruta':str(path.resolve()),'lr':q['lr'],'weight_decay':q['weight_decay'],'mejor_epoca':q['mejor_epoca'],'archivos':{x.name:sha(x) for x in (result,weights,history,pred)}},None

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--indice-bases',type=Path,required=True);p.add_argument('--adicionales',type=Path,required=True);p.add_argument('--nuevos',type=Path,required=True);p.add_argument('--salida',type=Path,required=True);p.add_argument('--congelar',action='store_true');a=p.parse_args()
 with a.indice_bases.open() as f: index=list(csv.DictReader(f))
 locations={}
 for x in index:
  key=(x['modelo'],int(x['semilla']))
  if key in locations:raise ValueError('Índice duplicado: '+str(key))
  locations[key]=Path(x['ruta'])
 runs=[];missing=[]
 for name in BASES+SIETE+NUEVOS:
  for seed in SEEDS:
   root=locations.get((name,seed)) if name in BASES else (a.adicionales if name in SIETE else a.nuevos)/'piloto'/f'{name}_{seed}'
   if root is None:missing.append({'modelo':name,'semilla':seed,'faltantes':['ruta en índice']});continue
   run,error=inspect_run(root,name,seed)
   if error:missing.append(error)
   else:runs.append(run)
 plan={'fecha_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'estado':'INVENTARIO','modelo_referencia':'resnet50','semillas_primarias':list(SEEDS),'regla':'Primeras cinco semillas comunes fijadas por diseño; no seleccionar por score. Las 22 bases se reportan adicionalmente, sin duplicarlas para emparejar.','reserva':'Partición heredada, expuesta en E1; evaluar solo tras congelar; sin reajustes posteriores.','margen_macro_f1':0.02,'bootstrap_paciente_minimo':2000,'alcance':'Comparación exploratoria; potencia y diferencia mínima detectable deben informarse.','corridas':runs,'faltantes':missing}
 print(json.dumps({'completas':len(runs),'esperadas':65,'faltantes':missing},ensure_ascii=False,indent=2))
 if not a.congelar:return
 if missing:raise RuntimeError('No se congela el conjunto de 13: faltan corridas. No se altera ningún entrenamiento.')
 for folder in {a.adicionales,a.nuevos,a.adicionales.parent/'NuCLS_E2_COLA_UNICA_V1'}:
  if (folder/'EJECUCION_ACTIVA.json').exists():raise RuntimeError('Hay una marca de ejecución. Verificar cierre; no se elimina.')
 # Detectar modificación concurrente antes de emitir el manifiesto.
 for run in runs:
  for filename,h in run['archivos'].items():
   if sha(Path(run['ruta'])/filename)!=h:raise RuntimeError('La evidencia cambió durante el inventario.')
 plan['estado']='CONGELADO_PREVIO_A_RESERVA'
 a.salida.parent.mkdir(parents=True,exist_ok=True)
 with a.salida.open('x') as f:json.dump(plan,f,ensure_ascii=False,indent=2)
 print('MANIFIESTO CREADO',a.salida,'SHA256',sha(a.salida))
if __name__=='__main__':main()
