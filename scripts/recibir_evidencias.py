"""Analiza ZIP exportados sin extraerlos ni modificar originales. Solo validación."""
import argparse,csv,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,accuracy_score,matthews_corrcoef,roc_auc_score,average_precision_score,confusion_matrix
CLASSES=['tumor','linfocito','estroma']
def rows(data):return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
def probabilities(rr):
 if not rr:raise ValueError('Predicciones vacías')
 keys=[(r['paciente'],r['archivo']) for r in rr]
 if len(set(keys))!=len(keys):raise ValueError('Predicciones duplicadas')
 labels={'tumor_nonMitotic':0,'lymphocyte':1,'nonTILnonMQ_stromal':2,'tumor':0,'linfocito':1,'estroma':2};y=np.array([labels[r['clase_real']] for r in rr]);p=np.array([[float(r['p_'+c]) for c in CLASSES] for r in rr])
 if not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('Probabilidades inválidas')
 return y,p

def metrics(y,p):
 pred=p.argmax(1);oh=np.eye(3)[y];confidence=p.max(1);correct=pred==y
 bins=np.minimum((confidence*10).astype(int),9);ece=0.;cal=[]
 for b in range(10):
  ix=bins==b
  if ix.any():
   acc=float(correct[ix].mean());conf=float(confidence[ix].mean());ece+=float(ix.mean())*abs(acc-conf);cal.append(dict(bin=b,limite_inferior=b/10,limite_superior=(b+1)/10,n=int(ix.sum()),accuracy=acc,confianza=conf))
 auc=[float(roc_auc_score(oh[:,c],p[:,c])) if len(np.unique(oh[:,c]))==2 else None for c in range(3)]
 ap=[float(average_precision_score(oh[:,c],p[:,c])) if oh[:,c].sum()>0 else None for c in range(3)]
 return dict(n=len(y),macro_f1=float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),accuracy=float(accuracy_score(y,pred)),mcc=float(matthews_corrcoef(y,pred)),brier_multiclase_suma=float(((p-oh)**2).sum(1).mean()),ece_10=ece,auc_por_clase=auc,ap_por_clase=ap,matriz_confusion=confusion_matrix(y,pred,labels=[0,1,2]).tolist(),calibracion=cal)

def analyze(archives,out):
 if out.exists():raise ValueError('Usa una carpeta nueva; no se sobrescriben informes')
 records=[];sources=[];seen={}
 for archive in archives:
  sources.append(dict(archivo=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
  with zipfile.ZipFile(archive) as z:
   if z.testzip():raise ValueError('ZIP corrupto')
   for name in sorted(z.namelist()):
    if Path(name).name!='predicciones_validacion.csv':continue
    parent=name.rsplit('/',1)[0];meta=parent+'/resultado.json'
    if meta not in z.namelist():raise ValueError('Falta resultado.json junto a '+name)
    r=json.loads(z.read(meta));rr=rows(z.read(name));y,p=probabilities(rr)
    identity=(r['modelo'],int(r['semilla']),parent)
    digest=hashlib.sha256(z.read(name)).hexdigest()
    if identity in seen:
     if seen[identity]!=digest:raise ValueError('Dos versiones incompatibles: '+str(identity))
     continue
    seen[identity]=digest
    m=metrics(y,p);reported=r.get('validacion',{}).get('macro_f1')
    if reported is not None and abs(m['macro_f1']-reported)>1e-7:raise ValueError('Macro F1 no coincide: '+name)
    patients=np.array([v['paciente'] for v in rr]);pm=[]
    for patient in sorted(set(patients)):
     ix=patients==patient;v=metrics(y[ix],p[ix]);pm.append(dict(paciente=patient,n=int(ix.sum()),macro_f1=v['macro_f1'],accuracy=v['accuracy']))
    phase='busqueda' if '/busqueda/' in '/'+parent+'/' else ('piloto' if '/piloto/' in '/'+parent+'/' else 'otro_validar_protocolo')
    records.append(dict(modelo=r['modelo'],semilla=int(r['semilla']),fase=phase,ruta=parent,predicciones_sha256=digest,metricas=m,por_paciente=pm))
 if not records:raise ValueError('No hay predicciones de validación con resultado.json; se requiere exportación completa')
 out.mkdir(parents=True)
 report=dict(estado='ANALISIS_DESCRIPTIVO_VALIDACION',fuentes=sources,corridas=records,limitaciones=['Validación utilizada para ajuste: no es test independiente','No se agrupan configuraciones de búsqueda con semillas de piloto','Sin inferencia confirmatoria por paciente ni ajuste de temperatura','No evalúa reserva ni accede a checkpoints'])
 (out/'recepcion.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
 flat=[dict(modelo=r['modelo'],semilla=r['semilla'],fase=r['fase'],ruta=r['ruta'],**{k:r['metricas'][k] for k in ['n','macro_f1','accuracy','mcc','brier_multiclase_suma','ece_10']}) for r in records]
 with (out/'metricas_validacion.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 return len(records)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--zip',nargs='+',type=Path,required=True);p.add_argument('--salida',type=Path,required=True);a=p.parse_args();print('Corridas analizadas:',analyze(a.zip,a.salida))
