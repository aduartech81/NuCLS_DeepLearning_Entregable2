"""Referencias exploratorias: mayoría de train y regresión logística con cuatro C."""
import argparse,sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'modelos'))
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import sklearn
import core_v2 as core

def features(root,rows):
 xs=[];ys=[]
 for r in rows:
  with Image.open(root/r['archivo']) as im:a=np.asarray(im.convert('RGB'),dtype=np.float32)/255
  # Medias por celdas de 8x8 y 16 bins RGB: representación fija, no entrenada.
  pooled=a.reshape(16,8,16,8,3).mean((1,3)).ravel()
  hist=np.concatenate([np.histogram(a[:,:,c],bins=16,range=(0,1),density=False)[0]/(128*128) for c in range(3)])
  xs.append(np.r_[pooled,hist]);ys.append(core.CLASSES.index(r['clase']))
 return np.asarray(xs),np.asarray(ys)
def main():
 p=argparse.ArgumentParser();p.add_argument('--datos',type=Path,required=True);p.add_argument('--salida',type=Path,required=True);a=p.parse_args()
 if a.salida.name!='NuCLS_E2_REFERENCIAS_SIMPLES':raise ValueError('Carpeta exclusiva requerida')
 if a.salida.is_symlink():raise ValueError('Salida enlazada no permitida')
 a.salida.mkdir(parents=True,exist_ok=True)
 if (a.salida/'resultado.json').exists() and all((a.salida/f).is_file() for f in ['logistica_predicciones.csv','mayoritaria_predicciones.csv','logistica.joblib']):print('Referencias ya calculadas');return
 groups=core.read_data(a.datos);x,y=features(a.datos,groups['train']);v,vy=features(a.datos,groups['validation'])
 scaler=StandardScaler().fit(x);x=scaler.transform(x);v=scaler.transform(v);trials=[];best=None
 for c in [.001,.01,.1,1.]:
  start=time.perf_counter();m=LogisticRegression(C=c,max_iter=2000,solver='lbfgs',random_state=7001);m.fit(x,y);prob=m.predict_proba(v)
  if int(m.n_iter_.max())>=2000:raise RuntimeError('La regresión no convergió; revisar antes de comparar')
  met=core.scores(vy,prob.argmax(1));trials.append(dict(C=c,segundos=time.perf_counter()-start,**met));print('Logística',c,met['macro_f1'],flush=True)
  if best is None or met['macro_f1']>best[0]:best=(met['macro_f1'],prob,c,m)
 majority=int(np.bincount(y,minlength=3).argmax());trivial=core.scores(vy,np.full(len(vy),majority))
 result=dict(estado='EXPLORATORIO_VALIDACION',sklearn=sklearn.__version__,busqueda_logistica=trials,C_elegido=best[2],mayoritaria=trivial,clase_mayoritaria=core.CLASSES[majority],nota='Deterministas: no se fabrican cinco repeticiones idénticas. Escalador ajustado únicamente a train.')
 for name,prob in [('logistica',best[1]),('mayoritaria',np.tile(np.eye(3)[majority],(len(vy),1)))]:
  records=[dict(paciente=r['paciente'],archivo=r['archivo'],clase_real=r['clase'],clase_predicha=core.CLASSES[int(prob[i].argmax())],p_tumor=float(prob[i,0]),p_linfocito=float(prob[i,1]),p_estroma=float(prob[i,2])) for i,r in enumerate(groups['validation'])];core.csv_write(a.salida/(name+'_predicciones.csv'),records)
 import joblib
 joblib.dump({'scaler':scaler,'model':best[3],'clases':core.CLASSES},a.salida/'logistica.joblib')
 core.save_json(a.salida/'resultado.json',result)
 print(core.archive(a.salida))
if __name__=='__main__':main()
