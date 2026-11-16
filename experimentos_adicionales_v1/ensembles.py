"""Análisis sin entrenamiento: ensembles fijados, incertidumbre y revisión humana."""
import argparse,csv,hashlib,itertools,json,zipfile
from pathlib import Path
import numpy as np
import estadistica_cierre as st
CLASSES=['tumor_nonMitotic','lymphocyte','nonTILnonMQ_stromal']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def uncertainty(p):
 p=np.asarray(p,float)
 if p.ndim!=3 or p.shape[2]!=3 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(-1),1,atol=1e-5):raise ValueError('Probabilidades inválidas')
 p=p/p.sum(-1,keepdims=True) # Corrección exclusivamente de redondeo de probabilidades
 mean=p.mean(0)
 def ent(q):return -(q*np.log(np.clip(q,1e-15,1))).sum(-1)
 entropy=ent(mean);mi=np.maximum(0,entropy-ent(p).mean(0))
 return mean,entropy,mi

def analyze(rows,models,out,scope):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 y=np.array([CLASSES.index(r['clase_real']) for r in rows]);patients=np.array([r['paciente'] for r in rows]);names=sorted(models);means={};summary=[];subgroups=[];risk=[];errors={}
 for name in names:
  mean,entropy,mi=uncertainty(models[name]);assert len(mean)==len(rows);means[name]=mean;pred=mean.argmax(1);errors[name]=pred!=y
  m=st.metrics(y,mean);summary.append(dict(modelo=name,n_semillas=len(models[name]),n=len(y),pacientes=len(set(patients)),macro_f1_ensemble=m['macro_f1'],macro_f1_individual_media=np.mean([st.f1cm(np.bincount(3*y+p.argmax(1),minlength=9).reshape(3,3)) for p in models[name]]),ece_fijo10=m['ece_fijo10'],brier=m['brier'],entropia_media=entropy.mean(),informacion_mutua_media=mi.mean()))
  st.save(out/(name+'_metricas.json'),m)
  st.table(out/(name+'_incertidumbre.csv'),[dict(archivo=r['archivo'],paciente=r['paciente'],entropia=entropy[i],informacion_mutua=mi[i],error=bool(pred[i]!=y[i])) for i,r in enumerate(rows)])
  for key in ['paciente','tss']:
   if not all(key in r for r in rows):continue
   for value in sorted({r[key] for r in rows}):
    ix=np.array([r[key]==value for r in rows]);cal=st.calibration(y[ix],mean[ix]);subgroups.append(dict(modelo=name,agrupacion=key,grupo=value,n=int(ix.sum()),macro_f1=st.f1cm(np.bincount(3*y[ix]+pred[ix],minlength=9).reshape(3,3)),ece=cal['ece_fijo10'],pendiente=cal['pendiente_toplabel'],intercepto=cal['intercepto_toplabel'],convergencia=cal['convergencia']))
  order=np.argsort(entropy,kind='stable')
  for fraction in [.25,.5,.75,1.]:
   ix=order[:max(1,int(len(y)*fraction))];risk.append(dict(modelo=name,fraccion_retenida=len(ix)/len(y),error_retenidos=float((pred[ix]!=y[ix]).mean()),nota='Curva descriptiva; no garantía de cobertura conforme'))
  # Simple random sample of errors: known inclusion probability, no morphology invented.
  candidates=np.flatnonzero(pred!=y);chosen=np.random.default_rng(15112026).choice(candidates,min(100,len(candidates)),replace=False);review=[]
  for i in sorted(chosen):review.append(dict(modelo=name,archivo=rows[i]['archivo'],paciente=patients[i],clase_real=CLASSES[y[i]],clase_predicha=CLASSES[pred[i]],confianza=mean[i].max(),probabilidad_inclusion=len(chosen)/len(candidates),categoria='',revisor='',observaciones=''))
  if review:st.table(out/(name+'_revision_errores.csv'),review)
 boot,_=st.patient_boot(y,np.array([means[n][None] for n in names]),patients)
 for j,r in enumerate(summary):r.update(IC95_paciente_inf=np.quantile(boot[:,j],.025),IC95_paciente_sup=np.quantile(boot[:,j],.975))
 st.table(out/'resumen_ensembles.csv',summary);st.table(out/'calibracion_subgrupos.csv',subgroups);st.table(out/'riesgo_retencion.csv',risk)
 pairs=[]
 for a,b in itertools.combinations(names,2):
  ea,eb=errors[a],errors[b];union=(ea|eb).sum();pairs.append(dict(modelo_a=a,modelo_b=b,errores_a=int(ea.sum()),errores_b=int(eb.sum()),errores_compartidos=int((ea&eb).sum()),jaccard_error=float((ea&eb).sum()/union) if union else 0,desacuerdo=float((means[a].argmax(1)!=means[b].argmax(1)).mean())))
 st.table(out/'errores_compartidos.csv',pairs)
 limitations=[]
 for name in names:
  gg=[r for r in subgroups if r['modelo']==name and r['agrupacion']=='paciente'];mm=next(r for r in summary if r['modelo']==name)
  limitations.append(dict(modelo=name,macro_f1_global=mm['macro_f1_ensemble'],min_macro_f1_paciente=min(r['macro_f1'] for r in gg),max_macro_f1_paciente=max(r['macro_f1'] for r in gg),brecha_global_peor_paciente=mm['macro_f1_ensemble']-min(r['macro_f1'] for r in gg),nota='Heterogeneidad descriptiva: soporte y clases por paciente distintos; no efecto causal ni límite del paper original'))
 st.table(out/'limitacion_agregacion_por_paciente.csv',limitations)
 st.save(out/'alcance.json',dict(alcance=scope,bootstrap_paciente=st.B,seleccion='Definida exclusivamente con validación; pesos uniformes',nota='MI entre semillas es un indicador de desacuerdo, no separa inequívocamente incertidumbre epistémica y aleatoria. Revisión humana pendiente. IC condicionado al protocolo y partición.'))
 (out/'LEER_REVISION.txt').write_text('Completar una categoría principal excluyente por error: ambigüedad morfológica, artefacto, recorte/contexto insuficiente, etiqueta probablemente incorrecta, confusión morfológica sin otro defecto, no evaluable. Registrar revisor y justificación; no cambiar la etiqueta original. Para tasas de ruido se requiere muestra aleatoria adicional de TODOS los casos; esta muestra de errores no estima ruido global. Los archivos corresponden a recortes originales en NuCLS_V2_DATOS.zip.\n')
 return summary

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cierre',type=Path,required=True);ap.add_argument('--salida',type=Path,required=True);a=ap.parse_args();source=a.cierre
 if a.salida.resolve()==source.resolve() or source.resolve() in a.salida.resolve().parents:raise ValueError('La salida debe estar separada del cierre fuente')
 if (source/'EJECUCION_ACTIVA.json').exists():raise RuntimeError('Cierre todavía activo')
 status=json.loads((source/'estado.json').read_text());assert status['estado']=='CIERRE_RESERVA_Y_ESTADISTICA_COMPLETADO','Esperar cierre completo'
 plan=json.loads((source/'entradas_fijadas.json').read_text());runs=plan['corridas'];seeds=plan['semillas'];top=sorted(plan['modelos'],key=lambda n:(-np.mean([r['validacion'] for r in runs if r['modelo']==n]),n))[:3]
 if len(runs)!=65 or len(seeds)!=5:raise ValueError('Se requieren 65 corridas del plan congelado')
 a.salida.mkdir(parents=True,exist_ok=True)
 identity=dict(plan_sha256=sha(source/'entradas_fijadas.json'),top3=top,codigo={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
 manifest=a.salida/'identidad.json'
 if manifest.exists() and json.loads(manifest.read_text())!=identity:raise ValueError('El análisis anterior tiene otra identidad')
 st.save(manifest,identity)
 rows=None;models={}
 for n in top:
  ps=[]
  for seed in seeds:
   folder=source/'corridas'/f'{n}_{seed}';mark=json.loads((folder/'completo.json').read_text())
   if mark['plan']!=identity['plan_sha256']:raise ValueError('Marca de otro plan')
   for f,h in mark['archivos'].items():
    if sha(folder/f)!=h:raise ValueError('Evidencia alterada: '+str(folder/f))
   rr=list(csv.DictReader((folder/'predicciones_reserva.csv').open()))
   keys=[(r['archivo'],r['paciente'],r['clase_real']) for r in rr]
   if rows is None:rows=rr
   elif keys!=[(r['archivo'],r['paciente'],r['clase_real']) for r in rows]:raise ValueError('Identidades desalineadas')
   with np.load(folder/'reserva.npz',allow_pickle=False) as z:
    if not np.array_equal(z['y'],[CLASSES.index(r['clase_real']) for r in rr]):raise ValueError('Etiquetas incompatibles')
    ps.append(z['p'])
  models[n]=np.array(ps)
 analyze(rows,models,a.salida,'Reserva heredada E1: análisis secundario de probabilidades congeladas; no nueva inferencia, selección ni ajuste')
 st.save(a.salida/'completo.json',dict(archivos={str(p.relative_to(a.salida)):sha(p) for p in a.salida.rglob('*') if p.is_file() and p.name!='completo.json'}))
 print('ENSEMBLES_COMPLETADOS',a.salida,flush=True)
if __name__=='__main__':main()
