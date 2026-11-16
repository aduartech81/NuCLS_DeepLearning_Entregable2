from pathlib import Path
import csv,json,hashlib,math,html,shutil
from datetime import datetime,timezone
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]; E=R/'evidencias/piloto_diez_20261115'
C=['tumor_nonMitotic','lymphocyte','nonTILnonMQ_stromal']; rows=[]; histories={}; reference=None
for d in sorted(E.iterdir()):
 if not d.is_dir():continue
 result=json.loads((d/'resultado.json').read_text()); hist=json.loads((d/'historia.json').read_text()); pred=list(csv.DictReader((d/'predicciones_validacion.csv').open()))
 assert result['estado']=='FASE_COMPLETADA_VALIDACION'
 assert d.name==f"{result['modelo']}_{result['semilla']}" and result['semilla'] in range(6101,6106)
 assert len(pred)==2049
 identities=sorted((p['paciente'],p['archivo'],p['clase_real']) for p in pred)
 assert len(set(x[1] for x in identities))==2049 and len(set(x[0] for x in identities))==21
 if reference is None:reference=identities
 assert identities==reference
 cm=np.zeros((3,3),int);brier=[];conf=[];correct=[]
 for p in pred:
  probs=np.array([float(p[k]) for k in ['p_tumor','p_linfocito','p_estroma']]); assert np.isfinite(probs).all() and (probs>=0).all() and (probs<=1).all() and abs(probs.sum()-1)<1e-5
  y=C.index(p['clase_real']);yp=C.index(p['clase_predicha']);assert probs.argmax()==yp
  cm[y,yp]+=1;brier.append(sum((probs-np.eye(3)[y])**2));conf.append(probs.max());correct.append(y==yp)
 f1=np.divide(2*np.diag(cm),cm.sum(0)+cm.sum(1),out=np.zeros(3),where=(cm.sum(0)+cm.sum(1))>0).mean()
 assert np.array_equal(cm,result['validacion']['confusion']) and abs(f1-result['validacion']['macro_f1'])<1e-10
 assert len(hist)==result['epocas_ejecutadas'] and [h['epoca'] for h in hist]==list(range(1,len(hist)+1))
 assert abs(hist[result['mejor_epoca']-1]['val_macro_f1']-f1)<1e-10 and abs(max(h['val_macro_f1'] for h in hist)-f1)<1e-10
 bins=np.minimum((np.array(conf)*10).astype(int),9);ece=sum(np.mean(bins==b)*abs(np.mean(np.array(conf)[bins==b])-np.mean(np.array(correct)[bins==b])) for b in range(10) if (bins==b).any())
 rows.append(dict(modelo=result['modelo'],semilla=result['semilla'],macro_f1=float(f1),accuracy=float(np.trace(cm)/cm.sum()),brier=float(np.mean(brier)),ece10=float(ece),epocas=len(hist),mejor_epoca=result['mejor_epoca'],segundos_registrados=result['segundos'],latencia_ms_reportada=result['inferencia_ms_por_recorte_mediana'],parametros=result['parametros']))
 histories[d.name]=hist
assert len(rows)==50
models=sorted(set(x['modelo'] for x in rows));assert len(models)==10
summ=[]
for m in models:
 rr=[x for x in rows if x['modelo']==m];assert sorted(x['semilla'] for x in rr)==list(range(6101,6106))
 vals=np.array([x['macro_f1'] for x in rr]);sd=vals.std(ddof=1);delta=t.ppf(.975,4)*sd/np.sqrt(5)
 summ.append(dict(modelo=m,R=5,media=float(vals.mean()),DE=float(sd),IC95_inf=float(vals.mean()-delta),IC95_sup=float(vals.mean()+delta),minimo=float(vals.min()),maximo=float(vals.max()),brier_media=float(np.mean([x['brier'] for x in rr])),ece10_media=float(np.mean([x['ece10'] for x in rr])),minutos_registrados=float(sum(x['segundos_registrados'] for x in rr)/60),latencia_ms_mediana=float(np.median([x['latencia_ms_reportada'] for x in rr])),parametros=rr[0]['parametros']))
summ.sort(key=lambda x:x['media'],reverse=True)
def savecsv(name,rr):
 with (E/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
savecsv('metricas_recalculadas.csv',rows);savecsv('resumen.csv',summ)
(E/'verificacion.json').write_text(json.dumps(dict(fecha_utc=datetime.now(timezone.utc).isoformat(),corridas=50,modelos=10,semillas=list(range(6101,6106)),predicciones_verificadas=50*2049,identidades=2049,pacientes=21,macro_f1_y_confusion_coinciden=True,historias_y_mejor_epoca_coinciden=True,reserva_consultada=False,checkpoints_descargados=False,Colab_intervenido=False,E1_modificado=False),indent=2))
(E/'SHA256.json').write_text(json.dumps({str(p.relative_to(E)):hashlib.sha256(p.read_bytes()).hexdigest() for p in E.rglob('*') if p.is_file() and p.name!='SHA256.json'},indent=2))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f5f8fb','axes.facecolor':'#f5f8fb','text.color':'#15334a','axes.labelcolor':'#15334a'})
fig,ax=plt.subplots(figsize=(11,7));
for i,s in enumerate(summ):
 vals=[r['macro_f1'] for r in rows if r['modelo']==s['modelo']]
 ax.plot([s['IC95_inf'],s['IC95_sup']],[i,i],color='#128c86',lw=3)
 ax.scatter(vals,np.arange(5)*.025+i-.05,color='#819aaf',s=22,zorder=3)
 ax.scatter(s['media'],i,color='#113951',s=65,zorder=4)
 ax.text(.91,i,f"{s['media']:.4f} ± {s['DE']:.4f}",va='center',fontsize=10)
ax.set_yticks(range(10),[s['modelo'] for s in summ]);ax.invert_yaxis();ax.set_xlim(.60,1.07);ax.set_xticks([.6,.7,.8,.9,1]);ax.grid(axis='x',alpha=.15);ax.set_xlabel('Macro-F1 en validación');ax.set_title('Diez modelos · cinco semillas por modelo',loc='left',fontweight='bold',pad=20)
fig.text(.02,.015,'Puntos: semillas 6101–6105. Línea: IC95 t de la media entre semillas; condicionado a selección en validación.',fontsize=9);fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(E/'comparacion.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(3,4,figsize=(15,10),sharex=True,sharey=True)
for ax,m in zip(axes.flat,models):
 for seed in range(6101,6106):
  h=histories[f'{m}_{seed}'];ax.plot([x['epoca'] for x in h],[x['val_macro_f1'] for x in h],label=str(seed),alpha=.8)
 ax.set_title(m,fontsize=9);ax.set_ylim(0,1);ax.grid(alpha=.15);ax.set_xlabel('Época');ax.set_ylabel('Macro-F1 validación')
axes.flat[-2].axis('off');axes.flat[-1].axis('off');axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc='center',title='Semilla')
fig.suptitle('Historias completas · parada temprana sin rellenar épocas ausentes',fontweight='bold');fig.tight_layout();fig.savefig(E/'curvas_validacion.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(3,4,figsize=(15,10),sharex=True)
for ax,m in zip(axes.flat,models):
 for seed in range(6101,6106):
  h=histories[f'{m}_{seed}'];ax.plot([x['epoca'] for x in h],[x['train_loss'] for x in h],color='#128c86',alpha=.5);ax.plot([x['epoca'] for x in h],[x['val_loss'] for x in h],color='#ca7246',alpha=.5,ls='--')
 ax.set_title(m,fontsize=9);ax.grid(alpha=.15);ax.set_xlabel('Época');ax.set_ylabel('Pérdida')
axes.flat[-2].axis('off');axes.flat[-1].axis('off');axes.flat[-1].text(.1,.5,'Verde: entrenamiento\nNaranja discontinua: validación\nCinco trayectorias por modelo',transform=axes.flat[-1].transAxes)
fig.suptitle('Pérdida por época · entrenamiento y validación',fontweight='bold');fig.tight_layout();fig.savefig(E/'curvas_perdida.png',dpi=150);plt.close(fig)

(E/"resumen.json").write_text(json.dumps(summ,indent=2))
print(json.dumps(summ,indent=2))
