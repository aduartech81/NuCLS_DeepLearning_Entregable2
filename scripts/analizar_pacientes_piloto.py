from pathlib import Path
import csv,json,statistics,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
p=argparse.ArgumentParser();p.add_argument('--paquete',type=Path,required=True);a=p.parse_args()
R=a.paquete;E=R/'evidencias/piloto_siete_20261115'
C=['tumor_nonMitotic','lymphocyte','nonTILnonMQ_stromal'];rows=[];supports={}
for d in sorted(E.iterdir()):
 if not d.is_dir():continue
 model,seed=d.name.rsplit('_',1);pred=list(csv.DictReader((d/'predicciones_validacion.csv').open()))
 for patient in sorted({p['paciente'] for p in pred}):
  pp=[p for p in pred if p['paciente']==patient];cm=np.zeros((3,3),int)
  for p in pp:cm[C.index(p['clase_real']),C.index(p['clase_predicha'])]+=1
  supports[patient]=cm.sum(1).tolist()
  f1=np.divide(2*np.diag(cm),cm.sum(0)+cm.sum(1),out=np.zeros(3),where=(cm.sum(0)+cm.sum(1))>0).mean()
  rows.append(dict(modelo=model,semilla=int(seed),paciente=patient,n=len(pp),clases_presentes=int((cm.sum(1)>0).sum()),macro_f1_tres_clases=float(f1),accuracy=float(np.trace(cm)/cm.sum())))
assert len(rows)==735
with (E/'metricas_paciente_recalculadas.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
models=sorted({r['modelo'] for r in rows});patients=sorted(supports)
means=np.array([[np.mean([r['macro_f1_tres_clases'] for r in rows if r['modelo']==m and r['paciente']==p]) for m in models] for p in patients])
fig,ax=plt.subplots(figsize=(12,10));im=ax.imshow(means,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
ax.set_xticks(range(7),models,rotation=35,ha='right',fontsize=9);ax.set_yticks(range(21),[f"{p} · n={sum(supports[p])}"+(' *' if sum(n>0 for n in supports[p])<3 else '') for p in patients],fontsize=9)
for i in range(21):
 for j in range(7):ax.text(j,i,f'{means[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if means[i,j]>.65 else '#132b40')
fig.colorbar(im,ax=ax,label='Macro-F1 medio de cinco semillas, tres clases fijas');ax.set_title('Variación entre pacientes · validación',pad=18,fontweight='bold');fig.text(.02,.01,'* Paciente sin las tres clases reales: macro-F1 fijo incluye ceros de clases ausentes; no usar para ordenar pacientes.',fontsize=9);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(E/'pacientes.png',dpi=150);plt.close(fig)
shutil.copy2(E/'pacientes.png',R/'book/figures/piloto_siete_pacientes.png')
