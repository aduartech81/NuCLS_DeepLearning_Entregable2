"""Estadística de checkpoints congelados: semillas y pacientes son ejes distintos."""
import itertools,json,csv,math,warnings
from pathlib import Path
import numpy as np
from scipy import stats,optimize,special
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score,matthews_corrcoef,roc_auc_score,average_precision_score,log_loss,confusion_matrix
B=2000
SEED=15112026

def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
 if isinstance(v,(np.integer,)):return int(v)
 if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
 return v

def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(p)
def table(p,rows):
 if not rows:return
 p=Path(p);tmp=p.with_suffix('.tmp')
 with tmp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(clean(rows))
 tmp.replace(p)
def holm(p):
 p=np.asarray(p,float);order=np.argsort(p);out=np.empty_like(p);out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))));return out

def f1cm(cm):
 cm=np.asarray(cm);tp=np.diagonal(cm,axis1=-2,axis2=-1);den=cm.sum(-1)+cm.sum(-2)
 return np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0).mean(-1)

def calibration(y,p):
 confidence=p.max(1);correct=(p.argmax(1)==y).astype(int);fixed=np.minimum((confidence*10).astype(int),9)
 groups=[np.where(fixed==i)[0] for i in range(10)]
 adaptive=np.array_split(np.argsort(confidence,kind='stable'),min(10,len(y)))
 def ece(parts):return sum(len(ix)/len(y)*abs(correct[ix].mean()-confidence[ix].mean()) for ix in parts if len(ix))
 slope=[]
 # Logística top-label: corrección frente a confianza. No es recalibración del modelo.
 x=np.log(np.clip(confidence,1e-7,1-1e-7)/np.clip(1-confidence,1e-7,1))
 if len(np.unique(correct))==2 and np.std(x)>1e-8:
  def obj(b):
   z=b[0]+b[1]*x;return np.logaddexp(0,z).sum()-correct@z
  fit=optimize.minimize(obj,[0.,1.],jac=lambda b:np.array([(special.expit(b[0]+b[1]*x)-correct).sum(),((special.expit(b[0]+b[1]*x)-correct)*x).sum()]),bounds=[(-30,30),(-30,30)],method='L-BFGS-B')
  ok=bool(fit.success and np.isfinite(fit.x).all() and max(abs(fit.x))<29.99)
  slope={'intercepto_toplabel':float(fit.x[0]) if ok else None,'pendiente_toplabel':float(fit.x[1]) if ok else None,'convergencia':ok}
 else:slope={'intercepto_toplabel':None,'pendiente_toplabel':None,'convergencia':False}
 return dict(ece_fijo10=ece(groups),ece_adaptativo10=ece(adaptive),**slope,fiabilidad=[dict(bin=i,n=len(ix),confianza=float(confidence[ix].mean()),accuracy=float(correct[ix].mean())) for i,ix in enumerate(groups) if len(ix)])

def metrics(y,p):
 pred=p.argmax(1);oh=np.eye(3)[y];cm=confusion_matrix(y,pred,labels=[0,1,2]);auc=[roc_auc_score(y==c,p[:,c]) if len(np.unique(y==c))==2 else np.nan for c in range(3)];ap=[average_precision_score(y==c,p[:,c]) if np.any(y==c) else np.nan for c in range(3)]
 return dict(macro_f1=f1cm(cm),f1_ponderado=f1_score(y,pred,average='weighted',zero_division=0),accuracy=accuracy_score(y,pred),balanced_accuracy=balanced_accuracy_score(y,pred),mcc=matthews_corrcoef(y,pred),auc_macro=np.mean(auc),ap_macro=np.mean(ap),auc_por_clase=auc,ap_por_clase=ap,log_loss=log_loss(y,p,labels=[0,1,2]),brier=float(((p-oh)**2).sum(1).mean()),confusion=cm.tolist(),**calibration(y,p))

def paired(a,b,alpha=.05):
 d=np.asarray(a)-np.asarray(b);n=len(d);sd=d.std(ddof=1);mean=d.mean()
 if sd>0:t=float(stats.ttest_rel(a,b).pvalue);dz=mean/sd;half=stats.t.ppf(1-alpha/2,n-1)*sd/np.sqrt(n)
 else:t=1. if mean==0 else 0.;dz=None;half=0.
 w=1. if np.all(d==0) else float(stats.wilcoxon(d,method='auto').pvalue)
 sh=1. if sd==0 else float(stats.shapiro(d).pvalue)
 # Permutación de signos exacta para las cinco semillas comunes.
 signs=np.array(list(itertools.product([-1,1],repeat=n))) if n<=15 else np.random.default_rng(SEED).choice([-1,1],(10000,n))
 perm=float(np.mean(np.abs((signs*d).mean(1))>=abs(mean)-1e-14))
 rng=np.random.default_rng(SEED);ds=d[rng.integers(0,n,(B,n))];sds=ds.std(1,ddof=1);valid=sds>0;eff=ds[valid].mean(1)/sds[valid]
 ci=np.quantile(eff,[.025,.975]).tolist() if len(eff) else [None,None]
 def power(effect,nn):
  crit=stats.t.ppf(1-.005/2,nn-1);nc=effect*np.sqrt(nn);return stats.nct.sf(crit,nn-1,nc)+stats.nct.cdf(-crit,nn-1,nc)
 if sd>0:
  mde=optimize.brentq(lambda effect:power(effect,n)-.8,1e-6,100)*sd
  ns=np.arange(2,1001);good=ns[power(.02/sd,ns)>=.8];req=int(good[0]) if len(good) else None;pow_now=float(power(.02/sd,n))
 else:mde=req=pow_now=None
 return dict(delta=float(mean),sd_diferencias=float(sd),IC95_t_inf=mean-half,IC95_t_sup=mean+half,p_t=t,p_wilcoxon=w,p_permutacion=perm,p_shapiro=sh,d_z=dz,d_z_IC95_inf=ci[0],d_z_IC95_sup=ci[1],bootstrap_efecto_validos=int(valid.sum()),potencia_delta002_alpha0005=pow_now,R_aproximado_potencia80=req,MDE_alpha0005=mde)

def patient_boot(y,probs,patients):
 # probs: modelos × semillas × imágenes × clases; remuestreo idéntico para todos.
 groups=np.unique(patients);cms=np.empty((probs.shape[0],probs.shape[1],len(groups),3,3),int)
 for i in range(probs.shape[0]):
  for j in range(probs.shape[1]):
   for g,name in enumerate(groups):
    ix=patients==name;cms[i,j,g]=confusion_matrix(y[ix],probs[i,j,ix].argmax(1),labels=[0,1,2])
 draw=np.random.default_rng(SEED).integers(0,len(groups),(B,len(groups)));weights=np.array([np.bincount(x,minlength=len(groups)) for x in draw])
 samples=f1cm(np.einsum('bg,msgij->bmsij',weights,cms)).mean(-1)
 return samples,cms

def delong_binary(y,a,b):
 # DeLong por placements. Diagnóstico iid por recorte: no inferencia clínica agrupada.
 pos=y==1;neg=~pos
 if pos.sum()<2 or neg.sum()<2:return {'delta_auc':None,'p_delong_iid':None}
 vs=[];ws=[];aucs=[]
 for score in [a,b]:
  diff=score[pos,None]-score[None,neg];h=(diff>0)+.5*(diff==0);vs.append(h.mean(1));ws.append(h.mean(0));aucs.append(h.mean())
 covariance=np.cov(np.array(vs),ddof=1)/pos.sum()+np.cov(np.array(ws),ddof=1)/neg.sum();v=np.array([1,-1])@covariance@np.array([1,-1]);delta=aucs[0]-aucs[1]
 p=2*stats.norm.sf(abs(delta)/np.sqrt(v)) if v>1e-18 else (1. if abs(delta)<1e-12 else 0.)
 return dict(delta_auc=delta,p_delong_iid=p)

def analyze(out,plan,all_predictions,y,patients,rows):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=Path(out);out.mkdir(exist_ok=True);names=plan['modelos'];base=names.index('resnet50');seeds=plan['semillas'];pred=np.stack([[all_predictions[(name,s)] for s in seeds] for name in names]);values=np.zeros((len(seeds),len(names)));runs=[];by_patient=[]
 for i,name in enumerate(names):
  for j,s in enumerate(seeds):
   m=metrics(y,pred[i,j]);values[j,i]=m['macro_f1'];save(out/f'{name}_{s}_metricas.json',m)
   runs.append(dict(modelo=name,semilla=s,**{k:m[k] for k in ['macro_f1','f1_ponderado','accuracy','balanced_accuracy','mcc','auc_macro','ap_macro','log_loss','brier','ece_fijo10','ece_adaptativo10','pendiente_toplabel','intercepto_toplabel']}))
   for pt in np.unique(patients):
    ix=patients==pt;by_patient.append(dict(modelo=name,semilla=s,paciente=pt,n=int(ix.sum()),macro_f1=f1cm(confusion_matrix(y[ix],pred[i,j,ix].argmax(1),labels=[0,1,2]))))
 table(out/'metricas_por_semilla.csv',runs);table(out/'metricas_por_paciente.csv',by_patient)
 if rows:
  sites=np.array([r['tss'] for r in rows]);by_site=[]
  for i,name in enumerate(names):
   for j,seed in enumerate(seeds):
    for site in np.unique(sites):
     ix=sites==site;by_site.append(dict(modelo=name,semilla=seed,tss=site,n=int(ix.sum()),pacientes=len(np.unique(patients[ix])),macro_f1=f1cm(confusion_matrix(y[ix],pred[i,j,ix].argmax(1),labels=[0,1,2]))))
  table(out/'metricas_por_tss.csv',by_site)
 boot,cms=patient_boot(y,pred,patients);summary=[];contrasts=[]
 for i,name in enumerate(names):
  ci=np.quantile(boot[:,i],[.025,.975]);summary.append(dict(modelo=name,R=len(seeds),macro_f1_media=values[:,i].mean(),macro_f1_sd=values[:,i].std(ddof=1),IC95_paciente_inf=ci[0],IC95_paciente_sup=ci[1]))
  if name not in plan['adicionales']:continue
  q=paired(values[:,i],values[:,base]);dif=boot[:,i]-boot[:,base];ci=np.quantile(dif,[.025,.975]);q.update(modelo=name,IC95_delta_paciente_inf=ci[0],IC95_delta_paciente_sup=ci[1])
  # Exacta de intercambiabilidad de modelos dentro de paciente (todas las semillas juntas).
  ds=[]
  for swap in itertools.product([False,True],repeat=len(np.unique(patients))):
   mask=np.array(swap)[None,:,None,None];a=np.where(mask,cms[base],cms[i]).sum(1);b=np.where(mask,cms[i],cms[base]).sum(1);ds.append((f1cm(a)-f1cm(b)).mean())
  q['p_permutacion_paciente']=np.mean(np.abs(ds)>=abs(q['delta'])-1e-12);contrasts.append(q)
 for test in ['t','wilcoxon','permutacion','permutacion_paciente']:
  for q,pv in zip(contrasts,holm([x['p_'+test] for x in contrasts])):q['p_'+test+'_Holm10']=pv
 table(out/'contrastes_vs_resnet50.csv',contrasts);table(out/'resumen_reserva.csv',summary)
 ranks=np.apply_along_axis(lambda x:stats.rankdata(-x),1,values);meanrank=ranks.mean(0);n,k=values.shape
 if np.all(np.ptp(values,axis=1)==0):fried={'estadistico':0.,'p':1.}
 else:
  f=stats.friedmanchisquare(*values.T);fried={'estadistico':f.statistic,'p':f.pvalue}
 se=np.sqrt(k*(k+1)/(6*n));cd=stats.studentized_range.ppf(.95,k,np.inf)/np.sqrt(2)*se
 nemenyi=[]
 for i in range(k):
  for j in range(i+1,k):nemenyi.append(dict(modelo_a=names[i],modelo_b=names[j],p_nemenyi=stats.studentized_range.sf(abs(meanrank[i]-meanrank[j])/se*np.sqrt(2),k,np.inf)))
 table(out/'nemenyi.csv',nemenyi);save(out/'omnibus.json',dict(Friedman=fried,CD95=cd,rangos=dict(zip(names,meanrank)),nota='Semillas como bloques condicionadas a partición; no equivalencia ni inferencia de población. Nemenyi se interpreta tras omnibus.'))
 order=np.argsort(meanrank);fig,ax=plt.subplots(figsize=(10,7));ax.scatter(meanrank[order],np.arange(k),color='#087c85');ax.set_yticks(np.arange(k),np.array(names)[order]);ax.set_xlim(.5,k+.5);ax.set_xlabel('Rango medio (menor = mejor)');ax.set_title('Rangos y diferencia crítica Nemenyi · reserva');ax.plot([1,1+cd],[-1,-1],color='#b85f13',lw=3);ax.text(1,-1.35,f'CD95 = {cd:.2f}; R={n}',fontsize=10);ax.invert_yaxis();fig.tight_layout();fig.savefig(out/'diferencia_critica.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,7));means=values.mean(0);ci=np.quantile(boot,[.025,.975],axis=0);order=np.argsort(means);ax.hlines(np.arange(k),ci[0,order],ci[1,order],color='#087c85');ax.scatter(means[order],np.arange(k),color='#087c85');ax.set_yticks(np.arange(k),np.array(names)[order]);ax.set_xlabel('Macro-F1: media de 5 semillas; IC95 por paciente');fig.tight_layout();fig.savefig(out/'benchmark_reserva.png',dpi=160);plt.close(fig)
 diagnostics=[]
 for name in plan['adicionales']:
  i=names.index(name);a=pred[i,0];b=pred[base,0];ac=a.argmax(1)==y;bc=b.argmax(1)==y;b01=int((~ac&bc).sum());b10=int((ac&~bc).sum());p_mc=stats.binomtest(b01,b01+b10,.5).pvalue if b01+b10 else 1.
  for c in range(3):diagnostics.append(dict(modelo=name,semilla=seeds[0],clase=c,**delong_binary((y==c).astype(int),a[:,c],b[:,c]),p_mcnemar_iid=p_mc,nota='Diagnóstico iid por recorte; no contraste clínico. Complementar con permutación por paciente.'))
 for q,pv in zip(diagnostics,holm([x['p_delong_iid'] if x['p_delong_iid'] is not None else 1 for x in diagnostics])):q['p_delong_Holm30_iid']=pv
 mc=holm([diagnostics[i*3]['p_mcnemar_iid'] for i in range(len(plan['adicionales']))])
 for i,q in enumerate(diagnostics):q['p_mcnemar_Holm10_iid']=mc[i//3]
 table(out/'diagnosticos_delong_mcnemar_iid.csv',diagnostics)
 aucnull=[]
 for i,name in enumerate(names):
  for c in range(3):
   truth=y==c;score=pred[i,0,:,c];a=score[truth];b=score[~truth]
   if len(a)<2 or len(b)<2:continue
   d=a[:,None]-b[None,:];h=(d>0)+.5*(d==0);auc=float(h.mean());var=h.mean(1).var(ddof=1)/len(a)+h.mean(0).var(ddof=1)/len(b)
   pv=stats.norm.sf((auc-.5)/np.sqrt(var)) if var>0 else (0. if auc>.5 else 1.)
   aucnull.append(dict(modelo=name,semilla=seeds[0],clase=c,auc=auc,p_auc_mayor_05_iid=pv,nota='Wald DeLong diagnóstico iid; no prueba clínica agrupada'))
 for r,pv in zip(aucnull,holm([r['p_auc_mayor_05_iid'] for r in aucnull])):r['p_Holm39_iid']=pv
 table(out/'diagnostico_auc_frente_05_iid.csv',aucnull)
 return summary
