"""Estadística exploratoria de validación, semillas pareadas; no inferencia por paciente."""
import argparse,csv,hashlib,io,itertools,json,math,zipfile
from pathlib import Path
import numpy as np
from scipy import stats,optimize,integrate

def holm(ps):
 order=np.argsort(ps); result=np.empty(len(ps));prev=0
 for i,j in enumerate(order):
  prev=max(prev,min(1.,(len(ps)-i)*ps[j]));result[j]=prev
 return result.tolist()
def power(n,d,alpha):
 c=stats.t.ppf(1-alpha/2,n-1)
 nc=d*np.sqrt(n);df=n-1
 return float(integrate.quad(lambda v:(stats.norm.sf(c*np.sqrt(v/df)-nc)+stats.norm.cdf(-c*np.sqrt(v/df)-nc))*stats.chi2.pdf(v,df),0,stats.chi2.ppf(1-1e-11,df),epsabs=1e-8)[0])
def write_csv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def analyze(archive,out):
 out.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(archive) as z:
  rows=list(csv.DictReader(io.StringIO(z.read('indice_validacion.csv').decode())))
  assert len(rows)==66 and len({(r['modelo'],r['semilla']) for r in rows})==66
  assert all(r['estado']=='completa' for r in rows)
  verified=0
  for r in rows:
   p=f"final/{r['modelo']}_{r['semilla']}/resultado.json"
   if p in z.namelist():
    assert hashlib.sha256(z.read(p)).hexdigest()==r['resultado_sha256'];verified+=1
 models=['softmaxcnn_bn','densenet121','resnet50']; data={}
 for m in models:
  rr=sorted([r for r in rows if r['modelo']==m],key=lambda r:int(r['semilla']))
  assert [int(r['semilla']) for r in rr]==list(range(6101,6123))
  data[m]=np.array([float(r['macro_f1_validacion']) for r in rr])
 summary=[]
 for m,x in data.items():
  se=stats.sem(x);ci=stats.t.interval(.95,21,loc=x.mean(),scale=se)
  summary.append(dict(modelo=m,n=22,media=float(x.mean()),sd=float(x.std(ddof=1)),IC95_inf=ci[0],IC95_sup=ci[1]))
 pairs=[];rng=np.random.default_rng(20261111)
 for a,b in [('resnet50','densenet121'),('resnet50','softmaxcnn_bn'),('densenet121','softmaxcnn_bn')]:
  d=data[a]-data[b];sd=d.std(ddof=1);se=stats.sem(d);ci=stats.t.interval(.95,21,loc=d.mean(),scale=se)
  # Monte Carlo bilateral de cambios de signo; 100000 réplicas, corrección +1.
  perm=(rng.choice([-1,1],(100000,22))*d).mean(1)
  md=optimize.brentq(lambda delta:power(22,delta/sd,.05/3)-.8,1e-8,1.)
  mde10=optimize.brentq(lambda delta:power(22,delta/sd,.05/10)-.8,1e-8,1.)
  required=next(n for n in range(3,2001) if power(n,.02/sd,.05/10)>=.8)
  rank=stats.rankdata(abs(d[d!=0]));rb=float(np.sum(rank*np.sign(d[d!=0]))/sum(rank))
  pairs.append(dict(comparacion=a+' - '+b,n=22,diferencia=float(d.mean()),sd_diferencias=float(sd),IC95_inf=ci[0],IC95_sup=ci[1],cohen_dz=float(d.mean()/sd),hedges_gz=float((1-3/(4*21-1))*d.mean()/sd),rank_biserial=rb,shapiro_p=float(stats.shapiro(d).pvalue),t_p=float(stats.ttest_rel(data[a],data[b]).pvalue),wilcoxon_p=float(stats.wilcoxon(d,method='exact').pvalue),permutacion_p=float((1+(abs(perm)>=abs(d.mean())).sum())/100001),MDE80_familia3=md,MDE80_familia10=mde10,potencia_delta002_familia10=power(22,.02/sd,.05/10),R_delta002_familia10=required))
 for test in ['t','wilcoxon','permutacion']:
  for r,p in zip(pairs,holm([r[test+'_p'] for r in pairs])):r[test+'_p_Holm']=p
 matrix=np.stack([data[m] for m in models],1);ranks=stats.rankdata(-matrix,axis=1).mean(0);nemenyi=[]
 for i,j in itertools.combinations(range(3),2):
  q=abs(ranks[i]-ranks[j])/np.sqrt(3*4/(6*22))*np.sqrt(2)
  nemenyi.append(dict(comparacion=models[i]+' - '+models[j],p_Nemenyi=float(stats.studentized_range.sf(q,3,np.inf))))
 fr=stats.friedmanchisquare(*[data[m] for m in models]);omni=dict(friedman_estadistico=float(fr.statistic),friedman_p=float(fr.pvalue),kendall_W=float(fr.statistic/(22*2)),rangos_medios=dict(zip(models,ranks)),nemenyi=nemenyi)
 write_csv(out/'resumen.csv',summary);write_csv(out/'contrastes_pareados.csv',pairs);write_csv(out/'nemenyi.csv',nemenyi)
 (out/'omnibus.json').write_text(json.dumps(omni,indent=2))
 report={'fuente_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'resultados_individuales_verificados':verified,'previos_solo_indice':66-verified,'resumen':summary,'contrastes':pairs,'omnibus':omni}
 (out/'analisis_completo.json').write_text(json.dumps(report,indent=2))
 body=['# Validación de los modelos base — 22 semillas','', 'Análisis exploratorio posterior al cierre. Semillas pareadas 6101–6122; misma partición. La validación seleccionó época y antes hiperparámetros: no es prueba independiente. Los intervalos entre semillas no representan incertidumbre entre pacientes. Reserva no evaluada en esta continuación.','', '|Modelo|Media|DE|IC95 entre semillas|','|---|---:|---:|---|']
 for r in summary:body.append(f"|{r['modelo']}|{r['media']:.4f}|{r['sd']:.4f}|[{r['IC95_inf']:.4f}; {r['IC95_sup']:.4f}]|")
 body+=['','## Comparaciones pareadas','Corrección Holm dentro de las tres comparaciones, por cada método. t es el análisis principal; Wilcoxon y permutación son sensibilidad, no tres oportunidades de declarar éxito. IC95 de diferencias sin ajuste simultáneo. Shapiro examina diferencias; no determina retrospectivamente qué prueba se publica.','', '|Comparación|Δ F1|t Holm|Wilcoxon Holm|Permutación Holm|dz|','|---|---:|---:|---:|---:|---:|']
 for r in pairs:body.append(f"|{r['comparacion']}|{r['diferencia']:.4f}|{r['t_p_Holm']:.5g}|{r['wilcoxon_p_Holm']:.5g}|{r['permutacion_p_Holm']:.5g}|{r['cohen_dz']:.3f}|")
 body+=['',f"Friedman: χ²={fr.statistic:.4f}, p={fr.pvalue:.5g}; Kendall W={omni['kendall_W']:.4f}. Post hoc Nemenyi en archivo separado.",'','## Potencia y relevancia','δ=0,02 es el umbral de relevancia heredado. MDE calculada con t no central, 80% de potencia y Bonferroni conservador (tres contrastes exploratorios o diez modelos contra baseline). No es potencia observada calculada a partir del efecto obtenido. Las dispersiones base son escenarios de planificación, no sustituyen cinco semillas piloto de los nuevos modelos.','']
 for r in pairs:body.append(f"- {r['comparacion']}: DE pareada {r['sd_diferencias']:.4f}; MDE con R=22/familia10={r['MDE80_familia10']:.4f}; R estimado para δ=0,02/familia10={r['R_delta002_familia10']}.")
 body+=['','## Alcance de evidencia',f'{verified} resultados nuevos verificados contra SHA256; {66-verified} previos solo referenciados por el índice. Los checkpoints permanecen en Drive y no se han verificado sus bytes aquí. No se calculan McNemar, DeLong ni bootstrap completo por paciente con medias por semilla: requieren las predicciones antiguas y nuevas alineadas. No se cuentan 22 semillas como 22 cohortes independientes.','', '## Próximo bloque','Diez candidatos en cuatro familias. Primero perfilado y piloto; fijar después el tamaño final con dispersión pareada y presupuesto observado. Si el plazo impide potencia 80%, declarar limitación y MDE, sin presentar ausencia de significación como equivalencia.']
 (out/'Informe_estadistico_E2_22.md').write_text('\n'.join(body)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--zip',type=Path,required=True);p.add_argument('--salida',type=Path,required=True);a=p.parse_args();analyze(a.zip,a.salida)
