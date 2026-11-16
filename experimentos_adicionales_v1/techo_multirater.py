"""Acuerdo leave-one-rater-out en macro-F1, como proxy; requiere anotaciones reales."""
import argparse,csv,collections,json
from pathlib import Path
import numpy as np
import estadistica_cierre as st
from ensembles import CLASSES

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--anotaciones',type=Path,required=True);ap.add_argument('--salida',type=Path,required=True);a=ap.parse_args();rr=list(csv.DictReader(a.anotaciones.open()));groups=collections.defaultdict(list);seen=set()
 for r in rr:
  for k in ['archivo','paciente','anotador','clase']:
   if not r.get(k):raise ValueError('Falta campo '+k)
  if r['clase'] not in CLASSES:raise ValueError('Mapeo de clases debe documentarse antes')
  key=(r['archivo'],r['anotador'])
  if key in seen:raise ValueError('Anotador duplicado por caso')
  seen.add(key);groups[r['archivo']].append(r)
 y=[];pred=[];patients=[];ties=0;insufficient=0
 for item,rows in sorted(groups.items()):
  if len({r['paciente'] for r in rows})!=1:raise ValueError('Paciente inconsistente')
  if len(rows)<3:insufficient+=1;continue
  for i,r in enumerate(rows):
   counts=collections.Counter(q['clase'] for j,q in enumerate(rows) if i!=j);best=counts.most_common()
   if len(best)>1 and best[0][1]==best[1][1]:ties+=1;continue
   y.append(CLASSES.index(best[0][0]));pred.append(CLASSES.index(r['clase']));patients.append(r['paciente'])
 if not y or len(set(patients))<2:raise ValueError('Sin suficientes consensos/pacientes para remuestreo agrupado')
 y=np.array(y);p=np.eye(3)[pred];boot,_=st.patient_boot(y,p[None,None],np.array(patients));m=st.f1cm(np.bincount(3*y+np.array(pred),minlength=9).reshape(3,3));a.salida.mkdir(exist_ok=True,parents=True)
 st.save(a.salida/'acuerdo_multirater.json',dict(macro_f1_leave_one_rater_out=m,IC95_paciente=np.quantile(boot[:,0],[.025,.975]),comparaciones=len(y),pacientes=len(set(patients)),items_con_menos_tres=insufficient,comparaciones_excluidas_empate=ties,nota='Proxy de acuerdo humano, no techo matemático. Comparaciones repetidas por núcleo remuestreadas conjuntamente por paciente. No calcular gamma contra modelos sobre otra población o referencia.'))
 print('ACUERDO_MULTIRATER_COMPLETADO',a.salida)
if __name__=='__main__':main()
