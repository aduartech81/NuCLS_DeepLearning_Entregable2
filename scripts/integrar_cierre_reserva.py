"""Importa un cierre completo verificado. Solo escribe en este paquete de E2."""
import argparse,csv,hashlib,io,json,re,shutil,subprocess,zipfile,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def readcsv(z,name):return list(csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig'))))
def validate(z):
 names=z.namelist()
 if len(set(names))!=len(names) or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):raise ValueError('ZIP con rutas ambiguas o inseguras')
 state=json.loads(z.read('estado.json'));plan=json.loads(z.read('entradas_fijadas.json'));pid=hashlib.sha256(z.read('entradas_fijadas.json')).hexdigest()
 if state.get('estado')!='CIERRE_RESERVA_Y_ESTADISTICA_COMPLETADO' or state.get('corridas')!=65:raise ValueError('No es un cierre completo de 65 corridas')
 if len(set(plan['modelos']))!=13 or plan['semillas']!=list(range(6101,6106)) or len(plan['adicionales'])!=10:raise ValueError('Lote incompatible')
 if json.loads(z.read('congelamiento_previo_reserva.json'))['plan_sha256']!=pid:raise ValueError('Congelamiento no coincide')
 expected={(n,s) for n in plan['modelos'] for s in plan['semillas']}
 if {(r['modelo'],r['semilla']) for r in plan['corridas']}!=expected or len(plan['corridas'])!=65:raise ValueError('Cobertura del plan incompleta')
 for model,seed in sorted(expected):
  prefix=f'corridas/{model}_{seed}/';marker=json.loads(z.read(prefix+'completo.json'))
  if marker['plan']!=pid:raise ValueError('Plan de corrida incompatible')
  if set(marker['archivos'])!={'reserva.npz','train_limpio.npz','predicciones_reserva.csv','brecha.json','costo.json'}:raise ValueError('Marcador de corrida incompleto')
  for name,h in marker['archivos'].items():
   if Path(name).name!=name or hashlib.sha256(z.read(prefix+name)).hexdigest()!=h:raise ValueError('Evidencia de corrida modificada')
 summary=readcsv(z,'resumen_reserva.csv');contrasts=readcsv(z,'contrastes_vs_resnet50.csv')
 if len(summary)!=13 or {r['modelo'] for r in summary}!=set(plan['modelos']) or any(int(r['R'])!=5 for r in summary):raise ValueError('Resumen incompleto')
 if len(contrasts)!=10 or {r['modelo'] for r in contrasts}!=set(plan['adicionales']):raise ValueError('Contrastes incompletos')
 return plan,summary,contrasts

def main():
 p=argparse.ArgumentParser();p.add_argument('zip',type=Path);p.add_argument('--compilar',action='store_true');a=p.parse_args()
 target=ROOT/'evidencias/cierre_reserva';digest=hashlib.sha256(a.zip.read_bytes()).hexdigest()
 with zipfile.ZipFile(a.zip) as z:
  plan,summary,contrasts=validate(z)
  if target.exists():
   if json.loads((target/'recepcion_local.json').read_text())['zip_sha256']!=digest:raise RuntimeError('Ya existe un cierre distinto; no se sobrescribe')
  else:
   temp=ROOT/'evidencias/.cierre_reserva_temporal'
   if temp.exists():raise RuntimeError('Hay una recepción temporal previa; revisar antes de continuar')
   temp.mkdir()
   for info in z.infolist():
    if info.is_dir():continue
    path=temp/info.filename;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(info))
   (temp/'recepcion_local.json').write_text(json.dumps({'zip_sha256':digest,'corridas_verificadas':65},indent=2));temp.rename(target)
 def table(rows,keys):
  return '| '+' | '.join(keys)+' |\n|'+'|'.join(['---']*len(keys))+'|\n'+''.join('| '+' | '.join(str(r[k]) for k in keys)+' |\n' for r in rows)
 chapter=ROOT/'book/09_cierre_reserva.md';text=chapter.read_text();text=re.sub(r'\n<!-- CIERRE_IMPORTADO -->.*', '',text,flags=re.S);text=text.replace('**Estado: procedimiento preparado; resultados de reserva de E2 pendientes de recepción.**','**Estado: cierre recibido y 65 corridas verificadas mediante huellas de archivo.**')
 text=text.replace('## Tabla principal pendiente','## Tabla principal').replace('No se insertan ceros, máximos de búsqueda ni resultados sintéticos como resultados de reserva. Al recibir el ZIP se incorporarán la tabla de 13 modelos, los diez contrastes, los diagramas y las discrepancias de validación a prueba. La importación comprobará el estado de cierre y la cobertura de 65 corridas.','La tabla y los contrastes recibidos se presentan al final de este capítulo. No sustituyen los análisis de desarrollo ni resuelven las limitaciones externas al cierre.')
 text+='\n<!-- CIERRE_IMPORTADO -->\n## Resultados del cierre recibido\n\n'+table(summary,['modelo','R','macro_f1_media','macro_f1_sd','IC95_paciente_inf','IC95_paciente_sup'])+'\n## Diez contrastes frente a ResNet50\n\n'+table(contrasts,['modelo','delta','p_t_Holm10','p_wilcoxon_Holm10','p_permutacion_paciente_Holm10'])+'\n![Benchmark de reserva](../evidencias/cierre_reserva/benchmark_reserva.png)\n\n![Diferencia crítica](../evidencias/cierre_reserva/diferencia_critica.png)\n'
 chapter.write_text(text)
 for page,prefix in [(ROOT/'dashboard/index.html','../'),(ROOT/'ABRIR_ENTREGA.html','')]:
  text=page.read_text();body=f'<p><strong>Cierre recibido: 65 corridas verificadas.</strong> Los límites de potencia y exposición previa se mantienen.</p><a class="button" href="{prefix}evidencias/cierre_reserva/informe.html">Abrir resultados del cierre →</a><img style="width:100%;height:auto" src="{prefix}evidencias/cierre_reserva/benchmark_reserva.png" alt="Benchmark de reserva con IC por paciente">'
  columns=['modelo','R','macro_f1_media','macro_f1_sd','IC95_paciente_inf','IC95_paciente_sup']
  body+='<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(row[k]))+'</td>' for k in columns)+'</tr>' for row in summary)+'</tbody></table></div>'
  if text.count('<!-- RESULTADOS_CIERRE_INICIO -->')!=1 or text.count('<!-- RESULTADOS_CIERRE_FIN -->')!=1:raise ValueError('Marcador de publicación ausente o duplicado')
  text=text.replace('Resultados de desarrollo; comparación final pendiente.','Cierre recibido; interpretación final y requisitos adicionales por revisar.')
  text=re.sub(r'<!-- RESULTADOS_CIERRE_INICIO -->.*?<!-- RESULTADOS_CIERRE_FIN -->','<!-- RESULTADOS_CIERRE_INICIO -->'+body+'<!-- RESULTADOS_CIERRE_FIN -->',text,flags=re.S);page.write_text(text)
 if a.compilar:subprocess.run(['jupyter-book','build',str(ROOT/'book')],check=True)
 print('Integración verificada en E2. Revisar discusión científica, compilar el libro y regenerar ZIP/huellas antes de entregar; los requisitos ajenos al cierre no cambian automáticamente.')
if __name__=='__main__':main()
