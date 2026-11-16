"""Copia solo evidencias pequeñas de las 66 corridas, sin escribir en sus carpetas."""
import csv,hashlib,json,zipfile
from pathlib import Path

def exportar(indice,salida):
 with indice.open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==66
 assert len({(r['modelo'],r['semilla']) for r in rows})==66
 if salida.name!='NuCLS_E2_BASES_66_EVIDENCIAS.zip':raise ValueError('Nombre de salida exclusivo requerido')
 tmp=salida.with_suffix('.tmp');missing=[]
 with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
  z.write(indice,'indice_validacion.csv')
  for r in rows:
   root=Path(r['ruta']);result=root/'resultado.json'
   if not result.exists() or hashlib.sha256(result.read_bytes()).hexdigest()!=r['resultado_sha256']:raise ValueError('Resultado falta o cambió: '+str(root))
   for name in ['resultado.json','historia.json','curvas.csv','predicciones_validacion.csv','metricas_por_paciente.csv']:
    f=root/name
    if not f.is_file():missing.append(str(f));continue
    z.write(f,f"final/{r['modelo']}_{r['semilla']}/{name}")
  z.writestr('verificacion.json',json.dumps({'corridas':66,'faltantes':missing},indent=2))
 if missing:tmp.unlink();raise ValueError('Evidencias faltantes: '+str(missing))
 tmp.replace(salida);print('Exportación de las 66 corridas:',salida)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--indice',type=Path,required=True);p.add_argument('--salida',type=Path,required=True);a=p.parse_args();exportar(a.indice,a.salida)
