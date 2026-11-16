"""Extrae exclusivamente train/validation a disco temporal; no extrae reserva."""
import argparse,csv,hashlib,shutil,zipfile
from pathlib import Path
EXPECTED='623adf36520751addc324e44cd1b89d95e47842e53cda494914c0248a8be86d4'
def main():
 p=argparse.ArgumentParser();p.add_argument('--zip',type=Path,required=True);p.add_argument('--salida',type=Path,required=True);a=p.parse_args()
 root=a.salida.resolve();root.mkdir(parents=True,exist_ok=True)
 local=root.parent/'nucls_e2_datos_fuente.zip'
 if not local.exists():
  tmp=local.with_suffix('.copia_tmp');shutil.copyfile(a.zip,tmp);tmp.replace(local)
 with zipfile.ZipFile(local) as z:
  raw=z.read('manifiesto.csv')
  if hashlib.sha256(raw).hexdigest()!=EXPECTED:raise ValueError('Manifiesto distinto. No se extraen imágenes.')
  allrows=list(csv.DictReader(raw.decode().splitlines()));rows=[r for r in allrows if r['particion'] in ('train','validation') and r['estado']=='incluida']
  assert len(rows)==16588
  for i,r in enumerate(rows,1):
   dst=(root/r['archivo']).resolve()
   if not dst.is_relative_to(root):raise ValueError('Ruta fuera de datos')
   if dst.is_file() and hashlib.sha256(dst.read_bytes()).hexdigest()==r['sha_archivo']:continue
   data=z.read(r['archivo'])
   if hashlib.sha256(data).hexdigest()!=r['sha_archivo']:raise ValueError('Imagen distinta: '+r['archivo'])
   dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_suffix('.tmp');tmp.write_bytes(data);tmp.replace(dst)
   if i%2000==0:print('Preparación de desarrollo:',i,'/',len(rows),flush=True)
  (root/'manifiesto.csv').write_bytes(raw)
 print('Datos listos: 16.588 imágenes de desarrollo; reserva no extraída.')
if __name__=='__main__':main()
