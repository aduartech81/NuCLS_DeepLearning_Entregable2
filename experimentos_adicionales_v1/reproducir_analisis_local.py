import csv,io,zipfile,json,hashlib
from pathlib import Path
import numpy as np
from ensembles import analyze
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--zip',type=Path,required=True);ap.add_argument('--salida',type=Path,required=True);a=ap.parse_args();source=a.zip;out=a.salida;out.mkdir(exist_ok=True,parents=True)
models={};rows=None;records=[]
with zipfile.ZipFile(source) as z:
 for name in ['softmaxcnn_bn','densenet121','resnet50']:
  ps=[]
  for seed in range(6113,6123):
   path=f'final/{name}_{seed}/predicciones_validacion.csv';raw=z.read(path);rr=list(csv.DictReader(io.StringIO(raw.decode())))
   if rows is None:rows=rr
   assert [(r['archivo'],r['paciente'],r['clase_real']) for r in rr]==[(r['archivo'],r['paciente'],r['clase_real']) for r in rows]
   ps.append([[float(r[k]) for k in ['p_tumor','p_linfocito','p_estroma']] for r in rr]);records.append(dict(archivo=path,sha256=hashlib.sha256(raw).hexdigest()))
  models[name]=np.array(ps)
summary=analyze(rows,models,out,'Provisional local: validación, 3 bases, diez semillas comunes6113–6122. No sustituye selección Top3 final ni evaluación en reserva.')
(out/'fuente.json').write_text(json.dumps(dict(zip=source.name,sha256=hashlib.sha256(source.read_bytes()).hexdigest(),archivos=records),indent=2))
print(json.dumps(summary,indent=2))
