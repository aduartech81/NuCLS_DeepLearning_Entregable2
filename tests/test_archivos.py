import unittest,tempfile,csv,hashlib,json,importlib.util
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
runner=module('runner',root/'scripts/adicionales_e2.py')
export=module('export',root/'scripts/exportar_bases.py')
class Archivos(unittest.TestCase):
 def test_configuracion_distinta_no_sobrescribe(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'config.json';runner.frozen(p,{'lr':.1})
   with self.assertRaises(ValueError):runner.frozen(p,{'lr':.2})
   self.assertEqual(json.loads(p.read_text()),{'lr':.1})
 def test_resultado_incompleto_no_se_omite(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'resultado.json').write_text(json.dumps({'modelo':'a','semilla':1,'estado':'FASE_COMPLETADA_VALIDACION'}))
   with self.assertRaises(ValueError):runner.validate_complete(p,'a',1)
 def test_exportacion_no_modifica_origen_y_verifica_hash(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);runs=p/'original';runs.mkdir();rows=[]
   for i in range(66):
    r=runs/str(i);r.mkdir();data=json.dumps({'id':i}).encode()
    for f in ['resultado.json','historia.json','curvas.csv','predicciones_validacion.csv','metricas_por_paciente.csv']:(r/f).write_bytes(data)
    rows.append(dict(modelo='m',semilla=i,ruta=str(r),resultado_sha256=hashlib.sha256(data).hexdigest()))
   index=p/'indice.csv'
   with index.open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
   before={str(f):f.read_bytes() for f in runs.rglob('*') if f.is_file()}
   out=p/'NuCLS_E2_BASES_66_EVIDENCIAS.zip';export.exportar(index,out)
   self.assertTrue(out.is_file());self.assertEqual(before,{str(f):f.read_bytes() for f in runs.rglob('*') if f.is_file()})
   (runs/'0/resultado.json').write_text('alterado')
   with self.assertRaises(ValueError):export.exportar(index,out)
if __name__=='__main__':unittest.main()
