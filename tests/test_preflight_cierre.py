import unittest,tempfile,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import preflight_cierre as p
class Preflight(unittest.TestCase):
 def fixture(self,root):
  (root/'NuCLS_V2_DATOS.zip').write_bytes(b'synthetic');base=root/'NuCLS_E2_CONTINUACION_BASES_22';base.mkdir();index=[]
  for m in p.BASES+p.SIETE+p.TRES:
   for seed in range(6101,6106):
    d=(base if m in p.BASES else root/('NuCLS_E2_TRES_NUEVOS_V1' if m in p.TRES else 'NuCLS_E2_ADICIONALES_EXPLORATORIO')/'piloto')/f'{m}_{seed}';d.mkdir(parents=True)
    (d/'resultado.json').write_text(json.dumps(dict(modelo=m,semilla=seed,estado='FASE_COMPLETADA_VALIDACION')))
    for n in ['historia.json','predicciones_validacion.csv','mejor.pt']:(d/n).write_bytes(b'synthetic')
    if m in p.BASES:index.append(dict(modelo=m,semilla=seed,ruta=str(d)))
  with (base/'indice_validacion.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=['modelo','semilla','ruta']);w.writeheader();w.writerows(index)
 def test_complete_readonly(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);self.fixture(r);before={str(p):p.read_bytes() for p in r.rglob('*') if p.is_file()};self.assertEqual(p.comprobar_fuentes(r)['corridas'],65);self.assertEqual(before,{str(p):p.read_bytes() for p in r.rglob('*') if p.is_file()})
 def test_missing_checkpoint(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);self.fixture(r);next(r.rglob('mejor.pt')).unlink()
   with self.assertRaisesRegex(RuntimeError,'64/65'):p.comprobar_fuentes(r)
 def test_wrong_identity(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);self.fixture(r);next(r.rglob('resultado.json')).write_text('{}')
   with self.assertRaises(ValueError):p.comprobar_fuentes(r)
 def test_wrong_account(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(FileNotFoundError):p.comprobar_fuentes(Path(d))
if __name__=='__main__':unittest.main()
