import unittest,tempfile,subprocess,sys,json,importlib.util
from pathlib import Path
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('runner',root/'scripts/adicionales_e2.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
class Protecciones(unittest.TestCase):
 def test_no_inicia_busqueda_sin_autorizacion(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'NuCLS_E2_TRES_NUEVOS_V1'
   p=subprocess.run([sys.executable,str(root/'scripts/adicionales_e2.py'),'--datos',str(Path(d)/'datos'),'--salida',str(out),'--fase','busqueda'],capture_output=True,text=True)
   self.assertNotEqual(p.returncode,0);self.assertIn('Entrenamiento bloqueado',p.stderr);self.assertFalse(out.exists())
 def test_bloqueo_previo_se_conserva(self):
  with tempfile.TemporaryDirectory() as d:
   prior=Path(d)/'NuCLS_E2_ADICIONALES_EXPLORATORIO';prior.mkdir();lock=prior/'EJECUCION_ACTIVA.json';lock.write_text('evidencia')
   out=Path(d)/'NuCLS_E2_TRES_NUEVOS_V1'
   p=subprocess.run([sys.executable,str(root/'scripts/adicionales_e2.py'),'--datos',str(Path(d)/'datos'),'--salida',str(out)],capture_output=True,text=True)
   self.assertIn('búsqueda anterior',p.stderr);self.assertEqual(lock.read_text(),'evidencia');self.assertFalse((out/'EJECUCION_ACTIVA.json').exists())
 def test_config_no_se_sobrescribe(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'config.json';runner.frozen(p,{'lr':.1})
   with self.assertRaises(ValueError):runner.frozen(p,{'lr':.2})
   self.assertEqual(json.loads(p.read_text()),{'lr':.1})
 def test_no_omite_evidencia_incompleta(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'resultado.json').write_text(json.dumps({'modelo':'x','semilla':7001,'estado':'FASE_COMPLETADA_VALIDACION'}))
   with self.assertRaises(ValueError):runner.validate_complete(p,'x',7001)
 def test_carpeta_anterior_rechazada(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'NuCLS_E2_ADICIONALES_EXPLORATORIO'
   p=subprocess.run([sys.executable,str(root/'scripts/adicionales_e2.py'),'--datos',str(Path(d)/'datos'),'--salida',str(out)],capture_output=True,text=True)
   self.assertIn('carpeta exclusiva',p.stderr);self.assertFalse(out.exists())
if __name__=='__main__':unittest.main()
