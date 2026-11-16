import sys,tempfile,unittest,zipfile,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from respaldo_urgente import respaldar
class TestUrgente(unittest.TestCase):
 def fixture(self,p,complete=True):
  with zipfile.ZipFile(p,'w') as z:
   z.writestr('estado.json',json.dumps({'estado':'CIERRE_RESERVA_Y_ESTADISTICA_COMPLETADO' if complete else 'INCOMPLETO','corridas':65}))
   for name in ['entradas_fijadas.json','congelamiento_previo_reserva.json','resumen_reserva.csv','contrastes_vs_resnet50.csv','informe.html','benchmark_reserva.png','diferencia_critica.png']:z.writestr(name,'SINTETICO')
 def test_respaldo_independiente_e_idempotente(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);source=root/'cierre.zip';self.fixture(source)
   target,receipt=respaldar(source,root);before=target.read_bytes()
   self.assertEqual(respaldar(source,root),(target,receipt))
   source.write_bytes(b'posterior');self.assertEqual(target.read_bytes(),before)
   self.assertEqual(hashlib.sha256(before).hexdigest(),json.loads(receipt.read_text())['sha256'])
 def test_cierre_incompleto_no_genera_respaldo(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);source=root/'cierre.zip';self.fixture(source,False)
   with self.assertRaises(ValueError):respaldar(source,root)
   self.assertFalse((root/'NuCLS_E2_ENTREGA_URGENTE').exists())
 def test_respaldo_alterado_no_se_sobrescribe(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);source=root/'cierre.zip';self.fixture(source);target,_=respaldar(source,root);target.write_bytes(b'ALTERADO')
   with self.assertRaises(ValueError):respaldar(source,root)
   self.assertEqual(target.read_bytes(),b'ALTERADO')
if __name__=='__main__':unittest.main()
