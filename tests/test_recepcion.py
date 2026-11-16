import sys,unittest,tempfile,zipfile,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from recibir_evidencias import metrics,probabilities,analyze
class TestRecepcion(unittest.TestCase):
 def test_perfect(self):
  m=metrics(np.array([0,1,2]),np.eye(3));self.assertEqual(m['macro_f1'],1);self.assertEqual(m['brier_multiclase_suma'],0);self.assertEqual(m['ece_10'],0)
 def test_invalid(self):
  with self.assertRaises(ValueError):probabilities([dict(paciente='a',archivo='a',clase_real='tumor',p_tumor='2',p_linfocito='0',p_estroma='0')])
 def test_empty_zip(self):
  with tempfile.TemporaryDirectory() as d:
   z=Path(d)/'a.zip'
   with zipfile.ZipFile(z,'w'):pass
   with self.assertRaises(ValueError):analyze([z],Path(d)/'out')
   self.assertFalse((Path(d)/'out').exists())
 def test_no_overwrite(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):analyze([],Path(d))
if __name__=='__main__':unittest.main()
