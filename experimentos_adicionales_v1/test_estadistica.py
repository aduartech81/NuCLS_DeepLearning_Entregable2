import unittest,tempfile,csv,json
from pathlib import Path
import numpy as np
import ensembles as e
import estadistica_cierre as st
class Tests(unittest.TestCase):
 def test_identical_seeds_zero_mi(self):
  p=np.tile(np.array([[.8,.1,.1],[.2,.7,.1]])[None],(5,1,1));m,h,mi=e.uncertainty(p);self.assertTrue(np.allclose(mi,0));self.assertTrue(np.allclose(m.sum(1),1))
 def test_disagreement_positive(self):
  _,_,mi=e.uncertainty(np.array([[[1,0,0]],[[0,1,0]]]));self.assertAlmostEqual(mi[0],np.log(2))
 def test_invalid(self):
  with self.assertRaises(ValueError):e.uncertainty(np.zeros((5,2,3)))
 def test_grouped_bootstrap(self):
  y=np.array([0,1,2,0,1,2]);p=np.eye(3)[y][None,None];boot,_=st.patient_boot(y,p,np.array(['a']*3+['b']*3));self.assertTrue(np.allclose(boot,1));self.assertEqual(boot.shape,(2000,1))
 def test_review_sample_and_manifest(self):
  rows=[dict(archivo=str(i),paciente=str(i//10),clase_real=e.CLASSES[i%3]) for i in range(120)];p=np.array([np.roll(np.eye(3)[i%3],1)*.8+.2/3 for i in range(120)]);ps=np.tile(p[None],(5,1,1))
  with tempfile.TemporaryDirectory() as d:
   e.analyze(rows,{'a':ps,'b':ps},d,'prueba');r=list(csv.DictReader((Path(d)/'a_revision_errores.csv').read_text().splitlines()));self.assertEqual(len(r),100);self.assertEqual(len({x['archivo'] for x in r}),100);self.assertTrue(all(x['categoria']=='' for x in r));self.assertAlmostEqual(float(r[0]['probabilidad_inclusion']),100/120)
 def test_calibration_degenerate(self):
  q=st.calibration(np.zeros(3,dtype=int),np.tile([1.,0.,0.],(3,1)));self.assertIsNone(q['pendiente_toplabel']);self.assertFalse(q['convergencia'])
if __name__=='__main__':unittest.main()
