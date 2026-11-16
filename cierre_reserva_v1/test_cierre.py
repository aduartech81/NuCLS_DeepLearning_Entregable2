import unittest,tempfile,csv,json,sys,types,importlib.util
from pathlib import Path
import numpy as np
import estadistica_cierre as s
from sklearn.metrics import f1_score

class Estadistica(unittest.TestCase):
 def test_holm(self):np.testing.assert_allclose(s.holm([.01,.04,.03]),[.03,.06,.06])
 def test_f1(self):
  y=np.array([0,0,1,1,2,2]);p=np.array([0,1,1,2,2,0]);cm=s.confusion_matrix(y,p,labels=[0,1,2]);self.assertAlmostEqual(s.f1cm(cm),f1_score(y,p,average='macro'))
 def test_patient_boot_pairing(self):
  y=np.tile([0,1,2],6);patients=np.repeat(np.arange(6),3);p=np.tile(np.eye(3)[y],(2,5,1,1));b,c=s.patient_boot(y,p,patients);self.assertEqual(b.shape,(2000,2));np.testing.assert_equal(b,1)
 def test_delong_equal(self):
  q=s.delong_binary(np.array([1,1,0,0]),np.array([.9,.4,.6,.1]),np.array([.9,.4,.6,.1]));self.assertEqual(q['p_delong_iid'],1)
 def test_zero_effect(self):
  q=s.paired(np.arange(5),np.arange(5));self.assertEqual(q['p_permutacion'],1);self.assertEqual(q['delta'],0)
 def test_five_pairs_exact(self):self.assertAlmostEqual(s.paired(np.ones(5),np.zeros(5))['p_permutacion'],.0625)
 def test_end_to_end_synthetic(self):
  rng=np.random.default_rng(4);y=np.tile([0,1,2],12);patients=np.repeat(np.arange(6),6);names=['resnet50']+[f'model{i}' for i in range(12)];seeds=list(range(6101,6106));plan={'modelos':names,'adicionales':names[3:],'semillas':seeds}
  pred={}
  for name in names:
   for seed in seeds:
    p=rng.random((len(y),3));p[np.arange(len(y)),y]+=.5;pred[name,seed]=p/p.sum(1)[:,None]
  with tempfile.TemporaryDirectory() as d:
   result=s.analyze(Path(d),plan,pred,y,patients,[{'tss':str(i//12)} for i in range(len(y))]);self.assertEqual(len(result),13);self.assertTrue((Path(d)/'metricas_por_tss.csv').is_file());self.assertTrue((Path(d)/'diferencia_critica.png').is_file());self.assertEqual(len(list(csv.DictReader((Path(d)/'metricas_por_semilla.csv').open()))),65);self.assertEqual(len(list(csv.DictReader((Path(d)/'contrastes_vs_resnet50.csv').open()))),10)
 def test_no_training_calls(self):
  import ast
  p=Path(__file__).parent
  for path in p.glob('*.py'):
   if path.name.startswith('test_'):continue
   tree=ast.parse(path.read_text())
   for node in ast.walk(tree):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):self.assertNotIn(node.func.attr,['fit','backward','step'],str(path))

if __name__=='__main__':unittest.main()
