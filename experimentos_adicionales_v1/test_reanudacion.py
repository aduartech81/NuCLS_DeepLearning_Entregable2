import tempfile,types,unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import core_v2 as c
class ResumeTest(unittest.TestCase):
 def test_interrupted_epoch_resume_matches(self):
  torch.set_num_threads(1)
  c.build=lambda name:torch.nn.Sequential(torch.nn.Conv2d(3,4,3,padding=1),torch.nn.ReLU(),torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(4,3))
  original_archive=c.archive;c.summary=lambda out:out
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);rows=[]
   for i in range(9):
    file=str(i)+'.png';Image.fromarray(np.random.default_rng(i).integers(0,255,(8,8,3),dtype=np.uint8)).save(root/file);rows.append(dict(archivo=file,paciente=str(i//3),clase=c.CLASSES[i%3]))
   groups={'train':rows,'validation':rows};one=root/'one';two=root/'two';one.mkdir();two.mkdir()
   def args(out):return types.SimpleNamespace(salida=out,datos=root,batch=3,lr=.001,weight_decay=.001,epocas=2,min_epocas=2,paciencia=5)
   c.run_one('toy',6101,groups,args(one),torch.device('cpu'))
   def interrupt(out):raise InterruptedError('Interrupción simulada después de guardar época')
   c.archive=interrupt
   with self.assertRaises(InterruptedError):c.run_one('toy',6101,groups,args(two),torch.device('cpu'))
   self.assertFalse((two/'toy_6101/resultado.json').exists());self.assertTrue((two/'toy_6101/ultimo.pt').exists());c.archive=original_archive
   c.run_one('toy',6101,groups,args(two),torch.device('cpu'))
   self.assertEqual((one/'toy_6101/predicciones_validacion.csv').read_bytes(),(two/'toy_6101/predicciones_validacion.csv').read_bytes())
   a=torch.load(one/'toy_6101/mejor.pt',weights_only=True);b=torch.load(two/'toy_6101/mejor.pt',weights_only=True)
   self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
   before=(two/'toy_6101/resultado.json').stat().st_mtime_ns;c.run_one('toy',6101,groups,args(two),torch.device('cpu'));self.assertEqual(before,(two/'toy_6101/resultado.json').stat().st_mtime_ns)
if __name__=='__main__':unittest.main()
