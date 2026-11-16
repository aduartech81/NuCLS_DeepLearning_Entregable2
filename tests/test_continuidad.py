import sys,unittest,tempfile,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'modelos'))
import torch,numpy as np
from PIL import Image
import core_v2 as core
class Continuidad(unittest.TestCase):
 def test_checkpoint_equivale_a_ejecucion_continua(self):
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
  original=(core.build,core.archive,core.summary)
  core.build=lambda name:torch.nn.Sequential(torch.nn.Conv2d(3,4,3),torch.nn.ReLU(),torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(4,3))
  core.archive=lambda p:p;core.summary=lambda p:p
  try:
   with tempfile.TemporaryDirectory() as td:
    root=Path(td);rows=[]
    for i in range(6):
     name=f'{i}.png';Image.fromarray(np.random.default_rng(i).integers(0,256,(128,128,3),dtype=np.uint8)).save(root/name)
     rows.append(dict(archivo=name,clase=core.CLASSES[i%3],paciente=str(i)))
    groups={'train':rows,'validation':rows}
    for folder,epochs in [('full',2),('resume',1),('resume',2)]:
     out=root/folder;out.mkdir(exist_ok=True)
     if epochs==2 and folder=='resume':(out/'toy_6101'/'resultado.json').unlink()
     args=argparse.Namespace(salida=out,datos=root,batch=3,epocas=epochs,min_epocas=5,paciencia=5,lr=.001,weight_decay=0.)
     core.run_one('toy',6101,groups,args,torch.device('cpu'))
    a=torch.load(root/'full/toy_6101/ultimo.pt',weights_only=False);b=torch.load(root/'resume/toy_6101/ultimo.pt',weights_only=False)
    self.assertTrue(all(torch.equal(a['model'][k],b['model'][k]) for k in a['model']))
    self.assertEqual(a['best'],b['best'])
  finally:core.build,core.archive,core.summary=original
if __name__=='__main__':unittest.main()
