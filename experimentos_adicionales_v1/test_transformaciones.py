import ast,types,random,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
root=Path(__file__).parent
classes=['tumor_nonMitotic','lymphocyte','nonTILnonMQ_stromal']
def selected(file,names,env):
 tree=ast.parse((root/file).read_text());tree.body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names];exec(compile(tree,file,'exec'),env);return env
base=selected('core_v2.py',['Crops'],dict(torch=torch,np=np,Image=Image,random=random,CLASSES=classes,Dataset=torch.utils.data.Dataset))
env=selected('experimentos.py',['AblationCrops','gradcam'],dict(torch=torch,np=np,Image=Image,random=random,core=types.SimpleNamespace(CLASSES=classes),MEAN=[.6392034726251884,.45917053999289786,.6277451963521469],SD=[.2139743515891908,.22149669223009963,.1742893379690752]))
C=env['AblationCrops'];cam=env['gradcam']
class Test(unittest.TestCase):
 def test_control_exact(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);Image.fromarray(np.random.default_rng(1).integers(0,256,(16,16,3),dtype=np.uint8)).save(p/'x.png');rows=[dict(archivo='x.png',clase=classes[0])]
   for s in range(20):
    C.condition='control';random.seed(s);a=base['Crops'](p,rows,True)[0][0];state=random.getstate();random.seed(s);b=C(p,rows,True)[0][0];self.assertTrue(torch.equal(a,b));self.assertEqual(state,random.getstate())
    for cond in ['sin_rotacion','sin_reflejo','sin_estandarizacion']:
     C.condition=cond;random.seed(s);C(p,rows,True)[0];self.assertEqual(state,random.getstate())
   C.condition='control'
 def test_gradcam_connected(self):
  model=torch.nn.Sequential(torch.nn.Conv2d(3,4,3,padding=1),torch.nn.ReLU(),torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(4,3));x=torch.randn(1,3,16,16,requires_grad=True)
  m,name,span=cam(model,x,1);self.assertEqual(m.shape,(16,16));self.assertTrue(torch.isfinite(m).all());self.assertEqual(name,'0');self.assertEqual(len(model[0]._forward_hooks),0)
 def test_unsupported(self):
  model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(12,3))
  with self.assertRaises(ValueError):cam(model,torch.ones(1,3,2,2,requires_grad=True),0)
if __name__=='__main__':unittest.main()
