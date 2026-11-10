import csv,hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('continuar',P/'scripts/continuar_semillas.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
HP={x:dict(lr=.001,weight_decay=1e-4 if x=='softmaxcnn_bn' else 0.) for x in m.MODELS}
def complete(path,model,seed):
 path.mkdir(parents=True,exist_ok=True);hist=[dict(epoca=i,val_macro_f1=1.) for i in range(1,6)]
 m.save(path/'historia.json',hist);m.save(path/'resultado.json',dict(modelo=model,semilla=seed,estado='FASE_COMPLETADA_VALIDACION',lr=HP[model]['lr'],weight_decay=HP[model]['weight_decay'],epocas_ejecutadas=5,mejor_epoca=1,validacion=dict(macro_f1=1.)))
 for f in ['curvas.csv','metricas_por_paciente.csv','mejor.pt']:(path/f).write_bytes(b'test fixture')
 with (path/'predicciones_validacion.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['paciente','archivo','clase_real','clase_predicha']);w.writeheader();w.writerows(dict(paciente='P'+str(i%21),archivo=str(i),clase_real=m.CLASSES[i%3],clase_predicha=m.CLASSES[i%3]) for i in range(2049))
class ResumeTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.old=self.root/'e1';self.new=self.root/'e2'
  for s in range(6101,6112):
   for model in m.MODELS:complete(self.old/'final'/f'{model}_{s}',model,s)
 def test_omits_all_33_complete(self):
  rows=m.inventory(self.old,self.new,HP);self.assertEqual(sum(x['estado']=='completa' for x in rows),33);self.assertEqual(sum(x['estado']=='nueva' for x in rows),33)
 def test_reuses_completed_6112_and_resumes_partial(self):
  for model in m.MODELS[:2]:complete(self.old/'final'/f'{model}_6112',model,6112)
  d=self.old/'final/resnet50_6112';d.mkdir();(d/'ultimo.pt').write_bytes(b'own checkpoint')
  rows=m.inventory(self.old,self.new,HP);self.assertEqual(sum(x['estado']=='completa' for x in rows),35);self.assertEqual(sum(x['estado']=='reanudar' for x in rows),1);self.assertEqual(sum(x['estado']=='nueva' for x in rows),30)
 def test_damaged_complete_does_not_retrain(self):
  (self.old/'final/resnet50_6101/mejor.pt').unlink()
  with self.assertRaises(ValueError):m.inventory(self.old,self.new,HP)
 def test_missing_old_closure_blocks(self):
  (self.old/'final/resnet50_6101/resultado.json').unlink()
  with self.assertRaises(ValueError):m.inventory(self.old,self.new,HP)
 def test_partial_copied_without_changing_source(self):
  d=self.old/'final/resnet50_6112';d.mkdir();(d/'ultimo.pt').write_bytes(b'saved epoch');(d/'mejor.pt').write_bytes(b'best');before={p.name:p.read_bytes() for p in d.iterdir()};target=self.new/'final/resnet50_6112';m.copy_resume(d,target)
  self.assertEqual(before,{p.name:p.read_bytes() for p in d.iterdir()});self.assertEqual((target/'ultimo.pt').read_bytes(),before['ultimo.pt'])
 def test_retry_interrupted_copy_uses_original_checkpoint(self):
  d=self.old/'final/resnet50_6112';d.mkdir();(d/'ultimo.pt').write_bytes(b'saved');target=self.new/'final/resnet50_6112';target.mkdir(parents=True);(target/'mejor.pt.copia_tmp').write_bytes(b'incomplete')
  r=next(x for x in m.inventory(self.old,self.new,HP) if x['modelo']=='resnet50' and x['semilla']==6112);self.assertEqual(r['ruta'],str(d))
 def test_new_completed_output_omitted_on_restart(self):
  complete(self.new/'final/resnet50_6112','resnet50',6112);r=next(x for x in m.inventory(self.old,self.new,HP) if x['modelo']=='resnet50' and x['semilla']==6112);self.assertEqual(r['estado'],'completa')
 def test_no_write_into_original_or_data(self):
  for out in [self.old,self.old/'nested',self.root]:
   with self.assertRaises(ValueError):m.separate(self.old,out,self.root/'data')
 def test_symlink_output_into_e1_blocks(self):
  self.new.mkdir();(self.new/'final').symlink_to(self.old/'final',target_is_directory=True)
  with self.assertRaises(ValueError):m.inventory(self.old,self.new,HP)
 def test_wrong_frozen_hash_blocks(self):
  with self.assertRaises(ValueError):m.completed(self.old/'final/resnet50_6101','resnet50',6101,HP,'0'*64)
 def test_wrong_hyperparameters_block(self):
  hp={**HP,'resnet50':dict(lr=.1,weight_decay=0.)}
  with self.assertRaises(ValueError):m.completed(self.old/'final/resnet50_6101','resnet50',6101,hp)
if __name__=='__main__':unittest.main()
