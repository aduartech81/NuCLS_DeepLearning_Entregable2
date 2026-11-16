"""Pruebas de protecciones de archivos sin simular ejecución GPU."""
import ast,csv,json,hashlib,types,tempfile,unittest,html
from pathlib import Path
import numpy as np
import estadistica_cierre as st
source=ast.parse((Path(__file__).parent/'cierre_unico.py').read_text())
keep={'sha','read_prediction','write_npz','cache_infer','inventory','baselines_saved','write_predictions','report'}
ns={'Path':Path,'np':np,'json':json,'csv':csv,'hashlib':hashlib,'st':st,'html':html}
exec(compile(ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in keep],type_ignores=[]),'production_helpers','exec'),ns)
class Protecciones(unittest.TestCase):
 def test_saved_reference_comparison_and_report(self):
  names=['resnet50']+[f'model{i}' for i in range(12)];seeds=list(range(6101,6106));ns.update(NAMES=names,SEEDS=seeds,core=types.SimpleNamespace(CLASSES=['a','b','c']))
  y=np.tile([0,1,2],6);patients=np.repeat(np.arange(6),3);rows=[dict(clase=['a','b','c'][c],paciente=str(patients[i]),tss='site',archivo=str(i)) for i,c in enumerate(y)]
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'out';out.mkdir()
   for name in names:
    for seed in seeds:
     folder=out/'corridas'/f'{name}_{seed}';folder.mkdir(parents=True);ns['write_npz'](folder/'reserva.npz',y=y,p=np.eye(3)[y])
   status=ns['baselines_saved'](Path(d),out,Path(d),{'train':rows},rows,y,patients,{'baseline_archivos':{}})
   self.assertEqual(status['logistica'],'PENDIENTE_NO_HAY_MODELO_GUARDADO');self.assertTrue((out/'contrastes_referencias_paciente.csv').is_file())
   ns['report'](out,[{'modelo':'resnet50','macro_f1_media':1.}],[],status);self.assertIn('NuCLS', (out/'informe.html').read_text())
 def test_cache_reused_without_inference(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'test.npz';ns['write_npz'](path,y=np.array([1]),p=np.array([[0.,1.,0.]]));identity={'checkpoint':'fijo'};st.save(path.with_suffix('.json'),dict(identidad=identity,predicciones_sha256=ns['sha'](path)))
   y,p=ns['cache_infer'](None,None,None,path,identity);self.assertEqual(y.tolist(),[1])
 def test_changed_cache_stops(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'test.npz';path.write_bytes(b'changed');st.save(path.with_suffix('.json'),dict(identidad={},predicciones_sha256='wrong'))
   with self.assertRaises(ValueError):ns['cache_infer'](None,None,None,path,{})
 def test_changed_plan_stops(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'test.npz';ns['write_npz'](path,y=np.array([1]),p=np.array([[0.,1.,0.]]));st.save(path.with_suffix('.json'),dict(identidad={'checkpoint':'old'},predicciones_sha256=ns['sha'](path)))
   with self.assertRaises(ValueError):ns['cache_infer'](None,None,None,path,{'checkpoint':'new'})
 def test_missing_checkpoint_stops_before_reserve(self):
  ns.update(BASES=['resnet50'],NAMES=['resnet50'],SEEDS=[6101],nuevos=types.SimpleNamespace(MODELS={}))
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'NuCLS_E2_CONTINUACION_BASES_22';root.mkdir();(root/'indice_validacion.csv').write_text('modelo,semilla,ruta,resultado_sha256\n')
   with self.assertRaisesRegex(RuntimeError,'No se abre reserva'):ns['inventory'](Path(d),{'validation':[]})
 def test_prediction_identity(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'p.csv';p.write_text('archivo,paciente,clase_real,p_tumor,p_linfocito,p_estroma\na,p,c,0,1,0\n')
   with self.assertRaises(ValueError):ns['read_prediction'](p,[dict(archivo='a',paciente='WRONG',clase='c')])
if __name__=='__main__':unittest.main()
