import ast,json,unittest,tempfile,sys,datetime,os,platform
from pathlib import Path
NB=Path(__file__).parent/'NuCLS_E2_Cola_Unica_V1.ipynb'
class ColaTests(unittest.TestCase):
 def setUp(self):self.nb=json.loads(NB.read_text())
 def test_celdas_compilan_sin_input(self):
  for c in self.nb['cells']:
   if c['cell_type']=='code':
    s=''.join(c['source']);compile(s,'cell','exec');self.assertNotIn('input(',s)
 def test_bloqueo_de_otro_host_conservado(self):
  s=''.join(self.nb['cells'][1]['source']);tree=ast.parse(s);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='comprobar_bloqueo');ns=dict(json=json,platform=platform,os=os,datetime=datetime)
  exec(compile(ast.Module(body=[fn],type_ignores=[]),'guard','exec'),ns)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'lock.json';p.write_text(json.dumps({'host':'otro_host','pid':123}));before=p.read_bytes()
   with self.assertRaises(RuntimeError):ns['comprobar_bloqueo'](p)
   self.assertEqual(before,p.read_bytes())
 def simulate(self,fail=None,profile=False):
  tree=ast.parse(''.join(self.nb['cells'][7]['source']));block=next(n for n in tree.body if isinstance(n,ast.Try));calls=[]
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);new=root/'nuevos';new.mkdir();lock=root/'lock';lock.touch()
   if profile:(new/'perfil.json').write_text('[{}, {}, {}]')
   def run(name,cmd):
    calls.append(name)
    if name==fail:raise RuntimeError('simulated failure')
   state={};ns=dict(json=json,sys=sys,datetime=datetime,ROOT=root,DRIVE=root,CODIGO_ANTERIOR=root,DATOS_ANTERIORES=root,SALIDA_ANTERIOR=root,CODIGO_NUEVOS=root,DATOS_NUEVOS=root,SALIDA_NUEVOS=new,HELPERS=root,INDICE=root/'index',estado=state,correr=run,fase=lambda *a:list(a),empaquetar=lambda:None,guardar=lambda:None,lock=lock)
   try:exec(compile(ast.Module(body=[block],type_ignores=[]),'queue','exec'),ns)
   except RuntimeError:
    if not fail:raise
   self.assertFalse(lock.exists());return calls,state
 def test_orden_y_cierre(self):
  calls,state=self.simulate();self.assertEqual(len(calls),9);self.assertEqual(state['estado'],'COLA_EXPLORATORIA_COMPLETADA')
 def test_no_continua_tras_fallo(self):
  calls,state=self.simulate('03_busqueda_tres');self.assertEqual(calls[-1],'03_busqueda_tres');self.assertEqual(state['estado'],'DETENIDA_CON_EVIDENCIA_CONSERVADA')
 def test_perfil_existente_no_repite(self):
  calls,_=self.simulate(profile=True);self.assertNotIn('02_perfil_tres',calls)
if __name__=='__main__':unittest.main()
