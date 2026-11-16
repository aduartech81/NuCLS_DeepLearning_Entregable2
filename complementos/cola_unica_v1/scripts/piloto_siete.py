"""Completa solo los siete candidatos retenidos; conserva motor y carpetas previas."""
import argparse,sys,json,hashlib,os,platform
from pathlib import Path
MODELS=['resnet18','mobilenetv3_small_100','efficientnet_b0','deit_tiny_patch16_224','swin_tiny_patch4_window7_224','mobilevit_xxs','mixer_s16_224']
def main():
 p=argparse.ArgumentParser();p.add_argument('--codigo',type=Path,required=True);p.add_argument('--datos',type=Path,required=True);p.add_argument('--salida',type=Path,required=True);a=p.parse_args()
 if a.salida.name!='NuCLS_E2_ADICIONALES_EXPLORATORIO':raise ValueError('Salida no autorizada')
 sys.path.insert(0,str(a.codigo/'modelos'))
 import core_v2 as core,adicionales_e2 as models,torch,numpy as np,torchvision,timm
 protocol=json.loads((a.salida/'protocolo_fijado.json').read_text());env=json.loads((a.salida/'entorno_fijado.json').read_text())
 for k,v in dict(torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,timm=timm.__version__,python=platform.python_version(),gpu=torch.cuda.get_device_name(0)).items():
  if env[k]!=v:raise ValueError('Entorno cambiado: '+k)
 if protocol['datos']!=str(a.datos.resolve()):raise ValueError('Ruta de datos diferente al protocolo')
 for name,h in protocol['motor'].items():
  f=a.codigo/('scripts' if name=='adicionales_e2.py' else 'modelos')/name
  # El protocolo original usa dos archivos homónimos; su diccionario conserva el de modelos.
  if name=='adicionales_e2.py':f=a.codigo/'modelos'/name
  if hashlib.sha256(f.read_bytes()).hexdigest()!=h:raise ValueError('Código distinto: '+name)
 if any(x.is_symlink() for x in a.salida.rglob('*')):raise ValueError('Salida enlazada')
 lock=a.salida/'EJECUCION_ACTIVA.json'
 with lock.open('x') as f:json.dump({'pid':os.getpid(),'host':platform.node()},f)
 try:
  torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  groups=core.read_data(a.datos);core.build=models.build;core.archive=lambda out:out;core.summary=lambda out:out
  choices=json.loads((a.salida/'seleccion_hiperparametros.json').read_text())
  for seed in range(6101,6106):
   for name in MODELS:
    folder=a.salida/'piloto';folder.mkdir(exist_ok=True);run=folder/f'{name}_{seed}';r=run/'resultado.json'
    if r.exists():
     result=json.loads(r.read_text());history=json.loads((run/'historia.json').read_text())
     if (result['modelo'],result['semilla'],result['estado'])!=(name,seed,'FASE_COMPLETADA_VALIDACION'):raise ValueError('Resultado incompatible')
     if len(history)!=result['epocas_ejecutadas'] or abs(max(v['val_macro_f1'] for v in history)-result['validacion']['macro_f1'])>1e-9:raise ValueError('Historia incompatible')
     for f in ['mejor.pt','ultimo.pt','predicciones_validacion.csv','curvas.csv','metricas_por_paciente.csv']:
      if not (run/f).is_file():raise ValueError('Evidencia incompleta '+str(run))
     print('OMITIDA COMPLETA',name,seed,flush=True);continue
    hp={k:choices[name][k] for k in ['lr','weight_decay']};args=argparse.Namespace(salida=folder,datos=a.datos,batch=32,epocas=25,paciencia=5,min_epocas=5,**hp)
    core.run_one(name,seed,groups,args,torch.device('cuda'))
    print('PILOTO GUARDADO',name,seed,flush=True)
 finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
