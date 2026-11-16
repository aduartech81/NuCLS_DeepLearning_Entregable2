"""Inventario temprano de solo lectura; no monta Drive, abre reserva ni inicia procesos."""
from pathlib import Path
import csv,json,hashlib
BASES=['softmaxcnn_bn','densenet121','resnet50']
SIETE=['resnet18','mobilenetv3_small_100','efficientnet_b0','deit_tiny_patch16_224','swin_tiny_patch4_window7_224','mobilevit_xxs','mixer_s16_224']
TRES=['convnext_tiny','inception_v3','sequencer2d_s']
def comprobar_fuentes(drive):
 drive=Path(drive)
 if not (drive/'NuCLS_V2_DATOS.zip').is_file():raise FileNotFoundError('Falta NuCLS_V2_DATOS.zip en esta cuenta. No se inició evaluación.')
 root=drive/'NuCLS_E2_CONTINUACION_BASES_22';idx=root/'indice_validacion.csv'
 if not idx.is_file():idx=root/'final/indice_validacion.csv'
 if not idx.is_file():raise FileNotFoundError('Falta el índice de las bases en esta cuenta de Drive.')
 mapping={}
 with idx.open() as f:
  for row in csv.DictReader(f):
   key=(row['modelo'],int(row['semilla']))
   if key in mapping:raise ValueError('Índice duplicado: '+str(key))
   mapping[key]=row
 missing=[];verified=[]
 for model in BASES+SIETE+TRES:
  for seed in range(6101,6106):
   item=mapping.get((model,seed))
   folder=Path(item['ruta']) if model in BASES and item else (None if model in BASES else drive/('NuCLS_E2_TRES_NUEVOS_V1' if model in TRES else 'NuCLS_E2_ADICIONALES_EXPLORATORIO')/'piloto'/f'{model}_{seed}')
   files=['resultado.json','historia.json','predicciones_validacion.csv','mejor.pt']
   if folder is None or any(not (folder/n).is_file() or (folder/n).stat().st_size==0 for n in files):missing.append(f'{model}_{seed}');continue
   q=json.loads((folder/'resultado.json').read_text())
   if (q.get('modelo'),q.get('semilla'),q.get('estado'))!=(model,seed,'FASE_COMPLETADA_VALIDACION'):raise ValueError('Resultado incompleto o de otra corrida: '+str(folder))
   if model in BASES and item.get('resultado_sha256') and hashlib.sha256((folder/'resultado.json').read_bytes()).hexdigest()!=item['resultado_sha256']:raise ValueError('Resultado distinto del índice: '+str(folder))
   verified.append(f'{model}_{seed}')
 if missing:raise RuntimeError(f'Preflight: {len(verified)}/65 presentes. Faltan: '+', '.join(missing)+'. No se entrenó ni se abrió reserva.')
 return {'estado':'FUENTES_65_PRESENTES','corridas':len(verified),'nota':'Presencia e identidad preliminares; el cierre aún verificará hashes, historias, predicciones y checkpoints.'}
