"""Ejecución secuencial v2: búsqueda, piloto, potencia, corridas y reserva.
Las imágenes originales permanecen intactas. El manifiesto está fijado por SHA256.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,hashlib,json,platform,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch,torchvision
from scipy.stats import t,nct
from core_v2 import (EXPECTED,CLASSES,build,read_data,run_one,save_json,csv_write,Crops,evaluate,scores,archive)
from torch.utils.data import DataLoader

MODELS=['softmaxcnn_bn','densenet121','resnet50']
GRID=[{'lr':lr,'weight_decay':wd} for lr in [1e-4,1e-3] for wd in [0.,1e-4]]

def result(out,name,seed):return json.loads((out/f'{name}_{seed}'/'resultado.json').read_text())
def source_hash():
    h=hashlib.sha256()
    for name in ['experimento_v2.py','core_v2.py','candidatos_v2.py']:h.update(name.encode());h.update(Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()

def required_runs(sd,delta=.02,alpha=.05/3,power=.8):
    if sd==0:return 5
    for n in range(5,501):
        cutoff=t.ppf(1-alpha/2,n-1);ncp=delta*np.sqrt(n)/sd
        if nct.sf(cutoff,n-1,ncp)+nct.cdf(-cutoff,n-1,ncp)>=power:return n
    return None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--datos',type=Path,required=True);ap.add_argument('--salida',type=Path,required=True);ap.add_argument('--device',choices=['auto','cuda','mps','cpu'],default='auto');a=ap.parse_args()
    a.datos=a.datos.expanduser().resolve();a.salida=a.salida.expanduser().resolve()
    if a.salida.is_relative_to(a.datos):ap.error('La salida debe estar fuera de las imágenes')
    device=torch.device(a.device if a.device!='auto' else 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True,warn_only=device.type=='mps')
    a.salida.mkdir(parents=True,exist_ok=True)
    config={'modelos':MODELS,'grid':GRID,'semilla_busqueda':4001,'semillas_piloto':list(range(5101,5106)),'semilla_inicial_final':6101,'epocas_maximas':25,'paciencia':5,'min_epocas':5,'batch':32,'delta':.02,'alpha_planificacion':.05/3,'potencia':.8,'max_corridas_por_modelo':50,'manifest_sha256':EXPECTED,'codigo_sha256':source_hash(),'entrada':'RGB 128x128; estandarización con medias y SD de train; rotaciones 90 y reflexión horizontal solo train','preentrenamiento':False,'criterio_checkpoint':'Mayor macro-F1 de validación; empate conserva primera época','criterio_hiperparametros':'Mayor macro-F1 validación; empate menor lr, luego menor weight_decay','criterio_baseline':'Mayor media final de macro-F1 validación; empate menor número de parámetros','reserva':'Una evaluación por checkpoint final dentro de un lote predefinido después de fijar todos los modelos; no selecciona modelos','device':str(device),'determinismo':'warn_only en MPS; estricto en CPU/CUDA, sujeto a entorno','torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,'python':sys.version,'platform':platform.platform()}
    cfg=a.salida/'configuracion.json'
    if cfg.exists():
        if json.loads(cfg.read_text())!=config:raise ValueError('Configuración o entorno cambiaron; no se mezclan ejecuciones')
    else:save_json(cfg,config)
    args=SimpleNamespace(datos=a.datos,batch=32,epocas=25,paciencia=5,min_epocas=5)
    print('Verificando entrenamiento y validación; no se abren imágenes de reserva.',flush=True)
    groups=read_data(a.datos);selected={};search=[]
    for model in MODELS:
        for i,hp in enumerate(GRID):
            args.salida=a.salida/'busqueda'/f'config_{i:02d}';args.salida.mkdir(parents=True,exist_ok=True);args.lr=hp['lr'];args.weight_decay=hp['weight_decay']
            run_one(model,4001,groups,args,device);q=result(args.salida,model,4001);search.append({'modelo':model,'configuracion':i,**hp,'macro_f1':q['validacion']['macro_f1']})
        candidates=[x for x in search if x['modelo']==model];selected[model]=sorted(candidates,key=lambda x:(-x['macro_f1'],x['lr'],x['weight_decay']))[0]
    csv_write(a.salida/'busqueda.csv',search);save_json(a.salida/'hiperparametros_seleccionados.json',selected)
    args.salida=a.salida/'piloto';args.salida.mkdir(exist_ok=True);pilot={}
    for model in MODELS:
        args.lr=selected[model]['lr'];args.weight_decay=selected[model]['weight_decay'];pilot[model]=[]
        for seed in config['semillas_piloto']:
            run_one(model,seed,groups,args,device);pilot[model].append(result(args.salida,model,seed)['validacion']['macro_f1'])
    pairs=[]
    import itertools
    for left,right in itertools.combinations(MODELS,2):
        sd=float(np.std(np.array(pilot[left])-np.array(pilot[right]),ddof=1));pairs.append({'A':left,'B':right,'sd_diferencias':sd,'R_requerido':required_runs(sd)})
    required=max((x['R_requerido'] or 501) for x in pairs);plan={'comparaciones':pairs,'R_por_modelo':required,'regla':'Máximo R para detectar delta .02 en tres diferencias pareadas con alpha .05/3 y potencia .8; mínimo cinco','limitacion':'Estimación basada en solo cinco pares; no incluye variabilidad entre pacientes','aprobado_por_limite_computacional':required<=50}
    save_json(a.salida/'plan_corridas.json',plan)
    if required>50:
        print('La precisión prevista requiere',required,'corridas/modelo. Se conserva evidencia y no se evalúa la reserva. ZIP:',archive(a.salida),flush=True);return
    args.salida=a.salida/'final';args.salida.mkdir(exist_ok=True);seeds=list(range(6101,6101+required));final=[]
    for seed in seeds:
        for model in MODELS:
            args.lr=selected[model]['lr'];args.weight_decay=selected[model]['weight_decay'];run_one(model,seed,groups,args,device);final.append(result(args.salida,model,seed))
    means={model:float(np.mean([x['validacion']['macro_f1'] for x in final if x['modelo']==model])) for model in MODELS};params={model:next(x['parametros'] for x in final if x['modelo']==model) for model in MODELS};baseline=sorted(MODELS,key=lambda m:(-means[m],params[m]))[0]
    # All checkpoint hashes are fixed before any reserve image is opened.
    weights={f'{model}_{seed}':hashlib.sha256((args.salida/f'{model}_{seed}'/'mejor.pt').read_bytes()).hexdigest() for model in MODELS for seed in seeds}
    frozen={'codigo_sha256':source_hash(),'manifest_sha256':EXPECTED,'modelos':MODELS,'semillas':seeds,'baseline_seleccionado_en_validacion':baseline,'media_macro_f1_validacion':means,'sha256_checkpoints':weights}
    freeze=a.salida/'evaluacion_reserva_fijada.json'
    if freeze.exists():
        if json.loads(freeze.read_text())!=frozen:raise ValueError('Cambió el lote de reserva; se bloquea una evaluación distinta')
    else:save_json(freeze,frozen)
    raw=(a.datos/'manifiesto.csv').read_bytes();assert hashlib.sha256(raw).hexdigest()==EXPECTED
    reserve=[r for r in csv.DictReader(raw.decode().splitlines()) if r['particion']=='reserva_candidata'];assert len(reserve)==969 and len({r['paciente'] for r in reserve})==6
    from PIL import Image
    for row in reserve:
        path=(a.datos/row['archivo']).resolve()
        if not path.is_relative_to(a.datos):raise ValueError('Ruta de reserva fuera de datos')
        with Image.open(path) as im:
            im.load();assert im.mode=='RGB' and im.size==(128,128);assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==row['sha_pixeles']
    testout=a.salida/'reserva';testout.mkdir(exist_ok=True);testresults=[]
    loader=DataLoader(Crops(a.datos,reserve),batch_size=32,num_workers=0)
    for model in MODELS:
        for seed in seeds:
            file=testout/f'{model}_{seed}.json'
            if file.exists():q=json.loads(file.read_text())
            else:
                net=build(model).to(device);net.load_state_dict(torch.load(args.salida/f'{model}_{seed}'/'mejor.pt',map_location=device,weights_only=True));loss,y,p=evaluate(net,loader,device)
                rows=[dict(paciente=row['paciente'],archivo=row['archivo'],clase_real=CLASSES[int(y[i])],clase_predicha=CLASSES[int(p[i].argmax())],p_tumor=float(p[i,0]),p_linfocito=float(p[i,1]),p_estroma=float(p[i,2])) for i,row in enumerate(reserve)]
                csv_write(testout/f'{model}_{seed}.csv',rows);q={'modelo':model,'semilla':seed,'checkpoint_sha256':weights[f'{model}_{seed}'],'metricas':scores(y,p.argmax(1)),'loss':loss};save_json(file,q);del net
            if q['checkpoint_sha256']!=weights[f'{model}_{seed}']:raise ValueError('Resultado de reserva incompatible')
            testresults.append(q)
    finalsummary=[]
    for model in MODELS:
        v=np.array([x['validacion']['macro_f1'] for x in final if x['modelo']==model]);ts=np.array([x['metricas']['macro_f1'] for x in testresults if x['modelo']==model]);half=t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
        finalsummary.append({'modelo':model,'R':required,'baseline':model==baseline,'macro_f1_val_media':float(v.mean()),'macro_f1_val_sd':float(v.std(ddof=1)),'IC95_semillas_inf':float(v.mean()-half),'IC95_semillas_sup':float(v.mean()+half),'macro_f1_reserva_media':float(ts.mean()),'macro_f1_reserva_sd':float(ts.std(ddof=1))})
    csv_write(a.salida/'resumen_experimento.csv',finalsummary);save_json(a.salida/'estado.json',{'estado':'EXPERIMENTO_COMPLETADO','baseline':baseline,'R':required,'advertencia':'IC por semillas condicionados a validación. Seis pacientes de reserva requieren análisis agrupado adicional; no son 969 pacientes.'})
    print('EXPERIMENTO COMPLETADO. ZIP:',archive(a.salida),flush=True)
if __name__=='__main__':main()
