"""Conserva un cierre íntegro antes de los experimentos largos; solo usa archivos locales."""
import hashlib,json,zipfile,shutil,os
from pathlib import Path

def respaldar(zip_cierre,drive):
    source=Path(zip_cierre);drive=Path(drive)
    with zipfile.ZipFile(source) as z:
        if z.testzip() is not None:raise ValueError('ZIP de cierre dañado')
        state=json.loads(z.read('estado.json'))
        if state.get('estado')!='CIERRE_RESERVA_Y_ESTADISTICA_COMPLETADO' or state.get('corridas')!=65:
            raise ValueError('El cierre no acredita 65 corridas completas')
        for name in ['entradas_fijadas.json','congelamiento_previo_reserva.json','resumen_reserva.csv','contrastes_vs_resnet50.csv','informe.html','benchmark_reserva.png','diferencia_critica.png']:
            if name not in z.namelist():raise ValueError('Salida de cierre ausente: '+name)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    folder=drive/'NuCLS_E2_ENTREGA_URGENTE'
    if folder.is_symlink():raise ValueError('Destino enlazado no permitido')
    folder.mkdir(exist_ok=True)
    target=folder/('CIERRE_65_'+digest[:16]+'.zip')
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise ValueError('Respaldo existente alterado; no se sobrescribe')
    else:
        temp=target.with_suffix('.tmp')
        with temp.open('xb') as out,source.open('rb') as inp:
            shutil.copyfileobj(inp,out);out.flush();os.fsync(out.fileno())
        if hashlib.sha256(temp.read_bytes()).hexdigest()!=digest:raise ValueError('Copia no coincide')
        temp.replace(target)
    receipt=folder/('RECIBO_'+digest[:16]+'.json')
    data={'estado':'CIERRE_URGENTE_RESPALDADO','zip':target.name,'sha256':digest,'corridas':65,
          'alcance':'Cierre y estadística; no certifica todos los requisitos ni los experimentos posteriores',
          'siguiente':'Continuar experimentos largos; este respaldo queda disponible para descargar'}
    if receipt.exists():
        if json.loads(receipt.read_text())!=data:raise ValueError('Recibo existente no coincide')
    else:
        with receipt.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2)
    return target,receipt
