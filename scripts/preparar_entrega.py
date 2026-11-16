"""Integra un cierre real en una copia nueva; compila y empaqueta. No accede a Colab."""
import argparse, hashlib, importlib.util, json, shutil, subprocess, sys, zipfile
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
ROOT = Path(__file__).resolve().parents[1]
class Links(HTMLParser):
    def __init__(self): super().__init__(); self.links=[]
    def handle_starttag(self, tag, attrs):
        self.links.extend(v for k,v in attrs if k in ('href','src') and v)
def comprobar_enlaces(root):
    count=0; missing=[]
    for f in root.rglob('*.html'):
        parser=Links(); parser.feed(f.read_text(errors='replace'))
        for link in parser.links:
            u=urlsplit(link)
            if u.scheme or u.netloc or not u.path or '{{' in link: continue
            count+=1
            if not (f.parent/unquote(u.path)).exists(): missing.append([str(f.relative_to(root)),link])
    if missing: raise ValueError('Enlaces locales rotos: '+repr(missing))
    return count

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('zip_cierre',type=Path)
    ap.add_argument('--destino',required=True,type=Path,help='Carpeta NUEVA de E2; nunca E1')
    ap.add_argument('--notebook-ejecutado',type=Path)
    a=ap.parse_args(); dest=a.destino.resolve(); out=dest.with_suffix('.zip')
    if dest.exists() or out.exists() or dest.is_relative_to(ROOT) or ROOT.is_relative_to(dest):
        raise ValueError('Destino existente o dentro del paquete original; elija una carpeta nueva')
    spec=importlib.util.spec_from_file_location('receptor',ROOT/'scripts/integrar_cierre_reserva.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    with zipfile.ZipFile(a.zip_cierre) as z:
        if z.testzip() is not None: raise ValueError('ZIP dañado')
        mod.validate(z)
        for n in ['informe.html','benchmark_reserva.png','diferencia_critica.png']:
            if n not in z.namelist(): raise ValueError('Falta salida para publicación: '+n)
    if a.notebook_ejecutado:
        nb=json.loads(a.notebook_ejecutado.read_text())
        if not any(c.get('cell_type')=='code' and c.get('outputs') and c.get('execution_count') is not None for c in nb.get('cells',[])):
            raise ValueError('El notebook no contiene evidencia de celdas ejecutadas con salidas')
    jb=shutil.which('jupyter-book')
    if not jb: raise RuntimeError('Instalar Jupyter Book en el equipo local antes de integrar')
    shutil.copytree(ROOT,dest,ignore=shutil.ignore_patterns('__pycache__','.doctrees','.jupyter_cache'))
    subprocess.run([sys.executable,str(dest/'scripts/integrar_cierre_reserva.py'),str(a.zip_cierre.resolve())],check=True)
    if a.notebook_ejecutado:
        folder=dest/'evidencias/notebooks_ejecutados';folder.mkdir(exist_ok=True)
        shutil.copy2(a.notebook_ejecutado,folder/'cierre_reserva_ejecutado.ipynb')
    subprocess.run([jb,'build',str(dest/'book')],check=True)
    links=comprobar_enlaces(dest)
    report={'cierre_importado':True,'enlaces_locales_verificados':links,
      'notebook_con_salidas_recibido':bool(a.notebook_ejecutado),
      'revision_cientifica_final_pendiente':True,
      'requisitos_ajenos_al_cierre':'No se marcan como completados automáticamente',
      'Colab_intervenido':False,'E1_modificado':False}
    (dest/'revision/Recepcion_para_entrega.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    manifest={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file() and p.name!='SHA256.json' and '__pycache__' not in p.parts and '.doctrees' not in p.parts}
    (dest/'SHA256.json').write_text(json.dumps(manifest,indent=2))
    temp=out.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted([*manifest,'SHA256.json']):z.write(dest/name,'NuCLS_Entregable2/'+name)
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:raise ValueError('ZIP de salida inválido')
    temp.replace(out)
    print('PAQUETE INTEGRADO:',out,'\nRevisar interpretación, matriz y notebook antes de enviar. No certifica cumplimiento total.')
if __name__=='__main__':main()
