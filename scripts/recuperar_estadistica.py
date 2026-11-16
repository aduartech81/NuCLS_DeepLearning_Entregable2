"""Recuperación numérica de E2 sin modificar código ni entradas congeladas.
NCT fallback: definición https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html
"""
from pathlib import Path
import sys, json, hashlib, math, datetime
import numpy as np
from scipy import stats, integrate, special

ORIGINAL_CDF=stats.nct.cdf
ORIGINAL_SF=stats.nct.sf
FALLBACKS=[]

def quadrature(x,df,nc,survival=False):
    if not (np.isfinite([x,df,nc]).all() and df>=1):
        raise ValueError('Parámetros NCT fuera del dominio de esta recuperación')
    # T=(Z+nc)/sqrt(V/df); Y=sqrt(V) tiene distribución chi.
    mode=math.sqrt(max(df-1.,0.))
    lognorm=(1-df/2)*math.log(2)-special.gammaln(df/2)
    def integrand(z):
        y=z+mode
        if y<=0:return 0.
        density=math.exp(lognorm+(df-1)*math.log(y)-y*y/2)
        a=x*y/math.sqrt(df)-nc
        return density*special.ndtr(-a if survival else a)
    # La masa fuera de ±12 del modo es despreciable para df>=1.
    val,err=integrate.quad(integrand,max(-mode,-12.),12.,epsabs=1e-12,epsrel=1e-10,limit=250,points=[0.])
    if not np.isfinite(val) or err>1e-8 or not -1e-10<=val<=1+1e-10:
        raise ValueError(f'Integración NCT no convergente: {val}, {err}')
    return float(np.clip(val,0,1))

def stable(original,survival):
    def evaluate(x,df,nc,*args,**kwargs):
        if args or kwargs: return original(x,df,nc,*args,**kwargs)
        xx,dd,cc=np.broadcast_arrays(np.asarray(x,float),np.asarray(df,float),np.asarray(nc,float))
        out=np.array(original(xx,dd,cc),dtype=float,copy=True)
        for ix in np.ndindex(out.shape):
            if not np.isfinite(out[ix]) or not 0<=out[ix]<=1:
                out[ix]=quadrature(float(xx[ix]),float(dd[ix]),float(cc[ix]),survival)
                FALLBACKS.append(dict(funcion='sf' if survival else 'cdf',x=float(xx[ix]),df=float(dd[ix]),nc=float(cc[ix]),valor=float(out[ix])))
        return float(out) if out.ndim==0 else out
    return evaluate

def main():
    codigo=Path('/content/nucls_e2_cierre_reserva_v1_codigo')
    drive=Path('/content/drive/MyDrive');out=drive/'NuCLS_E2_CIERRE_RESERVA_V1'
    planfile=out/'entradas_fijadas.json'
    if not planfile.is_file():raise RuntimeError('Falta el plan congelado. No se inició recuperación.')
    plan=json.loads(planfile.read_text());pid=hashlib.sha256(planfile.read_bytes()).hexdigest()
    if (out/'EJECUCION_ACTIVA.json').exists():raise RuntimeError('Hay una marca de ejecución: no se borra ni se inicia otra copia.')
    for name,h in plan['codigo'].items():
        if hashlib.sha256((codigo/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Código distinto del plan: '+name)
    runs=plan['corridas']
    if len(runs)!=65:raise RuntimeError('El plan no contiene 65 corridas.')
    for run in runs:
        dest=out/'corridas'/f"{run['modelo']}_{run['semilla']}"
        marker=json.loads((dest/'completo.json').read_text())
        if marker['plan']!=pid:raise RuntimeError('Marcador de otro plan')
        for name,h in marker['archivos'].items():
            if hashlib.sha256((dest/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Evidencia distinta: '+str(dest/name))
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    audit=out/('recuperacion_numerica_'+stamp+'.json')
    record=dict(plan_sha256=pid,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),metodo='NCT por integración de su definición solo cuando SciPy devuelve un valor inválido',corridas_completas_verificadas=65,codigo_congelado_modificado=False)
    def save():audit.write_text(json.dumps(dict(record,sustituciones=FALLBACKS),indent=2))
    save()
    stats.nct.cdf=stable(ORIGINAL_CDF,False);stats.nct.sf=stable(ORIGINAL_SF,True)
    sys.path.insert(0,str(codigo))
    import cierre_unico
    print('RECUPERACION: 65 cierres verificados. Se conserva el plan; se reutilizan inferencias.',flush=True)
    try:cierre_unico.main(drive)
    finally:save()
    print('RECUPERACION_ESTADISTICA_COMPLETADA. Continúa con la celda de respaldo urgente del mismo cuaderno.',flush=True)

if __name__=='__main__':main()
