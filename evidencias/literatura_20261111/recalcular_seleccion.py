import json,math,itertools,collections,pathlib
R=pathlib.Path(__file__).resolve().parent;meta=json.loads((R/'metadatos_candidatos.json').read_text())
# model,key,family,firstyear,domain quality-weighted evidence,source,eligible,reason
rows=[
('resnet18','resnet','CNN',2015,.6,'H2',True,''),
('resnext50_32x4d','resnext','CNN',2016,.4,'H6',True,''),
('mobilenetv3_small_100','mobilenetv3','CNN',2019,.3,'H5: Large; Small no verificado',True,''),
('efficientnet_b0','efficientnet','CNN',2019,.6,'H2: 4.01M; familia EfficientNet',True,''),
('deit_tiny_patch16_224','deit','Atención',2020,.3,'H7: variante no precisada; preprint',True,''),
('swin_tiny_patch4_window7_224','swin','Atención',2021,.8,'H1: variante exacta y 5 repeticiones',True,''),
('mobilevit_xxs','mobilevit','Híbrida CNN-atención',2021,.5,'H2: otra escala; validación externa',True,''),
('mobilevit_xs','mobilevit','Híbrida CNN-atención',2021,.5,'H2: otra escala',False,'Una variante por artículo fundacional; XXS conserva el mecanismo con menor costo'),
('mixer_s16_224','mixer','MLP visual',2021,.3,'H4: variante no precisada; partición por paciente no establecida',True,''),
('resmlp_12_224','resmlp','MLP visual',2021,0,'Sin evidencia histológica directa localizada del ResMLP de Touvron',True,''),
('convnext_tiny','convnext','CNN',2022,.8,'H1: variante exacta y 5 repeticiones',True,''),
('inception_v3','inception','CNN',2015,.8,'H1: variante exacta y 5 repeticiones',True,''),
('sequencer2d_s','sequencer','Recurrente',2022,.5,'H2: familia y tamaño concordante; validación externa',True,''),
('vgg16','vgg','CNN',2014,.4,'Antecedente E1',False,'Modelo denso de alto tamaño; fuera del cupo compacto operativo de esta fase'),
('vit_base_patch16_224','vit','Atención',2020,.6,'H2; H3',False,'85.8M en H2; fuera del cupo compacto operativo de esta fase'),
('cmt','cmt','Híbrida CNN-atención',2022,.5,'H2',False,'Sin implementación registrada en timm 1.0.22; integración y validación adicionales'),
('vmamba','vmamba','Estado espacial',2024,0,'Fundacional; sin extracción de aplicación histológica',False,'No registrado en timm 1.0.22; no confundir con MambaOut'),
('cellvit','cellvit','Atención',2023,1,'H8: tarea nuclear conjunta',False,'Segmentación y clasificación conjunta; requiere máscaras y cambia la tarea'),
('ctranspath','ctranspath','Atención',2022,.6,'Contexto de preentrenamiento histológico',False,'Representaciones preentrenadas externas; fuera del contraste desde cero'),
('uni','uni','Atención',2024,.6,'Contexto de modelos fundacionales',False,'Representaciones preentrenadas externas; fuera del contraste desde cero')]
a=[]
for model,key,fam,year,e,source,eligible,reason in rows:
 candidates=meta[key]
 # exact matched selected record; ResNet conference not preprint, remaining first verified title
 m=candidates[1] if key=='resnet' else candidates[0]
 a.append(dict(modelo=model,clave=key,familia=fam,primer_anio=year,doi=m['doi'],titulo=m['title'],openalex_id=m['id'],citas=m['cited_by_count'],evidencia=e,fuente_evidencia=source,elegible=eligible,exclusion=reason,codigo=1 if eligible or model=='mobilevit_xs' else 0))
elig=[r for r in a if r['elegible']];count=collections.Counter(r['familia'] for r in elig)
maxlog=max(math.log1p(r['citas']/(2026-r['primer_anio']+1)) for r in elig)
for r in a:
 r['citas_por_anio']=r['citas']/(2026-r['primer_anio']+1)
 r['C']=min(1,math.log1p(r['citas_por_anio'])/maxlog)
 r['A']=max(0,min(1,1-(2026-r['primer_anio'])/12))
 r['F']=1/count[r['familia']] if r['elegible'] else 0

def score(r,w):return w[0]*r['C']+w[1]*r['A']+w[2]*r['evidencia']+w[3]*r['codigo']+w[4]*r['F']
def choose(w):
 combos=[c for c in itertools.combinations(elig,10) if len({r['familia'] for r in c})>=4]
 return max(combos,key=lambda c:(sum(score(r,w) for r in c),tuple(sorted(r['modelo'] for r in c))))
w=(20,15,35,20,10);chosen=choose(w);ss={r['modelo'] for r in chosen}
for r in a:r['puntaje']=round(score(r,w),5) if r['elegible'] else None;r['seleccionado']=r['modelo'] in ss
sens=[]
for name,ww in [('principal',w),('evidencia',(10,10,50,20,10)),('impacto',(35,10,25,20,10)),('diversidad',(15,10,30,20,25))]:
 cc=choose(ww);sens.append(dict(escenario=name,pesos=ww,seleccion=[r['modelo'] for r in cc]))
(R/'seleccion.json').write_text(json.dumps({'fecha':'2026-11-11','estado':'SELECCION_BIBLIOGRAFICA_CONDICIONADA_A_PERFIL_GPU','pesos':w,'candidatos':a,'sensibilidad':sens},indent=2,ensure_ascii=False))
for r in sorted(elig,key=lambda r:-r['puntaje']):print(r['seleccionado'],r['modelo'],r['puntaje'])
print(json.dumps(sens,ensure_ascii=False))
