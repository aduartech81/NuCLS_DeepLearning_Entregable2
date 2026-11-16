from pathlib import Path
import csv,json,hashlib,zipfile,io,re,unicodedata,difflib,collections,math,html
import argparse
ap=argparse.ArgumentParser(description='Recalcula auditorías locales; no repite búsquedas ni abre Colab.');ap.add_argument('--zip',type=Path,required=True);args=ap.parse_args()
R=Path(__file__).resolve().parents[1];L=R/'evidencias/literatura_20261111';E=R/'evidencias/cierre_documental_20261115';E.mkdir(exist_ok=True)
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def table(p,rows):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def norm(x):return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',x.lower()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
base=read(L/'base_heredada/cribado_unificado.csv');ad=read(L/'base_heredada/adjudicacion_sintesis_20261108.csv');hs=read(L/'Cruce_dominio_E1_E2_20261112.csv')
titles={'H1':'From modern CNNs to vision transformers: Assessing the performance, robustness, and classification strategies of deep learning models in histopathology','H2':'Time to Embrace Natural Language Processing (NLP)-based Digital Pathology: Benchmarking NLP- and Convolutional Neural Network-based Deep Learning Pipelines','H4':'A multilayer perceptron-based model applied to histopathology image classification of lung adenocarcinoma subtypes','H6':'The value of deep neural networks in the pathological classification of thyroid tumors','H7':'ViT-DeiT: An Ensemble Model for Breast Cancer Histopathological Images Classification'}
ident=[]
for h in hs:
 matches=[b for b in base if b['doi'].strip().lower()==h['doi'].strip().lower()];title=titles.get(h['id_e2'],matches[0]['titulo'] if matches else h['titulo']);tm=[b for b in base if norm(b['titulo'])==norm(title)]
 ident.append(dict(id_e2=h['id_e2'],titulo_completo=title,doi=h['doi'],coincidencias_doi=';'.join(b['id'] for b in matches),coincidencias_titulo_normalizado=';'.join(b['id'] for b in tm),decision='Ya representado en E1; no sumar' if matches or tm else 'Sin coincidencia exacta en corpus exportado; no demuestra independencia de todas las versiones',fuente_identidad=('https://arxiv.org/abs/2302.10406' if h['id_e2']=='H2' else 'https://arxiv.org/abs/2211.00749' if h['id_e2']=='H7' else 'https://doi.org/'+h['doi'])))
table(E/'Identidades_dominio.csv',ident)
pairs=[]
for i,x in enumerate(ad):
 for y in ad[i+1:]:
  sim=difflib.SequenceMatcher(None,norm(x['titulo']),norm(y['titulo'])).ratio()
  if sim>.83:pairs.append(dict(informe_a=x['id'],informe_b=y['id'],similitud_titulo=sim,decision='No fusionar automáticamente',fundamento='DT-Yolact y StarDist: métodos distintos' if x['id']=='LIT-0545' else 'STi-HoverNet con distilación y TRG-DilatedHoVer: variantes distintas según fichas; independencia de datos no acreditada',fuente_a=x['fuente'],fuente_b=y['fuente']))
table(E/'Candidatos_similitud.csv',pairs)
counts=collections.Counter(x['grupo_sintesis'] for x in ad)
checks=dict(registros_identificados=1717,eliminados_automaticamente=69,duplicados=318,registros_unificados=len(base),lecturas_heredadas=len(ad),categorias=dict(counts),no_recuperados_E1=13,recuperados_E2=1,no_recuperados_actuales=12,referencias_dominio=len(hs),coincidencias_exactas_con_E1=sum(bool(x['coincidencias_doi']) for x in ident),pares_titulo_similares=len(pairs),doi_duplicados_entre166=sum(n-1 for n in collections.Counter(x['doi'].strip().lower() for x in ad if x['doi']).values()),titulo_duplicado_entre166=sum(n-1 for n in collections.Counter(norm(x['titulo']) for x in ad).values()),estudios_globales_independientes=None,limite='166 informes no equivalen automáticamente a166 estudios independientes; no reconstruir cadenas originales ausentes')
assert 1717==69+318+len(base);assert len(base)==820+330+1+len(ad)+13;assert sum(counts.values())==166
save(E/'Conciliacion_PRISMA.json',checks)
queries=[('identidad','"10.1016/j.media.2023.102809"','https://pubmed.ncbi.nlm.nih.gov/37201221/'),('identidad','"2302.10406"','https://arxiv.org/abs/2302.10406'),('identidad','"10.3389/fonc.2023.1172234"','https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2023.1172234/full'),('identidad','"10.1186/s13000-023-01380-2"','https://link.springer.com/article/10.1186/s13000-023-01380-2'),('recuperacion','site.irma-international.org "Histopathological Image Analysis in Medical Decision Making"','https://www.irma-international.org/viewtitle/344677/?isxn=9798369352618'),('recuperacion','"MedViT-HoVer" "pdf"','https://link.springer.com/book/10.1007/978-3-032-18834-2?page=2'),('recuperacion','"Synthetic data generation" "Derus" arxiv','https://cris.unibo.it/handle/11585/963147'),('mecanismo','site.arxiv.org speech recognition convolution transformer Conformer','https://www.isca-archive.org/interspeech_2020/gulati20_interspeech.pdf'),('mecanismo','site.arxiv.org long sequences structured state spaces S4 time series','https://arxiv.org/html/2111.00396v3'),('mecanismo','site.arxiv.org natural language knowledge distillation DistilBERT','https://arxiv.org/html/1910.01108v4')]
table(E/'Registro_consultas_20261115.csv',[dict(fecha='2026-11-15',proposito=t,consulta=q,plataforma='Buscador web; acceso directo posterior a fuente primaria',fuente_retenida=u,total_resultados='NR; sin exportación exhaustiva',alcance='Consulta dirigida; no nueva búsqueda institucional sistemática') for t,q,u in queries])
mechanisms=[dict(id='M1',modelo='Conformer',modalidad='Voz',fuente=queries[7][2],localizador='Sección2, encoder',mecanismo='Combina convolución local con atención global',transferencia_propuesta='Bloques híbridos para morfología local y contexto tisular',riesgo='La estructura temporal de voz no equivale a la geometría2D; requiere experimento independiente'),dict(id='M2',modelo='S4',modalidad='Secuencias y series temporales',fuente=queries[8][2],localizador='Sección3; apéndiceD.3.5',mecanismo='Estado estructurado para dependencias largas',transferencia_propuesta='Procesamiento de secuencias de parches con posición2D explícita',riesgo='Sensibilidad al recorrido de parches; resultados de pronóstico no validan clasificación nuclear'),dict(id='M3',modelo='DistilBERT',modalidad='Lenguaje',fuente=queries[9][2],localizador='Secciones2–3',mecanismo='Distilación de distribuciones y representaciones de un docente',transferencia_propuesta='Comprimir un clasificador histológico usando solo datos de desarrollo',riesgo='Puede heredar sesgos del docente; las mejoras de lenguaje no se trasladan numéricamente a NuCLS')]
table(E/'Mecanismos_transversales.csv',mechanisms)
# Metadata-only EDA from existing predictions. No images/reserve read.
source=args.zip;records=[];reference=None;allclasses=set();allpatients=set()
with zipfile.ZipFile(source) as z:
 files=sorted(n for n in z.namelist() if n.endswith('/predicciones_validacion.csv'))
 for f in files:
  raw=z.read(f);rr=list(csv.DictReader(io.StringIO(raw.decode())));ids=[(r['archivo'],r['paciente'],r['clase_real']) for r in rr];assert len(ids)==len(set(x[0] for x in ids));assert all(all(x) for x in ids)
  if reference is None:reference=ids
  assert ids==reference,'Identidades no alineadas'
  records.append(dict(archivo=f,sha256=hashlib.sha256(raw).hexdigest(),n=len(rr),identidad_alineada=True))
cc=collections.Counter(x[2] for x in reference);pc=collections.Counter(x[1] for x in reference);n=len(reference);entropy=-sum((v/n)*math.log2(v/n) for v in cc.values())
table(E/'Validacion_contrato_archivos.csv',records);table(E/'Validacion_clases.csv',[dict(clase=k,n=v,porcentaje=100*v/n,pacientes_con_clase=len({p for _,p,c in reference if c==k})) for k,v in sorted(cc.items())])
pat=[]
for p,count in sorted(pc.items()):
 c=collections.Counter(c for _,q,c in reference if q==p);pat.append(dict(paciente=p,n=count,clases_presentes=len(c),fraccion_recortes=count/n,**{k:c[k] for k in sorted(cc)}))
table(E/'Validacion_pacientes.csv',pat)
eda=dict(fuente_zip=source.name,sha256_zip=sha(source),archivos_verificados=len(records),recortes=n,pacientes=len(pc),clases=dict(cc),entropia_bits=entropy,razon_mayor_menor=max(cc.values())/min(cc.values()),min_recortes_paciente=min(pc.values()),max_recortes_paciente=max(pc.values()),fraccion_paciente_mayor=max(pc.values())/n,pacientes_con_menos_tres_clases=sum(p['clases_presentes']<3 for p in pat),alcance='Identidad/etiquetas registradas en predicciones de validación; no verifica imágenes, dHash, etiquetas histológicas ni partición train/reserva')
assert n==2049 and len(pc)==21 and len(cc)==3
save(E/'EDA_validacion_recalculado.json',eda)
# Random review form: labels blinded initially, no grading fabricated.
import random
sample=random.Random(15112026).sample(reference,200)
table(E/'Revision_etiquetas_validacion_200.csv',[dict(archivo=a,paciente=p,etiqueta_original=c,etiqueta_revisada='',motivo='',revisor='',probabilidad_inclusion=200/n) for a,p,c in sample])

print("AUDITORIA_DOCUMENTAL_COMPLETADA", E)
