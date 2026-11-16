from pathlib import Path
import json,csv,io,zipfile,hashlib,shutil,base64,html
from PIL import Image
import numpy as np
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--datos-historicos',type=Path,required=True);ap.add_argument('--predicciones-zip',type=Path,required=True);args=ap.parse_args()
BASE=args.datos_historicos
ORIG=BASE/'official/candidatos';CROPS=BASE/'piloto_interior128/20261105_225210'
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'evidencias/galeria_validacion_20261115').exists():raise RuntimeError('La galería ya existe; conservar el resultado verificado. Para regenerar use una copia sin esa galería y sin sus inserciones HTML.')
OUT=ROOT/'evidencias/galeria_validacion_20261115';OUT.mkdir(exist_ok=True)
ZIP=args.predicciones_zip
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={Path(r['archivo']).name:r for r in csv.DictReader((CROPS/'manifiesto.csv').open()) if r['archivo']}
orig={p.stem:p for p in (ORIG/'rgb').glob('*.png')};preds={};predbytes={};verified=[];source_records={}
classes=['tumor_nonMitotic','lymphocyte','nonTILnonMQ_stromal'];names=dict(zip(classes,['Tumor no mitótico','Linfocito','Estroma no TIL / no macrófago']))
with zipfile.ZipFile(ZIP) as z:
 for model in ['resnet50','densenet121','softmaxcnn_bn']:
  member=f'final/{model}_6113/predicciones_validacion.csv';raw=z.read(member);predbytes[model]=raw
  allrows=list(csv.DictReader(io.StringIO(raw.decode())));rows=[]
  for row in allrows:
   fov,idx=Path(row['archivo']).stem.rsplit('__',1)
   if fov not in orig:continue
   m=manifest[Path(row['archivo']).name];csvpath=ORIG/'csv'/(fov+'.csv')
   if fov not in source_records:
    source_records[fov]={a['']:a for a in csv.DictReader(csvpath.open())}
    assert sha(orig[fov])==m['sha_imagen'];assert sha(csvpath)==m['sha_anotaciones']
   a=source_records[fov][idx]
   assert a['main_classification']==row['clase_real']==m['clase']
   assert m['paciente']==row['paciente'] and row['archivo'].startswith('validation/')
   crop=CROPS/m['archivo'];im=Image.open(crop).convert('RGB');pixels=np.asarray(im)
   assert im.size==(128,128) and hashlib.sha256(pixels.tobytes()).hexdigest()==m['sha_pixeles']
   box=tuple(int(m[k]) for k in ['left','top','right','bottom']);reconstructed=Image.open(orig[fov]).convert('RGB').crop(box)
   assert np.array_equal(np.asarray(reconstructed),pixels)
   ps=[float(row[k]) for k in ['p_tumor','p_linfocito','p_estroma']]
   assert abs(sum(ps)-1)<1e-5 and classes[int(np.argmax(ps))]==row['clase_predicha']
   item=dict(row,modelo=model,semilla=6113,acierto=row['clase_real']==row['clase_predicha'],fila=idx,imagen=fov,sha_pixeles=m['sha_pixeles'],sha_imagen=m['sha_imagen'],sha_anotaciones=m['sha_anotaciones'],particion_historica=m['particion'],particion_evaluacion='validation',fuente_predicciones=member,sha_predicciones=hashlib.sha256(raw).hexdigest())
   verified.append(item);rows.append(item)
  preds[model]=sorted(rows,key=lambda x:x['archivo'])
selected=[]
for cl in classes:selected.append(next(x for x in preds['resnet50'] if x['clase_real']==cl and x['acierto']))
selected.append(next(x for x in preds['resnet50'] if not x['acierto']))
selected.append(next(x for x in preds['densenet121'] if not x['acierto'] and x['clase_real']=='tumor_nonMitotic'))
# Show the lymphocyte correctly classified by ResNet50, misclassified by CNN-BN, to illustrate disagreement.
id_correct=selected[1]['archivo'];selected.append(next(x for x in preds['softmaxcnn_bn'] if x['archivo']==id_correct and not x['acierto']))
for model,raw in predbytes.items():(OUT/(model+'_6113_predicciones_validacion.csv')).write_bytes(raw)
with (OUT/'Cotejo_198_predicciones.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(verified[0]));w.writeheader();w.writerows(verified)
cards=[];bookrows=[]
for i,row in enumerate(selected,1):
 folder=OUT/f'caso_{i:02d}';folder.mkdir(exist_ok=True);m=manifest[Path(row['archivo']).name];a=source_records[row['imagen']][row['fila']]
 shutil.copy2(CROPS/m['archivo'],folder/'limpia.png')
 shutil.copy2(orig[row['imagen']],folder/'campo_original.png')
 # Copy original CSV so annotation indices and coordinates are independently inspectable.
 shutil.copy2(ORIG/'csv'/(row['imagen']+'.csv'),folder/'anotaciones_originales.csv')
 (folder/'cotejo.json').write_text(json.dumps(dict(row,anotacion=a,ventana_recorte={k:m[k] for k in ['left','top','right','bottom']},nota='Cotejo automático contra referencia existente; no adjudicación humana nueva. Partición actual según predicción V2. El recorte histórico coincide píxel a píxel con el campo original y su manifiesto.'),ensure_ascii=False,indent=2))
 xs=[float(x)-int(m['left']) for x in a['coords_x'].split(',')];ys=[float(x)-int(m['top']) for x in a['coords_y'].split(',')];points=' '.join(f'{x},{y}' for x,y in zip(xs,ys))
 encoded=base64.b64encode((folder/'limpia.png').read_bytes()).decode()
 svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" role="img"><title>Anotación original: {names[row["clase_real"]]}, fila {row["fila"]}, tipo {a["type"]}</title><image href="data:image/png;base64,{encoded}" width="128" height="128"/><polygon points="{points}" fill="none" stroke="#f8e24a" stroke-width="1.5"/></svg>'
 (folder/'referencia.svg').write_text(svg)
 status='Acierto' if row['acierto'] else 'Discrepancia';probs=[float(row[k]) for k in ['p_tumor','p_linfocito','p_estroma']];confidence=max(probs)
 rel=f'../evidencias/galeria_validacion_20261115/caso_{i:02d}'
 card=f'''<article class="hist-card" data-clase="{row['clase_real']}" data-resultado="{'acierto' if row['acierto'] else 'error'}"><div class="hist-card-head"><span class="tag {'amber' if not row['acierto'] else ''}">{status}</span><strong>Caso {i:02d} · {names[row['clase_real']]}</strong></div><div class="hist-pair"><figure><img src="{rel}/limpia.png" alt="Recorte limpio del caso {i}" loading="lazy"><figcaption>Imagen limpia · 128 × 128</figcaption></figure><figure><img src="{rel}/referencia.svg" alt="Anotación de referencia del mismo núcleo" loading="lazy"><figcaption>Referencia · {a['type']}</figcaption></figure></div><p><b>Anotación:</b> {names[row['clase_real']]}<br><b>Predicción:</b> {names[row['clase_predicha']]}<br><b>Modelo:</b> {row['modelo']} · semilla 6113</p><p class="muted">Probabilidad de la clase predicha: {confidence:.1%}. No equivale a certeza diagnóstica.</p><details><summary>Identidad y cotejo</summary><p>{row['paciente']} · fila {row['fila']} · validación V2</p><p style="overflow-wrap:anywhere">{html.escape(row['archivo'])}</p><p><a href="{rel}/campo_original.png">Campo limpio completo</a> · <a href="{rel}/anotaciones_originales.csv">CSV original</a> · <a href="{rel}/cotejo.json">Trazabilidad</a></p></details></article>'''
 cards.append(card)
 bookrows.append(f'''### Caso {i:02d} · {status}\n\nReferencia: **{names[row['clase_real']]}**. Predicción: **{names[row['clase_predicha']]}**, {row['modelo']}, semilla6113. Probabilidad de la clase predicha: {confidence:.1%}.\n\n| Imagen limpia | Referencia existente |\n|---|---|\n| ![Recorte limpio](../evidencias/galeria_validacion_20261115/caso_{i:02d}/limpia.png) | ![Anotación original](../evidencias/galeria_validacion_20261115/caso_{i:02d}/referencia.svg) |\n\nPaciente {row['paciente']}; fila {row['fila']} del CSV de {row['imagen']}. [Cotejo y coordenadas](../evidencias/galeria_validacion_20261115/caso_{i:02d}/cotejo.json).\n''')
style='''<style>.hist-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}.hist-card{padding:22px;border:1px solid #dce7ed;border-radius:18px;background:white;min-width:0}.hist-card-head{display:flex;flex-wrap:wrap;gap:12px;align-items:center}.hist-pair{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:18px}.hist-pair figure{margin:0}.hist-pair img{width:100%;max-height:240px;object-fit:contain;background:#f3f5f8;border-radius:8px}.hist-pair figcaption{font-size:13px;margin-top:6px}.hist-card details{font-size:13px}.hist-controls{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}.hist-controls select{padding:10px;border:1px solid #abc;border-radius:8px;background:white}article[hidden]{display:none}@media(max-width:760px){.hist-grid{grid-template-columns:1fr}}</style>'''
section='''<section id="galeria-histologica"><div class="section-head"><h2>Del núcleo a la predicción</h2><span class="tag">Imágenes reales · validación V2</span></div><p>Las tres clases, tres aciertos y tres discrepancias cotejadas con las anotaciones originales. Los modelos reciben imágenes limpias; la caja amarilla se añade únicamente para mostrar la referencia.</p><p><strong>Muestra ilustrativa:</strong> seis comparaciones de cinco núcleos de un paciente. Se seleccionaron por disponibilidad y resultado, no aleatoriamente. No estima tasas de error ni sustituye una revisión histológica. Cajas rectangulares, no máscaras de segmentación ni Grad-CAM.</p><div class="hist-controls"><label>Clase <select id="hist-class"><option value="">Todas</option><option value="tumor_nonMitotic">Tumor no mitótico</option><option value="lymphocyte">Linfocito</option><option value="nonTILnonMQ_stromal">Estroma</option></select></label><label>Resultado <select id="hist-result"><option value="">Todos</option><option value="acierto">Aciertos</option><option value="error">Discrepancias</option></select></label></div><div class="hist-grid">'''+''.join(cards)+'''</div><p><a href="../book/_build/html/12_galeria_histologica.html">Método de cotejo, selección y límites →</a></p></section>'''
js='''<script>function filtrarHistologia(){const c=document.getElementById('hist-class').value;const r=document.getElementById('hist-result').value;document.querySelectorAll('.hist-card').forEach(e=>e.hidden=!!((c&&e.dataset.clase!==c)||(r&&e.dataset.resultado!==r)));}document.getElementById('hist-class').addEventListener('change',filtrarHistologia);document.getElementById('hist-result').addEventListener('change',filtrarHistologia);</script>'''
p=ROOT/'dashboard/index.html';s=p.read_text().replace('</head>',style+'</head>').replace('<main>','<main>'+section).replace('</body>',js+'</body>').replace('<a href="#resultados">','<a href="#galeria-histologica">Imágenes</a><a href="#resultados">');p.write_text(s)
standalone='<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NuCLS E2 · Galería histológica</title><link rel="stylesheet" href="style.css">'+style+'</head><body><main><a href="index.html">← Dashboard completo</a>'+section+'</main>'+js+'</body></html>'
(ROOT/'dashboard/galeria.html').write_text(standalone)
chapter='''# Imágenes limpias, anotaciones y predicciones cotejadas

La entrada del modelo es el recorte RGB limpio. Las referencias de NuCLS se usan como etiquetas y, en esta galería, como superposición ilustrativa. No se reanotaron núcleos ni se entrenó ningún modelo.

Se cotejaron **198 predicciones (66 núcleos × tres modelos, semilla6113)** contra los CSV originales disponibles localmente. La fila del CSV se vincula mediante el sufijo `__fila` del recorte; se verifican paciente, clase, coordenadas, hashes de imagen/anotaciones, probabilidad y argmax. Los recortes de128×128 coinciden píxel a píxel con la ventana del campo limpio y con el hash de su manifiesto histórico.

La galería incluye **seis comparaciones de cinco núcleos**, tres aciertos y tres discrepancias, de **TCGA-BH-A0BW**. Son ejemplos seleccionados de una muestra de conveniencia; no representan toda la validación ni permiten estimar nuevas tasas. La misma imagen aparece con dos modelos para ilustrar desacuerdo. Se fijó la semilla6113, disponible para las tres bases; no se eligió por mejor desempeño.

**Particiones:** estos recortes figuran como entrenamiento en el manifiesto histórico del piloto y como validación en las predicciones V2. La galería se identifica por la partición V2; no se mezclan métricas del piloto con V2 ni se concluye fuga en el entrenamiento V2 a partir de un nombre histórico. Esta comparación local no reaudita el ZIP de imágenes exacto cargado en Colab: valida la identidad y los píxeles de las fuentes históricas recuperadas.

Las anotaciones elegidas son rectángulos. Se dibujan sus coordenadas originales trasladadas a la ventana del recorte, no contornos segmentados. «Discrepancia» significa predicción distinta de la etiqueta existente, sin adjudicar la causa ni corregir la referencia automáticamente. La probabilidad del modelo no equivale a certeza clínica. No se visualizaron imágenes de reserva para esta galería.

[Tabla de cotejo de198 predicciones](../evidencias/galeria_validacion_20261115/Cotejo_198_predicciones.csv). [Resumen verificable](../evidencias/galeria_validacion_20261115/Verificacion.json).

'''+ '\n'.join(bookrows)
(ROOT/'book/12_galeria_histologica.md').write_text(chapter)
p=ROOT/'book/_toc.yml';p.write_text(p.read_text()+'  - file: 12_galeria_histologica\n')
p=ROOT/'book/02_eda.md';p.write_text(p.read_text()+'\n## Anotaciones disponibles y cotejo visual\n\nSe recuperaron imágenes limpias y CSV de anotaciones originales para66 núcleos de validación V2; no requieren volver a etiquetarse. El capítulo [Galería histológica](12_galeria_histologica.md) presenta tres clases, aciertos y discrepancias. Esta verificación automática no sustituye la adjudicación de causas ambiguas, la revisión de100 errores por modelo ni el acuerdo multi-rater.\n')
p=ROOT/'ABRIR_ENTREGA.html';s=p.read_text().replace('<main>','<main><section><h2>Galería histológica cotejada</h2><p>Imágenes limpias, anotaciones originales y predicciones de tres bases. Tres clases y seis comparaciones ilustrativas de validación.</p><a class="button" href="dashboard/galeria.html">Ver imágenes, aciertos y discrepancias →</a></section>');p.write_text(s)
report=dict(nucleos_cotejados=66,predicciones_cotejadas=198,modelos=list(preds),semilla=6113,casos_galeria=6,nucleos_galeria=len({x['archivo'] for x in selected}),pacientes_galeria=sorted({x['paciente'] for x in selected}),clases_galeria=sorted({x['clase_real'] for x in selected}),aciertos=3,discrepancias=3,fuente_zip=str(ZIP),sha_zip=sha(ZIP),hashes_origen_y_csv_verificados=True,recortes_pixeles_verificados=True,revision_humana_nueva=False,Colab_intervenido=False,E1_modificado=False,limitacion='Muestra de conveniencia; fuentes históricas cotejadas con identificadores de predicción V2, sin reauditar ZIP de datos actual en Colab.')
(OUT/'Verificacion.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
