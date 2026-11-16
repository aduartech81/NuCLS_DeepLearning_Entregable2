from pathlib import Path
import csv,io,json,zipfile,hashlib,shutil,collections,html,re,base64
from PIL import Image
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--paquete-anterior',type=Path,required=True);ap.add_argument('--salida',type=Path,required=True);ap.add_argument('--qc',type=Path,required=True);ap.add_argument('--predicciones',type=Path,required=True);args=ap.parse_args()
ROOT=args.salida
assert not ROOT.exists(), 'Usar destino nuevo'
shutil.copytree(args.paquete_anterior,ROOT)
OUT=ROOT/'evidencias/galeria_ampliada_20261115';OUT.mkdir(exist_ok=True)
QC=args.qc;PRED=args.predicciones
sha=lambda b:hashlib.sha256(b).hexdigest()
classes=['tumor_nonMitotic','lymphocyte','nonTILnonMQ_stromal'];labels=dict(zip(classes,['Tumor no mitótico','Linfocito','Estroma no TIL / no macrófago']))
models=['resnet50','densenet121','softmaxcnn_bn'];cache={};verified=[];predraw={}
q=zipfile.ZipFile(QC)
with zipfile.ZipFile(PRED) as z:
 for model in models:
  member=f'final/{model}_6113/predicciones_validacion.csv';raw=z.read(member);predraw[model]=raw
  for row in csv.DictReader(io.StringIO(raw.decode())):
   assert row['archivo'].startswith('validation/');fov,idx=Path(row['archivo']).stem.rsplit('__',1)
   if fov not in cache:
    annbytes=q.read('QC/csv/'+fov+'.csv');cache[fov]=(annbytes,{a['']:a for a in csv.DictReader(io.StringIO(annbytes.decode()))})
   annbytes,anns=cache[fov];ann=anns[idx]
   assert ann['main_classification']==row['clase_real'];assert fov.startswith(row['paciente']+'-')
   probs=[float(row[k]) for k in ['p_tumor','p_linfocito','p_estroma']];assert abs(sum(probs)-1)<1e-5 and classes[max(range(3),key=lambda i:probs[i])]==row['clase_predicha']
   verified.append(dict(row,modelo=model,semilla=6113,acierto=row['clase_real']==row['clase_predicha'],imagen=fov,fila=idx,tipo=ann['type'],sha_anotaciones=sha(annbytes),sha_predicciones=sha(raw),fuente_predicciones=member))
# Fixed strata, selection balances patients and forbids repeat nuclei across the gallery.
counts=collections.Counter();used=set();selected=[]
for outcome in [False,True]:
 for model in models:
  for cl in classes:
   group=[x for x in verified if x['modelo']==model and x['clase_real']==cl and x['acierto']==outcome]
   for _ in range(2):
    candidates=[x for x in group if x['archivo'] not in used];assert candidates,(model,cl,outcome)
    item=min(candidates,key=lambda x:(counts[x['paciente']],x['paciente'],x['archivo']))
    selected.append(item);counts[item['paciente']]+=1;used.add(item['archivo'])
selected.sort(key=lambda x:(classes.index(x['clase_real']),not x['acierto'],models.index(x['modelo']),x['archivo']))
print('Selección',len(selected),'pacientes',dict(counts),flush=True)
with (OUT/'Cotejo_predicciones.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(verified[0]));w.writeheader();w.writerows(verified)
for model,raw in predraw.items():(OUT/f'{model}_6113_predicciones_validacion.csv').write_bytes(raw)
cards=[];book=[];summary=[]
for i,row in enumerate(selected,1):
 folder=OUT/f'caso_{i:02d}';folder.mkdir(exist_ok=True);fov=row['imagen'];annbytes,anns=cache[fov];a=anns[row['fila']]
 rgb=q.read('QC/rgb/'+fov+'.png');(folder/'campo_limpio.png').write_bytes(rgb);(folder/'anotaciones_originales.csv').write_bytes(annbytes)
 visual=q.read('QC/visualization/'+fov+'.png');(folder/'visualizacion_original.png').write_bytes(visual)
 w,h=Image.open(io.BytesIO(rgb)).size
 xs=[float(x) for x in a['coords_x'].split(',')];ys=[float(y) for y in a['coords_y'].split(',')];assert len(xs)==len(ys)>2
 assert min(xs)>=0 and max(xs)<=w and min(ys)>=0 and max(ys)<=h
 points=' '.join(f'{x},{y}' for x,y in zip(xs,ys));data=base64.b64encode(rgb).decode()
 svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><title>Núcleo objetivo: fila {row["fila"]}, {labels[row["clase_real"]]}, anotación {a["type"]}</title><image href="data:image/png;base64,{data}" width="{w}" height="{h}"/><polygon points="{points}" fill="none" stroke="#fce24a" stroke-width="3"/></svg>'
 (folder/'referencia.svg').write_text(svg)
 record=dict(row,anotacion=a,sha_campo=sha(rgb),sha_visualizacion=sha(visual),dimensiones_campo=[w,h],vista='Campo original con núcleo objetivo señalado; no se afirma que el campo completo sea la entrada de128px del modelo',revision_humana_nueva=False)
 (folder/'cotejo.json').write_text(json.dumps(record,ensure_ascii=False,indent=2));summary.append(record)
 state='Acierto' if row['acierto'] else 'Discrepancia';result='acierto' if row['acierto'] else 'error';prob=max(float(row[k]) for k in ['p_tumor','p_linfocito','p_estroma']);rel=f'../evidencias/galeria_ampliada_20261115/caso_{i:02d}'
 card=f'''<article class="hist-card" data-clase="{row['clase_real']}" data-resultado="{result}" data-paciente="{row['paciente']}" data-modelo="{row['modelo']}"><div class="hist-card-head"><span class="tag {'amber' if not row['acierto'] else ''}">{state}</span><strong>{i:02d} · {labels[row['clase_real']]}</strong></div><p>{row['paciente']} · {row['modelo']} · semilla6113</p><div class="hist-pair"><figure><a href="{rel}/campo_limpio.png"><img src="{rel}/campo_limpio.png" alt="Campo RGB limpio, caso{i}" loading="lazy"></a><figcaption>Campo original limpio</figcaption></figure><figure><a href="{rel}/referencia.svg"><img src="{rel}/referencia.svg" alt="Núcleo objetivo según anotación original" loading="lazy"></a><figcaption>Objetivo amarillo · fila{row['fila']}</figcaption></figure></div><p><b>Referencia:</b> {labels[row['clase_real']]}<br><b>Predicción:</b> {labels[row['clase_predicha']]}<br><b>Probabilidad predicha:</b> {prob:.1%}</p><details><summary>Anotación original y trazabilidad</summary><p>Tipo: {a['type']}. La referencia no es Grad-CAM. El campo aporta contexto: la predicción corresponde solo al núcleo señalado.</p><p><a href="{rel}/visualizacion_original.png">Visualización oficial QC (todas las anotaciones)</a> · <a href="{rel}/anotaciones_originales.csv">CSV original</a> · <a href="{rel}/cotejo.json">Cotejo y huellas</a></p><p style="overflow-wrap:anywhere">{html.escape(row['archivo'])}</p></details></article>''';cards.append(card)
 book.append(f"| {i:02d} | {row['paciente']} | {row['modelo']} | {labels[row['clase_real']]} | {labels[row['clase_predicha']]} | {state} | [Ver](../evidencias/galeria_ampliada_20261115/caso_{i:02d}/referencia.svg) |")
patients=sorted(counts)
options=lambda vals:''.join(f'<option value="{x}">{x}</option>' for x in vals)
section=f'''<section id="galeria-histologica"><div class="section-head"><h2>Del núcleo a la predicción · galería ampliada</h2><span class="tag">36 casos · {len(patients)} pacientes · validación V2</span></div><p>18 aciertos y18 discrepancias: dos ejemplos por cada combinación de modelo, clase y resultado. Todos son núcleos distintos. Se cotejaron las6.147 predicciones de las tres bases sobre2.049 núcleos de21 pacientes con sus anotaciones QC.</p><p><strong>Cómo leer las imágenes:</strong> izquierda, campo RGB limpio original; derecha, el mismo campo con la anotación del núcleo objetivo resaltada en amarillo. La predicción corresponde a ese núcleo, no a todos los presentes. Se muestra contexto histológico, no se afirma que el campo completo sea la entrada de128px del modelo. La visualización oficial QC y el CSV también están disponibles en cada caso.</p><p>Muestra ilustrativa seleccionada por clase y resultado, equilibrando disponibilidad de pacientes. No estima prevalencia ni tasa de error. Las referencias son cajas o polígonos existentes; no son Grad-CAM ni nuevas adjudicaciones humanas.</p><div class="hist-controls"><label>Clase <select id="hist-class"><option value="">Todas</option>{options(classes)}</select></label><label>Resultado <select id="hist-result"><option value="">Todos</option><option value="acierto">Aciertos</option><option value="error">Discrepancias</option></select></label><label>Paciente <select id="hist-patient"><option value="">Todos</option>{options(patients)}</select></label><label>Modelo <select id="hist-model"><option value="">Todos</option>{options(models)}</select></label></div><p id="hist-count" aria-live="polite">36 de36 casos</p><div class="hist-grid">'''+''.join(cards)+'''</div><p><a href="../book/_build/html/12_galeria_histologica.html">Cotejo completo, selección y límites →</a></p></section>'''
js='''<script>function filtrarHistologia(){const pairs=[['hist-class','clase'],['hist-result','resultado'],['hist-patient','paciente'],['hist-model','modelo']];let n=0;document.querySelectorAll('.hist-card').forEach(e=>{e.hidden=pairs.some(([id,key])=>{const v=document.getElementById(id).value;return v&&e.dataset[key]!==v;});if(!e.hidden)n++;});document.getElementById('hist-count').textContent=n+' de36 casos';}['hist-class','hist-result','hist-patient','hist-model'].forEach(id=>document.getElementById(id).addEventListener('change',filtrarHistologia));</script>'''
for name in ['index.html','galeria.html']:
 p=ROOT/'dashboard'/name;s=p.read_text();s,n=re.subn(r'<section id="galeria-histologica">.*?</section>',lambda m:section,s,flags=re.S);assert n==1
 s,n=re.subn(r'<script>function filtrarHistologia\(\).*?</script>',lambda m:js,s,flags=re.S);assert n==1;p.write_text(s)
report=dict(predicciones_cotejadas=len(verified),nucleos_cotejados=len({x['archivo'] for x in verified}),pacientes_cotejados=len({x['paciente'] for x in verified}),casos=36,nucleos_distintos=36,pacientes_galeria=patients,soporte_pacientes=dict(counts),aciertos=18,discrepancias=18,semilla=6113,seleccion='Dos por modelo/clase/resultado; prioriza menor frecuencia acumulada de paciente y desempata por paciente/archivo. Sin repetir núcleos.',fuente_qc=str(QC),fuente_predicciones=str(PRED),sha_predicciones_zip=sha(PRED.read_bytes()),reserva_visualizada=False,E1_modificado=False,Colab_intervenido=False,revision_humana_nueva=False)
(OUT/'Verificacion.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(OUT/'Casos.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
chapter=f'''# Galería ampliada: imágenes limpias, anotaciones y predicciones

**36 núcleos diferentes de {len(patients)} pacientes**:18 aciertos y18 discrepancias. Dos casos por combinación de tres modelos, tres clases y dos resultados, con semilla6113 fija. Se priorizó diversidad de pacientes; el desempate es determinista por paciente y archivo. No se escogieron semillas por desempeño.

## Fuente y cotejo

Se recuperó el archivo original QC con campos RGB, visualizaciones oficiales y CSV. Se verificaron **6.147 predicciones sobre2.049 núcleos de21 pacientes de validación V2**: identificador de campo, fila de anotación, paciente, clase de referencia, suma de probabilidades y argmax. Se guardan los CSV de predicciones y las anotaciones exactas de cada imagen, con huellas. [Resumen verificable](../evidencias/galeria_ampliada_20261115/Verificacion.json) y [cotejo completo](../evidencias/galeria_ampliada_20261115/Cotejo_predicciones.csv).

Se muestra el **campo limpio original**, junto al mismo campo con la anotación del núcleo objetivo resaltada. La visualización oficial QC de todas las anotaciones está enlazada por caso. El modelo clasifica un recorte del núcleo identificado por el sufijo de fila; la imagen de contexto no se presenta como su entrada exacta de128×128. No se reabrió Colab ni se verificaron allí los bytes del ZIP de entrenamiento. La ampliación no depende del manifiesto histórico de recortes ni cambia ninguna partición.

## Alcance

Las imágenes y referencias son existentes, no generadas. El resaltado vectorial usa las coordenadas originales, sin inventar contornos. Una caja rectangular no equivale a segmentación; ninguna superposición es Grad-CAM. «Discrepancia» significa diferencia frente a la etiqueta registrada, no causa de error adjudicada. La selección estratificada por resultado no estima tasas, prevalencias ni rendimiento clínico. La revisión de100 errores por modelo y el acuerdo multi-rater siguen siendo tareas distintas. No se visualizaron imágenes de reserva.

## Distribución de los ejemplos

| Paciente | Casos |
|---|---:|
'''+''.join(f'| {p} | {counts[p]} |\n' for p in patients)+'''\n## Casos cotejados

| Caso | Paciente | Modelo | Referencia | Predicción | Resultado | Imagen anotada |
|---|---|---|---|---|---|---|
'''+ '\n'.join(book)+'\n\nLa galería interactiva del dashboard permite filtrar por paciente, clase, resultado y modelo. Las tablas de rendimiento completo permanecen separadas de esta selección ilustrativa.\n'
(ROOT/'book/12_galeria_histologica.md').write_text(chapter)
for path in ['book/02_eda.md','book/11_estado_entrega.md']:
 p=ROOT/path;s=p.read_text();s+=f'\n### Ampliación visual verificada\n\nEl archivo QC permitió ampliar el cotejo a2.049 núcleos de21 pacientes y6.147 predicciones. La galería actual reúne36 núcleos de{len(patients)} pacientes, con18 aciertos y18 discrepancias; sustituye la selección inicial de seis comparaciones de un paciente. Se muestran campos originales y objetivos anotados, no una reconstrucción del recorte usado en Colab.\n';p.write_text(s)
p=ROOT/'ABRIR_ENTREGA.html';s=p.read_text().replace('Tres clases y seis comparaciones ilustrativas de validación.',f'Tres clases y36 núcleos de{len(patients)} pacientes:18 aciertos y18 discrepancias cotejadas en validación.');p.write_text(s)
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
