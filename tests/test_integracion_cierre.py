"""Pruebas con datos sintéticos; no son resultados del experimento."""
import csv,hashlib,io,json,shutil,subprocess,sys,tempfile,unittest,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def fixture(path,corrupt=False):
    models=['resnet50','densenet121','softmaxcnn_bn']+['modelo_'+str(i) for i in range(10)]
    plan={'modelos':models,'adicionales':models[3:],'semillas':list(range(6101,6106)),
          'corridas':[{'modelo':m,'semilla':s} for m in models for s in range(6101,6106)]}
    raw=json.dumps(plan).encode();pid=hashlib.sha256(raw).hexdigest()
    files={'entradas_fijadas.json':raw,'estado.json':json.dumps({'estado':'CIERRE_RESERVA_Y_ESTADISTICA_COMPLETADO','corridas':65}).encode(),
           'congelamiento_previo_reserva.json':json.dumps({'plan_sha256':pid}).encode()}
    for row in plan['corridas']:
        prefix=f"corridas/{row['modelo']}_{row['semilla']}/";marks={}
        for n in ['reserva.npz','train_limpio.npz','predicciones_reserva.csv','brecha.json','costo.json']:
            files[prefix+n]=b'SINTETICO';marks[n]=hashlib.sha256(b'SINTETICO').hexdigest()
        files[prefix+'completo.json']=json.dumps({'plan':pid,'archivos':marks}).encode()
    def csvbytes(columns,rows):
        f=io.StringIO();w=csv.writer(f);w.writerow(columns);w.writerows(rows);return f.getvalue().encode()
    files['resumen_reserva.csv']=csvbytes(['modelo','R','macro_f1_media','macro_f1_sd','IC95_paciente_inf','IC95_paciente_sup'],[[m,5,.5,.1,.3,.7] for m in models])
    files['contrastes_vs_resnet50.csv']=csvbytes(['modelo','delta','p_t_Holm10','p_wilcoxon_Holm10','p_permutacion_paciente_Holm10'],[[m,0,1,1,1] for m in models[3:]])
    for name in ['informe.html','benchmark_reserva.png','diferencia_critica.png']:files[name]=b'SINTETICO'
    if corrupt:files['corridas/resnet50_6101/reserva.npz']=b'ALTERADO'
    with zipfile.ZipFile(path,'w') as z:
        for n,b in files.items():z.writestr(n,b)
class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name);self.root=self.p/'e2'
        for f in ['scripts/integrar_cierre_reserva.py','scripts/preparar_entrega.py','book/09_cierre_reserva.md','dashboard/index.html','ABRIR_ENTREGA.html']:
            out=self.root/f;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,out)
        (self.root/'evidencias').mkdir(exist_ok=True)
    def tearDown(self):self.tmp.cleanup()
    def test_import_is_idempotent_and_tables_present(self):
        z=self.p/'valid.zip';fixture(z)
        cmd=[sys.executable,str(self.root/'scripts/integrar_cierre_reserva.py'),str(z)]
        subprocess.run(cmd,check=True,capture_output=True)
        before=(self.root/'book/09_cierre_reserva.md').read_text()
        subprocess.run(cmd,check=True,capture_output=True)
        self.assertEqual(before,(self.root/'book/09_cierre_reserva.md').read_text())
        self.assertIn('<td>modelo_9</td>',(self.root/'dashboard/index.html').read_text())
        self.assertEqual(before.count('<!-- CIERRE_IMPORTADO -->'),1)
    def test_corrupt_evidence_rejected_before_copy(self):
        z=self.p/'bad.zip';fixture(z,True);dest=self.p/'nuevo'
        run=subprocess.run([sys.executable,str(self.root/'scripts/preparar_entrega.py'),str(z),'--destino',str(dest)],capture_output=True)
        self.assertNotEqual(run.returncode,0);self.assertFalse(dest.exists())
    def test_existing_destination_preserved(self):
        dest=self.p/'existente';dest.mkdir();(dest/'conservar').write_text('original')
        run=subprocess.run([sys.executable,str(self.root/'scripts/preparar_entrega.py'),'no.zip','--destino',str(dest)],capture_output=True)
        self.assertNotEqual(run.returncode,0);self.assertEqual((dest/'conservar').read_text(),'original')
if __name__=='__main__':unittest.main()
