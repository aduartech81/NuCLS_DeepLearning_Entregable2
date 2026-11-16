"""Motor de fases de desarrollo v2. Solo recibe entrenamiento y validación."""
import argparse, csv, hashlib, json, math, os, platform, random, sys, time, zipfile
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torchvision
import PIL
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from candidatos_v2 import build, CLASSES

EXPECTED = '623adf36520751addc324e44cd1b89d95e47842e53cda494914c0248a8be86d4'


def save_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)

def csv_write(path, rows):
    if not rows: return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def scores(y, p):
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (np.asarray(y), np.asarray(p)), 1)
    tp = np.diag(cm); support = cm.sum(1); predicted = cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(3), where=precision+recall > 0)
    return dict(macro_f1=float(f1.mean()), accuracy=float(tp.sum()/max(cm.sum(), 1)),
                balanced_accuracy=float(recall.mean()), confusion=cm.tolist(),
                clases=[dict(clase=c, precision=float(precision[i]), recall=float(recall[i]),
                             f1=float(f1[i]), soporte=int(support[i])) for i,c in enumerate(CLASSES)])

def read_data(root):
    raw = (root/'manifiesto.csv').read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError('El manifiesto no coincide con el dataset auditado; no se inicia entrenamiento.')
    rows = list(csv.DictReader(raw.decode().splitlines()))
    groups = {s:[r for r in rows if r['particion']==s and r['estado']=='incluida']
              for s in ('train','validation')}
    if [len(groups[s]) for s in groups] != [14539,2049]:
        raise ValueError('Conteos distintos de los auditados.')
    for key in ('paciente', 'tss', 'sha_pixeles'):
        if {r[key] for r in groups['train']} & {r[key] for r in groups['validation']}:
            raise ValueError('Solapamiento entre particiones: '+key)
    if [len({r['paciente'] for r in groups[s]}) for s in groups] != [83,21]:
        raise ValueError('Conteos de pacientes distintos.')
    for s, rr in groups.items():
        for i,r in enumerate(rr):
            if r['clase'] not in CLASSES: raise ValueError('Clase inesperada')
            p = (root/r['archivo']).resolve()
            if not p.is_relative_to(root.resolve()): raise ValueError('Ruta fuera del dataset')
            with Image.open(p) as im:
                if im.mode!='RGB' or im.size!=(128,128): raise ValueError('Formato incorrecto: '+str(p))
                digest = hashlib.sha256(np.asarray(im).tobytes()).hexdigest()
            if digest != r['sha_pixeles']: raise ValueError('Imagen modificada: '+str(p))
            if i%2000==0: print('Verificando',s,i,'/',len(rr),flush=True)
    return groups

class Crops(Dataset):
    def __init__(self, root, rows, augment=False): self.root,self.rows,self.augment=root,rows,augment
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r=self.rows[i]
        with Image.open(self.root/r['archivo']) as im: x=np.asarray(im).copy()
        if self.augment:
            x=np.rot90(x, random.randrange(4))
            if random.random()<.5: x=x[:,::-1]
        t=torch.from_numpy(x.copy()).permute(2,0,1).float()/255
        mean=torch.tensor([0.6392034726251884,0.45917053999289786,0.6277451963521469])[:,None,None]
        sd=torch.tensor([0.2139743515891908,0.22149669223009963,0.1742893379690752])[:,None,None]
        return (t-mean)/sd, CLASSES.index(r['clase'])

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def synchronize(device):
    if device.type=='cuda': torch.cuda.synchronize()
    elif device.type=='mps': torch.mps.synchronize()

def evaluate(model, loader, device):
    model.eval(); probs=[]; ys=[]; loss=0.; n=0
    with torch.no_grad():
        for x,y in loader:
            logits=model(x.to(device)); ll=nn.functional.cross_entropy(logits,y.to(device))
            if not torch.isfinite(ll): raise ValueError('Pérdida de validación no finita')
            probs.extend(logits.softmax(1).cpu().tolist());ys.extend(y.tolist())
            loss+=float(ll)*len(y);n+=len(y)
    a=np.asarray(probs)
    if not np.isfinite(a).all() or not np.allclose(a.sum(1),1,atol=1e-5): raise ValueError('Probabilidades inválidas')
    return loss/n, np.asarray(ys), a
