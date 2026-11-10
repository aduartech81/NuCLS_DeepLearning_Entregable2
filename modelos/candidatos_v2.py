"""Candidatos trazables. No inicia entrenamiento ni modifica los modelos del piloto.
La CNN conserva la arquitectura softmaxCNN_IN27 de Basha et al. (2018)
y añade pooling adaptativo sin parámetros para aceptar ventanas nativas 128x128.
"""
from torch import nn
from torchvision import models
CLASSES=['tumor_nonMitotic','lymphocyte','nonTILnonMQ_stromal']

class NativeGridPool(nn.Module):
    """30x30 -> 5x5: bloques 6x6 disjuntos; 5x5 -> identidad."""
    def forward(self, x):
        if x.shape[-2:] == (30, 30):
            return nn.functional.avg_pool2d(x, kernel_size=6, stride=6)
        if x.shape[-2:] == (5, 5):
            return x
        raise ValueError('Se esperan mapas 30x30 (entrada 128) o 5x5 (entrada 27).')

class SoftmaxCNN(nn.Module):
    def __init__(self,n_classes=3,adapt_native=True):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3,36,4),nn.ReLU(),nn.BatchNorm2d(36),nn.MaxPool2d(2),nn.Conv2d(36,48,3),nn.ReLU(),nn.BatchNorm2d(48),nn.MaxPool2d(2))
        self.adapter=NativeGridPool() if adapt_native else nn.Identity()
        self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(1200,512),nn.ReLU(),nn.BatchNorm1d(512),nn.Dropout(.5),nn.Linear(512,512),nn.ReLU(),nn.BatchNorm1d(512),nn.Dropout(.5),nn.Linear(512,n_classes))
    def forward(self,x):return self.classifier(self.adapter(self.features(x)))

def build(name,n_classes=3):
    if name=='softmaxcnn_bn':return SoftmaxCNN(n_classes)
    if name=='densenet121':
        m=models.densenet121(weights=None);m.classifier=nn.Linear(m.classifier.in_features,n_classes);return m
    if name=='resnet50':
        m=models.resnet50(weights=None);m.fc=nn.Linear(m.fc.in_features,n_classes);return m
    if name=='resnet18':
        m=models.resnet18(weights=None);m.fc=nn.Linear(m.fc.in_features,n_classes);return m
    raise ValueError(name)
