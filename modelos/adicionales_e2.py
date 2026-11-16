"""Diez candidatos con variantes definidas. Sin pesos externos, entrada nativa 128."""
import timm
from torch import nn
MODELS = {
 'resnet18': ('CNN residual','1512.03385'),
 'resnext50_32x4d': ('CNN residual','1611.05431'),
 'mobilenetv3_small_100': ('CNN eficiente','1905.02244'),
 'efficientnet_b0': ('CNN eficiente','1905.11946'),
 'deit_tiny_patch16_224': ('Atencion','2012.12877'),
 'swin_tiny_patch4_window7_224': ('Atencion','2103.14030'),
 'mobilevit_xxs': ('Hibrida CNN-atencion','2110.02178'),
 'mobilevit_xs': ('Hibrida CNN-atencion','2110.02178'),
 'mixer_s16_224': ('MLP de vision','2105.01601'),
 'resmlp_12_224': ('MLP de vision','2105.03404'),
}
def build(name,n_classes=3):
 if name not in MODELS:raise ValueError(name)
 kw={'img_size':128} if name.startswith(('deit','swin','mixer','resmlp')) else {}
 # Desactivar kernels de atención fusionada para determinismo estricto en CUDA.
 timm.layers.set_fused_attn(False)
 return timm.create_model(name,pretrained=False,num_classes=n_classes,**kw)
