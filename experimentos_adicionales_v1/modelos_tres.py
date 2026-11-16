"""Complemento bibliográfico E2: tres modelos, sin pesos externos, entrada128."""
import timm
MODELS = {
 'convnext_tiny': ('CNN','2201.03545'),
 'inception_v3': ('CNN','1512.00567'),
 'sequencer2d_s': ('Recurrente','2205.01972'),
}
def build(name,n_classes=3):
 if name not in MODELS:raise ValueError(name)
 timm.layers.set_fused_attn(False)
 return timm.create_model(name,pretrained=False,num_classes=n_classes)
