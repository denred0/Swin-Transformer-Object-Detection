import timm
from pprint import pprint

model_names = timm.list_models(pretrained=True)
# model_names = timm.list_models('*cnn*')

pprint(model_names)
#
# model = 'swin_base_patch4_window7_224_in22k'
# m = timm.create_model(model, pretrained=True)
# print(model)
# pprint(m.default_cfg)
