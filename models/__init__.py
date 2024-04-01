from .detr import build

def build_model(args, ec_weight):
    return build(args, ec_weight)
