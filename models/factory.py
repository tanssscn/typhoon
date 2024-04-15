# from .registry import is_model, model_entrypoint
import os
import torch
from timm import is_model, model_entrypoint


def split_model_name(model_name):
    model_split = model_name.split(':', 1)
    if len(model_split) == 1:
        return '', model_split[0]
    else:
        source_name, model_name = model_split
        assert source_name in ('timm', 'hf_hub')
        return source_name, model_name


def safe_model_name(model_name, remove_source=True):
    def make_safe(name):
        return ''.join(c if c.isalnum() else '_' for c in name).rstrip('_')

    if remove_source:
        model_name = split_model_name(model_name)[-1]
    return make_safe(model_name)


def create_model_bak(model_name, device):
    source_name, model_name = split_model_name(model_name)
    if is_model(model_name):
        create_fn = model_entrypoint(model_name)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

    model = create_fn().to(device)
    return model


def create_model(model_name, device, weights_path, status):
    # split_model_name()：将model_name分割为来源名称和模型名称
    source_name, model_name = split_model_name(model_name)
    print('split_model_name', source_name, model_name)
    if is_model(model_name):
        # model_entrypoint：获取模型的入口函数
        create_fn = model_entrypoint(model_name)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

    model = create_fn().to(device)
    if status == "train" and weights_path:
        load_weights(model, weights_path, device)
    return model


def load_swin_dict(model, checkpoint_path, device, strict=False):
    if not os.path.exists(checkpoint_path):
        raise RuntimeError('checkpoint_path does not exist')
    weights_dict = torch.load(checkpoint_path, map_location=device)["model"]
    # 删除有关分类类别的权重
    for k in list(weights_dict.keys()):
        if "head" in k:
            del weights_dict[k]
        if "norm.weight" in k:
            del weights_dict[k]
        if "norm.bias" in k:
            del weights_dict[k]
    print(model.load_state_dict(weights_dict, strict=strict))


def load_transxnet_dict(model, checkpoint_path, device, strict=False):
    if not os.path.exists(checkpoint_path):
        raise RuntimeError('checkpoint_path does not exist')
    weights_dict = torch.load(checkpoint_path, map_location=device)
    print(weights_dict.keys())

    # 删除有关分类类别的权重
    deldic = ['classifier.0.weight', 'classifier.0.bias', 'classifier.2.weight', 'classifier.2.bias']
    for j in deldic:
        del weights_dict[j]
    print(weights_dict.keys())
    print('加载预训练权重成功', model.load_state_dict(weights_dict, strict=strict))

def load_weights(model, checkpoint_path, device, strict=False):
    if not os.path.exists(checkpoint_path):
        raise RuntimeError('checkpoint_path does not exist')

    weights_dict = torch.load(checkpoint_path, map_location=device)

    print('加载预训练权重成功:', model.load_state_dict(weights_dict, strict=False))


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}
