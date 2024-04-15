import os

# 项目根路径
root = os.path.dirname(os.path.abspath(__file__)) + '/'
# 模型名
swinlstm = 'swinlstm'
Predictor='Predictor'
model = Predictor
targets_len = 4
inputs_len = 4
# 数据集路径
data_path = '/home/ubuntu18/tan/storm_request/data/img_224'
# 预训练权重
# pre_weights = root + 'weights/resnext50.pth'
# pre_weights = root+'weights/swin_tiny_patch4_window7_224.pth'
# pre_weights = root+'weights/biformer_small_best.pth'm
pre_weights = root + 'runs/train/exp102/model-best.pth'
# pre_weights = ''
# pre_weights = r2ot + 'runs/weight/mynet/model-best.pth'

# 权重
# weights = root + 'runs/weight/locnet/model-best.pth'
# weights = root + 'runs/weight/swinlstm/model-best.pth'
# weights = root + 'runs/weight/resnext50/model-best.pth'
# weights = root + 'runs/weight/convnext/model-best.pth'
# weights = root + 'runs/weight/convnextv2/model-best.pth'
weights = root + 'runs/train/exp15/model-best.pth'
# weights = root + 'runs/weight/mynet/model-best.pth'
# weights = root + 'runs/weight/biformer_small/model-best.pth'
# weights = root + 'runs/weight/rmt/model-best.pth'
project = root + 'runs/train'
img_size = 224
