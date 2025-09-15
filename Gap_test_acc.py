
from nntool.api import NNGraph
from gap_utils.gap_utils import gap_loader,gap_test
from pytorch_utils.dataset import cifar_set
import numpy as np
import pickle
import os

load_name = 'mob_t2.onnx'
result_name = 'mob_t'
LOAD = False

train_set, val_set, num_classes = cifar_set('cifar100')

train_loader = gap_loader(train_set,transpose_to_hwc=False,with_label=False)
test_loader = gap_loader(val_set,transpose_to_hwc=False,with_label=True)

model = NNGraph.load_graph(
            load_name,
            load_quantization=False,
        )

model.adjust_order()
model.fusions('expression_matcher')
model.fusions('scaled_match_group')
# model.fusions('expression_matcher')

# 开始量化图
if LOAD:
    with open(result_name+'_statis','rb') as f:
        statistics = pickle.load(f)
else:
    statistics = model.collect_statistics(train_loader)
    with open(result_name+'_statis','wb') as f:
        pickle.dump(statistics,f)

model.quantize(
    statistics,
    schemes=['scaled'], # 使用的方案
    graph_options={
        # "use_ne16": False,
        "hwc": False,
        "weight_bits": 8,
    },
    )

model.adjust_order()
model.fusions('expression_matcher')
model.fusions('scaled_match_group')

acc = gap_test(model,test_loader,dequantize=True)
print(acc)