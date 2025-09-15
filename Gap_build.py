
from nntool.api import NNGraph
from nntool.api.utils import model_settings
from gap_utils.gap_utils import gap_loader,transform_32,transform_224
from pytorch_utils.dataset import cifar_set
import numpy as np
import pickle
import os

transform=transform_224

load_name = 'mob.onnx'
result_name = 'mob'
LOAD = False

train_set, val_set, num_classes = cifar_set('cifar10_debug',train_transform=transform['train'],val_transform=transform['val'])

train_loader = gap_loader(val_set,transpose_to_hwc=False,with_label=False)
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

# model.draw(expressions='quantized',fusions=True,quant_labels=True,filepath=result_name+'_quantized')
log = str(model.show())
print(log)

input_data = []

for idx,(test_image,_) in enumerate(test_loader):
    qout = model.execute([test_image], quantize=True)
    input_data.append(qout[0][0])
    if len(input_data) == 1:
        break

qout = model.execute([test_image], quantize=True)
print(qout[-1][:]) #看输出

# setting = model_settings(cluster_stack_size=4096,l1_size=64000,l2_size=400000)
result = model.execute_on_target(input_tensors=input_data, directory='./mob', pmsis_os='pulpos',source='gap8_v3.sh', output_tensors=False, dont_run=True,
                            #   settings = setting,
                              )


# print(result.stdout)

# acc = gap_test(model,test_loader2,dequantize=True)
# print(acc)