import sys
import os.path as osp

from measure import measure_model
sys.path.insert(0, osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))), '../'))
from models.mobile import mobilenetv3_large, mobilenetv3_small
from models.vgg import Encoder

import datetime
start = datetime.datetime.now()


# mobile_model = mobilenetv3_large()
vgg_model = Encoder()
n_flops, n_convops, n_params = measure_model(vgg_model, 417, 417)
print('FLOPs: {:.4f}M, Conv_FLOPs: {:.4f}M, Params: {:.4f}M'.
format(n_flops / 1e6, n_convops / 1e6, n_params / 1e6))

end = datetime.datetime.now()
print (end-start)