from __future__ import division

import os
import warnings
from collections import OrderedDict
from config import return_args, args
from scipy.ndimage.filters import gaussian_filter
from torchvision import transforms
from utils import setup_seed
import nni
from nni.utils import merge_parameter
import util.misc as utils
import torch
import numpy as np
import cv2
import torch.nn as nn
from Networks.CDETR import build_model
import torch.onnx
import onnx
import torch_tensorrt

from datetime import datetime
img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tensor_transform = transforms.ToTensor()

warnings.filterwarnings('ignore')
'''fixed random seed '''
setup_seed(args.seed)

def change_input_dim(model):
    # Use some symbolic name not used for any other dimension
    sym_batch_dim = "N"
    # or an actal value
    actual_batch_dim = 4 

    # The following code changes the first dimension of every input to be batch-dim
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_dim 
    inputs = model.graph.input
    for input in inputs:
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = input.type.tensor_type.shape.dim[0]
        # update dim to be a symbolic value
        dim1.dim_param = sym_batch_dim
        # or update it to be an actual value:
        # dim1.dim_value = actual_batch_dim


def apply(transform, infile, outfile):
    model = onnx.load(infile)
    transform(model)
    onnx.save(model, outfile)

      
def main(args):

    utils.init_distributed_mode(return_args)
    model, criterion, postprocessors = build_model(return_args)
    model = model.cuda()
    
    if args['pre']:
        if os.path.isfile(args['pre']):
            
            #for layer_id in range(model.transformer.decoder.num_layers - 1):
            #    model.transformer.decoder.layers[layer_id + 1].ca_qpos_proj = None
            
            checkpoint = torch.load(args['pre'], map_location='cuda')['state_dict']
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
               # if 'backbone' in k or 'transformer' in k:
                name = k.replace('bbox', 'point') # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
                name = name.replace('module.', '')
                new_state_dict[name] = v

            print("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'], map_location='cuda')
            model.load_state_dict(new_state_dict)

            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            print("=> no checkpoint found at '{}'".format(args['pre']))

    model.eval()
    
    model.aux_loss = False
    model.half()
    traced_script_module = torch.jit.trace(model, torch.rand(1, 3, 256, 256).cuda().half())
    
    traced_script_module.save("cltr.pt")

if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()

    params = vars(merge_parameter(return_args, tuner_params))

    main(params)
