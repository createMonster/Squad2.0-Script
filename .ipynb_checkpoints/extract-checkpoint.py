import torch.nn as nn
import argparse
import glob
import logging
import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import transformers
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers.trainer_utils import is_main_process


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
path_base = '/private/wenlong/squad/candidate_output/'

device = torch.device("cuda")

MODEL_LIST = ['electra_finetune_dev_bs12_lr3e-6_ep3_seed1996', 'tmp']
MODEL_LIST = [path_base+x for x in MODEL_LIST]

class MyExtract(nn.Module):
    
    def __init__(self):
        super(MyExtract, self).__init__()
        self.fc1 = nn.Linear(29696, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x
    
def get_hidden_states_as_input(model_path):
    
    config = AutoConfig.from_pretrained(
        model_path,
        cache_dir=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        do_lower_case=True,
        cache_dir=None,
        use_fast=False,  # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
    )
    single_model = AutoModelForQuestionAnswering.from_pretrained(
        model_path,
        from_tf=False,
        config=config,
        cache_dir=None,
    )
    
    
    '''
    tensor_list = [torch.load(x)[-1] for x in MODEL_LIST]
    tensor_list = [x.view(x.size()[1], -1) for x in tensor_list]
    hidden_input = torch.cat(tensor_list, dim=1)
    
    print (hidden_input.size())
    
    return hidden_input
    '''
    
            
    
hidden_input = get_hidden_states_as_input()
model = MyExtract()

model.train()




'''

config = AutoConfig.from_pretrained(
        model_path,
        cache_dir=None,
    )
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    do_lower_case=True,
    cache_dir=None,
    use_fast=False,  # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
)
single_model = AutoModelForQuestionAnswering.from_pretrained(
    model_path,
    from_tf=False,
    config=config,
    cache_dir=None,
)

print (single_model)

single_model.train()

for param in single_model.parameters():
    param.requires_grad_(False)
    
input = 

'''