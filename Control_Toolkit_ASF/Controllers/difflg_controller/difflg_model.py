import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from datetime import datetime

import time
import os
import socket
from zoneinfo import ZoneInfo
import ast

import torch.profiler
############################ CONFIG ########################

############################ DATA ########################
# from load_cp_data import MyCPSDataset
# train_data_dir = "D:\\zihao\\telluride_25\\diff_logic\\npc25-difflogic-4-control\\cartpole_recording\\Train\\"
# test_data_dir = "D:\\zihao\\telluride_25\\diff_logic\\npc25-difflogic-4-control\\cartpole_recording\\Test\\"

from dotenv import dotenv_values
config = { **dotenv_values(".env"), **os.environ }

# PASS_INPUT_TO_ALL_LAYERS=1 C_INIT="XAVIER_U" C_SPARSITY=10 G_SPARSITY=1 OPT_GATE16_CODEPATH=3 KINETO_LOG_LEVEL=99 GATE_ARCHITECTURE="[2000,2000]" INTERCONNECT_ARCHITECTURE="[]" PRINTOUT_EVERY=211 EPOCHS=300 uv run mnist.py
# CONNECTIVITY_GAIN=1 LEARNING_RATE=0.001 PASS_RESIDUAL=0 PASS_INPUT_TO_ALL_LAYERS=0 C_INIT="NORMAL" C_SPARSITY=100 C_INIT_PARAM=0 GATE_ARCHITECTURE="[250,250,250,250,250,250]"PRINTOUT_EVERY=55 VALIDATE_EVERY=211 EPOCHS=200 uv run mnist.py

LOG_TAG = config.get("LOG_TAG", "MNIST")
TIMEZONE = config.get("TIMEZONE", "UTC")

BINARIZE_IMAGE_TRESHOLD = float(config.get("BINARIZE_IMAGE_TRESHOLD", 0.75))
IMG_WIDTH = int(config.get("IMG_WIDTH", 28)) # 16 is suitable for Tiny Tapeout
IMG_CROP = int(config.get("IMG_CROP", 28))
# INPUT_SIZE = IMG_WIDTH * IMG_WIDTH
INPUT_SIZE = 7 * 32
DATA_SPLIT_SEED = int(config.get("DATA_SPLIT_SEED", 42))
TRAIN_FRACTION = float(config.get("TRAIN_FRACTION", 0.9))
NUMBER_OF_CATEGORIES = int(config.get("NUMBER_OF_CATEGORIES", 1))
ONLY_USE_DATA_SUBSET = config.get("ONLY_USE_DATA_SUBSET", "0").lower() in ("true", "1", "yes")

SEED = config.get("SEED", random.randint(0, 1024*1024))
LOG_NAME = f"{LOG_TAG}_{SEED}"
GATE_ARCHITECTURE = ast.literal_eval(config.get("GATE_ARCHITECTURE", "[1000,1000]")) # previous: [1300,1300,1300] resnet95: [2000, 2000] conn_gain96: [2250, 2250, 2250] power_law_fixed97: [8000,8000,8000, 8000,8000,8000] auto_scale_logits_97_5 [8000]
INTERCONNECT_ARCHITECTURE = ast.literal_eval(config.get("INTERCONNECT_ARCHITECTURE", "[[],[]]")) # previous: [[32, 325], [26, 52], [26, 52]])) resnet95, conn_gain96, auto_scale_logits_97_5: [] power_law_fixed97: [[],[-1],[-1], [-1],[-1],[-1]] p
if not INTERCONNECT_ARCHITECTURE or INTERCONNECT_ARCHITECTURE == []:
    INTERCONNECT_ARCHITECTURE = [[] for g in GATE_ARCHITECTURE]
assert len(GATE_ARCHITECTURE) == len(INTERCONNECT_ARCHITECTURE)
BATCH_SIZE = int(config.get("BATCH_SIZE", 256))

EPOCHS = int(config.get("EPOCHS", 30))
EPOCH_STEPS = math.floor((135030 * TRAIN_FRACTION) / BATCH_SIZE) # MNIST consists of 60K images
TRAINING_STEPS = EPOCHS*EPOCH_STEPS
PRINTOUT_EVERY = int(config.get("PRINTOUT_EVERY", EPOCH_STEPS))
VALIDATE_EVERY = int(config.get("VALIDATE_EVERY", EPOCH_STEPS * 5))

LEARNING_RATE = float(config.get("LEARNING_RATE", 0.01))

SUPPRESS_PASSTHROUGH = config.get("SUPPRESS_PASSTHROUGH", "0").lower() in ("true", "1", "yes")
SUPPRESS_CONST = config.get("SUPPRESS_CONST", "0").lower() in ("true", "1", "yes")

PROFILE = config.get("PROFILE", "0").lower() in ("true", "1", "yes")
if PROFILE: prof = torch.profiler.profile(schedule=torch.profiler.schedule(skip_first=10, wait=3, warmup=1, active=1, repeat=1000), record_shapes=True, with_flops=True) #, with_stack=True, with_modules=True)
PROFILER_ROWS = int(config.get("PROFILER_ROWS", 20))

FORCE_CPU = config.get("FORCE_CPU", "0").lower() in ("true", "1", "yes")
COMPILE_MODEL = config.get("COMPILE_MODEL", "0").lower() in ("true", "1", "yes")

C_INIT = config.get("C_INIT", "NORMAL") # NORMAL, DIRAC
G_INIT = config.get("G_INIT", "NORMAL") # NORMAL, UNIFORM, PASSTHROUGH, XOR
C_SPARSITY = float(config.get("C_SPARSITY", 1.0)) # NOTE: 1.0 works well only for SHALLOW nets, 3.0 for deeper is necessary to binarize well
G_SPARSITY = float(config.get("G_SPARSITY", 1.0))

PASS_INPUT_TO_ALL_LAYERS = config.get("PASS_INPUT_TO_ALL_LAYERS", "0").lower() in ("true", "1", "yes") # previous: 1
PASS_RESIDUAL = config.get("PASS_RESIDUAL", "0").lower() in ("true", "1", "yes")

TAU = float(config.get("TAU", -1))

config_printout_keys = ["LOG_NAME", "TIMEZONE",
               "BINARIZE_IMAGE_TRESHOLD", "IMG_WIDTH", "INPUT_SIZE", "IMG_CROP", "DATA_SPLIT_SEED", "TRAIN_FRACTION", "NUMBER_OF_CATEGORIES", "ONLY_USE_DATA_SUBSET",
               "SEED", "GATE_ARCHITECTURE", "INTERCONNECT_ARCHITECTURE", "BATCH_SIZE",
               "EPOCHS", "EPOCH_STEPS", "TRAINING_STEPS", "PRINTOUT_EVERY", "VALIDATE_EVERY",
               "LEARNING_RATE",
               "C_INIT", "G_INIT", "C_SPARSITY", "G_SPARSITY",
               "PASS_INPUT_TO_ALL_LAYERS", "PASS_RESIDUAL",
               "TAU",
               "SUPPRESS_PASSTHROUGH", "SUPPRESS_CONST",
               "PROFILE", "FORCE_CPU", "COMPILE_MODEL"]
config_printout_dict = {key: globals()[key] for key in config_printout_keys}

############################ LOG ########################
log = print

############################ DEVICE ########################

try:
    device = torch.device(
                    "cuda" if torch.cuda.is_available()         and not FORCE_CPU else 
                    "mps"  if torch.backends.mps.is_available() and not FORCE_CPU else 
                    "cpu")
except:
    device = torch.device("cpu")

#################### TENSOR BINARIZATION ##################

def binarize_inplace(x, dim=-1, bin_value=1):
    ones_at = torch.argmax(x, dim=dim)
    x.data.zero_()
    x.data.scatter_(dim=dim, index=ones_at.unsqueeze(dim), value=bin_value)

############################ MODEL ########################

class FixedPowerLawInterconnect(nn.Module):
    def __init__(self, inputs, outputs, alpha, x_min=1.0, name=''):
        super(FixedPowerLawInterconnect, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.alpha = alpha

        max_length = inputs
        size = outputs
        r = torch.rand(size)
        if alpha > 1:
            magnitudes = x_min * (1 - r) ** (-1 / (alpha - 1))          # Power law distribution
            signs = torch.randint(low=0, high=2, size=(size,)) * 2 - 1  # -1 or +1
            offsets = magnitudes * signs * max_length
            indices = torch.arange(start=0, end=size) + offsets.long()
            indices = indices % max_length
        else:
            c       = torch.randperm(outputs) % inputs
            indices = torch.randperm(inputs)[c]
        self.register_buffer("indices", indices)
        self.binarized = False

    @torch.profiler.record_function("mnist::Fixed::FWD")
    def forward(self, x):
        return x[:, self.indices] if not self.binarized else torch.matmul(x, self.c)

        # Performance comparison
        # 1) x[:, self.indices]
        # MPS: 4.29 ms per iteration [300,300], tiny bit faster
        # MPS: 9.93 ms per iteration [3000,3000]
        # Takes significantly less memory though!

        # 2)
        # batch_size = x.shape[0]
        # if self.batch_indices.shape[0] != batch_size:
        #     self.batch_indices = self.indices.repeat(batch_size, 1)
        # return torch.gather(x, dim=1, index=self.batch_indices) 
        # MPS: 4.57 ms per iteration [300,300]
        # MPS: 9.32 ms per iteration [3000,3000], tiny bit faster

    def binarize(self, bin_value=1):
        with torch.no_grad():
            c = torch.zeros((self.inputs, self.outputs), dtype=torch.float32, device=device)
            c.scatter_(dim=0, index=self.indices.unsqueeze(0), value=bin_value)
            self.register_buffer("c", c)
            self.binarized = True

    def __repr__(self):
        with torch.no_grad():
            i = self.indices.view(self.outputs // 2, 2) # [batch_size, number_of_gates, 2]
            A = i[:,0]
            B = i[:,1]

            d = torch.abs(A-B)
            d = torch.minimum(d, self.inputs - d)
            return f"FixedPowerLawInterconnect({self.inputs} -> {self.outputs // 2}x2, α={self.alpha}, mean={d.float().mean().long()} median={d.float().median().long()})"

class LearnableInterconnect(nn.Module):
    def __init__(self, inputs, outputs, name=''):
        super(LearnableInterconnect, self).__init__()
        self.c = nn.Parameter(torch.zeros((inputs, outputs), dtype=torch.float32))
        if C_INIT == "DIRAC" or C_INIT == "UNIQUE":
            with torch.no_grad():
                nn.init.normal_(self.c, mean=0.0, std=.01) # add a very small amount of noise as a tie-breaker
                idx = torch.randperm(outputs) % inputs
                idx = torch.randperm(inputs)[idx]
                idx = idx.unsqueeze(0)
                self.c.data.scatter_add_(dim=0, index=idx, src=torch.ones_like(idx, dtype=torch.float32) * 10.0) # 10 here is inspired by the min/max range of a gaussian
        else:
            nn.init.normal_(self.c, mean=0.0, std=1)
        self.name = name
        self.binarized = False
    
    @torch.profiler.record_function("mnist::Sparse::FWD")
    def forward(self, x):
        connections = F.softmax(self.c * C_SPARSITY, dim=0) if not self.binarized else self.c
        return torch.matmul(x, connections)

    def binarize(self, bin_value=1):
        binarize_inplace(self.c, dim=0, bin_value=bin_value)
        self.binarized = True

    def __repr__(self):
        return f"LearnableInterconnect({self.c.shape[0]} -> {self.c.shape[1] // 2}x2)"

class LearnableGate16Array(nn.Module):
    def __init__(self, number_of_gates, name=''):
        super(LearnableGate16Array, self).__init__()
        self.number_of_gates = number_of_gates
        self.number_of_inputs = number_of_gates * 2
        self.name = name
        self.w = nn.Parameter(torch.zeros((16, self.number_of_gates), dtype=torch.float32)) # [16, W]
        self.zeros = torch.empty(0)
        self.ones = torch.empty(0)
        self.binarized = False
        nn.init.normal_(self.w, mean=0, std=1)
        if G_INIT == "UNIFORM":
            nn.init.uniform_(self.w, a=0.0, b=1.0)
        elif G_INIT == "PASS" or G_INIT == "PASSTHROUGH":
            with torch.no_grad(): self.w[3, :] = 5 # 5 will roughly result in a 95% percent once softmax() applied
        elif G_INIT == "XOR":
            with torch.no_grad(): self.w[6, :] = 5 # 5 will roughly result in a 95% percent once softmax() applied
        else:
            nn.init.normal_(self.w, mean=0.0, std=1)

        # g0  = 
        # g1  =         AB
        # g2  =   A    -AB
        # g3  =   A
        # g4  =      B -AB
        # g5  =      B
        # g6  =   A  B-2AB
        # g7  =   A  B -AB
        # g8  = 1-A -B  AB
        # g9  = 1-A -B 2AB
        # g10 = 1   -B
        # g11 = 1   -B  AB
        # g12 = 1-A
        # g13 = 1-A     AB
        # g14 = 1      -AB
        # g15 = 1

        W = torch.zeros(4, 16)
        # 1 weights
        W[0, 8:16] =            1
        # A weights
        W[1, [2, 3,  6,  7]] =  1
        W[1, [8, 9, 12, 13]] = -1
        # B weights
        W[2, [4, 5,  6,  7]] =  1
        W[2, [8, 9, 10, 11]] = -1
        # A*B weights
        W[3, 1] =               1
        W[3, 6] =              -2
        W[3, [2, 4, 7,14]] =   -1
        W[3, [1, 8,11,13]] =    1
        W[3, 9] =               2 

        self.W16_to_4 = W

    @torch.profiler.record_function("mnist::LearnableGate16::FWD")
    def forward(self, x):
        # batch_size = x.shape[0]
        # x = x.view(batch_size, self.number_of_gates, 2) # [batch_size, number_of_gates, 2]

        # A = x[:,:,0] # [batch_size, number_of_gates]
        # B = x[:,:,1] # [batch_size, number_of_gates]
        
        # weights_t = F.softmax(self.w * G_SPARSITY, dim=0).transpose(0,1) if not self.binarized else self.w.transpose(0,1)
        # weights = torch.matmul(weights_t, self.W16_to_4.transpose(0,1))  # [number_of_gates, 4]
        # result = weights * torch.stack([torch.ones_like(A), A, B, A*B], dim=2)
        # return result.sum(dim=2)

        batch_size = x.shape[0]
        x = x.view(batch_size, self.number_of_gates, 2) # [batch_size, number_of_gates, 2]

        A = x[:,:,0]          # [batch_size, number_of_gates]
        B = x[:,:,1]          # [batch_size, number_of_gates]
        
        weights = F.softmax(self.w * G_SPARSITY, dim=0) if not self.binarized else self.w
        weights = torch.matmul(self.W16_to_4, weights) # [4, number_of_gates]
        result = weights * torch.stack([torch.ones_like(A), A, B, A*B], dim=1)
        return result.sum(dim=1)

    def binarize(self, bin_value=1):
        binarize_inplace(self.w, dim=0, bin_value=bin_value)
        self.binarized = True

    def __repr__(self):
        return f"LearnableGate16Array({self.number_of_gates})"

class Model(nn.Module):
    def __init__(self, seed, gate_architecture, interconnect_architecture, number_of_categories, input_size):
        super(Model, self).__init__()
        self.gate_architecture = gate_architecture
        self.interconnect_architecture = interconnect_architecture
        self.first_layer_gates = self.gate_architecture[0]
        self.last_layer_gates = self.gate_architecture[-1]
        self.number_of_categories = number_of_categories
        self.input_size = input_size
        self.seed = seed
        
        self.outputs_per_category = self.last_layer_gates // self.number_of_categories
        assert self.last_layer_gates == self.number_of_categories * self.outputs_per_category

        layers_ = []
        layer_inputs = input_size
        R = [input_size]
        for layer_idx, (layer_gates, interconnect_params) in enumerate(zip(gate_architecture, interconnect_architecture)):
            if len(interconnect_params) == 1 and interconnect_params[0] > 0:
                interconnect = FixedPowerLawInterconnect(layer_inputs, layer_gates*2, alpha= interconnect_params[0],  name=f"i_{layer_idx}")
            else:
                interconnect = LearnableInterconnect    (layer_inputs, layer_gates*2,                                 name=f"i_{layer_idx}")
            layers_.append(interconnect)
            layers_.append(LearnableGate16Array(layer_gates, f"g_{layer_idx}"))
            layer_inputs = layer_gates
            R.append(layer_gates)
            if PASS_INPUT_TO_ALL_LAYERS:
                layer_inputs += input_size
            if PASS_RESIDUAL and (layer_idx > 0 or not PASS_INPUT_TO_ALL_LAYERS):
                layer_inputs += R[-2]
            
        layers_.append(nn.Linear(layer_inputs, 1, bias=True))
        self.layers = nn.ModuleList(layers_)

    @torch.profiler.record_function("mnist::Model::FWD")
    def forward(self, X):
        with torch.no_grad():
            S = torch.sum(X, dim=-1)
            # self.log_input_mean = torch.mean(S).item()
            # self.log_input_std = torch.std(S).item()
            # self.log_input_norm = torch.norm(S).item()
        I = X
        R = [I]
        for layer_idx in range(0, len(self.layers)):
            X = self.layers[layer_idx](X)
            if type(self.layers[layer_idx]) is LearnableGate16Array:
                R.append(X)
                # TODO: fix unreadable logic with layer_idx
                # NOTE: ugly layer_idx > 1 which differ from layer_idx > 0 in the Model constructor, but has the same meaning
                if PASS_INPUT_TO_ALL_LAYERS and layer_idx < len(self.layers)-2:
                    X = torch.cat([X, I], dim=-1)
                if PASS_RESIDUAL and (layer_idx > 1 or not PASS_INPUT_TO_ALL_LAYERS) and layer_idx < len(self.layers)-2:
                    X = torch.cat([X, R[-2]], dim=-1)

        # print ("TRAINING: ", self.training)
        if not self.training:   # INFERENCE ends here! Everything past this line will only concern training
            return X            # Finishing inference here is both:
                                # 1) an OPTIMISATION and
                                # 2) it ensures no discrepancy between VALIDATION step during training vs STANDALONE inference
        # X = X.view(X.size(0), self.number_of_categories, self.outputs_per_category).sum(dim=-1)

        if TAU < 0:
            gain = np.sqrt(self.outputs_per_category / 6.0)
        else:
            gain = TAU

        # with torch.no_grad():
            # self.log_applied_gain = gain
            # self.log_pregain_mean = torch.mean(X).item()
            # self.log_pregain_std = torch.std(X).item()
            # self.log_pregain_min = torch.min(X).item()
            # self.log_pregain_max = torch.max(X).item()
            # self.log_pregain_norm = torch.norm(X).item()
        # X = X / gain

        # with torch.no_grad():
        #     self.log_logits_mean = torch.mean(X).item()
        #     self.log_logits_std = torch.std(X).item()
        #     self.log_logits_norm = torch.norm(X).item()

        return X

    def clone_and_binarize(self, device, bin_value=1):
        model_binarized = Model(self.seed, self.gate_architecture, self.interconnect_architecture, self.number_of_categories, self.input_size).to(device)
        model_binarized.load_state_dict(self.state_dict())
        for layer in model_binarized.layers:
            if hasattr(layer, 'binarize') and callable(layer.binarize):
                layer.binarize(bin_value)
        return model_binarized

    def get_passthrough_fraction(self):
        pass_fraction_array = []
        indices = torch.tensor([3, 5, 10, 12], dtype=torch.long)
        for model_layer in self.layers:
            if hasattr(model_layer, 'w'):
                weights_after_softmax = F.softmax(model_layer.w, dim=0)
                pass_weight = (weights_after_softmax[indices, :]).sum()
                total_weight = weights_after_softmax.sum()
                pass_fraction_array.append(pass_weight / total_weight)
        return pass_fraction_array
    
    def get_unique_fraction(self):
        unique_fraction_array = []
        with torch.no_grad():
            for model_layer in self.layers:
                # TODO: better to measure only unique indices that point to the previous layer and ignore skip connections:
                # ... = sum(unique_indices < previous_layer.c.shape[1])
                if hasattr(model_layer, 'c'):
                    unique_indices = torch.unique(torch.argmax(model_layer.c, dim=0)).numel()
                    unique_fraction_array.append(unique_indices / model_layer.c.shape[0])
                elif hasattr(model_layer, 'top_c') and hasattr(model_layer, 'top_indices'):
                    outputs = model_layer.top_c.shape[1]
                    top1 = torch.argmax(model_layer.top_c, dim=0)
                    indices = model_layer.top_indices[top1, torch.arange(outputs)]
                    max_inputs = indices.max().item() # approximation
                    unique_indices = torch.unique(indices).numel()
                    unique_fraction_array.append(unique_indices / max_inputs)
                elif hasattr(model_layer, 'indices'):
                    max_inputs = model_layer.indices.max().item() # approximation
                    unique_indices = torch.unique(model_layer.indices).numel()
                    unique_fraction_array.append(unique_indices / max_inputs)
        return unique_fraction_array

    def compute_selected_gates_fraction(self, selected_gates):
        gate_fraction_array = []
        indices = torch.tensor(selected_gates, dtype=torch.long)
        for model_layer in self.layers:
            if hasattr(model_layer, 'w'):
                weights_after_softmax = F.softmax(model_layer.w, dim=0)
                pass_weight = (weights_after_softmax[indices, :]).sum()
                total_weight = weights_after_softmax.sum()
                gate_fraction_array.append(pass_weight / total_weight)
        return torch.mean(torch.tensor(gate_fraction_array)).item()
