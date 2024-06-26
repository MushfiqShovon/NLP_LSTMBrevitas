import sys
import os
import time
import random
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse

import pandas as pd

from models import Gen, GenQLSTM, GenQLSTM_2bit, GenQLSTM_1bit

parser = argparse.ArgumentParser(description="Testing script")
parser.add_argument('--bit_width', type=int, default=8, help='bit width for quantization')
args = parser.parse_args()

data_dict = torch.load(os.path.join('data', 'ag_news', 'data', 'traindata.v40000.l80.s5000'))
traindata = data_dict['traindata']
trainlabel = data_dict['trainlabel']
validdata = data_dict['validdata']
validlabel = data_dict['validlabel']
testdata = data_dict['testdata']
testlabel = data_dict['testlabel']
vocab_size = data_dict['vocabsize']

word_emb_dim = 100  # size of word embeddings
label_emb_dim = 100  # size of label embeddings
hid_dim = 100  # number of hidden units
nlayers = 1  # number of lstm layers
nclass = 4  # number of classes
dropout = 0
use_cuda = torch.cuda.is_available()
tied = False
use_bias = False
concat_label = 'hidden'
avg_loss = False
one_hot = False
bit_width=args.bit_width

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = Gen(vocab_size, word_emb_dim, label_emb_dim, hid_dim, nlayers, nclass, dropout, use_cuda, tied, use_bias, concat_label, avg_loss, one_hot, bit_width).to(device)

criterion = nn.CrossEntropyLoss(reduce=False).to(device)

model_path = 'ModelParameterLSTM_FP.pth'

# Load the state dictionary from the file
state_dict = torch.load(model_path)

# Load the state dictionary into the model
model.load_state_dict(state_dict)

def batches(data, label, batch_size, is_eval = False, nclass = 0):
    d_l = list(zip(data, label))
    random.shuffle(d_l)
    data, label = zip(*d_l)
    data, label = list(data), list(label)

    for i in range(0, len(data), batch_size):
        sentences = data[i:i + batch_size]
        labels = label[i:i + batch_size]

        s_l = zip(sentences, labels)
        s_l = sorted(s_l, key = lambda l: len(l[0]), reverse=True)

        sentences, labels = zip(*s_l)

        sentences = list(sentences)
        labels = list(labels)

        # x_pred: predicted ground truth, padding in the end
        # y_ext: extend label to the length of sentence length for concatnation
        # seq_len: pred_seq_len = actual seq len - 1

        y_ext = []
        for idx, d in enumerate(sentences):
            y_ext.append([labels[idx]] * (len(d) - 1))

        if is_eval:
            y_exts = []
            for y_label in range(nclass):
                y_ext = []
                for d in sentences:
                    y_ext.append(torch.LongTensor([y_label] * (len(d) - 1)))
                y_exts.append(y_ext)

            yield [torch.LongTensor(s) for s in sentences], y_exts, torch.LongTensor(labels)
        else:
            yield [torch.LongTensor(s) for s in sentences], \
                [torch.LongTensor(y) for y in y_ext], torch.LongTensor(labels)

def evaluate(validdata, validlabel, model, criterion, args):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    total_correct = 0.

    cnt = 0
    with torch.no_grad():
        for sents, y_exts, labels in batches(validdata, validlabel, args.batch_size, True, args.nclass):
            hidden = model.init_hidden(len(sents))

            x = nn.utils.rnn.pack_sequence([s[:-1] for s in sents])
            x_pred = nn.utils.rnn.pack_sequence([s[1:] for s in sents])

            # p_y = torch.FloatTensor([0.071] * len(seq_len))

            losses = []
            for y_ext in y_exts:
                y_ext = nn.utils.rnn.pack_sequence(y_ext)

                if args.device.type == 'cuda':
                    x, y_ext, x_pred, labels = x.cuda(), y_ext.cuda(), x_pred.cuda(), labels.cuda()

                # output (batch_size, )
                hidden = model.init_hidden(len(sents))
                #if args.device.type == 'cuda':
                #    hidden = hidden.cuda()
                #    model=model.cuda()
                loss = model(x, x_pred, y_ext, hidden, criterion, True)
                losses.append(loss)

            losses = torch.cat(losses, dim=0).view(-1, len(sents))
            prediction = torch.argmin(losses, dim=0)

            num_correct = (prediction == labels).float().sum()

            total_loss += torch.sum(torch.min(losses, dim=0)[0]).item()
            total_correct += num_correct.item()
            cnt += 1

    return total_loss / cnt, total_correct / len(validlabel) * 100.0

def save_model(bit_width, model, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, f'ModelParameter_{bit_width}bit.pth')
    torch.save(model.state_dict(), filename)
    logging.info(f"Model saved to {filename}")

class var:
    batch_size=32
    nclass = len(open(os.path.join('data', 'ag_news', 'classes.txt'), 'r').readlines())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

var=var()

model.eval()
test_start_time = time.time()
test_loss, test_acc = evaluate(testdata, testlabel, model, criterion, var)
test_time=time.time()-test_start_time
print('=' * 89)
print('| Test | test loss ', test_loss, ' | test acc ', test_acc)
print('=' * 89)

quantized_model = GenQLSTM(vocab_size, word_emb_dim, label_emb_dim, hid_dim, nlayers, nclass, dropout, use_cuda, tied, use_bias, concat_label, avg_loss, one_hot, bit_width).to(device)

# Ensure that the hidden_dim is defined based on your model configuration
hidden_dim = hid_dim

# Transfer parameters from pretrained model to quantized model
quantized_model.encoder.weight.data = model.encoder.weight.data
quantized_model.label_encoder.weight.data = model.label_encoder.weight.data

# LSTM weights and biases
quantized_model.rnn.layers[0][0].input_gate_params.input_weight.weight.data = model.rnn.weight_ih_l0[:hidden_dim, :].data
quantized_model.rnn.layers[0][0].forget_gate_params.input_weight.weight.data = model.rnn.weight_ih_l0[hidden_dim:2*hidden_dim, :].data
quantized_model.rnn.layers[0][0].cell_gate_params.input_weight.weight.data = model.rnn.weight_ih_l0[2*hidden_dim:3*hidden_dim, :].data
quantized_model.rnn.layers[0][0].output_gate_params.input_weight.weight.data = model.rnn.weight_ih_l0[3*hidden_dim:, :].data

quantized_model.rnn.layers[0][0].input_gate_params.hidden_weight.weight.data = model.rnn.weight_hh_l0[:hidden_dim, :].data
quantized_model.rnn.layers[0][0].forget_gate_params.hidden_weight.weight.data = model.rnn.weight_hh_l0[hidden_dim:2*hidden_dim, :].data
quantized_model.rnn.layers[0][0].cell_gate_params.hidden_weight.weight.data = model.rnn.weight_hh_l0[2*hidden_dim:3*hidden_dim, :].data
quantized_model.rnn.layers[0][0].output_gate_params.hidden_weight.weight.data = model.rnn.weight_hh_l0[3*hidden_dim:, :].data

quantized_model.rnn.layers[0][0].input_gate_params.bias.data = model.rnn.bias_ih_l0[:hidden_dim].data + model.rnn.bias_hh_l0[:hidden_dim].data
quantized_model.rnn.layers[0][0].forget_gate_params.bias.data = model.rnn.bias_ih_l0[hidden_dim:2*hidden_dim].data + model.rnn.bias_hh_l0[hidden_dim:2*hidden_dim].data
quantized_model.rnn.layers[0][0].cell_gate_params.bias.data = model.rnn.bias_ih_l0[2*hidden_dim:3*hidden_dim].data + model.rnn.bias_hh_l0[2*hidden_dim:3*hidden_dim].data
quantized_model.rnn.layers[0][0].output_gate_params.bias.data = model.rnn.bias_ih_l0[3*hidden_dim:].data + model.rnn.bias_hh_l0[3*hidden_dim:].data

quantized_model.decoder.weight.data = model.decoder.weight.data

quantized_model.eval()
test_start_time = time.time()
test_loss, test_acc = evaluate(testdata, testlabel, quantized_model, criterion, var)
test_time=time.time()-test_start_time
print('=' * 89)
print('| Test | test loss ', test_loss, ' | test acc ', test_acc)
print('=' * 89)

model_dir=f'./ModelParameter/FULL_Quantized_woCal/{bit_width}bit/'
save_model(bit_width, model, model_dir)
accuracies = []
accuracies.append({'bit_width': bit_width, 'Validation Accuracy': test_acc, 'Epoch Time': test_time})
df_accuracies = pd.DataFrame(accuracies)
accuracy_save_path = model_dir + f'accuracies_{bit_width}bit.csv'
df_accuracies.to_csv(accuracy_save_path, index=False)
print(f'Accuracies saved at "{accuracy_save_path}"')
