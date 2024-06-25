import torch
import pandas as pd
import re
import spacy
from torchtext.data.utils import get_tokenizer
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim
import torchtext
import time
import site
import os
import argparse
from torchtext import data

os.environ['SP_DIR'] = '/opt/conda/lib/python3.11/site-packages'

import warnings as wrn
wrn.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Training script")

parser.add_argument('--bit_width', type=int, default=8, help='bit width for quantization')
parser.add_argument('--model', type=str, default='QLSTM', choices=['QLSTM', 'QLSTM_2bit', 'QLSTM_1bit', 'QLSTM_OnlyLSTM', 'QLSTM_OnlyLSTM_1bit', 'QLSTM_OnlyLSTM_2bit', 'QLSTM_Others', 'QLSTM_Others_1bit', 'QLSTM_Others_2bit'], help='model class to use')

args = parser.parse_args()

from models import LSTMNet

SEED = 2021

torch.manual_seed(SEED)
torch.backends.cuda.deterministic = True

def clean_text(text):
    cleaned_text = re.sub(r'[^A-Za-z0-9]+', ' ', str(text))
    return cleaned_text

# Load and preprocess the data files
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path, header=None, delimiter='\t') # Assuming tab-separated values in .data files
    df[1] = df[1].apply(clean_text) # Assuming the text is in the second column
    cleaned_file_path = file_path.replace('.data', '_cleaned.data')
    df.to_csv(cleaned_file_path, index=False, header=False)
    return cleaned_file_path

cleaned_train_file = load_and_preprocess('./data/train.data')
cleaned_valid_file = load_and_preprocess('./data/valid.data')
cleaned_test_file = load_and_preprocess('./data/test.data')

spacy_en = spacy.load('en_core_web_sm')

def spacy_tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

LABEL = data.LabelField()
TEXT = data.Field(tokenize=spacy_tokenizer, batch_first=True, include_lengths=True)
fields = [("label", LABEL), ("text", TEXT)]

training_data = data.TabularDataset(path=cleaned_train_file, format="csv", fields=fields, skip_header=True)
validation_data = data.TabularDataset(path=cleaned_valid_file, format="csv", fields=fields, skip_header=True)
test_data = data.TabularDataset(path=cleaned_test_file, format="csv", fields=fields, skip_header=True)

print(vars(training_data.examples[0]))

train_data,valid_data = training_data.split(split_ratio=0.75,
                                            random_state=random.seed(SEED))

TEXT.build_vocab(train_data,
                 min_freq=5)

LABEL.build_vocab(train_data)
# Count the number of instances per class
label_counts = {LABEL.vocab.itos[i]: LABEL.vocab.freqs[LABEL.vocab.itos[i]] for i in range(len(LABEL.vocab))}
print("Number of instances per class:", label_counts)


print("Size of text vocab:",len(TEXT.vocab))

print("Size of label vocab:",len(LABEL.vocab))

TEXT.vocab.freqs.most_common(10)

# Creating GPU variable
#device = torch.device("cuda")
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')

Sent_SIZE=32
print("Batch size initialized")

train_iterator,validation_iterator = data.BucketIterator.splits(
    (train_data,valid_data),
    batch_size = Sent_SIZE,
    # Sort key is how to sort the samples
    sort_key = lambda x:len(x.text),
    sort_within_batch = True,
    device = device
)

test_iterator = data.BucketIterator(
    test_data,
    batch_size=Sent_SIZE,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=device
)

SIZE_OF_VOCAB = len(TEXT.vocab)
EMBEDDING_DIM = 100
NUM_HIDDEN_NODES = 100
NUM_OUTPUT_NODES = len(LABEL.vocab)
NUM_LAYERS = 1
BIDIRECTION = False
DROPOUT = 0.2
BIT_WIDTH = args.bit_width

print(SIZE_OF_VOCAB)
print(NUM_OUTPUT_NODES)

model = LSTMNet(SIZE_OF_VOCAB, EMBEDDING_DIM, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, NUM_LAYERS, BIDIRECTION, DROPOUT, BIT_WIDTH)
criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()

print(torch.cuda.is_available())

model = model.to(device)
optimizer = optim.Adam(model.parameters(),lr=1e-3)
#criterion = nn.BCELoss()
#criterion = criterion.to(device)

model

def multi_class_accuracy(preds, y):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == y).float()
    acc = correct.sum() / len(correct)
    return acc

def train(model,iterator,optimizer,criterion):
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    model.train()
    
    for batch in iterator:
        
        # cleaning the cache of optimizer
        optimizer.zero_grad()
        
        text,text_lengths = batch.text
        #print("Text Length:", text_lengths[0].item())
        global Sent_SIZE
        #Sent_SIZE=text_lengths[0].item()
        #print("Sent Length:", Sent_SIZE)
        #print("Iterator Batch Size:", batch.batch_size)
        batch.batch_size=Sent_SIZE
        #print("Iterator Batch Size:", batch.batch_size)
        iterator = data.BucketIterator(
            train_data,
            batch_size=Sent_SIZE,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            device=device
        )
        
        # forward propagation and squeezing
        predictions = model(text,text_lengths).squeeze()
        
        # computing loss / backward propagation
        loss = criterion(predictions, batch.label)
        #loss = criterion(predictions,batch.type)
        loss.backward()
        
        # accuracy
        acc = multi_class_accuracy(predictions,batch.label)
        
        # updating params
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    # It'll return the means of loss and accuracy
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model,iterator,criterion):
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    # deactivate the dropouts
    model.eval()
    
    # Sets require_grad flat False
    with torch.no_grad():
        for batch in iterator:
            text,text_lengths = batch.text
            
            predictions = model(text,text_lengths).squeeze()
              
            #compute loss and accuracy
            loss = criterion(predictions, batch.label)
            acc = multi_class_accuracy(predictions, batch.label)
            
            #keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def infer(model, iterator, criterion):
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze()
            loss = criterion(predictions, batch.label)
            acc = multi_class_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Create dataset and iterator for test data
test_fields = [("label", LABEL), ("text", TEXT)]
test_data = data.TabularDataset(path=cleaned_test_file, format="csv", fields=test_fields, skip_header=True)

checkpoint = torch.load('./modelParameter_LSTM_AGNEWS_FP.pth')
model.load_state_dict(checkpoint)

for name, param in model.named_parameters():
    print(name)

import brevitas.nn as qnn

from models import QLSTM, QLSTM_2bit, QLSTM_1bit, QLSTM_OnlyLSTM, QLSTM_OnlyLSTM_2bit, QLSTM_OnlyLSTM_1bit, QLSTM_Others, QLSTM_Others_2bit, QLSTM_Others_1bit

# Initialize the quantized model
if args.model == 'QLSTM':
    quantized_model = QLSTM(SIZE_OF_VOCAB, EMBEDDING_DIM, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, NUM_LAYERS, BIDIRECTION, DROPOUT, BIT_WIDTH)
    MODEL_DIR = f'ModelParameterDisc_PTQ/FULL_Quantized'
elif args.model == 'QLSTM_2bit':
    quantized_model = QLSTM_2bit(SIZE_OF_VOCAB, EMBEDDING_DIM, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, NUM_LAYERS, BIDIRECTION, DROPOUT, BIT_WIDTH)
    MODEL_DIR = f'ModelParameterDisc_PTQ/FULL_Quantized'
elif args.model == 'QLSTM_1bit':
    quantized_model = QLSTM_1bit(SIZE_OF_VOCAB, EMBEDDING_DIM, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, NUM_LAYERS, BIDIRECTION, DROPOUT, BIT_WIDTH)
    MODEL_DIR = f'ModelParameterDisc_PTQ/FULL_Quantized'
elif args.model == 'QLSTM_OnlyLSTM':
    quantized_model = QLSTM_OnlyLSTM(SIZE_OF_VOCAB, EMBEDDING_DIM, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, NUM_LAYERS, BIDIRECTION, DROPOUT, BIT_WIDTH)
    MODEL_DIR = f'ModelParameterDisc_PTQ/LSTMLayer_Quantized'
elif args.model == 'QLSTM_OnlyLSTM_2bit':
    quantized_model = QLSTM_OnlyLSTM_2bit(SIZE_OF_VOCAB, EMBEDDING_DIM, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, NUM_LAYERS, BIDIRECTION, DROPOUT, BIT_WIDTH)
    MODEL_DIR = f'ModelParameterDisc_PTQ/LSTMLayer_Quantized'
elif args.model == 'QLSTM_OnlyLSTM_1bit':
    quantized_model = QLSTM_OnlyLSTM_1bit(SIZE_OF_VOCAB, EMBEDDING_DIM, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, NUM_LAYERS, BIDIRECTION, DROPOUT, BIT_WIDTH)
    MODEL_DIR = f'ModelParameterDisc_PTQ/LSTMLayer_Quantized'
elif args.model == 'QLSTM_Others':
    quantized_model = QLSTM_Others(SIZE_OF_VOCAB, EMBEDDING_DIM, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, NUM_LAYERS, BIDIRECTION, DROPOUT, BIT_WIDTH)
    MODEL_DIR = f'ModelParameterDisc_PTQ/OtherLayers_Quantized'
elif args.model == 'QLSTM_Others_2bit':
    quantized_model = QLSTM_Others_2bit(SIZE_OF_VOCAB, EMBEDDING_DIM, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, NUM_LAYERS, BIDIRECTION, DROPOUT, BIT_WIDTH)
    MODEL_DIR = f'ModelParameterDisc_PTQ/OtherLayers_Quantized'
elif args.model == 'QLSTM_Others_1bit':
    quantized_model = QLSTM_Others_1bit(SIZE_OF_VOCAB, EMBEDDING_DIM, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, NUM_LAYERS, BIDIRECTION, DROPOUT, BIT_WIDTH)
    MODEL_DIR = f'ModelParameterDisc_PTQ/OtherLayers_Quantized'
    

quantized_model.to(device)

for name, param in quantized_model.named_parameters():
    print(name)


hidden_dim = NUM_HIDDEN_NODES
if args.model == 'QLSTM_Others' or args.model == 'QLSTM_Others_2bit' or args.model == 'QLSTM_Others_1bit':
    quantized_model.embedding.weight.data = model.embedding.weight.data
    quantized_model.lstm.weight_ih_l0 = model.lstm.weight_ih_l0
    quantized_model.lstm.weight_hh_l0 = model.lstm.weight_hh_l0
    quantized_model.lstm.bias_ih_l0 = model.lstm.bias_ih_l0
    quantized_model.lstm.bias_hh_l0 = model.lstm.bias_hh_l0
    quantized_model.fc.weight.data = model.fc.weight.data

else:
    quantized_model.embedding.weight.data = model.embedding.weight.data
    quantized_model.quantLSTM.layers[0][0].input_gate_params.input_weight.weight.data = model.lstm.weight_ih_l0[:hidden_dim, :].data
    quantized_model.quantLSTM.layers[0][0].forget_gate_params.input_weight.weight.data = model.lstm.weight_ih_l0[hidden_dim:2*hidden_dim, :].data
    quantized_model.quantLSTM.layers[0][0].cell_gate_params.input_weight.weight.data = model.lstm.weight_ih_l0[2*hidden_dim:3*hidden_dim, :].data
    quantized_model.quantLSTM.layers[0][0].output_gate_params.input_weight.weight.data = model.lstm.weight_ih_l0[3*hidden_dim:, :].data
    quantized_model.quantLSTM.layers[0][0].input_gate_params.hidden_weight.weight.data = model.lstm.weight_hh_l0[:hidden_dim, :].data
    quantized_model.quantLSTM.layers[0][0].forget_gate_params.hidden_weight.weight.data = model.lstm.weight_hh_l0[hidden_dim:2*hidden_dim, :].data
    quantized_model.quantLSTM.layers[0][0].cell_gate_params.hidden_weight.weight.data = model.lstm.weight_hh_l0[2*hidden_dim:3*hidden_dim, :].data
    quantized_model.quantLSTM.layers[0][0].output_gate_params.hidden_weight.weight.data = model.lstm.weight_hh_l0[3*hidden_dim:, :].data
    quantized_model.quantLSTM.layers[0][0].input_gate_params.bias.data = model.lstm.bias_ih_l0[:hidden_dim].data + model.lstm.bias_hh_l0[:hidden_dim].data
    quantized_model.quantLSTM.layers[0][0].forget_gate_params.bias.data = model.lstm.bias_ih_l0[hidden_dim:2*hidden_dim].data + model.lstm.bias_hh_l0[hidden_dim:2*hidden_dim].data
    quantized_model.quantLSTM.layers[0][0].cell_gate_params.bias.data = model.lstm.bias_ih_l0[2*hidden_dim:3*hidden_dim].data + model.lstm.bias_hh_l0[2*hidden_dim:3*hidden_dim].data
    quantized_model.quantLSTM.layers[0][0].output_gate_params.bias.data = model.lstm.bias_ih_l0[3*hidden_dim:].data + model.lstm.bias_hh_l0[3*hidden_dim:].data
    quantized_model.fc.weight.data = model.fc.weight.data



def calibrate(model, iterator):
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            _ = model(text, text_lengths)

calibrate(quantized_model, train_iterator)

# Create the directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

quantized_model_path = f'./{MODEL_DIR}/model_FullQ_{BIT_WIDTH}bit.pth'
torch.save(quantized_model.state_dict(), quantized_model_path)

test_loss, test_acc = infer(quantized_model, test_iterator, criterion)
print(f'\t Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}%')

def save_results_to_csv(bit_width, test_loss, test_acc, file_path=f'./{MODEL_DIR}/accuracies.csv'):
    results_df = pd.DataFrame({
        'Bit Width': [bit_width],
        'Test Loss': [test_loss],
        'Test Accuracy': [test_acc]
    })
    
    if os.path.exists(file_path):
        results_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(file_path, mode='w', header=True, index=False)

# Save the results to CSV
save_results_to_csv(BIT_WIDTH, test_loss, test_acc)
