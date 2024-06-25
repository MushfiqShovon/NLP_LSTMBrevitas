import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn

from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint, SignedBinaryWeightPerTensorConst, SignedBinaryActPerTensorConst, SignedTernaryWeightPerTensorConst, SignedTernaryActPerTensorConst, Uint8ActPerTensorFixedPoint, Int8WeightPerTensorFloat

class Gen(nn.Module):
    def __init__(self, ntoken, ninp, nlabelembed, nhid, nlayers, nclass, dropout, use_cuda, 
        tied, use_bias, concat_label, avg_loss, one_hot, bit_width):
        
        super(Gen, self).__init__()
        
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if one_hot: # one_hot
            self.label_encoder = self.get_one_hot
        else:           
            self.label_encoder = nn.Embedding(nclass, nlabelembed)
        if use_bias:
            self.bias_encoder = nn.Embedding(nclass, ntoken)
        if concat_label != 'hidden':
            self.rnn = nn.LSTM(ninp + nlabelembed, nhid, nlayers, dropout=dropout)
        else:
            self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        if concat_label != 'input':
            if one_hot:
                self.decoder = nn.Linear((nhid + nclass), ntoken, False)
            else:
                self.decoder = nn.Linear((nhid + nlabelembed), ntoken, False)
        else:
            self.decoder = nn.Linear(nhid, ntoken, False)
        self.nlayers = nlayers
        self.nhid = nhid
        self.loss = nn.CrossEntropyLoss()
        self.use_cuda = use_cuda
        self.nclass = nclass
        self.avg_loss = avg_loss
        self.use_bias = use_bias
        self.concat_label = concat_label
        self.init_weights()
        if tied:
            if nhid + nlabelembed != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

    def get_one_hot(self, batch):
        ones = torch.eye(self.nclass)
        if self.use_cuda:
            ones = ones.cuda()
        return ones.index_select(0,batch)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, x_pred, y_ext, hidden, criterion, is_infer = False):
        embedded_sents = self.encoder(x.data)

        embedded_label = self.label_encoder(y_ext.data)

        if self.concat_label != 'hidden':
            embedded_sents = torch.cat((embedded_sents, embedded_label), 1)

        embedded_sents = nn.utils.rnn.PackedSequence(
            embedded_sents, x.batch_sizes)

        output, (_, _) = self.rnn(embedded_sents, hidden)

        if self.concat_label != 'input':
            hidden_data = torch.cat((output.data, embedded_label), 1)
        else:
            hidden_data = output.data
        hidden_data = self.drop(hidden_data)

        # out: seq_len * n_token.
        out = self.decoder(hidden_data)
        if self.use_bias:
            out += self.bias_encoder(y_ext.data)

        loss = criterion(out, x_pred.data)

        if is_infer:
            LM_loss = nn.utils.rnn.pad_packed_sequence(nn.utils.rnn.PackedSequence(
                loss, x.batch_sizes))[0].transpose(0,1)
            total_loss = torch.sum(LM_loss, dim = 1)
            return total_loss
        else:
            if self.avg_loss:
                return torch.mean(loss)
            else:
                return torch.sum(loss)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))

class GenQLSTM(nn.Module):
    def __init__(self, ntoken, ninp, nlabelembed, nhid, nlayers, nclass, dropout, use_cuda, 
        tied, use_bias, concat_label, avg_loss, one_hot, bit_width):
        
        super(GenQLSTM, self).__init__()

        self.forward_pass_counter = 0

        if bit_width==1:
            weight_quant_type = SignedBinaryWeightPerTensorConst
            act_quant_type = SignedBinaryActPerTensorConst
            sigmoid_act_quant_type = SignedBinaryActPerTensorConst
        elif bit_width==2:
            weight_quant_type = SignedTernaryWeightPerTensorConst
            act_quant_type = SignedTernaryActPerTensorConst
            sigmoid_act_quant_type = SignedTernaryActPerTensorConst
        else:
            weight_quant_type = Int8WeightPerTensorFixedPoint
            act_quant_type = Int8ActPerTensorFixedPoint
            sigmoid_act_quant_type = Uint8ActPerTensorFixedPoint
            
        self.drop = nn.Dropout(dropout)
        self.encoder = qnn.QuantEmbedding(ntoken, ninp, weight_quant=weight_quant_type, weight_bit_width=bit_width)
        if one_hot:  # one_hot
            self.label_encoder = self.get_one_hot
        else:
            self.label_encoder = qnn.QuantEmbedding(nclass, nlabelembed, weight_quant=weight_quant_type, weight_bit_width=bit_width)
        if use_bias:
            self.bias_encoder = qnn.QuantEmbedding(nclass, ntoken, weight_quant=weight_quant_type, weight_bit_width=bit_width)
        if concat_label != 'hidden':
            self.rnn = qnn.QuantLSTM(
                ninp + nlabelembed, 
                nhid, 
                nlayers,
                bidirectional=False, 
                weight_bit_width=bit_width,
                io_bit_width=bit_width,
                gate_acc_bit_width=bit_width,
                sigmoid_bit_width=bit_width,
                tanh_bit_width=bit_width,
                weight_quant = weight_quant_type, 
                io_quant = act_quant_type,
                gate_acc_quant=act_quant_type,
                sigmoid_quant=sigmoid_act_quant_type,
                tanh_quant=act_quant_type,
                dropout=dropout)
        else:
            self.rnn = qnn.QuantLSTM(
                ninp, 
                nhid, 
                nlayers, 
                bidirectional=False, 
                weight_bit_width=bit_width,
                io_bit_width=bit_width,
                gate_acc_bit_width=bit_width,
                sigmoid_bit_width=bit_width,
                tanh_bit_width=bit_width,
                weight_quant = weight_quant_type, 
                io_quant = act_quant_type,
                gate_acc_quant=act_quant_type,
                sigmoid_quant=sigmoid_act_quant_type,
                tanh_quant=act_quant_type,
                dropout=dropout)
        if concat_label != 'input':
            if one_hot:
                self.decoder = qnn.QuantLinear((nhid + nclass), ntoken, bias=False, weight_quant = weight_quant_type, weight_bit_width=bit_width)
            else:
                self.decoder = qnn.QuantLinear((nhid + nlabelembed), ntoken, bias=False, weight_quant = weight_quant_type, weight_bit_width=bit_width)
        else:
            self.decoder = qnn.QuantLinear(nhid, ntoken, bias=False, weight_quant = weight_quant_type, weight_bit_width=bit_width)
        self.nlayers = nlayers
        self.nhid = nhid
        self.loss = nn.CrossEntropyLoss()
        self.use_cuda = use_cuda
        self.nclass = nclass
        self.avg_loss = avg_loss
        self.use_bias = use_bias
        self.concat_label = concat_label
        self.init_weights()
        if tied:
            if nhid + nlabelembed != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

    def get_one_hot(self, batch):
        ones = torch.eye(self.nclass)
        if self.use_cuda:
            ones = ones.cuda()
        return ones.index_select(0, batch)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, x_pred, y_ext, hidden, criterion, is_infer=False, cal=False):
        self.forward_pass_counter += 1
        print(f"Forward pass count: {self.forward_pass_counter}")
        embedded_sents = self.encoder(x.data) #Embedding

        #print("Embedded Sents Size:", embedded_sents.size())

        embedded_label = self.label_encoder(y_ext.data)

        #print("Embedded label Size:", embedded_label.size())

        if self.concat_label != 'hidden':
            embedded_sents = torch.cat((embedded_sents, embedded_label), 1)
            #print("Embedded Sents Size after cat:", embedded_sents.size())

        embedded_sents = nn.utils.rnn.pad_sequence(embedded_sents, batch_first=True)
        #print("Embedded Sents Size after packing:", type(embedded_sents), embedded_sents.size())

        #print("hidden size:", hidden.size())

        output, hidden = self.rnn(embedded_sents)  # Adjusted to handle RNN output

        #print("output of rnn:", output.data.size())

        first_part = output.data[:, 0, :]
        last_part = output.data[:, -1, :]
        
        # Now concatenate them along the second dimension
        result = (first_part + last_part) / 2
        #result = result.view(-1, 100)
        #print("Result Size:", result.size())

        if self.concat_label != 'input':
            if cal: 
                embedded_label = embedded_label[:, 0, :]
            else:
                embedded_label=embedded_label
            hidden_data = torch.cat((result, embedded_label), 1)
        else:
            hidden_data = result
        hidden_data = self.drop(hidden_data)

        out = self.decoder(hidden_data)
        if self.use_bias:
            out += self.bias_encoder(y_ext.data)

        if criterion is not None:
            loss = criterion(out, x_pred.data)
        else:
            loss = None

        if is_infer:
            if cal:
                return loss
            else:
                LM_loss = nn.utils.rnn.pad_packed_sequence(nn.utils.rnn.PackedSequence(
                loss, x.batch_sizes))[0].transpose(0, 1)
                total_loss = torch.sum(LM_loss, dim=1)
                return total_loss
        else:
            if self.avg_loss:
                return torch.mean(loss)
            else:
                return torch.sum(loss)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, bsz, self.nhid)  # Simplified for RNN


class GenQLSTM_OnlyLSTM(nn.Module):
    def __init__(self, ntoken, ninp, nlabelembed, nhid, nlayers, nclass, dropout, use_cuda, 
        tied, use_bias, concat_label, avg_loss, one_hot, bit_width):
        
        super(GenQLSTM_OnlyLSTM, self).__init__()

        self.forward_pass_counter = 0

        if bit_width==1:
            weight_quant_type = SignedBinaryWeightPerTensorConst
            act_quant_type = SignedBinaryActPerTensorConst
            sigmoid_act_quant_type = SignedBinaryActPerTensorConst
        elif bit_width==2:
            weight_quant_type = SignedTernaryWeightPerTensorConst
            act_quant_type = SignedTernaryActPerTensorConst
            sigmoid_act_quant_type = SignedTernaryActPerTensorConst
        else:
            weight_quant_type = Int8WeightPerTensorFixedPoint
            act_quant_type = Int8ActPerTensorFixedPoint
            sigmoid_act_quant_type = Uint8ActPerTensorFixedPoint

        print(weight_quant_type)
        print(act_quant_type)
        print(sigmoid_act_quant_type)
        
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if one_hot:  # one_hot
            self.label_encoder = self.get_one_hot
        else:
            self.label_encoder = nn.Embedding(nclass, nlabelembed)
        if use_bias:
            self.bias_encoder = nn.Embedding(nclass, ntoken)
        if concat_label != 'hidden':
            self.rnn = qnn.QuantLSTM(
                ninp + nlabelembed, 
                nhid, 
                nlayers,
                bidirectional=False, 
                weight_bit_width=bit_width,
                io_bit_width=bit_width,
                gate_acc_bit_width=bit_width,
                sigmoid_bit_width=bit_width,
                tanh_bit_width=bit_width,
                weight_quant = weight_quant_type, 
                io_quant = act_quant_type,
                gate_acc_quant=act_quant_type,
                sigmoid_quant=sigmoid_act_quant_type,
                tanh_quant=act_quant_type,
                dropout=dropout)
        else:
            self.rnn = qnn.QuantLSTM(
                ninp, 
                nhid, 
                nlayers, 
                bidirectional=False, 
                weight_bit_width=bit_width,
                io_bit_width=bit_width,
                gate_acc_bit_width=bit_width,
                sigmoid_bit_width=bit_width,
                tanh_bit_width=bit_width,
                weight_quant = weight_quant_type, 
                io_quant = act_quant_type,
                gate_acc_quant=act_quant_type,
                sigmoid_quant=sigmoid_act_quant_type,
                tanh_quant=act_quant_type,
                dropout=dropout)
        if concat_label != 'input':
            if one_hot:
                self.decoder = nn.Linear((nhid + nclass), ntoken, bias=False, )
            else:
                self.decoder = nn.Linear((nhid + nlabelembed), ntoken, bias=False)
        else:
            self.decoder = nn.Linear(nhid, ntoken, bias=False)
            
        self.nlayers = nlayers
        self.nhid = nhid
        self.loss = nn.CrossEntropyLoss()
        self.use_cuda = use_cuda
        self.nclass = nclass
        self.avg_loss = avg_loss
        self.use_bias = use_bias
        self.concat_label = concat_label
        self.init_weights()
        if tied:
            if nhid + nlabelembed != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

    def get_one_hot(self, batch):
        ones = torch.eye(self.nclass)
        if self.use_cuda:
            ones = ones.cuda()
        return ones.index_select(0, batch)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, x_pred, y_ext, hidden, criterion, is_infer=False, cal=False):
        self.forward_pass_counter += 1
        print(f"Forward pass count: {self.forward_pass_counter}")
        embedded_sents = self.encoder(x.data) #Embedding

        #print("Embedded Sents Size:", embedded_sents.size())

        embedded_label = self.label_encoder(y_ext.data)

        #print("Embedded label Size:", embedded_label.size())

        if self.concat_label != 'hidden':
            embedded_sents = torch.cat((embedded_sents, embedded_label), 1)
            #print("Embedded Sents Size after cat:", embedded_sents.size())

        embedded_sents = nn.utils.rnn.pad_sequence(embedded_sents, batch_first=True)
        #print("Embedded Sents Size after packing:", type(embedded_sents), embedded_sents.size())

        #print("hidden size:", hidden.size())

        output, hidden = self.rnn(embedded_sents)  # Adjusted to handle RNN output

        #print("output of rnn:", output.data.size())

        first_part = output.data[:, 0, :]
        last_part = output.data[:, -1, :]
        
        # Now concatenate them along the second dimension
        result = (first_part + last_part) / 2
        #result = result.view(-1, 100)
        #print("Result Size:", result.size())

        if self.concat_label != 'input':
            if cal: 
                embedded_label = embedded_label[:, 0, :]
            else:
                embedded_label=embedded_label
            hidden_data = torch.cat((result, embedded_label), 1)
        else:
            hidden_data = result
        hidden_data = self.drop(hidden_data)

        out = self.decoder(hidden_data)
        if self.use_bias:
            out += self.bias_encoder(y_ext.data)

        if criterion is not None:
            loss = criterion(out, x_pred.data)
        else:
            loss = None

        if is_infer:
            if cal:
                return loss
            else:
                LM_loss = nn.utils.rnn.pad_packed_sequence(nn.utils.rnn.PackedSequence(
                loss, x.batch_sizes))[0].transpose(0, 1)
                total_loss = torch.sum(LM_loss, dim=1)
                return total_loss
        else:
            if self.avg_loss:
                return torch.mean(loss)
            else:
                return torch.sum(loss)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, bsz, self.nhid)  # Simplified for RNN


class GenQLSTM_Others(nn.Module):
    def __init__(self, ntoken, ninp, nlabelembed, nhid, nlayers, nclass, dropout, use_cuda, 
        tied, use_bias, concat_label, avg_loss, one_hot, bit_width):
        
        super(GenQLSTM_Others, self).__init__()

        self.forward_pass_counter = 0

        if bit_width==1:
            weight_quant_type = SignedBinaryWeightPerTensorConst
            act_quant_type = SignedBinaryActPerTensorConst
            sigmoid_act_quant_type = SignedBinaryActPerTensorConst
        elif bit_width==2:
            weight_quant_type = SignedTernaryWeightPerTensorConst
            act_quant_type = SignedTernaryActPerTensorConst
            sigmoid_act_quant_type = SignedTernaryActPerTensorConst
        else:
            weight_quant_type = Int8WeightPerTensorFixedPoint
            act_quant_type = Int8ActPerTensorFixedPoint
            sigmoid_act_quant_type = Uint8ActPerTensorFixedPoint
        print(weight_quant_type)
        print(act_quant_type)
        print(sigmoid_act_quant_type)
        
        self.drop = nn.Dropout(dropout)
        self.encoder = qnn.QuantEmbedding(ntoken, ninp, weight_quant=weight_quant_type, weight_bit_width=bit_width)
        if one_hot:  # one_hot
            self.label_encoder = self.get_one_hot
        else:
            self.label_encoder = qnn.QuantEmbedding(nclass, nlabelembed, weight_quant=weight_quant_type, weight_bit_width=bit_width)
        if use_bias:
            self.bias_encoder = qnn.QuantEmbedding(nclass, ntoken, weight_quant=weight_quant_type, weight_bit_width=bit_width)
        if concat_label != 'hidden':
            self.rnn = nn.LSTM(ninp + nlabelembed, nhid, nlayers, dropout=dropout)
        else:
            self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        if concat_label != 'input':
            if one_hot:
                self.decoder = qnn.QuantLinear((nhid + nclass), ntoken, bias=False, weight_quant = weight_quant_type, weight_bit_width=bit_width)
            else:
                self.decoder = qnn.QuantLinear((nhid + nlabelembed), ntoken, bias=False, weight_quant = weight_quant_type, weight_bit_width=bit_width)
        else:
            self.decoder = qnn.QuantLinear(nhid, ntoken, bias=False, weight_quant = weight_quant_type, weight_bit_width=bit_width)
        self.nlayers = nlayers
        self.nhid = nhid
        self.loss = nn.CrossEntropyLoss()
        self.use_cuda = use_cuda
        self.nclass = nclass
        self.avg_loss = avg_loss
        self.use_bias = use_bias
        self.concat_label = concat_label
        self.init_weights()
        if tied:
            if nhid + nlabelembed != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

    def get_one_hot(self, batch):
        ones = torch.eye(self.nclass)
        if self.use_cuda:
            ones = ones.cuda()
        return ones.index_select(0, batch)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, x_pred, y_ext, hidden, criterion, is_infer=False, cal=False):
        self.forward_pass_counter += 1
        print(f"Forward pass count: {self.forward_pass_counter}")
        embedded_sents = self.encoder(x.data) #Embedding

        #print("Embedded Sents Size:", embedded_sents.size())

        embedded_label = self.label_encoder(y_ext.data)

        #print("Embedded label Size:", embedded_label.size())

        if self.concat_label != 'hidden':
            embedded_sents = torch.cat((embedded_sents, embedded_label), 1)
            #print("Embedded Sents Size after cat:", embedded_sents.size())

        embedded_sents = nn.utils.rnn.pad_sequence(embedded_sents, batch_first=True)
        #print("Embedded Sents Size after packing:", type(embedded_sents), embedded_sents.size())

        #print("hidden size:", hidden.size())

        output, hidden = self.rnn(embedded_sents)  # Adjusted to handle RNN output

        #print("output of rnn:", output.data.size())

        first_part = output.data[:, 0, :]
        last_part = output.data[:, -1, :]
        
        # Now concatenate them along the second dimension
        result = (first_part + last_part) / 2
        #result = result.view(-1, 100)
        #print("Result Size:", result.size())

        if self.concat_label != 'input':
            if cal: 
                embedded_label = embedded_label[:, 0, :]
            else:
                embedded_label=embedded_label
            hidden_data = torch.cat((result, embedded_label), 1)
        else:
            hidden_data = result
        hidden_data = self.drop(hidden_data)

        out = self.decoder(hidden_data)
        if self.use_bias:
            out += self.bias_encoder(y_ext.data)

        if criterion is not None:
            loss = criterion(out, x_pred.data)
        else:
            loss = None

        if is_infer:
            if cal:
                return loss
            else:
                LM_loss = nn.utils.rnn.pad_packed_sequence(nn.utils.rnn.PackedSequence(
                loss, x.batch_sizes))[0].transpose(0, 1)
                total_loss = torch.sum(LM_loss, dim=1)
                return total_loss
        else:
            if self.avg_loss:
                return torch.mean(loss)
            else:
                return torch.sum(loss)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, bsz, self.nhid)  # Simplified for RNN


class GenQLSTM_2bit(nn.Module):
    def __init__(self, ntoken, ninp, nlabelembed, nhid, nlayers, nclass, dropout, use_cuda, 
        tied, use_bias, concat_label, avg_loss, one_hot, bit_width):
        
        super(GenQLSTM_2bit, self).__init__()

        self.forward_pass_counter = 0
        self.drop = nn.Dropout(dropout)
        self.encoder = qnn.QuantEmbedding(ntoken, ninp, weight_quant=SignedTernaryWeightPerTensorConst, weight_bit_width=bit_width)
        if one_hot:  # one_hot
            self.label_encoder = self.get_one_hot
        else:
            self.label_encoder = qnn.QuantEmbedding(nclass, nlabelembed, weight_quant=SignedTernaryWeightPerTensorConst, weight_bit_width=bit_width)
        if use_bias:
            self.bias_encoder = qnn.QuantEmbedding(nclass, ntoken, weight_quant=SignedTernaryWeightPerTensorConst, weight_bit_width=bit_width)
        if concat_label != 'hidden':
            self.rnn = qnn.QuantLSTM(
                ninp + nlabelembed, 
                nhid, 
                nlayers,
                bidirectional=False, 
                weight_bit_width=bit_width,
                io_bit_width=bit_width,
                gate_acc_bit_width=bit_width,
                sigmoid_bit_width=bit_width,
                tanh_bit_width=bit_width,
                weight_quant = SignedTernaryWeightPerTensorConst, 
                io_quant = SignedTernaryActPerTensorConst,
                gate_acc_quant=SignedTernaryActPerTensorConst,
                sigmoid_quant=SignedTernaryActPerTensorConst,
                tanh_quant=SignedTernaryActPerTensorConst,
                dropout=dropout)
        else:
            self.rnn = qnn.QuantLSTM(
                ninp, 
                nhid, 
                nlayers, 
                bidirectional=False, 
                weight_bit_width=bit_width,
                io_bit_width=bit_width,
                gate_acc_bit_width=bit_width,
                sigmoid_bit_width=bit_width,
                tanh_bit_width=bit_width,
                weight_quant = SignedTernaryWeightPerTensorConst, 
                io_quant = SignedTernaryActPerTensorConst,
                gate_acc_quant=SignedTernaryActPerTensorConst,
                sigmoid_quant=SignedTernaryActPerTensorConst,
                tanh_quant=SignedTernaryActPerTensorConst,
                dropout=dropout)
        if concat_label != 'input':
            if one_hot:
                self.decoder = qnn.QuantLinear((nhid + nclass), ntoken, bias=False, weight_quant = SignedTernaryWeightPerTensorConst, weight_bit_width=bit_width)
            else:
                self.decoder = qnn.QuantLinear((nhid + nlabelembed), ntoken, bias=False, weight_quant = SignedTernaryWeightPerTensorConst, weight_bit_width=bit_width)
        else:
            self.decoder = qnn.QuantLinear(nhid, ntoken, bias=False, weight_quant = SignedTernaryWeightPerTensorConst, weight_bit_width=bit_width)
        self.nlayers = nlayers
        self.nhid = nhid
        self.loss = nn.CrossEntropyLoss()
        self.use_cuda = use_cuda
        self.nclass = nclass
        self.avg_loss = avg_loss
        self.use_bias = use_bias
        self.concat_label = concat_label
        self.init_weights()
        if tied:
            if nhid + nlabelembed != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

    def get_one_hot(self, batch):
        ones = torch.eye(self.nclass)
        if self.use_cuda:
            ones = ones.cuda()
        return ones.index_select(0, batch)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, x_pred, y_ext, hidden, criterion, is_infer=False, cal=False):
        self.forward_pass_counter += 1
        print(f"Forward pass count: {self.forward_pass_counter}")
        embedded_sents = self.encoder(x.data) #Embedding

        #print("Embedded Sents Size:", embedded_sents.size())

        embedded_label = self.label_encoder(y_ext.data)

        #print("Embedded label Size:", embedded_label.size())

        if self.concat_label != 'hidden':
            embedded_sents = torch.cat((embedded_sents, embedded_label), 1)
            #print("Embedded Sents Size after cat:", embedded_sents.size())

        embedded_sents = nn.utils.rnn.pad_sequence(embedded_sents, batch_first=True)
        #print("Embedded Sents Size after packing:", type(embedded_sents), embedded_sents.size())

        #print("hidden size:", hidden.size())

        output, hidden = self.rnn(embedded_sents)  # Adjusted to handle RNN output

        #print("output of rnn:", output.data.size())

        first_part = output.data[:, 0, :]
        last_part = output.data[:, -1, :]
        
        # Now concatenate them along the second dimension
        result = (first_part + last_part) / 2
        #result = result.view(-1, 100)
        #print("Result Size:", result.size())

        if self.concat_label != 'input':
            if cal: 
                embedded_label = embedded_label[:, 0, :]
            else:
                embedded_label=embedded_label
            hidden_data = torch.cat((result, embedded_label), 1)
        else:
            hidden_data = result
        hidden_data = self.drop(hidden_data)

        out = self.decoder(hidden_data)
        if self.use_bias:
            out += self.bias_encoder(y_ext.data)

        if criterion is not None:
            loss = criterion(out, x_pred.data)
        else:
            loss = None

        if is_infer:
            if cal:
                return loss
            else:
                LM_loss = nn.utils.rnn.pad_packed_sequence(nn.utils.rnn.PackedSequence(
                loss, x.batch_sizes))[0].transpose(0, 1)
                total_loss = torch.sum(LM_loss, dim=1)
                return total_loss
        else:
            if self.avg_loss:
                return torch.mean(loss)
            else:
                return torch.sum(loss)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, bsz, self.nhid)  # Simplified for RNN


class GenQLSTM_1bit(nn.Module):
    def __init__(self, ntoken, ninp, nlabelembed, nhid, nlayers, nclass, dropout, use_cuda, 
        tied, use_bias, concat_label, avg_loss, one_hot, bit_width):
        
        super(GenQLSTM_1bit, self).__init__()

        self.forward_pass_counter = 0
        self.drop = nn.Dropout(dropout)
        self.encoder = qnn.QuantEmbedding(ntoken, ninp, weight_quant=SignedBinaryWeightPerTensorConst, weight_bit_width=bit_width)
        if one_hot:  # one_hot
            self.label_encoder = self.get_one_hot
        else:
            self.label_encoder = qnn.QuantEmbedding(nclass, nlabelembed, weight_quant=SignedBinaryWeightPerTensorConst, weight_bit_width=bit_width)
        if use_bias:
            self.bias_encoder = qnn.QuantEmbedding(nclass, ntoken, weight_quant=SignedBinaryWeightPerTensorConst, weight_bit_width=bit_width)
        if concat_label != 'hidden':
            self.rnn = qnn.QuantLSTM(
                ninp + nlabelembed, 
                nhid, 
                nlayers,
                bidirectional=False, 
                weight_bit_width=bit_width,
                io_bit_width=bit_width,
                gate_acc_bit_width=bit_width,
                sigmoid_bit_width=bit_width,
                tanh_bit_width=bit_width,
                weight_quant = SignedBinaryWeightPerTensorConst, 
                io_quant = SignedBinaryActPerTensorConst,
                gate_acc_quant=SignedBinaryActPerTensorConst,
                sigmoid_quant=SignedBinaryActPerTensorConst,
                tanh_quant=SignedBinaryActPerTensorConst,
                dropout=dropout)
        else:
            self.rnn = qnn.QuantLSTM(
                ninp, 
                nhid, 
                nlayers, 
                bidirectional=False, 
                weight_bit_width=bit_width,
                io_bit_width=bit_width,
                gate_acc_bit_width=bit_width,
                sigmoid_bit_width=bit_width,
                tanh_bit_width=bit_width,
                weight_quant = SignedBinaryWeightPerTensorConst, 
                io_quant = SignedBinaryActPerTensorConst,
                gate_acc_quant=SignedBinaryActPerTensorConst,
                sigmoid_quant=SignedBinaryActPerTensorConst,
                tanh_quant=SignedBinaryActPerTensorConst,
                dropout=dropout)
        if concat_label != 'input':
            if one_hot:
                self.decoder = qnn.QuantLinear((nhid + nclass), ntoken, bias=False, weight_quant = SignedBinaryWeightPerTensorConst, weight_bit_width=bit_width)
            else:
                self.decoder = qnn.QuantLinear((nhid + nlabelembed), ntoken, bias=False, weight_quant = SignedBinaryWeightPerTensorConst, weight_bit_width=bit_width)
        else:
            self.decoder = qnn.QuantLinear(nhid, ntoken, bias=False, weight_quant = SignedBinaryWeightPerTensorConst, weight_bit_width=bit_width)
        self.nlayers = nlayers
        self.nhid = nhid
        self.loss = nn.CrossEntropyLoss()
        self.use_cuda = use_cuda
        self.nclass = nclass
        self.avg_loss = avg_loss
        self.use_bias = use_bias
        self.concat_label = concat_label
        self.init_weights()
        if tied:
            if nhid + nlabelembed != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

    def get_one_hot(self, batch):
        ones = torch.eye(self.nclass)
        if self.use_cuda:
            ones = ones.cuda()
        return ones.index_select(0, batch)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, x_pred, y_ext, hidden, criterion, is_infer=False, cal=False):
        self.forward_pass_counter += 1
        print(f"Forward pass count: {self.forward_pass_counter}")
        embedded_sents = self.encoder(x.data) #Embedding

        #print("Embedded Sents Size:", embedded_sents.size())

        embedded_label = self.label_encoder(y_ext.data)

        #print("Embedded label Size:", embedded_label.size())

        if self.concat_label != 'hidden':
            embedded_sents = torch.cat((embedded_sents, embedded_label), 1)
            #print("Embedded Sents Size after cat:", embedded_sents.size())

        embedded_sents = nn.utils.rnn.pad_sequence(embedded_sents, batch_first=True)
        #print("Embedded Sents Size after packing:", type(embedded_sents), embedded_sents.size())

        #print("hidden size:", hidden.size())

        output, hidden = self.rnn(embedded_sents)  # Adjusted to handle RNN output

        #print("output of rnn:", output.data.size())

        first_part = output.data[:, 0, :]
        last_part = output.data[:, -1, :]
        
        # Now concatenate them along the second dimension
        result = (first_part + last_part) / 2
        #result = result.view(-1, 100)
        #print("Result Size:", result.size())

        if self.concat_label != 'input':
            if cal: 
                embedded_label = embedded_label[:, 0, :]
            else:
                embedded_label=embedded_label
            hidden_data = torch.cat((result, embedded_label), 1)
        else:
            hidden_data = result
        hidden_data = self.drop(hidden_data)

        out = self.decoder(hidden_data)
        if self.use_bias:
            out += self.bias_encoder(y_ext.data)

        if criterion is not None:
            loss = criterion(out, x_pred.data)
        else:
            loss = None

        if is_infer:
            if cal:
                return loss
            else:
                LM_loss = nn.utils.rnn.pad_packed_sequence(nn.utils.rnn.PackedSequence(
                loss, x.batch_sizes))[0].transpose(0, 1)
                total_loss = torch.sum(LM_loss, dim=1)
                return total_loss
        else:
            if self.avg_loss:
                return torch.mean(loss)
            else:
                return torch.sum(loss)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, bsz, self.nhid)  # Simplified for RNN


class ModelBase(nn.Module):
    def __init__(self, ntoken, word_emb_dim, label_emb_dim, c_emb_dim, nhid, nlayers, nclass, n_c, dropout, use_EM, infer_method, use_cuda, one_hot, bilinear):
        super(ModelBase, self).__init__()

        self.nlayers = nlayers
        self.use_cuda = use_cuda
        self.nhid = nhid
        self.n_c = n_c

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, word_emb_dim)
        self.rnn = nn.LSTM(word_emb_dim, nhid, nlayers, dropout=dropout)
        self.condition_encoder = nn.Embedding(n_c, c_emb_dim)

        self.use_EM = use_EM
        self.infer_method = infer_method
        self.bilinear = bilinear
        self.initrange = 0.1
        

    @staticmethod
    def get_model(mode, ntoken, word_emb_dim, label_emb_dim, c_emb_dim, nhid, nlayers, nclass, n_c, dropout, use_EM, infer_method, use_cuda, one_hot, bilinear):
        if mode == 'joint':
            return JointLatentModel(ntoken, word_emb_dim, label_emb_dim, c_emb_dim, nhid, nlayers, nclass, n_c, dropout, use_EM, infer_method, use_cuda, one_hot, bilinear)
        elif mode == 'auxiliary':
            return AuxiliaryLatentModel(ntoken, word_emb_dim, label_emb_dim, c_emb_dim, nhid, nlayers, nclass, n_c, dropout, use_EM, infer_method, use_cuda, one_hot, bilinear)
        elif mode == 'middle':
            return MiddleLatentModel(ntoken, word_emb_dim, label_emb_dim, c_emb_dim, nhid, nlayers, nclass, n_c, dropout, use_EM, infer_method, use_cuda, one_hot, bilinear)
        elif mode == 'hierarchy':
            return HierarchyLatentModel(ntoken, word_emb_dim, label_emb_dim, c_emb_dim, nhid, nlayers, nclass, n_c, dropout, use_EM, infer_method, use_cuda, one_hot, bilinear)
    def init_weights(self):
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.condition_encoder.weight.data.uniform_(-self.initrange, self.initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))

class HierarchyLatentModel(ModelBase):
    def __init__(self, ntoken, word_emb_dim, label_emb_dim, c_emb_dim, nhid, nlayers, nclass, n_c, dropout, use_EM, infer_method, use_cuda, one_hot, bilinear):
        super(HierarchyLatentModel, self).__init__(ntoken, word_emb_dim, label_emb_dim, c_emb_dim, nhid, nlayers, nclass, n_c, dropout, use_EM, infer_method, use_cuda, one_hot, bilinear)

        self.label_encoder = nn.Embedding(nclass, label_emb_dim)
        self.decoder = nn.Linear(nhid + label_emb_dim + c_emb_dim, ntoken)
        self.c_y_score = nn.Linear(c_emb_dim, nclass)
        self.init_weights()

    def init_weights(self):
        super(HierarchyLatentModel, self).init_weights()
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.label_encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.c_y_score.weight.data.uniform_(-self.initrange, self.initrange)

    def forward(self, sent, y_label, hidden, criterion, is_infer = False, is_interpret = False, is_generare=False):
        x = nn.utils.rnn.pack_sequence([s[:-1] for s in sent])
        x_pred = nn.utils.rnn.pack_sequence([s[1:] for s in sent])

        embedded_sents = self.encoder(x.data)
        embedded_sents = nn.utils.rnn.PackedSequence(embedded_sents, x.batch_sizes)
        output, (_, _) = self.rnn(embedded_sents, hidden)
        
        losses = []

        for c_label in range(self.n_c):
            c_ext = []
            y_ext = []
            for d, y in zip(sent, y_label):
                c_ext.append(torch.LongTensor([c_label] * (len(d) - 1)))
                y_ext.append(torch.LongTensor([y] * (len(d) - 1)))
            c_ext = nn.utils.rnn.pack_sequence(c_ext)
            y_ext = nn.utils.rnn.pack_sequence(y_ext)

            if self.use_cuda:
                c_ext = c_ext.cuda()
                y_ext = y_ext.cuda()
            embedded_c_s = self.condition_encoder(c_ext.data)
            embedded_y_s = self.label_encoder(y_ext.data)

            hidden_data = torch.cat((output.data, embedded_c_s, embedded_y_s), 1)
            hidden_data = self.drop(hidden_data)
            out = self.decoder(hidden_data)
            loss = criterion(out, x_pred.data)
            loss = nn.utils.rnn.pad_packed_sequence(nn.utils.rnn.PackedSequence(loss, x.batch_sizes))[0].transpose(0,1)

            # batch_size * seq_len
            loss = torch.sum(loss, dim=1)
            losses.append(-loss)

        logp_x_cy = torch.cat(losses, dim=0).view(-1, len(sent))

        logp_c_y = F.log_softmax(self.c_y_score(self.condition_encoder.weight), dim=0)
        y_label = y_label.unsqueeze(0).repeat(self.n_c,1).view(self.n_c, -1)

        if self.use_cuda:
            y_label = y_label.cuda()
        logp_c_y = logp_c_y.gather(1, y_label)

        logsums = logp_x_cy + logp_c_y

        if is_interpret:
            # given groundtruth x, y, argmax_c p(c, x|y)
            # given x, argmax_c argmax_y p(x|y)
            # given x, argmax_c sum_y p(x|y)

            c = torch.argmax(logp_x_cy + logp_c_y, dim = 0)
            logq_c_xy = logsums - torch.logsumexp(logsums, 0)
            return c, torch.exp(logq_c_xy)

        if self.use_EM:
            # EM has different inference and training objective
            if is_infer:
                final_loss = - torch.logsumexp(logsums, 0)
            else:
                with torch.no_grad():
                    # #_c * batch_size 
                    logq_c_xy = logsums - torch.logsumexp(logsums, 0)
                final_loss = - torch.sum(logsums * torch.exp(logq_c_xy), 0)          
        else:
            # can use another inference method, instead of marginalize over c, we can use argmax_c
            final_loss = - torch.logsumexp(logsums, 0)

        return final_loss

class MiddleLatentModel(ModelBase):
    def __init__(self, ntoken, word_emb_dim, label_emb_dim, c_emb_dim, nhid, nlayers, nclass, n_c, dropout, use_EM, infer_method, use_cuda, one_hot, bilinear):
        super(MiddleLatentModel, self).__init__(ntoken, word_emb_dim, label_emb_dim, c_emb_dim, nhid, nlayers, nclass, n_c, dropout, use_EM, infer_method, use_cuda, one_hot, bilinear)

        self.decoder = nn.Linear(nhid + c_emb_dim, ntoken)


        if not self.bilinear:
            self.c_y_score = nn.Linear(c_emb_dim, nclass)
        else:
            self.label_encoder = nn.Embedding(nclass, label_emb_dim)
            self.bilinear_weight = nn.Parameter(torch.FloatTensor(label_emb_dim, c_emb_dim).uniform_(-0.1, 0.1))
            
        self.init_weights()

    def init_weights(self):
        super(MiddleLatentModel, self).init_weights()
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        if not self.bilinear:
            self.c_y_score.weight.data.uniform_(-self.initrange, self.initrange)
        else:
            self.label_encoder.weight.data.uniform_(-self.initrange, self.initrange)

    def forward(self, sent, y_label, hidden, criterion, is_infer = False, is_interpret = False, is_generare=False):
        x = nn.utils.rnn.pack_sequence([s[:-1] for s in sent])
        x_pred = nn.utils.rnn.pack_sequence([s[1:] for s in sent])

        embedded_sents = self.encoder(x.data)
        embedded_sents = nn.utils.rnn.PackedSequence(embedded_sents, x.batch_sizes)
        output, (_, _) = self.rnn(embedded_sents, hidden)
        
        losses = []

        for c_label in range(self.n_c):
            c_ext = []
            for d, y in zip(sent, y_label):
                c_ext.append(torch.LongTensor([c_label] * (len(d) - 1)))
            c_ext = nn.utils.rnn.pack_sequence(c_ext)

            if self.use_cuda:
                c_ext = c_ext.cuda()
            embedded_c_s = self.condition_encoder(c_ext.data)

            hidden_data = torch.cat((output.data, embedded_c_s), 1)
            hidden_data = self.drop(hidden_data)
            out = self.decoder(hidden_data)
            loss = criterion(out, x_pred.data)
            loss = nn.utils.rnn.pad_packed_sequence(nn.utils.rnn.PackedSequence(loss, x.batch_sizes))[0].transpose(0,1)

            # batch_size * seq_len
            loss = torch.sum(loss, dim=1)
            losses.append(-loss)

        logp_x_c = torch.cat(losses, dim=0).view(-1, len(sent))


        if not self.bilinear:
            logp_c_y = F.log_softmax(self.c_y_score(self.condition_encoder.weight), dim=0)
        else:
            mid_value = torch.matmul(self.condition_encoder.weight, self.bilinear_weight.transpose(0,1))
            ycs = torch.matmul(self.label_encoder.weight, mid_value.transpose(0,1))
            logp_c_y = torch.log_softmax(ycs, dim=0).transpose(0,1)

        y_label = y_label.unsqueeze(0).repeat(self.n_c,1).view(self.n_c,-1)
        if self.use_cuda:
            y_label = y_label.cuda()
        logp_c_y = logp_c_y.gather(1, y_label)


        if is_interpret:
            # given groundtruth x, y, argmax_c p(c, x|y)
            # given x, argmax_c argmax_y p(x|y)
            # given x, argmax_c sum_y p(x|y)
            logsums = logp_c_y + logp_x_c

            c = torch.argmax(logsums, dim = 0)
            logp_c_xy = logsums - torch.logsumexp(logsums, 0)
            return c, torch.exp(logp_c_xy).transpose(1,0), logsums


        final_loss = - torch.logsumexp((logp_x_c + logp_c_y), 0)

        return final_loss


class JointLatentModel(ModelBase):
    def __init__(self, ntoken, word_emb_dim, label_emb_dim, c_emb_dim, nhid, nlayers, nclass, n_c, dropout, use_EM, infer_method, use_cuda, one_hot, bilinear):
        super(JointLatentModel, self).__init__(ntoken, word_emb_dim, label_emb_dim, c_emb_dim, nhid, nlayers, nclass, n_c, dropout, use_EM, infer_method, use_cuda, one_hot, bilinear)

        if not self.bilinear:
            self.c_l_score = nn.Linear(c_emb_dim, nclass)
        else:
            self.label_encoder = nn.Embedding(nclass, label_emb_dim)
            self.bilinear_weight = nn.Parameter(torch.FloatTensor(label_emb_dim, c_emb_dim).uniform_(-0.1, 0.1))
        self.decoder = nn.Linear((nhid + c_emb_dim), ntoken)
        # self.c_score = nn.Parameter(torch.zeros(1, n_c, dtype=torch.float, requires_grad=True))
        self.c_score = nn.Linear(c_emb_dim, 1)

        self.init_weights()

    def init_weights(self):
        super(JointLatentModel, self).init_weights()
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)       
        self.c_score.weight.data.uniform_(-self.initrange, self.initrange)
        if not self.bilinear:
            self.c_l_score.weight.data.uniform_(-self.initrange, self.initrange)
        else:
            self.label_encoder.weight.data.uniform_(-self.initrange, self.initrange)

    def forward(self, sent, y_label, hidden, criterion, is_infer = False, is_interpret = False):
        x = nn.utils.rnn.pack_sequence([s[:-1] for s in sent])
        x_pred = nn.utils.rnn.pack_sequence([s[1:] for s in sent])

        embedded_sents = self.encoder(x.data)
        embedded_sents = nn.utils.rnn.PackedSequence(embedded_sents, x.batch_sizes)
        output, (_, _) = self.rnn(embedded_sents, hidden)
        
        losses = []

        for c_label in range(self.n_c):
            c_ext = []
            for d in sent:
                c_ext.append(torch.LongTensor([c_label] * (len(d) - 1)))
            c_ext = nn.utils.rnn.pack_sequence(c_ext)

            if self.use_cuda:
                c_ext = c_ext.cuda()
            embedded_c_s = self.condition_encoder(c_ext.data)
            hidden_data = torch.cat((output.data, embedded_c_s), 1)
            hidden_data = self.drop(hidden_data)
            out = self.decoder(hidden_data)
            loss = criterion(out, x_pred.data)
            loss = nn.utils.rnn.pad_packed_sequence(nn.utils.rnn.PackedSequence(loss, x.batch_sizes))[0].transpose(0,1)

            # batch_size * seq_len
            loss = torch.sum(loss, dim=1)
            losses.append(-loss)

        logp_x_c = torch.cat(losses, dim=0).view(-1, len(sent))

        logp_c = F.log_softmax(self.c_score(self.condition_encoder.weight), dim=0)
        logp_c = logp_c.repeat(1, len(sent))


        if not self.bilinear:
            logp_y_c = F.log_softmax(self.c_l_score(self.condition_encoder.weight), dim=1)
        else:
            mid_value = torch.matmul(self.condition_encoder.weight, self.bilinear_weight.transpose(0,1))
            ycs = torch.matmul(self.label_encoder.weight, mid_value.transpose(0,1))
            logp_y_c = torch.log_softmax(ycs, dim=1).transpose(0,1)

        y_label = y_label.unsqueeze(0).repeat(self.n_c,1)
        if self.use_cuda:
            y_label = y_label.cuda()
        logp_y_c = logp_y_c.gather(1, y_label.view(self.n_c,-1))

        logsums = logp_x_c + logp_y_c + logp_c


        if is_interpret:
            # given groundtruth x, y, argmax_c p(c, x|y)
            # given x, argmax_c argmax_y p(x|y)
            # given x, argmax_c sum_y p(x|y)
            c = torch.argmax(logsums, dim = 0)
            logp_c_xy = logsums - torch.logsumexp(logsums, 0)
            return c, torch.exp(logp_c_xy).transpose(1,0), logsums

        if is_infer:
            final_loss = - torch.logsumexp(logsums, 0)
        else:
            with torch.no_grad():
                logp_xy = torch.logsumexp(logsums, 0)
                # #_c * batch_size 
                logq_c_xy = logsums - logp_xy
            final_loss = - torch.sum(logsums * torch.exp(logq_c_xy), 0)
        return final_loss


class AuxiliaryLatentModel(ModelBase):
    def __init__(self, ntoken, word_emb_dim, label_emb_dim, c_emb_dim, nhid, nlayers, nclass, n_c, dropout, use_EM, infer_method, use_cuda, one_hot, bilinear):
        super(AuxiliaryLatentModel, self).__init__(ntoken, word_emb_dim, label_emb_dim, c_emb_dim, nhid, nlayers, nclass, n_c, dropout, use_EM, infer_method, use_cuda, one_hot, bilinear)

        if one_hot: # one_hot
            self.label_encoder = self.get_one_hot
            self.decoder = nn.Linear(nhid + nclass + c_emb_dim, ntoken)
        else:           
            self.label_encoder = nn.Embedding(nclass, label_emb_dim)
            self.decoder = nn.Linear(nhid + label_emb_dim + c_emb_dim, ntoken)
        # self.c_score = nn.Parameter(torch.zeros(1, n_c, dtype=torch.float, requires_grad=True))
        self.c_score = nn.Linear(c_emb_dim, 1)
        self.one_hot = one_hot
        self.nclass = nclass
        self.init_weights()

    def init_weights(self):
        super(AuxiliaryLatentModel, self).init_weights()
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        if not self.one_hot:
            self.label_encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.c_score.weight.data.uniform_(-self.initrange, self.initrange)

    def get_one_hot(self, batch):
        ones = torch.eye(self.nclass)
        if self.use_cuda:
            ones = ones.cuda()
        return ones.index_select(0,batch)

    def forward(self, sent, y_label, hidden, criterion, is_infer = False, is_interpret = False, is_generare=False):

        if is_generare:
            generate_texts = []
            for c_label in range(self.n_c):
                hidden = self.init_hidden(1)

                input = torch.tensor([[2]], dtype=torch.long)
                if self.use_cuda:
                    input = input.cuda()
                generate_text = []
                for _ in range(80):
                    emb = self.encoder(input)
                    output, hidden = self.rnn(emb, hidden)
                    # print(output)

                    c_ext = torch.LongTensor([c_label])
                    y_ext = torch.LongTensor(y_label)

                    if self.use_cuda:
                        c_ext = c_ext.cuda()
                        y_ext = y_ext.cuda()

                    embedded_c_s = self.condition_encoder(c_ext)
                    embedded_y_s = self.label_encoder(y_ext)

                    hidden_data = torch.cat((output.squeeze(0), embedded_c_s, embedded_y_s), 1)
                    out = self.decoder(hidden_data)

                    word_weights = out.squeeze().div(0.7).exp().cpu()

                    word_idx = torch.argmax(out,dim=1)
                    word_idx = torch.multinomial(word_weights, 1).view(out.size(0))

                    input.fill_(word_idx[0])
                
                    generate_text.append(word_idx.tolist()[0])

                    if word_idx[0] == 3:
                        break
                generate_texts.append(generate_text)
            return generate_texts

        x = nn.utils.rnn.pack_sequence([s[:-1] for s in sent])
        x_pred = nn.utils.rnn.pack_sequence([s[1:] for s in sent])

        embedded_sents = self.encoder(x.data)
        embedded_sents = nn.utils.rnn.PackedSequence(embedded_sents, x.batch_sizes)
        output, hidden = self.rnn(embedded_sents, hidden)

        losses = []
        for c_label in range(self.n_c):
            c_ext = []
            y_ext = []
            for d, y in zip(sent, y_label):
                c_ext.append(torch.LongTensor([c_label] * (len(d) - 1)))
                y_ext.append(torch.LongTensor([y] * (len(d) - 1)))
            c_ext = nn.utils.rnn.pack_sequence(c_ext)
            y_ext = nn.utils.rnn.pack_sequence(y_ext)

            if self.use_cuda:
                c_ext = c_ext.cuda()
                y_ext = y_ext.cuda()
            embedded_c_s = self.condition_encoder(c_ext.data)
            embedded_y_s = self.label_encoder(y_ext.data)

            hidden_data = torch.cat((output.data, embedded_c_s, embedded_y_s), 1)
            hidden_data = self.drop(hidden_data)
            out = self.decoder(hidden_data)

            loss = criterion(out, x_pred.data)
            loss = nn.utils.rnn.pad_packed_sequence(nn.utils.rnn.PackedSequence(loss, x.batch_sizes))[0].transpose(0,1)

            # batch_size * seq_len
            loss = torch.sum(loss, dim=1)
            losses.append(-loss)

        logp_x_cy = torch.cat(losses, dim=0).view(-1, len(sent))

        logp_c = F.log_softmax(self.c_score(self.condition_encoder.weight),
            dim=0)       
        logp_c = logp_c.repeat(1, len(sent))

        # if p(y) where y in Y is uniformly distributed, than p(x,c,y) can be treated as p(sum)
        logsums = logp_x_cy + logp_c

        if is_interpret:
            # given groundtruth x, y, argmax_c p(c, x|y)
            # given x, argmax_c argmax_y p(x|y)
            # given x, argmax_c sum_y p(x|y)

            c = torch.argmax(logp_x_cy + logp_c, dim = 0)
            logp_c_xy = logsums - torch.logsumexp(logsums, 0)
            return c, torch.exp(logp_c_xy).transpose(1,0), logsums

        if self.use_EM:
            # EM has different inference and training objective
            if is_infer:
                final_loss = - torch.logsumexp(logsums, 0)
            else:
                with torch.no_grad():
                    # #_c * batch_size 
                    logq_c_xy = logsums - torch.logsumexp(logsums, 0)
                final_loss = - torch.sum(logsums * torch.exp(logq_c_xy), 0)          
        else:
            if is_infer:
                # can use another inference method, instead of marginalize over c, we can use argmax_c
                if self.infer_method == 'sum':
                    final_loss = - torch.logsumexp(logsums, 0)
                elif self.infer_method == 'max':
                    final_loss = torch.min(-logsums, dim=0)[0]
                else:
                    logq_c_xy = logsums - torch.logsumexp(logsums, 0)
                    final_loss = - torch.logsumexp(logp_x_cy + logq_c_xy, 0)

            else:
                final_loss = - torch.logsumexp(logsums, 0)
        return final_loss
