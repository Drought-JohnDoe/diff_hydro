import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .dropout import DropMask, createMask
from . import cnn
import csv
import numpy as np
import random
import warnings


class LSTMcell_untied(torch.nn.Module):
    def __init__(self,
                 *,
                 inputSize,
                 hiddenSize,
                 train=True,
                 dr=0.5,
                 drMethod='gal+sem',
                 gpu=0):
        super(LSTMcell_untied, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = inputSize
        self.dr = dr

        self.w_xi = Parameter(torch.Tensor(hiddenSize, inputSize))
        self.w_xf = Parameter(torch.Tensor(hiddenSize, inputSize))
        self.w_xo = Parameter(torch.Tensor(hiddenSize, inputSize))
        self.w_xc = Parameter(torch.Tensor(hiddenSize, inputSize))

        self.w_hi = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_hf = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_ho = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_hc = Parameter(torch.Tensor(hiddenSize, hiddenSize))

        self.b_i = Parameter(torch.Tensor(hiddenSize))
        self.b_f = Parameter(torch.Tensor(hiddenSize))
        self.b_o = Parameter(torch.Tensor(hiddenSize))
        self.b_c = Parameter(torch.Tensor(hiddenSize))

        self.drMethod = drMethod.split('+')
        self.gpu = gpu
        self.train = train
        if gpu >= 0:
            self = self.cuda(gpu)
            self.is_cuda = True
        else:
            self.is_cuda = False
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hiddenSize)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def init_mask(self, x, h, c):
        self.maskX_i = createMask(x, self.dr)
        self.maskX_f = createMask(x, self.dr)
        self.maskX_c = createMask(x, self.dr)
        self.maskX_o = createMask(x, self.dr)

        self.maskH_i = createMask(h, self.dr)
        self.maskH_f = createMask(h, self.dr)
        self.maskH_c = createMask(h, self.dr)
        self.maskH_o = createMask(h, self.dr)

        self.maskC = createMask(c, self.dr)

        self.maskW_xi = createMask(self.w_xi, self.dr)
        self.maskW_xf = createMask(self.w_xf, self.dr)
        self.maskW_xc = createMask(self.w_xc, self.dr)
        self.maskW_xo = createMask(self.w_xo, self.dr)
        self.maskW_hi = createMask(self.w_hi, self.dr)
        self.maskW_hf = createMask(self.w_hf, self.dr)
        self.maskW_hc = createMask(self.w_hc, self.dr)
        self.maskW_ho = createMask(self.w_ho, self.dr)

    def forward(self, x, hidden):
        h0, c0 = hidden
        doDrop = self.training and self.dr > 0.0

        if doDrop:
            self.init_mask(x, h0, c0)

        if doDrop and 'drH' in self.drMethod:
            h0_i = h0.mul(self.maskH_i)
            h0_f = h0.mul(self.maskH_f)
            h0_c = h0.mul(self.maskH_c)
            h0_o = h0.mul(self.maskH_o)
        else:
            h0_i = h0
            h0_f = h0
            h0_c = h0
            h0_o = h0

        if doDrop and 'drX' in self.drMethod:
            x_i = x.mul(self.maskX_i)
            x_f = x.mul(self.maskX_f)
            x_c = x.mul(self.maskX_c)
            x_o = x.mul(self.maskX_o)
        else:
            x_i = x
            x_f = x
            x_c = x
            x_o = x

        if doDrop and 'drW' in self.drMethod:
            w_xi = self.w_xi.mul(self.maskW_xi)
            w_xf = self.w_xf.mul(self.maskW_xf)
            w_xc = self.w_xc.mul(self.maskW_xc)
            w_xo = self.w_xo.mul(self.maskW_xo)
            w_hi = self.w_hi.mul(self.maskW_hi)
            w_hf = self.w_hf.mul(self.maskW_hf)
            w_hc = self.w_hc.mul(self.maskW_hc)
            w_ho = self.w_ho.mul(self.maskW_ho)
        else:
            w_xi = self.w_xi
            w_xf = self.w_xf
            w_xc = self.w_xc
            w_xo = self.w_xo
            w_hi = self.w_hi
            w_hf = self.w_hf
            w_hc = self.w_hc
            w_ho = self.w_ho

        gate_i = F.linear(x_i, w_xi) + F.linear(h0_i, w_hi) + self.b_i
        gate_f = F.linear(x_f, w_xf) + F.linear(h0_f, w_hf) + self.b_f
        gate_c = F.linear(x_c, w_xc) + F.linear(h0_c, w_hc) + self.b_c
        gate_o = F.linear(x_o, w_xo) + F.linear(h0_o, w_ho) + self.b_o

        gate_i = F.sigmoid(gate_i)
        gate_f = F.sigmoid(gate_f)
        gate_c = F.tanh(gate_c)
        gate_o = F.sigmoid(gate_o)

        if doDrop and 'drC' in self.drMethod:
            gate_c = gate_c.mul(self.maskC)

        c1 = (gate_f * c0) + (gate_i * gate_c)
        h1 = gate_o * F.tanh(c1)

        return h1, c1


class LSTMcell_tied(torch.nn.Module):
    def __init__(self,
                 *,
                 inputSize,
                 hiddenSize,
                 mode='train',
                 dr=0.5,
                 drMethod='drX+drW+drC',
                 gpu=1):
        super(LSTMcell_tied, self).__init__()

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.dr = dr

        self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
        self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
        self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
        self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))

        self.drMethod = drMethod.split('+')
        self.gpu = gpu
        self.mode = mode
        if mode == 'train':
            self.train(mode=True)
        elif mode == 'test':
            self.train(mode=False)
        elif mode == 'drMC':
            self.train(mode=False)

        if gpu >= 0:
            self = self.cuda()
            self.is_cuda = True
        else:
            self.is_cuda = False
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def reset_mask(self, x, h, c):
        self.maskX = createMask(x, self.dr)
        self.maskH = createMask(h, self.dr)
        self.maskC = createMask(c, self.dr)
        self.maskW_ih = createMask(self.w_ih, self.dr)
        self.maskW_hh = createMask(self.w_hh, self.dr)

    def forward(self, x, hidden, *, resetMask=True, doDropMC=False):
        if self.dr > 0 and (doDropMC is True or self.training is True):
            doDrop = True
        else:
            doDrop = False

        batchSize = x.size(0)
        h0, c0 = hidden
        if h0 is None:
            h0 = x.new_zeros(batchSize, self.hiddenSize, requires_grad=False)
        if c0 is None:
            c0 = x.new_zeros(batchSize, self.hiddenSize, requires_grad=False)

        if self.dr > 0 and self.training is True and resetMask is True:
            self.reset_mask(x, h0, c0)

        if doDrop is True and 'drH' in self.drMethod:
            h0 = DropMask.apply(h0, self.maskH, True)

        if doDrop is True and 'drX' in self.drMethod:
            x = DropMask.apply(x, self.maskX, True)

        if doDrop is True and 'drW' in self.drMethod:
            w_ih = DropMask.apply(self.w_ih, self.maskW_ih, True)
            w_hh = DropMask.apply(self.w_hh, self.maskW_hh, True)
        else:
            # self.w are parameters, while w are not
            w_ih = self.w_ih
            w_hh = self.w_hh

        gates = F.linear(x, w_ih, self.b_ih) + \
            F.linear(h0, w_hh, self.b_hh)
        gate_i, gate_f, gate_c, gate_o = gates.chunk(4, 1)

        gate_i = torch.sigmoid(gate_i)
        gate_f = torch.sigmoid(gate_f)
        gate_c = torch.tanh(gate_c)
        gate_o = torch.sigmoid(gate_o)

        if self.training is True and 'drC' in self.drMethod:
            gate_c = gate_c.mul(self.maskC)

        c1 = (gate_f * c0) + (gate_i * gate_c)
        h1 = gate_o * torch.tanh(c1)

        return h1, c1


class CudnnLstm(torch.nn.Module):
    def __init__(self, *, inputSize, hiddenSize, dr=0.5, drMethod='drW',
                 gpu=0):
        super(CudnnLstm, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.dr = dr
        self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
        self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
        self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
        self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]
        self.cuda()

        self.reset_mask()
        self.reset_parameters()

    def _apply(self, fn):
        ret = super(CudnnLstm, self)._apply(fn)
        return ret

    def __setstate__(self, d):
        super(CudnnLstm, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]

    def reset_mask(self):
        self.maskW_ih = createMask(self.w_ih, self.dr)
        self.maskW_hh = createMask(self.w_hh, self.dr)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, input, hx=None, cx=None, doDropMC=False, dropoutFalse=False):
        # dropoutFalse: it will ensure doDrop is false, unless doDropMC is true
        if dropoutFalse and (not doDropMC):
            doDrop = False
        elif self.dr > 0 and (doDropMC is True or self.training is True):
            doDrop = True
        else:
            doDrop = False

        batchSize = input.size(1)

        if hx is None:
            hx = input.new_zeros(
                1, batchSize, self.hiddenSize, requires_grad=False)
        if cx is None:
            cx = input.new_zeros(
                1, batchSize, self.hiddenSize, requires_grad=False)

        # cuDNN backend - disabled flat weight
        # handle = torch.backends.cudnn.get_handle()
        if doDrop is True:
            self.reset_mask()
            weight = [
                DropMask.apply(self.w_ih, self.maskW_ih, True),
                DropMask.apply(self.w_hh, self.maskW_hh, True), self.b_ih,
                self.b_hh
            ]
        else:
            weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

        # output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
            # input, weight, 4, None, hx, cx, torch.backends.cudnn.CUDNN_LSTM,
            # self.hiddenSize, 1, False, 0, self.training, False, (), None)
        if torch.__version__ < "1.8":
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input, weight, 4, None, hx, cx, 2,  # 2 means LSTM
                self.hiddenSize, 1, False, 0, self.training, False, (), None)
        else:
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input, weight, 4, None, hx, cx, 2,  # 2 means LSTM
                self.hiddenSize, 0, 1, False, 0, self.training, False, (), None)		
        return output, (hy, cy)

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights]
                for weights in self._all_weights]

class CNN1dkernel(torch.nn.Module):
    def __init__(self,
                 *,
                 ninchannel=1,
                 nkernel=3,
                 kernelSize=3,
                 stride=1,
                 padding=0):
        super(CNN1dkernel, self).__init__()
        self.cnn1d = torch.nn.Conv1d(
            in_channels=ninchannel,
            out_channels=nkernel,
            kernel_size=kernelSize,
            padding=padding,
            stride=stride,
        )

    def forward(self, x):
        output = F.relu(self.cnn1d(x))
        return output

class CudnnLstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CudnnLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        # self.drtest = torch.nn.Dropout(p=0.4)

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        x0 = F.relu(self.linearIn(x))
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC, dropoutFalse=dropoutFalse)
        # outLSTMdr = self.drtest(outLSTM)
        out = self.linearOut(outLSTM)
        return out


class SafeLstmModel(torch.nn.Module):
    """
    Runtime-safe LSTM wrapper used for dynamic parameter heads.

    This keeps the same high-level structure as ``CudnnLstmModel``
    (linearIn -> LSTM -> linearOut) but relies on standard PyTorch
    ``nn.LSTM`` instead of the legacy CUDA-only ``_cudnn_rnn`` path.
    """

    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(SafeLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.dr = dr
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = torch.nn.LSTM(
            input_size=hiddenSize,
            hidden_size=hiddenSize,
            num_layers=1,
            dropout=0.0)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self._force_cpu_fallback = False

    def _forward_impl(self, x, doDropMC=False, dropoutFalse=False):
        x0 = F.relu(self.linearIn(x))
        if self.dr > 0 and (doDropMC is True or self.training is True) and not dropoutFalse:
            x0 = F.dropout(x0, p=self.dr, training=True)
        outLSTM, _ = self.lstm(x0)
        out = self.linearOut(outLSTM)
        return out

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        if self._force_cpu_fallback:
            if next(self.parameters()).device.type != 'cpu':
                self.cpu()
            out = self._forward_impl(x.cpu(), doDropMC=doDropMC, dropoutFalse=dropoutFalse)
            return out.to(x.device)

        try:
            return self._forward_impl(x, doDropMC=doDropMC, dropoutFalse=dropoutFalse)
        except RuntimeError as err:
            msg = str(err).lower()
            if x.device.type == 'cuda' and ('cublas runtime error' in msg or 'cuda' in msg):
                self._force_cpu_fallback = True
                self.cpu()
                warnings.warn(
                    'SafeLstmModel falling back to CPU after CUDA linear/LSTM failure; '
                    'continuing training with CPU dynamic LG head.',
                    RuntimeWarning)
                out = self._forward_impl(x.cpu(), doDropMC=doDropMC, dropoutFalse=dropoutFalse)
                return out.to(x.device)
            raise


class CNN1dLSTMmodel(torch.nn.Module):
    def __init__(self, *, nx, ny, nobs, hiddenSize,
                 nkernel=(10,5), kernelSize=(3,3), stride=(2,1), dr=0.5, poolOpt=None):
        # two convolutional layer
        super(CNN1dLSTMmodel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.obs = nobs
        self.hiddenSize = hiddenSize
        nlayer = len(nkernel)
        self.features = nn.Sequential()
        ninchan = 1
        Lout = nobs
        for ii in range(nlayer):
            ConvLayer = CNN1dkernel(
                ninchannel=ninchan, nkernel=nkernel[ii], kernelSize=kernelSize[ii], stride=stride[ii])
            self.features.add_module('CnnLayer%d' % (ii + 1), ConvLayer)
            ninchan = nkernel[ii]
            Lout = cnn.calConvSize(lin=Lout, kernel=kernelSize[ii], stride=stride[ii])
            self.features.add_module('Relu%d' % (ii + 1), nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d' % (ii + 1), nn.MaxPool1d(poolOpt[ii]))
                Lout = cnn.calPoolSize(lin=Lout, kernel=poolOpt[ii])
        self.Ncnnout = int(Lout*nkernel[-1]) # total CNN feature number after convolution
        Nf = self.Ncnnout + nx
        self.linearIn = torch.nn.Linear(Nf, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, z, doDropMC=False):
        nt, ngrid, nobs = z.shape
        z = z.view(nt*ngrid, 1, nobs)
        z0 = self.features(z)
        # z0 = (ntime*ngrid) * nkernel * sizeafterconv
        z0 = z0.view(nt, ngrid, self.Ncnnout)
        x0 = torch.cat((x, z0), dim=2)
        x0 = F.relu(self.linearIn(x0))
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        out = self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out

class CNN1dLSTMInmodel(torch.nn.Module):
    # Directly add the CNN extracted features into LSTM inputSize
    def __init__(self, *, nx, ny, nobs, hiddenSize,
                 nkernel=(10,5), kernelSize=(3,3), stride=(2,1), dr=0.5, poolOpt=None, cnndr=0.0):
        # two convolutional layer
        super(CNN1dLSTMInmodel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.obs = nobs
        self.hiddenSize = hiddenSize
        nlayer = len(nkernel)
        self.features = nn.Sequential()
        ninchan = 1
        Lout = nobs
        for ii in range(nlayer):
            ConvLayer = CNN1dkernel(
                ninchannel=ninchan, nkernel=nkernel[ii], kernelSize=kernelSize[ii], stride=stride[ii])
            self.features.add_module('CnnLayer%d' % (ii + 1), ConvLayer)
            if cnndr != 0.0:
                self.features.add_module('dropout%d' % (ii + 1), nn.Dropout(p=cnndr))
            ninchan = nkernel[ii]
            Lout = cnn.calConvSize(lin=Lout, kernel=kernelSize[ii], stride=stride[ii])
            self.features.add_module('Relu%d' % (ii + 1), nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d' % (ii + 1), nn.MaxPool1d(poolOpt[ii]))
                Lout = cnn.calPoolSize(lin=Lout, kernel=poolOpt[ii])
        self.Ncnnout = int(Lout*nkernel[-1]) # total CNN feature number after convolution
        Nf = self.Ncnnout + hiddenSize
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=Nf, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, z, doDropMC=False):
        nt, ngrid, nobs = z.shape
        z = z.view(nt*ngrid, 1, nobs)
        z0 = self.features(z)
        # z0 = (ntime*ngrid) * nkernel * sizeafterconv
        z0 = z0.view(nt, ngrid, self.Ncnnout)
        x = F.relu(self.linearIn(x))
        x0 = torch.cat((x, z0), dim=2)
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        out = self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out

class CNN1dLCmodel(torch.nn.Module):
    # add the CNN extracted features into original LSTM input, then pass through linear layer
    def __init__(self, *, nx, ny, nobs, hiddenSize,
                 nkernel=(10,5), kernelSize=(3,3), stride=(2,1), dr=0.5, poolOpt=None, cnndr=0.0):
        # two convolutional layer
        super(CNN1dLCmodel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.obs = nobs
        self.hiddenSize = hiddenSize
        nlayer = len(nkernel)
        self.features = nn.Sequential()
        ninchan = 1 # need to modify the hardcode: 4 for smap and 1 for FDC
        Lout = nobs
        for ii in range(nlayer):
            ConvLayer = CNN1dkernel(
                ninchannel=ninchan, nkernel=nkernel[ii], kernelSize=kernelSize[ii], stride=stride[ii])
            self.features.add_module('CnnLayer%d' % (ii + 1), ConvLayer)
            if cnndr != 0.0:
                self.features.add_module('dropout%d' % (ii + 1), nn.Dropout(p=cnndr))
            ninchan = nkernel[ii]
            Lout = cnn.calConvSize(lin=Lout, kernel=kernelSize[ii], stride=stride[ii])
            self.features.add_module('Relu%d' % (ii + 1), nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d' % (ii + 1), nn.MaxPool1d(poolOpt[ii]))
                Lout = cnn.calPoolSize(lin=Lout, kernel=poolOpt[ii])
        self.Ncnnout = int(Lout*nkernel[-1]) # total CNN feature number after convolution
        Nf = self.Ncnnout + nx
        self.linearIn = torch.nn.Linear(Nf, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, z, doDropMC=False):
        # z = ngrid*nVar add a channel dimension
        ngrid = z.shape[0]
        rho, BS, Nvar = x.shape
        if len(z.shape) == 2: # for FDC, else 3 dimension for smap
            z = torch.unsqueeze(z, dim=1)
        z0 = self.features(z)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0 = z0.view(ngrid, self.Ncnnout).repeat(rho,1,1)
        x = torch.cat((x, z0), dim=2)
        x0 = F.relu(self.linearIn(x))
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        out = self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out

class CNN1dLCInmodel(torch.nn.Module):
    # Directly add the CNN extracted features into LSTM inputSize
    def __init__(self, *, nx, ny, nobs, hiddenSize,
                 nkernel=(10,5), kernelSize=(3,3), stride=(2,1), dr=0.5, poolOpt=None, cnndr=0.0):
        # two convolutional layer
        super(CNN1dLCInmodel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.obs = nobs
        self.hiddenSize = hiddenSize
        nlayer = len(nkernel)
        self.features = nn.Sequential()
        ninchan = 1
        Lout = nobs
        for ii in range(nlayer):
            ConvLayer = CNN1dkernel(
                ninchannel=ninchan, nkernel=nkernel[ii], kernelSize=kernelSize[ii], stride=stride[ii])
            self.features.add_module('CnnLayer%d' % (ii + 1), ConvLayer)
            if cnndr != 0.0:
                self.features.add_module('dropout%d' % (ii + 1), nn.Dropout(p=cnndr))
            ninchan = nkernel[ii]
            Lout = cnn.calConvSize(lin=Lout, kernel=kernelSize[ii], stride=stride[ii])
            self.features.add_module('Relu%d' % (ii + 1), nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d' % (ii + 1), nn.MaxPool1d(poolOpt[ii]))
                Lout = cnn.calPoolSize(lin=Lout, kernel=poolOpt[ii])
        self.Ncnnout = int(Lout*nkernel[-1]) # total CNN feature number after convolution
        Nf = self.Ncnnout + hiddenSize
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=Nf, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, z, doDropMC=False):
        # z = ngrid*nVar add a channel dimension
        ngrid, nobs = z.shape
        rho, BS, Nvar = x.shape
        z = torch.unsqueeze(z, dim=1)
        z0 = self.features(z)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0 = z0.view(ngrid, self.Ncnnout).repeat(rho,1,1)
        x = F.relu(self.linearIn(x))
        x0 = torch.cat((x, z0), dim=2)
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        out = self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out

class CudnnInvLstmModel(torch.nn.Module):
    # using cudnnLstm to extract features from SMAP observations
    def __init__(self, *, nx, ny, hiddenSize, ninv, nfea, hiddeninv, dr=0.5, drinv=0.5):
        # two LSTM
        super(CudnnInvLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ninv = ninv
        self.nfea = nfea
        self.hiddeninv = hiddeninv
        self.lstminv = CudnnLstmModel(
            nx=ninv, ny=nfea, hiddenSize=hiddeninv, dr=drinv)
        self.lstm = CudnnLstmModel(
            nx=nfea+nx, ny=ny, hiddenSize=hiddenSize, dr=dr)
        self.gpu = 1

    def forward(self, x, z, doDropMC=False):
        Gen = self.lstminv(z)
        dim = x.shape;
        nt = dim[0]
        invpara = Gen[-1, :, :].repeat(nt, 1, 1)
        x1 = torch.cat((x, invpara), dim=2)
        out = self.lstm(x1)
        # out = rho/time * batchsize * Ntargetvar
        return out


class LstmCloseModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, fillObs=True):
        super(LstmCloseModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx + 1, hiddenSize)
        # self.lstm = CudnnLstm(
        #     inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.lstm = LSTMcell_tied(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr, drMethod='drW')
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        self.fillObs = fillObs

    def forward(self, x, y=None):
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1).cuda()
        out = torch.zeros(nt, ngrid, self.ny).cuda()
        ht = None
        ct = None
        resetMask = True
        for t in range(nt):
            if self.fillObs is True:
                ytObs = y[t, :, :]
                mask = ytObs == ytObs
                yt[mask] = ytObs[mask]
            xt = torch.cat((x[t, :, :], yt), 1)
            x0 = F.relu(self.linearIn(xt))
            ht, ct = self.lstm(x0, hidden=(ht, ct), resetMask=resetMask)
            yt = self.linearOut(ht)
            resetMask = False
            out[t, :, :] = yt
        return out


class AnnModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize):
        super(AnnModel, self).__init__()
        self.hiddenSize = hiddenSize
        self.i2h = nn.Linear(nx, hiddenSize)
        self.h2h = nn.Linear(hiddenSize, hiddenSize)
        self.h2o = nn.Linear(hiddenSize, ny)
        self.ny = ny

    def forward(self, x, y=None):
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1).cuda()
        out = torch.zeros(nt, ngrid, self.ny).cuda()
        for t in range(nt):
            xt = x[t, :, :]
            ht = F.relu(self.i2h(xt))
            ht2 = self.h2h(ht)
            yt = self.h2o(ht2)
            out[t, :, :] = yt
        return out


class AnnCloseModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, fillObs=True):
        super(AnnCloseModel, self).__init__()
        self.hiddenSize = hiddenSize
        self.i2h = nn.Linear(nx + 1, hiddenSize)
        self.h2h = nn.Linear(hiddenSize, hiddenSize)
        self.h2o = nn.Linear(hiddenSize, ny)
        self.fillObs = fillObs
        self.ny = ny

    def forward(self, x, y=None):
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1).cuda()
        out = torch.zeros(nt, ngrid, self.ny).cuda()
        for t in range(nt):
            if self.fillObs is True:
                ytObs = y[t, :, :]
                mask = ytObs == ytObs
                yt[mask] = ytObs[mask]
            xt = torch.cat((x[t, :, :], yt), 1)
            ht = F.relu(self.i2h(xt))
            ht2 = self.h2h(ht)
            yt = self.h2o(ht2)
            out[t, :, :] = yt
        return out


class LstmCnnCond(nn.Module):
    def __init__(self,
                 *,
                 nx,
                 ny,
                 ct,
                 opt=1,
                 hiddenSize=64,
                 cnnSize=32,
                 cp1=(64, 3, 2),
                 cp2=(128, 5, 2),
                 dr=0.5):
        super(LstmCnnCond, self).__init__()

        # opt == 1: cnn output as initial state of LSTM (h0)
        # opt == 2: cnn output as additional output of LSTM
        # opt == 3: cnn output as constant input of LSTM

        if opt == 1:
            cnnSize = hiddenSize

        self.nx = nx
        self.ny = ny
        self.ct = ct
        self.ctRm = False
        self.hiddenSize = hiddenSize
        self.opt = opt

        self.cnn = cnn.Cnn1d(nx=nx, nt=ct, cnnSize=cnnSize, cp1=cp1, cp2=cp2)

        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        if opt == 3:
            self.linearIn = torch.nn.Linear(nx + cnnSize, hiddenSize)
        else:
            self.linearIn = torch.nn.Linear(nx, hiddenSize)
        if opt == 2:
            self.linearOut = torch.nn.Linear(hiddenSize + cnnSize, ny)
        else:
            self.linearOut = torch.nn.Linear(hiddenSize, ny)

    def forward(self, x, xc):
        # x- [nt,ngrid,nx]
        x1 = xc
        x1 = self.cnn(x1)
        x2 = x
        if self.opt == 1:
            x2 = F.relu(self.linearIn(x2))
            x2, (hn, cn) = self.lstm(x2, hx=x1[None, :, :])
            x2 = self.linearOut(x2)
        elif self.opt == 2:
            x1 = x1[None, :, :].repeat(x2.shape[0], 1, 1)
            x2 = F.relu(self.linearIn(x2))
            x2, (hn, cn) = self.lstm(x2)
            x2 = self.linearOut(torch.cat([x2, x1], 2))
        elif self.opt == 3:
            x1 = x1[None, :, :].repeat(x2.shape[0], 1, 1)
            x2 = torch.cat([x2, x1], 2)
            x2 = F.relu(self.linearIn(x2))
            x2, (hn, cn) = self.lstm(x2)
            x2 = self.linearOut(x2)

        return x2


class LstmCnnForcast(nn.Module):
    def __init__(self,
                 *,
                 nx,
                 ny,
                 ct,
                 opt=1,
                 hiddenSize=64,
                 cnnSize=32,
                 cp1=(64, 3, 2),
                 cp2=(128, 5, 2),
                 dr=0.5):
        super(LstmCnnForcast, self).__init__()

        if opt == 1:
            cnnSize = hiddenSize

        self.nx = nx
        self.ny = ny
        self.ct = ct
        self.ctRm = True
        self.hiddenSize = hiddenSize
        self.opt = opt
        self.cnnSize = cnnSize

        if opt == 1:
            self.cnn = cnn.Cnn1d(
                nx=nx + 1, nt=ct, cnnSize=cnnSize, cp1=cp1, cp2=cp2)
        if opt == 2:
            self.cnn = cnn.Cnn1d(
                nx=1, nt=ct, cnnSize=cnnSize, cp1=cp1, cp2=cp2)

        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearIn = torch.nn.Linear(nx + cnnSize, hiddenSize)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)

    def forward(self, x, y):
        # x- [nt,ngrid,nx]
        nt, ngrid, nx = x.shape
        ct = self.ct
        pt = nt - ct

        if self.opt == 1:
            x1 = torch.cat((y, x), dim=2)
        elif self.opt == 2:
            x1 = y

        x1out = torch.zeros([pt, ngrid, self.cnnSize]).cuda()
        for k in range(pt):
            x1out[k, :, :] = self.cnn(x1[k:k + ct, :, :])

        x2 = x[ct:nt, :, :]
        x2 = torch.cat([x2, x1out], 2)
        x2 = F.relu(self.linearIn(x2))
        x2, (hn, cn) = self.lstm(x2)
        x2 = self.linearOut(x2)

        return x2

class CudnnLstmModel_R2P(torch.nn.Module):
    pass

class CpuLstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CpuLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = LSTMcell_tied(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr, drMethod='drW', gpu=-1)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = -1

    def forward(self, x, doDropMC=False):
        # x0 = F.relu(self.linearIn(x))
        # outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        # out = self.linearOut(outLSTM)
        # return out
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1)
        out = torch.zeros(nt, ngrid, self.ny)
        ht = None
        ct = None
        resetMask = True
        for t in range(nt):
            xt = x[t, :, :]
            x0 = F.relu(self.linearIn(xt))
            ht, ct = self.lstm(x0, hidden=(ht, ct), resetMask=resetMask)
            yt = self.linearOut(ht)
            resetMask = False
            out[t, :, :] = yt
        return out


def UH_conv(x,UH,viewmode=1):
    # UH is a vector indicating the unit hydrograph
    # the convolved dimension will be the last dimension
    # UH convolution is
    # Q(t)=\integral(x(\tao)*UH(t-\tao))d\tao
    # conv1d does \integral(w(\tao)*x(t+\tao))d\tao
    # hence we flip the UH
    # https://programmer.group/pytorch-learning-conv1d-conv2d-and-conv3d.html
    # view
    # x: [batch, var, time]
    # UH:[batch, var, uhLen]
    # batch needs to be accommodated by channels and we make use of groups
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    # https://pytorch.org/docs/stable/nn.functional.html

    mm= x.shape; nb=mm[0]
    m = UH.shape[-1]
    padd = m-1
    if viewmode==1:
        xx = x.view([1,nb,mm[-1]])
        w  = UH.view([nb,1,m])
        groups = nb

    y = F.conv1d(xx, torch.flip(w,[2]), groups=groups, padding=padd, stride=1, bias=None)
    y=y[:,:,0:-padd]
    return y.view(mm)


def UH_gamma(a,b,lenF=10):
    # UH. a [time (same all time steps), batch, var]
    m = a.shape
    w = torch.zeros([lenF, m[1],m[2]], device=a.device, dtype=a.dtype)
    aa = F.relu(a[0:lenF,:,:]).view([lenF, m[1],m[2]])+0.1 # minimum 0.1. First dimension of a is repeat
    theta = F.relu(b[0:lenF,:,:]).view([lenF, m[1],m[2]])+0.5 # minimum 0.5
    t = torch.arange(0.5,lenF*1.0).view([lenF,1,1]).repeat([1,m[1],m[2]])
    t = t.to(device=aa.device, dtype=aa.dtype)
    denom = (aa.lgamma().exp())*(theta**aa)
    mid= t**(aa-1)
    right=torch.exp(-t/theta)
    w = 1/denom*mid*right
    w = w/w.sum(0) # scale to 1 for each UH

    return w

class SimpAnn(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize):
        super(SimpAnn, self).__init__()
        self.hiddenSize = hiddenSize
        self.i2h = nn.Linear(nx, hiddenSize)
        self.h2h = nn.Linear(hiddenSize, hiddenSize)
        self.h2o = nn.Linear(hiddenSize, ny)
        self.ny = ny

    def forward(self, x):
        ht = F.relu(self.i2h(x))
        ht2 = F.relu(self.h2h(ht))
        out = F.relu(self.h2o(ht2))
        return out


class HBVMul(torch.nn.Module):
    """Multi-component HBV model implemented in PyTorch by Dapeng Feng"""

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMul, self).__init__()

    def forward(self, x, parameters, mu, muwts, rtwts, bufftime=0, outstate=False, routOpt=False, comprout=False,
                corrwts=None, pcorr=None):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     corrwts:P correction weights; pcorr: P correction opt
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS = 1e-5 # keep the numerical calculation stable
        device = x.device
        dtype = x.dtype

        # Initialization for warm up states
        if bufftime > 0:
            with torch.no_grad():
                xinit = x[0:bufftime, :, :]
                initmodel = HBVMul()
                Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = initmodel(xinit, parameters, mu, muwts, rtwts,
                                                                      bufftime=0, outstate=True, routOpt=False, comprout=False,
                                                                      corrwts=corrwts, pcorr=pcorr)
        else:
            # Without warm-up bufftime=0, initialize state variables with zeros
            Ngrid = x.shape[1]
            SNOWPACK = torch.zeros([Ngrid, mu], dtype=dtype, device=device) + 0.001
            MELTWATER = torch.zeros([Ngrid, mu], dtype=dtype, device=device) + 0.001
            SM = torch.zeros([Ngrid, mu], dtype=dtype, device=device) + 0.001
            SUZ = torch.zeros([Ngrid, mu], dtype=dtype, device=device) + 0.001
            SLZ = torch.zeros([Ngrid, mu], dtype=dtype, device=device) + 0.001
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()

        P = x[bufftime:, :, 0]
        Nstep, Ngrid = P.size()
        if pcorr is not None:
            parPCORR = pcorr[0] + corrwts[:,0]*(pcorr[1]-pcorr[0])
            P = parPCORR.unsqueeze(0).repeat(Nstep, 1) * P
            # print('P corrected')

        Pm= P.unsqueeze(2).repeat(1,1,mu) # precip
        T = x[bufftime:, :, 1]
        Tm = T.unsqueeze(2).repeat(1,1,mu) # temperature
        ETpot = x[bufftime:, :, 2]
        ETpm = ETpot.unsqueeze(2).repeat(1,1,mu) # potential ET


        ## scale the parameters to real vales
        parascaLst = [[1,6], [50,1000], [0.05,0.9], [0.01,0.5], [0.001,0.2], [0.2,1],
                        [0,10], [0,100], [-2.5,2.5], [0.5,10], [0,0.1], [0,0.2]] # HBV para
        routscaLst = [[0,2.9], [0,6.5]] # routing para
        # dim of each para is [Nbasin*Ncomponent]
        parBETA = parascaLst[0][0] + parameters[:,0,:]*(parascaLst[0][1]-parascaLst[0][0])
        parFC = parascaLst[1][0] + parameters[:,1,:]*(parascaLst[1][1]-parascaLst[1][0])
        parK0 = parascaLst[2][0] + parameters[:,2,:]*(parascaLst[2][1]-parascaLst[2][0])
        parK1 = parascaLst[3][0] + parameters[:,3,:]*(parascaLst[3][1]-parascaLst[3][0])
        parK2 = parascaLst[4][0] + parameters[:,4,:]*(parascaLst[4][1]-parascaLst[4][0])
        parLP = parascaLst[5][0] + parameters[:,5,:]*(parascaLst[5][1]-parascaLst[5][0])
        parPERC = parascaLst[6][0] + parameters[:,6,:]*(parascaLst[6][1]-parascaLst[6][0])
        parUZL = parascaLst[7][0] + parameters[:,7,:]*(parascaLst[7][1]-parascaLst[7][0])
        parTT = parascaLst[8][0] + parameters[:,8,:]*(parascaLst[8][1]-parascaLst[8][0])
        parCFMAX = parascaLst[9][0] + parameters[:,9,:]*(parascaLst[9][1]-parascaLst[9][0])
        parCFR = parascaLst[10][0] + parameters[:,10,:]*(parascaLst[10][1]-parascaLst[10][0])
        parCWH = parascaLst[11][0] + parameters[:,11,:]*(parascaLst[11][1]-parascaLst[11][0])

        # Initialize time series of model variables
        Qsimmu = torch.zeros(Pm.size(), dtype=dtype, device=device) + 0.001
        ETmu = torch.zeros(Pm.size(), dtype=dtype, device=device) + 0.001

        # Output the three simulated components of total Q
        Qsimmu0 = torch.zeros(Pm.size(), dtype=dtype, device=device) + 0.001
        Qsimmu1 = torch.zeros(Pm.size(), dtype=dtype, device=device) + 0.001
        Qsimmu2 = torch.zeros(Pm.size(), dtype=dtype, device=device) + 0.001


        for t in range(Nstep):
            # Separate precipitation into liquid and solid components
            PRECIP = Pm[t, :, :]  # need to check later, seems repeating with line 52
            RAIN = torch.mul(PRECIP, (Tm[t, :, :] >= parTT).type(torch.float32))
            SNOW = torch.mul(PRECIP, (Tm[t, :, :] < parTT).type(torch.float32))

            # Snow
            SNOWPACK = SNOWPACK + SNOW
            melt = parCFMAX * (Tm[t, :, :] - parTT)
            melt = torch.clamp(melt, min=0.0)
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = parCFR * parCFMAX * (parTT - Tm[t, :, :])
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (parCWH * SNOWPACK)
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation
            soil_wetness = (SM / parFC) ** parBETA
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness

            SM = SM + RAIN + tosoil - recharge
            excess = SM - parFC
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            evapfactor = SM / (parLP * parFC)
            evapfactor  = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = ETpm[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=PRECS) # SM can not be zero for gradient tracking
            ETmu[t, :, :] = ETact

            # Groundwater boxes
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, parPERC)
            SUZ = SUZ - PERC
            Q0 = parK0 * torch.clamp(SUZ - parUZL, min=0.0)
            SUZ = SUZ - Q0
            Q1 = parK1 * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = parK2 * SLZ
            SLZ = SLZ - Q2
            Qsimmu[t, :, :] = Q0 + Q1 + Q2

            # save components
            Qsimmu0[t, :, :] = Q0
            Qsimmu1[t, :, :] = Q1
            Qsimmu2[t, :, :] = Q2


        Qsimave0 = Qsimmu0.mean(-1, keepdim=True)
        Qsimave1 = Qsimmu1.mean(-1, keepdim=True)
        Qsimave2 = Qsimmu2.mean(-1, keepdim=True)
        ETave = ETmu.mean(-1, keepdim=True)

        # get the initial component average
        if muwts is None:
            Qsimave = Qsimmu.mean(-1)
        else:
            Qsimave = (Qsimmu * muwts).sum(-1)

        if routOpt is True: # routing
            if comprout is True:
                # do routing to all the components, reshape the matrix to [Time, gage*multi]
                Qsim = Qsimmu.view(Nstep, Ngrid * mu)
            else:
                # average the components first, then do routing
                Qsim = Qsimave
            # scale learned routing parameter
            tempa = routscaLst[0][0] + rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb = routscaLst[1][0] + rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            UH = UH_gamma(routa, routb, lenF=15)  # lenF: folter
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])   # dim:gage*var*time
            UH = UH.permute([1, 2, 0])  # dim: gage*var*time
            Qsrout = UH_conv(rf, UH).permute([2, 0, 1])

            if comprout is True: # Qs is [time, [gage*mult], var] now
                Qstemp = Qsrout.view(Nstep, Ngrid, mu)
                if muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp * muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else: # no routing, output the initial average simulations

            Qs = torch.unsqueeze(Qsimave, -1) # add a dimension

        if outstate is True:
            return Qs, SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            Qall = torch.cat((Qs, Qsimave0, Qsimave1, Qsimave2, ETave), dim=-1)
            return Qall


class HBVMulET(torch.nn.Module):
    """Multi-component HBV Model PyTorch version"""
    # Add an ET shape parameter; others are the same as class HBVMul()
    # refer HBVMul() for detailed comments

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulET, self).__init__()

    def forward(self, x, parameters, mu, muwts, rtwts, bufftime=0, outstate=False, routOpt=False, comprout=False):

        PRECS = 1e-5

        # Initialization
        if bufftime > 0:
            with torch.no_grad():
                xinit = x[0:bufftime, :, :]
                initmodel = HBVMulET()
                Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = initmodel(xinit, parameters, mu, muwts, rtwts,
                                                                      bufftime=0, outstate=True, routOpt=False, comprout=False)
        else:

            # Without buff time, initialize state variables with zeros
            Ngrid = x.shape[1]
            SNOWPACK = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            MELTWATER = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SM = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SUZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SLZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()

        P = x[bufftime:, :, 0]
        Pm= P.unsqueeze(2).repeat(1,1,mu)
        T = x[bufftime:, :, 1]
        Tm = T.unsqueeze(2).repeat(1,1,mu)
        ETpot = x[bufftime:, :, 2]
        ETpm = ETpot.unsqueeze(2).repeat(1,1,mu)


        ## scale the parameters
        parascaLst = [[1,6], [50,1000], [0.05,0.9], [0.01,0.5], [0.001,0.2], [0.2,1],
                        [0,10], [0,100], [-2.5,2.5], [0.5,10], [0,0.1], [0,0.2], [0.3,5]]
        routscaLst = [[0,2.9], [0,6.5]]

        parBETA = parascaLst[0][0] + parameters[:,0,:]*(parascaLst[0][1]-parascaLst[0][0])
        parFC = parascaLst[1][0] + parameters[:,1,:]*(parascaLst[1][1]-parascaLst[1][0])
        parK0 = parascaLst[2][0] + parameters[:,2,:]*(parascaLst[2][1]-parascaLst[2][0])
        parK1 = parascaLst[3][0] + parameters[:,3,:]*(parascaLst[3][1]-parascaLst[3][0])
        parK2 = parascaLst[4][0] + parameters[:,4,:]*(parascaLst[4][1]-parascaLst[4][0])
        parLP = parascaLst[5][0] + parameters[:,5,:]*(parascaLst[5][1]-parascaLst[5][0])
        parPERC = parascaLst[6][0] + parameters[:,6,:]*(parascaLst[6][1]-parascaLst[6][0])
        parUZL = parascaLst[7][0] + parameters[:,7,:]*(parascaLst[7][1]-parascaLst[7][0])
        parTT = parascaLst[8][0] + parameters[:,8,:]*(parascaLst[8][1]-parascaLst[8][0])
        parCFMAX = parascaLst[9][0] + parameters[:,9,:]*(parascaLst[9][1]-parascaLst[9][0])
        parCFR = parascaLst[10][0] + parameters[:,10,:]*(parascaLst[10][1]-parascaLst[10][0])
        parCWH = parascaLst[11][0] + parameters[:,11,:]*(parascaLst[11][1]-parascaLst[11][0])

        # The added ET parameter
        parBETAET = parascaLst[12][0] + parameters[:,12,:]*(parascaLst[12][1]-parascaLst[12][0])

        Nstep, Ngrid = P.size()

        # Initialize time series of model variables
        Qsimmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()

        for t in range(Nstep):
            # Separate precipitation into liquid and solid components
            PRECIP = Pm[t, :, :]  # need to check later, seems repeating with line 52
            RAIN = torch.mul(PRECIP, (Tm[t, :, :] >= parTT).type(torch.float32))
            SNOW = torch.mul(PRECIP, (Tm[t, :, :] < parTT).type(torch.float32))

            # Snow
            SNOWPACK = SNOWPACK + SNOW
            melt = parCFMAX * (Tm[t, :, :] - parTT)
            # melt[melt < 0.0] = 0.0
            melt = torch.clamp(melt, min=0.0)
            # melt[melt > SNOWPACK] = SNOWPACK[melt > SNOWPACK]
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = parCFR * parCFMAX * (parTT - Tm[t, :, :])
            # refreezing[refreezing < 0.0] = 0.0
            # refreezing[refreezing > MELTWATER] = MELTWATER[refreezing > MELTWATER]
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (parCWH * SNOWPACK)
            # tosoil[tosoil < 0.0] = 0.0
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation
            soil_wetness = (SM / parFC) ** parBETA
            # soil_wetness[soil_wetness < 0.0] = 0.0
            # soil_wetness[soil_wetness > 1.0] = 1.0
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness


            SM = SM + RAIN + tosoil - recharge
            excess = SM - parFC
            # excess[excess < 0.0] = 0.0
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            # MODIFY HERE. Add the shape para parBETAET for ET equation
            evapfactor = (SM / (parLP * parFC)) ** parBETAET
            # evapfactor = SM / (parLP * parFC)
            # evapfactor[evapfactor < 0.0] = 0.0
            # evapfactor[evapfactor > 1.0] = 1.0
            evapfactor  = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = ETpm[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=PRECS) # SM can not be zero for gradient tracking

            # Groundwater boxes
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, parPERC)
            SUZ = SUZ - PERC
            Q0 = parK0 * torch.clamp(SUZ - parUZL, min=0.0)
            SUZ = SUZ - Q0
            Q1 = parK1 * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = parK2 * SLZ
            SLZ = SLZ - Q2
            Qsimmu[t, :, :] = Q0 + Q1 + Q2


        # get the initial average
        if muwts is None:
            Qsimave = Qsimmu.mean(-1)
        else:
            Qsimave = (Qsimmu * muwts).sum(-1)

        if routOpt is True: # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim = Qsimmu.view(Nstep, Ngrid * mu)
            else:
                # average the components, then do routing
                Qsim = Qsimave

            tempa = routscaLst[0][0] + rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb = routscaLst[1][0] + rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            UH = UH_gamma(routa, routb, lenF=15)  # lenF: folter
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])   # dim:gage*var*time
            UH = UH.permute([1, 2, 0])  # dim: gage*var*time
            Qsrout = UH_conv(rf, UH).permute([2, 0, 1])

            if comprout is True: # Qs is [time, [gage*mult], var] now
                Qstemp = Qsrout.view(Nstep, Ngrid, mu)
                if muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp * muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else: # no routing, output the initial average simulations

            Qs = torch.unsqueeze(Qsimave, -1) # add a dimension

        if outstate is True:
            return Qs, SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            return Qs # total streamflow


class SIMHYD7Differentiable(nn.Module):
    """
    Standard 7-parameter SIMHYD with smooth operators for autograd stability.

    Parameter order:
        [INSC, COEFF, SQ, SMSC, SUB, CRAK, K]

    Inputs:
        inputs[..., 0] = precipitation P, mm/day
        inputs[..., 1] = PET, mm/day
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4):
        super(SIMHYD7Differentiable, self).__init__()
        assert mode in ('normal', 'analysis')
        self.mode = mode
        self.theta_is_raw = theta_is_raw
        self.smooth = smooth
        self.eps = eps

    def _pos(self, x):
        if self.smooth:
            return 0.5 * (x + torch.sqrt(x * x + self.eps ** 2))
        return torch.relu(x)

    def _min(self, a, b):
        return a - self._pos(a - b)

    def _expand(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)

        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = 0.0 + theta[:, 2:3] * (6.0 - 0.0)
        SMSC = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = 0.0 + theta[:, 4:5] * (1.0 - 0.0)
        CRAK = 0.0 + theta[:, 5:6] * (1.0 - 0.0)
        K = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        return INSC, COEF, SQ, SMSC, SUB, CRAK, K

    @torch.no_grad()
    def denorm_params(self, theta):
        return torch.cat(self._expand(theta), dim=1)

    def forward(self, inputs, theta, initial_state=None):
        B, T, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        INSC, COEF, SQ, SMSC, SUB, CRAK, K = self._expand(theta)

        P = self._pos(inputs[:, :, 0:1])
        E0 = self._pos(inputs[:, :, 1:2])

        if initial_state is None:
            SMS = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
        else:
            SMS = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]

        q_hist = []
        sms_hist = []
        gw_hist = []

        for t in range(T):
            Pt = P[:, t, :]
            E0t = E0[:, t, :]

            SMS0 = self._min(self._pos(SMS), SMSC)
            GW0 = self._pos(GW)

            IMAX = self._min(INSC, E0t)
            INT = self._min(IMAX, Pt)
            INR = self._pos(Pt - INT)

            wetness = SMS0 / (SMSC + 1e-8)
            infil_cap = COEF * torch.exp(-SQ * wetness)
            RMO = self._min(infil_cap, INR)
            IRUN = self._pos(INR - RMO)

            SRUN = SUB * wetness * RMO
            REC = CRAK * wetness * (RMO - SRUN)
            REC = self._pos(REC)
            SMF = self._pos(RMO - SRUN - REC)

            POT = self._pos(E0t - INT)
            ETS_cap = 10.0 * wetness
            ETS = self._min(ETS_cap, POT)
            ETS = self._min(ETS, SMS0 + SMF)

            SMS_pre = SMS0 + SMF - ETS
            SOIL_EXCESS = self._pos(SMS_pre - SMSC)
            SMS_next = self._pos(SMS_pre - SOIL_EXCESS)
            REC_total = REC + SOIL_EXCESS

            BAS = K * GW0
            GW_next = self._pos(GW0 + REC_total - BAS)
            Q = self._pos(IRUN + SRUN + BAS)

            SMS = SMS_next
            GW = GW_next

            q_hist.append(Q)
            sms_hist.append(SMS)
            gw_hist.append(GW)

        q_seq = torch.stack(q_hist, dim=1)
        sms_seq = torch.stack(sms_hist, dim=1)
        gw_seq = torch.stack(gw_hist, dim=1)

        if self.mode == 'normal':
            return q_seq

        return torch.cat([sms_seq, gw_seq, q_seq], dim=-1)


class MultiInv_SIMHYDModel(torch.nn.Module):
    """
    Reuse the inversion LSTM to infer one static 7-parameter SIMHYD vector per basin.
    """

    def __init__(self, *, ninv, hiddeninv=256, inittime=0, drinv=0.5):
        super(MultiInv_SIMHYDModel, self).__init__()
        self.lstminv = CudnnLstmModel(nx=ninv, ny=7, hiddenSize=hiddeninv, dr=drinv)
        self.simhyd = SIMHYD7Differentiable(mode='normal', theta_is_raw=False)
        self.simhyd_analysis = SIMHYD7Differentiable(mode='analysis', theta_is_raw=False)
        self.hiddeninv = hiddeninv
        self.inittime = inittime
        self.ny = 1

    def forward(self, x, z, doDropMC=False):
        param_seq = self.lstminv(z)
        theta = torch.sigmoid(param_seq[-1, :, :])

        x_bt = x.permute(1, 0, 2)
        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            state_hist = self.simhyd_analysis(warm_inputs, theta)
            initial_state = state_hist[:, -1, 0:2]
            q_seq = self.simhyd(main_inputs, theta, initial_state=initial_state)
        else:
            q_seq = self.simhyd(x_bt, theta)

        return q_seq.permute(1, 0, 2)


class SnowSIMHYD8Differentiable(nn.Module):
    """
    SIMHYD with a groundwater-loss term plus an HBV-style snow module.

    Parameter order:
        [INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH]

    Inputs:
        inputs[..., 0] = precipitation P, mm/day
        inputs[..., 1] = temperature T, deg C
        inputs[..., 2] = PET, mm/day
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0):
        super(SnowSIMHYD8Differentiable, self).__init__()
        assert mode in ('normal', 'analysis')
        self.mode = mode
        self.theta_is_raw = theta_is_raw
        self.smooth = smooth
        self.eps = eps
        self.rain_snow_gain = rain_snow_gain

    def _pos(self, x):
        if self.smooth:
            return 0.5 * (x + torch.sqrt(x * x + self.eps ** 2))
        return torch.relu(x)

    def _min(self, a, b):
        return a - self._pos(a - b)

    def _expand(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)

        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = 0.0 + theta[:, 2:3] * (6.0 - 0.0)
        SMSC = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = 0.0 + theta[:, 4:5] * (1.0 - 0.0)
        CRAK = 0.0 + theta[:, 5:6] * (1.0 - 0.0)
        K = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        LG = 0.0 + theta[:, 7:8] * (1.0 - 0.0)
        TT = -2.5 + theta[:, 8:9] * (2.5 - (-2.5))
        CFMAX = 0.5 + theta[:, 9:10] * (10.0 - 0.5)
        CFR = 0.0 + theta[:, 10:11] * (0.1 - 0.0)
        CWH = 0.0 + theta[:, 11:12] * (0.2 - 0.0)
        return INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH

    @torch.no_grad()
    def denorm_params(self, theta):
        return torch.cat(self._expand(theta), dim=1)

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.5):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH = self._expand(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SMS = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
        else:
            SMS = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]
            SNOWPACK = initial_state[:, 2:3]
            MELTWATER = initial_state[:, 3:4]

        if lg_dyn_seq is not None:
            if lg_dyn_seq.shape[0] != B or lg_dyn_seq.shape[1] != Tlen:
                raise ValueError('lg_dyn_seq must have shape [B, T, 1] matching inputs')

        q_hist = []
        sms_hist = []
        gw_hist = []
        snow_hist = []
        meltwater_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SMS0 = self._min(self._pos(SMS), SMSC)
            GW0 = self._pos(GW)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()

            RAIN = Pt * frac_rain
            SNOW = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + SNOW

            melt_pot = CFMAX * self._pos(Tt - TT)
            melt = self._min(melt_pot, SNOWPACK1)

            MELTWATER1 = MELTWATER0 + melt
            SNOWPACK2 = SNOWPACK1 - melt

            refreeze_pot = CFR * CFMAX * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)

            SNOWPACK3 = SNOWPACK2 + refreezing
            MELTWATER2 = MELTWATER1 - refreezing

            water_holding = CWH * SNOWPACK3
            tosoil = self._pos(MELTWATER2 - water_holding)

            MELTWATER_next = self._pos(MELTWATER2 - tosoil)
            SNOWPACK_next = self._pos(SNOWPACK3)

            Peff = RAIN + tosoil

            IMAX = self._min(INSC, E0t)
            INT = self._min(IMAX, Peff)
            INR = self._pos(Peff - INT)

            wetness = SMS0 / (SMSC + 1e-8)
            infil_cap = COEF * torch.exp(-SQ * wetness)
            RMO = self._min(infil_cap, INR)
            IRUN = self._pos(INR - RMO)

            SRUN = SUB * wetness * RMO
            REC = CRAK * wetness * (RMO - SRUN)
            REC = self._pos(REC)
            SMF = self._pos(RMO - SRUN - REC)

            POT = self._pos(E0t - INT)
            ETS_cap = 10.0 * wetness
            ETS = self._min(ETS_cap, POT)
            ETS = self._min(ETS, SMS0 + SMF)

            SMS_pre = SMS0 + SMF - ETS
            SOIL_EXCESS = self._pos(SMS_pre - SMSC)
            SMS_next = self._pos(SMS_pre - SOIL_EXCESS)
            REC_total = REC + SOIL_EXCESS

            BAS = K * GW0
            if lg_dyn_seq is None:
                LG_t = LG
            else:
                LG_dyn_t = lg_dyn_seq[:, t, :]
                LG_t = (1.0 - lg_dyn_weight) * LG + lg_dyn_weight * LG_dyn_t
            GW_next = self._pos(GW0 + REC_total - BAS - LG_t)
            Q = self._pos(IRUN + SRUN + BAS)

            SMS = SMS_next
            GW = GW_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q)
            sms_hist.append(SMS)
            gw_hist.append(GW)
            snow_hist.append(SNOWPACK)
            meltwater_hist.append(MELTWATER)

        q_seq = torch.stack(q_hist, dim=1)
        sms_seq = torch.stack(sms_hist, dim=1)
        gw_seq = torch.stack(gw_hist, dim=1)
        snow_seq = torch.stack(snow_hist, dim=1)
        meltwater_seq = torch.stack(meltwater_hist, dim=1)

        if self.mode == 'normal':
            return q_seq

        return torch.cat([sms_seq, gw_seq, snow_seq, meltwater_seq, q_seq], dim=-1)


class MultiInv_SnowSIMHYDModel(torch.nn.Module):
    """
    Inversion LSTM infers one static Snow-SIMHYD parameter vector per basin.
    """

    def __init__(self, *, ninv, hiddeninv=256, inittime=0, drinv=0.5):
        super(MultiInv_SnowSIMHYDModel, self).__init__()
        self.lstminv = CudnnLstmModel(nx=ninv, ny=12, hiddenSize=hiddeninv, dr=drinv)
        self.simhyd = SnowSIMHYD8Differentiable(mode='normal', theta_is_raw=False, smooth=True)
        self.simhyd_analysis = SnowSIMHYD8Differentiable(mode='analysis', theta_is_raw=False, smooth=True)
        self.hiddeninv = hiddeninv
        self.inittime = inittime
        self.ny = 1

    def forward(self, x, z, doDropMC=False):
        param_seq = self.lstminv(z)
        theta = torch.sigmoid(param_seq[-1, :, :])

        x_bt = x.permute(1, 0, 2)
        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            state_hist = self.simhyd_analysis(warm_inputs, theta)
            initial_state = state_hist[:, -1, 0:4]
            q_seq = self.simhyd(main_inputs, theta, initial_state=initial_state)
        else:
            q_seq = self.simhyd(x_bt, theta)

        return q_seq.permute(1, 0, 2)


class SnowSIMHYD11NoLossDifferentiable(nn.Module):
    """
    Simple SIMHYD + HBV-style snow module, with no explicit loss sink.

    Parameter order:
        [INSC, COEF, SQ, SMSC, SUB, CRAK, K, TT, CFMAX, CFR, CWH]

    Inputs:
        inputs[..., 0] = precipitation P, mm/day
        inputs[..., 1] = temperature T, deg C
        inputs[..., 2] = PET, mm/day
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0):
        super(SnowSIMHYD11NoLossDifferentiable, self).__init__()
        assert mode in ('normal', 'analysis')
        self.mode = mode
        self.theta_is_raw = theta_is_raw
        self.smooth = smooth
        self.eps = eps
        self.rain_snow_gain = rain_snow_gain

    def _pos(self, x):
        if self.smooth:
            return 0.5 * (x + torch.sqrt(x * x + self.eps ** 2))
        return torch.relu(x)

    def _min(self, a, b):
        return a - self._pos(a - b)

    def _expand(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)
        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = 0.0 + theta[:, 2:3] * 6.0
        SMSC = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = theta[:, 4:5]
        CRAK = theta[:, 5:6]
        K = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        TT = -2.5 + theta[:, 7:8] * 5.0
        CFMAX = 0.5 + theta[:, 8:9] * (10.0 - 0.5)
        CFR = theta[:, 9:10] * 0.1
        CWH = theta[:, 10:11] * 0.2
        return INSC, COEF, SQ, SMSC, SUB, CRAK, K, TT, CFMAX, CFR, CWH

    def forward(self, inputs, theta, initial_state=None, return_diagnostics=False, return_final_state=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        INSC, COEF, SQ, SMSC, SUB, CRAK, K, TT, CFMAX, CFR, CWH = self._expand(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SMS = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
        else:
            SMS = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]
            SNOWPACK = initial_state[:, 2:3]
            MELTWATER = initial_state[:, 3:4]

        q_hist = []
        if return_diagnostics:
            diag_hist = {k: [] for k in [
                'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release_to_soil',
                'interception_evaporation', 'actual_ET', 'infiltration', 'infiltration_excess',
                'surface_runoff', 'recharge_to_groundwater', 'soil_overflow',
                'baseflow_raw', 'baseflow_capped', 'total_discharge',
                'soil_moisture', 'groundwater', 'snowpack', 'meltwater',
                'INSC', 'COEF', 'SQ', 'SMSC', 'SUB', 'CRAK', 'K', 'TT', 'CFMAX', 'CFR', 'CWH',
                'water_balance_residual',
            ]}

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SMS0 = self._min(self._pos(SMS), SMSC)
            GW0 = self._pos(GW)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            storage0 = SMS0 + GW0 + SNOWPACK0 + MELTWATER0

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()
            RAIN = Pt * frac_rain
            SNOW = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + SNOW
            melt_pot = CFMAX * self._pos(Tt - TT)
            melt = self._min(melt_pot, SNOWPACK1)
            MELTWATER1 = MELTWATER0 + melt
            SNOWPACK2 = self._pos(SNOWPACK1 - melt)

            refreeze_pot = CFR * CFMAX * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            SNOWPACK3 = SNOWPACK2 + refreezing
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)

            water_holding = CWH * SNOWPACK3
            tosoil_raw = self._pos(MELTWATER2 - water_holding)
            tosoil = self._min(tosoil_raw, MELTWATER2)
            MELTWATER_next = self._pos(MELTWATER2 - tosoil)
            SNOWPACK_next = self._pos(SNOWPACK3)
            Peff = RAIN + tosoil

            IMAX = self._min(INSC, E0t)
            INT = self._min(IMAX, Peff)
            INR = self._pos(Peff - INT)

            wetness = SMS0 / (SMSC + 1e-8)
            infil_cap = COEF * torch.exp(-SQ * wetness)
            RMO = self._min(infil_cap, INR)
            IRUN = self._pos(INR - RMO)

            SRUN = SUB * wetness * RMO
            REC = CRAK * wetness * (RMO - SRUN)
            REC = self._pos(REC)
            SMF = self._pos(RMO - SRUN - REC)

            POT = self._pos(E0t - INT)
            ETS_cap = 10.0 * wetness
            ETS = self._min(ETS_cap, POT)
            ETS = self._min(ETS, SMS0 + SMF)

            SMS_pre = SMS0 + SMF - ETS
            SOIL_EXCESS = self._pos(SMS_pre - SMSC)
            SMS_next = self._pos(SMS_pre - SOIL_EXCESS)
            REC_total = REC + SOIL_EXCESS

            GW_available = self._pos(GW0 + REC_total)
            BAS_raw = K * GW_available
            BAS = self._min(BAS_raw, GW_available)
            GW_next = self._pos(GW_available - BAS)
            Q = self._pos(IRUN + SRUN + BAS)

            storage1 = SMS_next + GW_next + SNOWPACK_next + MELTWATER_next
            delta_storage = storage1 - storage0
            residual = Pt - INT - ETS - Q - delta_storage

            SMS = SMS_next
            GW = GW_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next
            q_hist.append(Q)

            if return_diagnostics:
                diag_vals = {
                    'precipitation': Pt,
                    'rainfall': RAIN,
                    'snowfall': SNOW,
                    'snowmelt': melt,
                    'refreezing': refreezing,
                    'snow_release_to_soil': tosoil,
                    'interception_evaporation': INT,
                    'actual_ET': ETS,
                    'infiltration': RMO,
                    'infiltration_excess': IRUN,
                    'surface_runoff': SRUN + IRUN,
                    'recharge_to_groundwater': REC_total,
                    'soil_overflow': SOIL_EXCESS,
                    'baseflow_raw': BAS_raw,
                    'baseflow_capped': BAS,
                    'total_discharge': Q,
                    'soil_moisture': SMS_next,
                    'groundwater': GW_next,
                    'snowpack': SNOWPACK_next,
                    'meltwater': MELTWATER_next,
                    'INSC': INSC,
                    'COEF': COEF,
                    'SQ': SQ,
                    'SMSC': SMSC,
                    'SUB': SUB,
                    'CRAK': CRAK,
                    'K': K,
                    'TT': TT,
                    'CFMAX': CFMAX,
                    'CFR': CFR,
                    'CWH': CWH,
                    'water_balance_residual': residual,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SMS, GW, SNOWPACK, MELTWATER], dim=1)

        if return_diagnostics:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_final_state:
                return q_seq, diag_out, final_state
            return q_seq, diag_out

        if self.mode == 'normal':
            if return_final_state:
                return q_seq, final_state
            return q_seq

        if return_diagnostics:
            state_seq = torch.cat([
                diag_out['soil_moisture'],
                diag_out['groundwater'],
                diag_out['snowpack'],
                diag_out['meltwater'],
                q_seq], dim=-1)
        else:
            state_seq = None
        if return_final_state:
            return state_seq, final_state
        return state_seq


class MultiInv_SnowSIMHYDNoLossModel(torch.nn.Module):
    """
    Simple single-component SIMHYD + snow model with no explicit loss sink.
    """

    def __init__(self, *, ninv, hiddeninv=256, inittime=0, drinv=0.5):
        super(MultiInv_SnowSIMHYDNoLossModel, self).__init__()
        self.lstminv = CudnnLstmModel(nx=ninv, ny=11, hiddenSize=hiddeninv, dr=drinv)
        self.simhyd = SnowSIMHYD11NoLossDifferentiable(mode='normal', theta_is_raw=False, smooth=True)
        self.simhyd_analysis = SnowSIMHYD11NoLossDifferentiable(mode='analysis', theta_is_raw=False, smooth=True)
        self.hiddeninv = hiddeninv
        self.inittime = inittime
        self.ny = 1

    def forward(self, x, z, doDropMC=False, return_diagnostics=False):
        param_seq = self.lstminv(z)
        theta = torch.sigmoid(param_seq[-1, :, :])

        x_bt = x.permute(1, 0, 2)
        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            _, warm_diag, warm_state = self.simhyd_analysis(
                warm_inputs, theta, return_diagnostics=True, return_final_state=True)
            if return_diagnostics:
                q_seq, diag_out = self.simhyd(
                    main_inputs, theta, initial_state=warm_state, return_diagnostics=True)
            else:
                q_seq = self.simhyd(main_inputs, theta, initial_state=warm_state)
                diag_out = None
        else:
            if return_diagnostics:
                q_seq, diag_out = self.simhyd(x_bt, theta, return_diagnostics=True)
            else:
                q_seq = self.simhyd(x_bt, theta)
                diag_out = None

        out = q_seq.permute(1, 0, 2)
        if return_diagnostics:
            diag_out = {k: v.permute(1, 0, 2) for k, v in diag_out.items()}
            return out, diag_out
        return out


class MultiInv_SnowSIMHYDMulTDModel(torch.nn.Module):
    """
    Experimental multi-component Snow-SIMHYD.

    Differences from MultiInv_SnowSIMHYDModel:
      - multiple Snow-SIMHYD components per basin
      - optional learned component weights
      - optional routing after component mixing
      - semi-dynamic LG: only groundwater loss changes through time
    """

    def __init__(self, *, ninv, nfea=12, nmul=4, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.5):
        super(MultiInv_SnowSIMHYDMulTDModel, self).__init__()
        self.ninv = ninv
        self.nfea = nfea
        self.nmul = nmul
        self.hiddeninv = hiddeninv
        self.inittime = inittime
        self.routOpt = routOpt
        self.comprout = comprout
        self.compwts = compwts
        self.lgdyn = lgdyn
        self.lgdynweight = lgdynweight
        self.ny = 1

        self.nstaticpm = nfea * nmul
        self.nroutpm = nmul * 2 if comprout else 2
        self.nwtspm = nmul if compwts else 0
        self.ndynpm = nmul if lgdyn else 0
        self.ntp = self.nstaticpm + self.nroutpm + self.nwtspm + self.ndynpm

        self.lstminv = CudnnLstmModel(nx=ninv, ny=self.ntp, hiddenSize=hiddeninv, dr=drinv)
        self.simhyd = SnowSIMHYD8Differentiable(mode='normal', theta_is_raw=False, smooth=True)
        self.simhyd_analysis = SnowSIMHYD8Differentiable(mode='analysis', theta_is_raw=False, smooth=True)

    def _route_q(self, qin, rtwts):
        # qin: [time, batch, 1], rtwts: [batch, 2] in [0, 1]
        Nstep = qin.shape[0]
        lenF = 15
        routscaLst = [[0, 2.9], [0, 6.5]]
        rf = qin.permute([1, 2, 0])  # [batch, 1, time]
        tempa = routscaLst[0][0] + rtwts[:, 0] * (routscaLst[0][1] - routscaLst[0][0])
        tempb = routscaLst[1][0] + rtwts[:, 1] * (routscaLst[1][1] - routscaLst[1][0])
        rept = max(Nstep, lenF)
        routa = tempa.repeat(rept, 1).unsqueeze(-1)
        routb = tempb.repeat(rept, 1).unsqueeze(-1)
        UH = UH_gamma(routa, routb, lenF=lenF).permute([1, 2, 0])
        qout = UH_conv(rf, UH).permute([2, 0, 1])
        return qout

    def forward(self, x, z, doDropMC=False):
        gen = self.lstminv(z)
        nt_dyn, ngage, _ = gen.shape
        nt_x = x.shape[0]

        params0 = gen[-1, :, :]
        static0 = params0[:, 0:self.nstaticpm]
        snowpara = torch.sigmoid(static0).view(ngage, self.nfea, self.nmul)

        cursor = self.nstaticpm
        routpara0 = params0[:, cursor:cursor + self.nroutpm]
        if self.comprout is False:
            routpara = torch.sigmoid(routpara0)
        else:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        cursor += self.nroutpm

        if self.compwts is False:
            wts = None
        else:
            wtspara = params0[:, cursor:cursor + self.nwtspm]
            wts = F.softmax(wtspara, dim=-1)
            cursor += self.nwtspm

        if self.lgdyn is False:
            lg_dyn = None
        else:
            lg_dyn = torch.sigmoid(gen[:, :, cursor:cursor + self.ndynpm])  # [Tmain, B, mu]

        x_rep = x.unsqueeze(2).repeat(1, 1, self.nmul, 1).view(nt_x, ngage * self.nmul, x.shape[2])
        x_bt = x_rep.permute(1, 0, 2)
        theta = snowpara.permute(0, 2, 1).contiguous().view(ngage * self.nmul, self.nfea)

        if lg_dyn is None:
            lg_bt = None
        else:
            lg_bt = lg_dyn.permute(1, 2, 0).contiguous().view(ngage * self.nmul, nt_dyn, 1)

        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            if lg_bt is None:
                main_lg = None
            else:
                main_lg = lg_bt
            state_hist = self.simhyd_analysis(
                warm_inputs,
                theta,
                lg_dyn_seq=None,
                lg_dyn_weight=self.lgdynweight)
            initial_state = state_hist[:, -1, 0:4]
            q_seq = self.simhyd(
                main_inputs,
                theta,
                initial_state=initial_state,
                lg_dyn_seq=main_lg,
                lg_dyn_weight=self.lgdynweight)
        else:
            if lg_bt is not None and lg_bt.shape[1] != x_bt.shape[1]:
                raise ValueError('dynamic parameter length must match x when inittime=0')
            q_seq = self.simhyd(
                x_bt,
                theta,
                lg_dyn_seq=lg_bt,
                lg_dyn_weight=self.lgdynweight)

        q_comp = q_seq.view(ngage, self.nmul, q_seq.shape[1], 1).permute(2, 0, 1, 3)

        if self.routOpt is True and self.comprout is True:
            q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
            q_routed = self._route_q(q_for_routing, routpara)
            q_routed = q_routed.view(q_comp.shape[0], ngage, self.nmul, 1)
            if wts is None:
                out = torch.mean(q_routed, dim=2)
            else:
                out = torch.sum(q_routed * wts.unsqueeze(0).unsqueeze(-1), dim=2)
            return out

        if wts is None:
            q_mix = torch.mean(q_comp, dim=2)
        else:
            q_mix = torch.sum(q_comp * wts.unsqueeze(0).unsqueeze(-1), dim=2)

        if self.routOpt is True:
            out = self._route_q(q_mix, routpara)
        else:
            out = q_mix
        return out


class MultiInv_SnowSIMHYDMulTDHeterModel(torch.nn.Module):
    """
    Heterogeneity-aware multi-component Snow-SIMHYD.

    Key change from MultiInv_SnowSIMHYDMulTDModel:
      - static parameters, routing, and component weights come from a direct
        basin-attribute encoder instead of only from the shared inversion LSTM
      - dynamic LG(t) is produced by a separate sequence head, then shifted by
        an attribute-conditioned bias so it can vary across basins
      - component-specific learnable biases break permutation symmetry
    """

    def __init__(self, *, ninv, nfea=12, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.5):
        super(MultiInv_SnowSIMHYDMulTDHeterModel, self).__init__()
        self.ninv = ninv
        self.nfea = nfea
        self.nmul = nmul
        self.nattr = nattr
        self.hiddeninv = hiddeninv
        self.inittime = inittime
        self.routOpt = routOpt
        self.comprout = comprout
        self.compwts = compwts
        self.lgdyn = lgdyn
        self.lgdynweight = lgdynweight
        self.ny = 1

        self.nstaticpm = nfea * nmul
        if comprout is False:
            self.nroutpm = 2
        else:
            self.nroutpm = nmul * 2
        if compwts is False:
            self.nwtspm = 0
        else:
            self.nwtspm = nmul
        if lgdyn is False:
            self.ndynpm = 0
        else:
            self.ndynpm = nmul

        self.staticFeat = nn.Sequential(
            nn.Linear(nattr, hiddeninv),
            nn.ReLU(),
            nn.Linear(hiddeninv, hiddeninv),
            nn.ReLU())
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)

        if self.lgdyn is True:
            # Use the runtime-safe LSTM path for dynamic LG(t) so large global runs
            # do not depend on the legacy CUDA-only CudnnLstm implementation.
            self.lstmdyn = SafeLstmModel(nx=ninv, ny=self.ndynpm, hiddenSize=hiddeninv, dr=drinv)
            self.lgAttr = nn.Linear(nattr, self.ndynpm)

        self.simhyd = SnowSIMHYD8Differentiable(mode='normal', theta_is_raw=False, smooth=True)
        self.simhyd_analysis = SnowSIMHYD8Differentiable(mode='analysis', theta_is_raw=False, smooth=True)

        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.compWeightBias = nn.Parameter(torch.linspace(-0.2, 0.2, nmul)) if self.nwtspm > 0 else None

    def _route_q(self, qin, rtwts):
        Nstep = qin.shape[0]
        lenF = 15
        routscaLst = [[0, 2.9], [0, 6.5]]
        rf = qin.permute([1, 2, 0])
        tempa = routscaLst[0][0] + rtwts[:, 0] * (routscaLst[0][1] - routscaLst[0][0])
        tempb = routscaLst[1][0] + rtwts[:, 1] * (routscaLst[1][1] - routscaLst[1][0])
        rept = max(Nstep, lenF)
        routa = tempa.repeat(rept, 1).unsqueeze(-1)
        routb = tempb.repeat(rept, 1).unsqueeze(-1)
        UH = UH_gamma(routa, routb, lenF=lenF).permute([1, 2, 0])
        qout = UH_conv(rf, UH).permute([2, 0, 1])
        return qout

    def forward(self, x, z, doDropMC=False):
        nt_x = x.shape[0]
        ngage = z.shape[1]
        basin_attr = z[-1, :, -self.nattr:]

        staticFeat = self.staticFeat(basin_attr)
        staticParams0 = self.staticOut(staticFeat)

        cursor = 0
        static0 = staticParams0[:, cursor:cursor + self.nstaticpm].view(ngage, self.nfea, self.nmul)
        static0 = static0 + self.compStaticBias
        snowpara = torch.sigmoid(static0)
        cursor += self.nstaticpm

        routpara0 = staticParams0[:, cursor:cursor + self.nroutpm]
        if self.comprout is False:
            routpara = torch.sigmoid(routpara0)
        else:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        cursor += self.nroutpm

        if self.nwtspm == 0:
            wts = None
        else:
            wtspara = staticParams0[:, cursor:cursor + self.nwtspm] + self.compWeightBias
            wts = F.softmax(wtspara, dim=-1)

        if self.lgdyn is False:
            lg_dyn = None
        else:
            lg_dyn_seq = self.lstmdyn(z)
            lg_attr_bias = self.lgAttr(basin_attr).unsqueeze(0).repeat(lg_dyn_seq.shape[0], 1, 1)
            lg_dyn = torch.sigmoid(lg_dyn_seq + lg_attr_bias)

        x_rep = x.unsqueeze(2).repeat(1, 1, self.nmul, 1).view(nt_x, ngage * self.nmul, x.shape[2])
        x_bt = x_rep.permute(1, 0, 2)
        theta = snowpara.permute(0, 2, 1).contiguous().view(ngage * self.nmul, self.nfea)

        if lg_dyn is None:
            lg_bt = None
        else:
            lg_bt = lg_dyn.permute(1, 2, 0).contiguous().view(ngage * self.nmul, lg_dyn.shape[0], 1)
        if lg_bt is not None and self.inittime > 0:
            lg_bt_main = lg_bt[:, self.inittime:, :]
        else:
            lg_bt_main = lg_bt
        if lg_bt is not None and self.inittime > 0:
            lg_bt_main = lg_bt[:, self.inittime:, :]
        else:
            lg_bt_main = lg_bt

        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            state_hist = self.simhyd_analysis(
                warm_inputs,
                theta,
                lg_dyn_seq=None,
                lg_dyn_weight=self.lgdynweight)
            initial_state = state_hist[:, -1, 0:4]
            q_seq = self.simhyd(
                main_inputs,
                theta,
                initial_state=initial_state,
                lg_dyn_seq=lg_bt,
                lg_dyn_weight=self.lgdynweight)
        else:
            q_seq = self.simhyd(
                x_bt,
                theta,
                lg_dyn_seq=lg_bt,
                lg_dyn_weight=self.lgdynweight)

        q_comp = q_seq.view(ngage, self.nmul, q_seq.shape[1], 1).permute(2, 0, 1, 3)

        if self.routOpt is True and self.comprout is True:
            q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
            q_routed = self._route_q(q_for_routing, routpara)
            q_routed = q_routed.view(q_comp.shape[0], ngage, self.nmul, 1)
            if wts is None:
                out = torch.mean(q_routed, dim=2)
            else:
                out = torch.sum(q_routed * wts.unsqueeze(0).unsqueeze(-1), dim=2)
            return out

        if wts is None:
            q_mix = torch.mean(q_comp, dim=2)
        else:
            q_mix = torch.sum(q_comp * wts.unsqueeze(0).unsqueeze(-1), dim=2)

        if self.routOpt is True:
            out = self._route_q(q_mix, routpara)
        else:
            out = q_mix
        return out


class DynamicSimHydDifferentiable(nn.Module):
    """
    Dynamic Snow-SIMHYD process model.

    Relative to SnowSIMHYD8Differentiable, this class can switch selected
    process parameters from static to dynamic:
      - Scheme A: COEF_t and SQ_t
      - Scheme B: ETGAM_t
      - Scheme C: SUB_t, CRAK_t, K_t
      - Scheme D: SG_CRIT for groundwater disconnection

    Dynamic controls are predicted from per-step forcings, internal states,
    and seasonal sin/cos features. Seasonal features can be passed in the
    input tensor as channels 3 and 4; if omitted, a simple repeating annual
    cycle based on the step index is used as a fallback.
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0,
                 dynamic_coef_sq=False, dynamic_etgam=False, dynamic_groundwater=False,
                 groundwater_disconnection=False, nn_runoff_partition=False, dynamic_all=False,
                 dyn_hidden=32):
        super(DynamicSimHydDifferentiable, self).__init__()
        assert mode in ('normal', 'analysis')
        self.mode = mode
        self.theta_is_raw = theta_is_raw
        self.smooth = smooth
        self.eps = eps
        self.rain_snow_gain = rain_snow_gain
        self.dynamic_coef_sq = dynamic_coef_sq or dynamic_all
        self.dynamic_etgam = dynamic_etgam or dynamic_all
        self.dynamic_groundwater = dynamic_groundwater or dynamic_all
        self.groundwater_disconnection = groundwater_disconnection
        self.nn_runoff_partition = nn_runoff_partition
        if self.nn_runoff_partition is True:
            raise NotImplementedError('nn_runoff_partition is reserved for a future experiment')

        self.dyn_out_dim = 0
        self.dyn_slices = dict()
        if self.dynamic_coef_sq is True:
            self.dyn_slices['coef_sq'] = slice(self.dyn_out_dim, self.dyn_out_dim + 2)
            self.dyn_out_dim += 2
        if self.dynamic_etgam is True:
            self.dyn_slices['etgam'] = slice(self.dyn_out_dim, self.dyn_out_dim + 1)
            self.dyn_out_dim += 1
        if self.dynamic_groundwater is True:
            self.dyn_slices['groundwater'] = slice(self.dyn_out_dim, self.dyn_out_dim + 3)
            self.dyn_out_dim += 3

        if self.dyn_out_dim > 0:
            self.dynHead = nn.Sequential(
                nn.Linear(8, dyn_hidden),
                nn.ReLU(),
                nn.Linear(dyn_hidden, dyn_hidden),
                nn.ReLU(),
                nn.Linear(dyn_hidden, self.dyn_out_dim))
        else:
            self.dynHead = None

    def _pos(self, x):
        if self.smooth:
            return 0.5 * (x + torch.sqrt(x * x + self.eps ** 2))
        return torch.relu(x)

    def _min(self, a, b):
        return a - self._pos(a - b)

    def _expand(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)

        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = 0.0 + theta[:, 2:3] * (6.0 - 0.0)
        SMSC = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = 0.0 + theta[:, 4:5] * (1.0 - 0.0)
        CRAK = 0.0 + theta[:, 5:6] * (1.0 - 0.0)
        K = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        LG = 0.0 + theta[:, 7:8] * (1.0 - 0.0)
        TT = -2.5 + theta[:, 8:9] * (2.5 - (-2.5))
        CFMAX = 0.5 + theta[:, 9:10] * (10.0 - 0.5)
        CFR = 0.0 + theta[:, 10:11] * (0.1 - 0.0)
        CWH = 0.0 + theta[:, 11:12] * (0.2 - 0.0)
        if theta.shape[1] >= 13:
            SG_CRIT = 0.0 + theta[:, 12:13] * 500.0
        else:
            SG_CRIT = None
        return INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH, SG_CRIT

    @torch.no_grad()
    def denorm_params(self, theta):
        params = self._expand(theta)
        if params[-1] is None:
            return torch.cat(params[:-1], dim=1)
        return torch.cat(params, dim=1)

    def _seasonal_feats(self, inputs, t, Tlen, device, dtype):
        if inputs.shape[-1] >= 5:
            sin_t = inputs[:, t, 3:4]
            cos_t = inputs[:, t, 4:5]
            return sin_t, cos_t
        ang = 2.0 * math.pi * float(t % 365) / 365.0
        sin_t = torch.full((inputs.shape[0], 1), math.sin(ang), device=device, dtype=dtype)
        cos_t = torch.full((inputs.shape[0], 1), math.cos(ang), device=device, dtype=dtype)
        return sin_t, cos_t

    def _apply_dynamic_controls(self, dyn_raw, COEF, SQ, SUB, CRAK, K, SMS0, SMSC):
        COEF_t = COEF
        SQ_t = SQ
        SUB_t = SUB
        CRAK_t = CRAK
        K_t = K
        ETGAM_t = torch.ones_like(COEF)

        if self.dynamic_coef_sq is True:
            raw = dyn_raw[:, self.dyn_slices['coef_sq']]
            coef_mult = 0.5 + 1.5 * torch.sigmoid(raw[:, 0:1])
            sq_mult = 0.5 + 1.5 * torch.sigmoid(raw[:, 1:2])
            COEF_t = COEF * coef_mult
            SQ_t = SQ * sq_mult

        if self.dynamic_etgam is True:
            raw = dyn_raw[:, self.dyn_slices['etgam']]
            ETGAM_t = 0.25 + 3.75 * torch.sigmoid(raw[:, 0:1])

        if self.dynamic_groundwater is True:
            raw = dyn_raw[:, self.dyn_slices['groundwater']]
            sub_mult = 0.5 + 1.5 * torch.sigmoid(raw[:, 0:1])
            crak_mult = 0.5 + 1.5 * torch.sigmoid(raw[:, 1:2])
            k_mult = 0.5 + 1.5 * torch.sigmoid(raw[:, 2:3])
            SUB_t = torch.clamp(SUB * sub_mult, 0.0, 1.0)
            CRAK_t = torch.clamp(CRAK * crak_mult, 0.0, 1.0)
            K_t = torch.clamp(K * k_mult, 0.003, 0.3)

        return COEF_t, SQ_t, ETGAM_t, SUB_t, CRAK_t, K_t

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.5,
                return_diagnostics=False, return_final_state=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH, SG_CRIT = self._expand(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SMS = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
        else:
            SMS = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]
            SNOWPACK = initial_state[:, 2:3]
            MELTWATER = initial_state[:, 3:4]

        if lg_dyn_seq is not None:
            if lg_dyn_seq.shape[0] != B or lg_dyn_seq.shape[1] != Tlen:
                raise ValueError('lg_dyn_seq must have shape [B, T, 1] matching inputs')

        q_hist = []
        diag_hist = dict()
        if return_diagnostics is True:
            for name in [
                    'snowpack', 'soil_moisture', 'groundwater', 'interception_storage',
                    'rainfall', 'snowfall', 'snowmelt', 'interception_evaporation', 'actual_ET',
                    'infiltration', 'recharge_to_groundwater', 'surface_runoff', 'interflow',
                    'baseflow', 'groundwater_loss', 'total_unrouted_discharge',
                    'COEF_t', 'SQ_t', 'ETGAM_t', 'SUB_t', 'CRAK_t', 'K_t', 'LG_t', 'SG_CRIT']:
                diag_hist[name] = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SMS0 = self._min(self._pos(SMS), SMSC)
            GW0 = self._pos(GW)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()

            RAIN = Pt * frac_rain
            SNOW = Pt * (1.0 - frac_rain)
            SNOWPACK1 = SNOWPACK0 + SNOW

            melt_pot = CFMAX * self._pos(Tt - TT)
            melt = self._min(melt_pot, SNOWPACK1)
            MELTWATER1 = MELTWATER0 + melt
            SNOWPACK2 = SNOWPACK1 - melt

            refreeze_pot = CFR * CFMAX * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            SNOWPACK3 = SNOWPACK2 + refreezing
            MELTWATER2 = MELTWATER1 - refreezing

            water_holding = CWH * SNOWPACK3
            tosoil = self._pos(MELTWATER2 - water_holding)
            MELTWATER_next = self._pos(MELTWATER2 - tosoil)
            SNOWPACK_next = self._pos(SNOWPACK3)

            Peff = RAIN + tosoil

            sin_t, cos_t = self._seasonal_feats(inputs, t, Tlen, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0,
                Tt / 20.0,
                E0t / 10.0,
                SMS0 / (SMSC + 1e-8),
                GW0 / (SMSC + 1e-8),
                SNOWPACK0 / (SMSC + 1e-8),
                sin_t,
                cos_t], dim=1)
            if self.dynHead is None:
                dyn_raw = None
            else:
                dyn_raw = self.dynHead(dyn_in)

            if dyn_raw is None:
                COEF_t, SQ_t = COEF, SQ
                ETGAM_t = torch.ones_like(COEF)
                SUB_t, CRAK_t, K_t = SUB, CRAK, K
            else:
                COEF_t, SQ_t, ETGAM_t, SUB_t, CRAK_t, K_t = self._apply_dynamic_controls(
                    dyn_raw, COEF, SQ, SUB, CRAK, K, SMS0, SMSC)

            IMAX = self._min(INSC, E0t)
            INT = self._min(IMAX, Peff)
            INR = self._pos(Peff - INT)

            wetness = SMS0 / (SMSC + 1e-8)
            infil_cap = COEF_t * torch.exp(-SQ_t * wetness)
            RMO = self._min(infil_cap, INR)
            IRUN = self._pos(INR - RMO)

            SRUN = SUB_t * wetness * RMO
            REC = CRAK_t * wetness * (RMO - SRUN)
            REC = self._pos(REC)
            SMF = self._pos(RMO - SRUN - REC)

            POT = self._pos(E0t - INT)
            if self.dynamic_etgam is True:
                wetness_eff = torch.clamp(wetness, min=0.0, max=1.0)
                et_scale = torch.pow(torch.clamp(wetness_eff, min=1e-6), ETGAM_t)
                et_scale = torch.clamp(et_scale, max=1.0)
                ETS = POT * et_scale
                ETS = self._min(ETS, SMS0 + SMF)
            else:
                ETS_cap = 10.0 * wetness
                ETS = self._min(ETS_cap, POT)
                ETS = self._min(ETS, SMS0 + SMF)

            SMS_pre = SMS0 + SMF - ETS
            SOIL_EXCESS = self._pos(SMS_pre - SMSC)
            SMS_next = self._pos(SMS_pre - SOIL_EXCESS)
            REC_total = REC + SOIL_EXCESS

            if lg_dyn_seq is None:
                LG_t = LG
            else:
                LG_dyn_t = lg_dyn_seq[:, t, :]
                LG_t = (1.0 - lg_dyn_weight) * LG + lg_dyn_weight * LG_dyn_t

            if self.groundwater_disconnection is True and SG_CRIT is not None:
                BAS = K_t * F.softplus(GW0 - SG_CRIT)
                GWLOSS = LG_t * F.softplus(SG_CRIT - GW0)
            else:
                BAS = K_t * GW0
                GWLOSS = LG_t

            GW_next = self._pos(GW0 + REC_total - BAS - GWLOSS)
            Q = self._pos(IRUN + SRUN + BAS)

            SMS = SMS_next
            GW = GW_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q)
            if return_diagnostics is True:
                diag_vals = {
                    'snowpack': SNOWPACK_next,
                    'soil_moisture': SMS_next,
                    'groundwater': GW_next,
                    'interception_storage': torch.zeros_like(INT),
                    'rainfall': RAIN,
                    'snowfall': SNOW,
                    'snowmelt': melt,
                    'interception_evaporation': INT,
                    'actual_ET': ETS,
                    'infiltration': RMO,
                    'recharge_to_groundwater': REC_total,
                    'surface_runoff': IRUN,
                    'interflow': SRUN,
                    'baseflow': BAS,
                    'groundwater_loss': GWLOSS,
                    'total_unrouted_discharge': Q,
                    'COEF_t': COEF_t,
                    'SQ_t': SQ_t,
                    'ETGAM_t': ETGAM_t,
                    'SUB_t': SUB_t,
                    'CRAK_t': CRAK_t,
                    'K_t': K_t,
                    'LG_t': LG_t,
                    'SG_CRIT': SG_CRIT if SG_CRIT is not None else torch.zeros_like(Q),
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SMS, GW, SNOWPACK, MELTWATER], dim=1)

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out

        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModel(torch.nn.Module):
    """
    Dynamic SimHyd variant built from the heterogeneity-aware Snow-SIMHYD
    architecture, with optional time-varying process parameters.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.5,
                 dynamic_coef_sq=False, dynamic_etgam=False, dynamic_groundwater=False,
                 groundwater_disconnection=False, nn_runoff_partition=False, dynamic_all=False):
        super(MultiInv_DynamicSimHydModel, self).__init__()
        self.ninv = ninv
        self.nmul = nmul
        self.nattr = nattr
        self.hiddeninv = hiddeninv
        self.inittime = inittime
        self.routOpt = routOpt
        self.comprout = comprout
        self.compwts = compwts
        self.lgdyn = lgdyn
        self.lgdynweight = lgdynweight
        self.dynamic_coef_sq = dynamic_coef_sq or dynamic_all
        self.dynamic_etgam = dynamic_etgam or dynamic_all
        self.dynamic_groundwater = dynamic_groundwater or dynamic_all
        self.groundwater_disconnection = groundwater_disconnection
        self.nn_runoff_partition = nn_runoff_partition
        self.dynamic_all = dynamic_all
        self.nfea = 13 if groundwater_disconnection else 12
        self.ny = 1

        self.nstaticpm = self.nfea * nmul
        self.nroutpm = nmul * 2 if comprout else 2
        self.nwtspm = nmul if compwts else 0
        self.ndynpm = nmul if lgdyn else 0

        self.staticFeat = nn.Sequential(
            nn.Linear(nattr, hiddeninv),
            nn.ReLU(),
            nn.Linear(hiddeninv, hiddeninv),
            nn.ReLU())
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)

        if self.lgdyn is True:
            self.lstmdyn = SafeLstmModel(nx=ninv, ny=self.ndynpm, hiddenSize=hiddeninv, dr=drinv)
            self.lgAttr = nn.Linear(nattr, self.ndynpm)

        self.simhyd = DynamicSimHydDifferentiable(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_coef_sq=self.dynamic_coef_sq,
            dynamic_etgam=self.dynamic_etgam,
            dynamic_groundwater=self.dynamic_groundwater,
            groundwater_disconnection=self.groundwater_disconnection,
            nn_runoff_partition=self.nn_runoff_partition,
            dynamic_all=self.dynamic_all)
        self.simhyd_analysis = DynamicSimHydDifferentiable(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_coef_sq=self.dynamic_coef_sq,
            dynamic_etgam=self.dynamic_etgam,
            dynamic_groundwater=self.dynamic_groundwater,
            groundwater_disconnection=self.groundwater_disconnection,
            nn_runoff_partition=self.nn_runoff_partition,
            dynamic_all=self.dynamic_all)

        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.compWeightBias = nn.Parameter(torch.linspace(-0.2, 0.2, nmul)) if self.nwtspm > 0 else None

    def _route_q(self, qin, rtwts):
        Nstep = qin.shape[0]
        lenF = 15
        routscaLst = [[0, 2.9], [0, 6.5]]
        rf = qin.permute([1, 2, 0])
        tempa = routscaLst[0][0] + rtwts[:, 0] * (routscaLst[0][1] - routscaLst[0][0])
        tempb = routscaLst[1][0] + rtwts[:, 1] * (routscaLst[1][1] - routscaLst[1][0])
        rept = max(Nstep, lenF)
        routa = tempa.repeat(rept, 1).unsqueeze(-1)
        routb = tempb.repeat(rept, 1).unsqueeze(-1)
        UH = UH_gamma(routa, routb, lenF=lenF).permute([1, 2, 0])
        qout = UH_conv(rf, UH).permute([2, 0, 1])
        return qout

    def _mix_component_tensor(self, tensor_comp, ngage):
        tensor4 = tensor_comp.view(ngage, self.nmul, tensor_comp.shape[1], tensor_comp.shape[2]).permute(2, 0, 1, 3)
        if self.nwtspm == 0:
            return torch.mean(tensor4, dim=2)
        return torch.sum(tensor4 * self._last_wts.unsqueeze(0).unsqueeze(-1), dim=2)

    def forward(self, x, z, doDropMC=False, return_diagnostics=False):
        nt_x = x.shape[0]
        ngage = z.shape[1]
        basin_attr = z[-1, :, -self.nattr:]

        staticFeat = self.staticFeat(basin_attr)
        staticParams0 = self.staticOut(staticFeat)

        cursor = 0
        static0 = staticParams0[:, cursor:cursor + self.nstaticpm].view(ngage, self.nfea, self.nmul)
        static0 = static0 + self.compStaticBias
        snowpara = torch.sigmoid(static0)
        cursor += self.nstaticpm

        routpara0 = staticParams0[:, cursor:cursor + self.nroutpm]
        if self.comprout is False:
            routpara = torch.sigmoid(routpara0)
        else:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        cursor += self.nroutpm

        if self.nwtspm == 0:
            wts = None
        else:
            wtspara = staticParams0[:, cursor:cursor + self.nwtspm] + self.compWeightBias
            wts = F.softmax(wtspara, dim=-1)
        self._last_wts = wts

        if self.lgdyn is False:
            lg_dyn = None
        else:
            lg_dyn_seq = self.lstmdyn(z)
            lg_attr_bias = self.lgAttr(basin_attr).unsqueeze(0).repeat(lg_dyn_seq.shape[0], 1, 1)
            lg_dyn = torch.sigmoid(lg_dyn_seq + lg_attr_bias)

        x_rep = x.unsqueeze(2).repeat(1, 1, self.nmul, 1).view(nt_x, ngage * self.nmul, x.shape[2])
        x_bt = x_rep.permute(1, 0, 2)
        theta = snowpara.permute(0, 2, 1).contiguous().view(ngage * self.nmul, self.nfea)

        if lg_dyn is None:
            lg_bt = None
        else:
            lg_bt = lg_dyn.permute(1, 2, 0).contiguous().view(ngage * self.nmul, lg_dyn.shape[0], 1)

        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            x_use = x[self.inittime:, :, :]
            _, warm_state = self.simhyd_analysis(
                warm_inputs,
                theta,
                lg_dyn_seq=None,
                lg_dyn_weight=self.lgdynweight,
                return_final_state=True)
            if return_diagnostics is True:
                q_seq, diag_comp = self.simhyd(
                    main_inputs,
                    theta,
                    initial_state=warm_state,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight,
                    return_diagnostics=True)
            else:
                q_seq = self.simhyd(
                    main_inputs,
                    theta,
                    initial_state=warm_state,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight)
        else:
            if return_diagnostics is True:
                q_seq, diag_comp = self.simhyd(
                    x_bt,
                    theta,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight,
                    return_diagnostics=True)
            else:
                q_seq = self.simhyd(
                    x_bt,
                    theta,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight)

        q_comp = q_seq.view(ngage, self.nmul, q_seq.shape[1], 1).permute(2, 0, 1, 3)

        if self.routOpt is True and self.comprout is True:
            q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
            q_routed = self._route_q(q_for_routing, routpara)
            q_routed = q_routed.view(q_comp.shape[0], ngage, self.nmul, 1)
            if wts is None:
                out = torch.mean(q_routed, dim=2)
            else:
                out = torch.sum(q_routed * wts.unsqueeze(0).unsqueeze(-1), dim=2)
        else:
            if wts is None:
                q_mix = torch.mean(q_comp, dim=2)
            else:
                q_mix = torch.sum(q_comp * wts.unsqueeze(0).unsqueeze(-1), dim=2)
            if self.routOpt is True:
                out = self._route_q(q_mix, routpara)
            else:
                out = q_mix

        if return_diagnostics is not True:
            return out

        diag_out = dict()
        for name, tensor_comp in diag_comp.items():
            diag_out[name] = self._mix_component_tensor(tensor_comp, ngage)
        if 'total_unrouted_discharge' not in diag_out:
            if wts is None:
                diag_out['total_unrouted_discharge'] = torch.mean(q_comp, dim=2)
            else:
                diag_out['total_unrouted_discharge'] = torch.sum(q_comp * wts.unsqueeze(0).unsqueeze(-1), dim=2)
        diag_out['total_discharge'] = out
        return out, diag_out


class DynamicSimHydModelFiveDifferentiable(nn.Module):
    """
    Model Five: selective dynamic Snow-SIMHYD process model.

    Design goals:
      - keep most parameters static and identifiable
      - allow only targeted dynamics (SQ, ETGAM, constrained partition, LG)
      - optional snow-only CFMAX dynamics and optional routing-scale dynamics
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True, dynamic_cfmax_snow=True,
                 dynamic_routing_scale=False, dynamic_all=False, dyn_hidden=32):
        super(DynamicSimHydModelFiveDifferentiable, self).__init__()
        assert mode in ('normal', 'analysis')
        self.mode = mode
        self.theta_is_raw = theta_is_raw
        self.smooth = smooth
        self.eps = eps
        self.rain_snow_gain = rain_snow_gain

        self.dynamic_sq = dynamic_sq or dynamic_all
        self.dynamic_etgam = dynamic_etgam or dynamic_all
        self.dynamic_partition = dynamic_partition or dynamic_all
        self.dynamic_cfmax_snow = dynamic_cfmax_snow or dynamic_all
        self.dynamic_routing_scale = dynamic_routing_scale or dynamic_all

        self.dyn_out_dim = 0
        self.dyn_slices = dict()
        if self.dynamic_sq is True:
            self.dyn_slices['sq'] = slice(self.dyn_out_dim, self.dyn_out_dim + 1)
            self.dyn_out_dim += 1
        if self.dynamic_etgam is True:
            self.dyn_slices['etgam'] = slice(self.dyn_out_dim, self.dyn_out_dim + 1)
            self.dyn_out_dim += 1
        if self.dynamic_partition is True:
            self.dyn_slices['partition'] = slice(self.dyn_out_dim, self.dyn_out_dim + 3)
            self.dyn_out_dim += 3
        if self.dynamic_cfmax_snow is True:
            self.dyn_slices['cfmax'] = slice(self.dyn_out_dim, self.dyn_out_dim + 1)
            self.dyn_out_dim += 1

        if self.dyn_out_dim > 0:
            self.dynHead = nn.Sequential(
                nn.Linear(9, dyn_hidden),
                nn.ReLU(),
                nn.Linear(dyn_hidden, dyn_hidden),
                nn.ReLU(),
                nn.Linear(dyn_hidden, self.dyn_out_dim))
            self._dynhead_force_cpu_fallback = False
        else:
            self.dynHead = None
            self._dynhead_force_cpu_fallback = False

    def _pos(self, x):
        if self.smooth:
            return 0.5 * (x + torch.sqrt(x * x + self.eps ** 2))
        return torch.relu(x)

    def _min(self, a, b):
        return a - self._pos(a - b)

    def _run_dyn_head(self, dyn_in):
        if self.dynHead is None:
            return None
        if self._dynhead_force_cpu_fallback:
            if next(self.dynHead.parameters()).device.type != 'cpu':
                self.dynHead.cpu()
            out = self.dynHead(dyn_in.cpu())
            return out.to(dyn_in.device)
        try:
            return self.dynHead(dyn_in)
        except RuntimeError as err:
            msg = str(err).lower()
            if dyn_in.device.type == 'cuda' and ('cublas runtime error' in msg or 'cuda' in msg):
                self._dynhead_force_cpu_fallback = True
                self.dynHead.cpu()
                warnings.warn(
                    'DynamicSimHydModelFiveDifferentiable dynHead falling back to CPU after CUDA linear failure; '
                    'continuing training with CPU dynamic process head.',
                    RuntimeWarning)
                out = self.dynHead(dyn_in.cpu())
                return out.to(dyn_in.device)
            raise

    def _expand(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)

        # Static parameters and baselines (all in physical ranges)
        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = 0.0 + theta[:, 2:3] * (6.0 - 0.0)
        SMSC = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = 0.0 + theta[:, 4:5] * (1.0 - 0.0)
        CRAK = 0.0 + theta[:, 5:6] * (1.0 - 0.0)
        K = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        LG = 0.0 + theta[:, 7:8] * (0.2 - 0.0)
        TT = -2.5 + theta[:, 8:9] * (2.5 - (-2.5))
        CFMAX = 0.5 + theta[:, 9:10] * (10.0 - 0.5)
        CFR = 0.0 + theta[:, 10:11] * (0.1 - 0.0)
        CWH = 0.0 + theta[:, 11:12] * (0.2 - 0.0)
        SG_CRIT = 0.0 + theta[:, 12:13] * (300.0 - 0.0)
        return INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH, SG_CRIT

    @torch.no_grad()
    def denorm_params(self, theta):
        return torch.cat(self._expand(theta), dim=1)

    def _seasonal_feats(self, inputs, t, device, dtype):
        if inputs.shape[-1] >= 5:
            return inputs[:, t, 3:4], inputs[:, t, 4:5]
        ang = 2.0 * math.pi * float(t % 365) / 365.0
        sin_t = torch.full((inputs.shape[0], 1), math.sin(ang), device=device, dtype=dtype)
        cos_t = torch.full((inputs.shape[0], 1), math.cos(ang), device=device, dtype=dtype)
        return sin_t, cos_t

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH, SG_CRIT = self._expand(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SMS = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
        else:
            SMS = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]
            SNOWPACK = initial_state[:, 2:3]
            MELTWATER = initial_state[:, 3:4]

        if lg_dyn_seq is not None and (lg_dyn_seq.shape[0] != B or lg_dyn_seq.shape[1] != Tlen):
            raise ValueError('lg_dyn_seq must have shape [B, T, 1] matching inputs')

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = dict()
        if return_diagnostics is True:
            for name in [
                    'snowpack', 'interception_storage', 'soil_moisture', 'groundwater',
                    'rainfall', 'snowfall', 'snowmelt', 'interception_evaporation', 'actual_ET',
                    'infiltration', 'recharge_to_groundwater', 'surface_runoff', 'interflow',
                    'baseflow', 'groundwater_loss', 'channel_loss', 'total_discharge',
                    'COEF_t', 'SQ_t', 'ETGAM_t', 'SUB_t', 'CRAK_t', 'K_t', 'LG_t', 'SG_CRIT',
                    'CFMAX_t']:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        route_mult_hist = []
        recharge_frac_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SMS0 = self._min(self._pos(SMS), SMSC)
            GW0 = self._pos(GW)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            wetness = torch.clamp(SMS0 / (SMSC + 1e-8), min=0.0, max=1.5)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0,
                Tt / 20.0,
                E0t / 10.0,
                SMS0 / 300.0,
                wetness,
                GW0 / 300.0,
                SNOWPACK0 / 300.0,
                sin_t,
                cos_t], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            # Dynamic SQ
            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)

            # Dynamic ET exponent
            if self.dynamic_etgam is True:
                ETGAM_t = 0.25 + 3.75 * torch.sigmoid(dyn_raw[:, self.dyn_slices['etgam']])
            else:
                ETGAM_t = torch.ones_like(SQ)

            # Dynamic CFMAX for snow basins only
            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX * m_cf_eff

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()

            RAIN = Pt * frac_rain
            SNOW = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + SNOW
            melt_pot = CFMAX_t * self._pos(Tt - TT)
            melt = self._min(melt_pot, SNOWPACK1)
            MELTWATER1 = MELTWATER0 + melt
            SNOWPACK2 = SNOWPACK1 - melt

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            SNOWPACK3 = SNOWPACK2 + refreezing
            MELTWATER2 = MELTWATER1 - refreezing

            water_holding = CWH * SNOWPACK3
            tosoil = self._pos(MELTWATER2 - water_holding)
            MELTWATER_next = self._pos(MELTWATER2 - tosoil)
            SNOWPACK_next = self._pos(SNOWPACK3)
            Peff = RAIN + tosoil

            IMAX = self._min(INSC, E0t)
            INT = self._min(IMAX, Peff)
            INR = self._pos(Peff - INT)

            infil_cap = COEF * torch.exp(-SQ_t * wetness)
            RMO = self._min(infil_cap, INR)
            IRUN_excess = self._pos(INR - RMO)

            # Constrained dynamic partition using static SUB/CRAK as biases
            available = wetness * RMO
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_inter = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([
                torch.log(base_surface),
                torch.log(base_inter),
                torch.log(base_recharge)], dim=1)
            if self.dynamic_partition is True:
                part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']]
            else:
                part_logits = base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]

            SRUN = IRUN_excess + f_surface * available
            IFLOW = f_inter * available
            REC = f_recharge * available
            SMF = self._pos(RMO - available)

            POT = self._pos(E0t - INT)
            et_scale = torch.pow(torch.clamp(wetness, min=1e-6, max=1.0), ETGAM_t)
            et_scale = torch.clamp(et_scale, max=1.0)
            ETS = POT * et_scale
            ETS = self._min(ETS, SMS0 + SMF)

            SMS_pre = SMS0 + SMF - ETS
            SOIL_EXCESS = self._pos(SMS_pre - SMSC)
            SMS_next = self._pos(SMS_pre - SOIL_EXCESS)
            REC_total = REC + SOIL_EXCESS

            if lg_dyn_seq is None:
                LG_t = LG
            else:
                LG_dyn_t = torch.clamp(lg_dyn_seq[:, t, :], 0.0, 1.0)
                LG_eff = (1.0 - lg_dyn_weight) * LG + lg_dyn_weight * (LG * LG_dyn_t * 1.5)
                LG_t = torch.clamp(LG_eff, 0.0, 0.2)

            BAS = K * F.softplus(GW0 - SG_CRIT)
            GWLOSS = LG_t * F.softplus(SG_CRIT - GW0)
            GW_next = self._pos(GW0 + REC_total - BAS - GWLOSS)
            Q = self._pos(SRUN + IFLOW + BAS)

            SMS = SMS_next
            GW = GW_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)

            if return_diagnostics is True:
                diag_vals = {
                    'snowpack': SNOWPACK_next,
                    'interception_storage': torch.zeros_like(INT),
                    'soil_moisture': SMS_next,
                    'groundwater': GW_next,
                    'rainfall': RAIN,
                    'snowfall': SNOW,
                    'snowmelt': melt,
                    'interception_evaporation': INT,
                    'actual_ET': ETS,
                    'infiltration': RMO,
                    'recharge_to_groundwater': REC_total,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'baseflow': BAS,
                    'groundwater_loss': GWLOSS,
                    'channel_loss': torch.zeros_like(Q),
                    'total_discharge': Q,
                    'COEF_t': COEF,
                    'SQ_t': SQ_t,
                    'ETGAM_t': ETGAM_t,
                    'SUB_t': f_surface,
                    'CRAK_t': f_recharge,
                    'K_t': K,
                    'LG_t': LG_t,
                    'SG_CRIT': SG_CRIT,
                    'CFMAX_t': CFMAX_t,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SMS, GW, SNOWPACK, MELTWATER], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(route_mult_hist) > 0:
            rt_seq = torch.stack(route_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((rt_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((rt_seq[:, 1:, :] - rt_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)

        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': reg_part
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out

        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelFive(torch.nn.Module):
    """
    Model Five wrapper with selective dynamic process controls.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3):
        super(MultiInv_DynamicSimHydModelFive, self).__init__()
        self.ninv = ninv
        self.nmul = nmul
        self.nattr = nattr
        self.hiddeninv = hiddeninv
        self.inittime = inittime
        self.routOpt = routOpt
        self.comprout = comprout
        self.compwts = compwts
        self.lgdyn = lgdyn
        self.lgdynweight = lgdynweight
        self.dynamic_sq = dynamic_sq or dynamic_all
        self.dynamic_etgam = dynamic_etgam or dynamic_all
        self.dynamic_partition = dynamic_partition or dynamic_all
        self.dynamic_cfmax_snow = dynamic_cfmax_snow or dynamic_all
        self.dynamic_routing_scale = dynamic_routing_scale or dynamic_all
        self.dynamic_all = dynamic_all
        self.reg_amp_w = reg_amp_w
        self.reg_smooth_w = reg_smooth_w
        self.reg_part_w = reg_part_w
        self._last_aux_loss = None
        self._last_aux_terms = None

        self.nfea = 13
        self.ny = 1
        self.nstaticpm = self.nfea * nmul
        self.nroutpm = nmul * 2 if comprout else 2
        self.nwtspm = nmul if compwts else 0
        self.ndynpm = nmul if lgdyn else 0

        self.staticFeat = nn.Sequential(
            nn.Linear(nattr, hiddeninv),
            nn.ReLU(),
            nn.Linear(hiddeninv, hiddeninv),
            nn.ReLU())
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)

        if self.lgdyn is True:
            self.lstmdyn = SafeLstmModel(nx=ninv, ny=self.ndynpm, hiddenSize=hiddeninv, dr=drinv)
            self.lgAttr = nn.Linear(nattr, self.ndynpm)

        if self.dynamic_routing_scale is True:
            self.routeDynHead = nn.Sequential(
                nn.Linear(5, hiddeninv // 2),
                nn.ReLU(),
                nn.Linear(hiddeninv // 2, 1))
        else:
            self.routeDynHead = None

        self.simhyd = DynamicSimHydModelFiveDifferentiable(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq,
            dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition,
            dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale,
            dynamic_all=self.dynamic_all)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiable(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq,
            dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition,
            dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale,
            dynamic_all=self.dynamic_all)

        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.compWeightBias = nn.Parameter(torch.linspace(-0.2, 0.2, nmul)) if self.nwtspm > 0 else None

    def get_auxiliary_loss(self):
        if self._last_aux_loss is None:
            return None
        return self._last_aux_loss

    def _route_q(self, qin, rtwts):
        Nstep = qin.shape[0]
        lenF = 15
        routscaLst = [[0, 2.9], [0, 6.5]]
        rf = qin.permute([1, 2, 0])
        tempa = routscaLst[0][0] + rtwts[:, 0] * (routscaLst[0][1] - routscaLst[0][0])
        tempb = routscaLst[1][0] + rtwts[:, 1] * (routscaLst[1][1] - routscaLst[1][0])
        rept = max(Nstep, lenF)
        routa = tempa.repeat(rept, 1).unsqueeze(-1)
        routb = tempb.repeat(rept, 1).unsqueeze(-1)
        UH = UH_gamma(routa, routb, lenF=lenF).permute([1, 2, 0])
        qout = UH_conv(rf, UH).permute([2, 0, 1])
        return qout

    def _route_q_dynamic_scale(self, qin, rtwts, x_base):
        # qin: [T, B, 1], x_base: [T, B, nx]
        T, B, _ = qin.shape
        routscaLst = [[0, 2.9], [0, 6.5]]
        route_b_static = routscaLst[1][0] + rtwts[:, 1:2] * (routscaLst[1][1] - routscaLst[1][0])
        q_out = []
        q_prev = torch.zeros(B, 1, device=qin.device, dtype=qin.dtype)
        route_mult_hist = []
        for t in range(T):
            qin_t = qin[t, :, :]
            p_t = x_base[t, :, 0:1]
            smsc_t = torch.clamp(x_base[t, :, 2:3], min=1e-6)
            sm_t = x_base[t, :, 1:2]
            wetness_t = torch.clamp(sm_t / smsc_t, 0.0, 2.0)
            sin_t = x_base[t, :, 3:4] if x_base.shape[-1] >= 5 else torch.zeros_like(qin_t)
            cos_t = x_base[t, :, 4:5] if x_base.shape[-1] >= 5 else torch.ones_like(qin_t)
            dyn_in = torch.cat([p_t / 20.0, wetness_t, qin_t / 20.0, sin_t, cos_t], dim=1)
            m_route = 0.75 + 0.75 * torch.sigmoid(self.routeDynHead(dyn_in))
            route_b_t = torch.clamp(route_b_static * m_route, min=0.1, max=12.0)
            alpha = torch.exp(-1.0 / route_b_t)
            q_now = alpha * q_prev + (1.0 - alpha) * qin_t
            q_out.append(q_now)
            q_prev = q_now
            route_mult_hist.append(m_route)
        q_seq = torch.stack(q_out, dim=0)
        route_seq = torch.stack(route_mult_hist, dim=0)
        return q_seq, route_seq

    def _mix_component_tensor(self, tensor_comp, ngage):
        tensor4 = tensor_comp.view(ngage, self.nmul, tensor_comp.shape[1], tensor_comp.shape[2]).permute(2, 0, 1, 3)
        if self.nwtspm == 0:
            return torch.mean(tensor4, dim=2)
        return torch.sum(tensor4 * self._last_wts.unsqueeze(0).unsqueeze(-1), dim=2)

    def forward(self, x, z, doDropMC=False, return_diagnostics=False):
        nt_x = x.shape[0]
        ngage = z.shape[1]
        basin_attr = z[-1, :, -self.nattr:]

        # Optional raw snow fraction channel in z (last non-attr feature)
        if z.shape[2] > self.nattr:
            snow_frac_raw = torch.clamp(z[-1, :, -self.nattr - 1:-self.nattr], min=0.0, max=1.0)
        else:
            snow_frac_raw = None

        staticFeat = self.staticFeat(basin_attr)
        staticParams0 = self.staticOut(staticFeat)

        cursor = 0
        static0 = staticParams0[:, cursor:cursor + self.nstaticpm].view(ngage, self.nfea, self.nmul)
        static0 = static0 + self.compStaticBias
        snowpara = torch.sigmoid(static0)
        cursor += self.nstaticpm

        routpara0 = staticParams0[:, cursor:cursor + self.nroutpm]
        if self.comprout is False:
            routpara = torch.sigmoid(routpara0)
        else:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        cursor += self.nroutpm

        if self.nwtspm == 0:
            wts = None
        else:
            wtspara = staticParams0[:, cursor:cursor + self.nwtspm] + self.compWeightBias
            wts = F.softmax(wtspara, dim=-1)
        self._last_wts = wts

        if self.lgdyn is False:
            lg_dyn = None
        else:
            lg_dyn_seq = self.lstmdyn(z)
            lg_attr_bias = self.lgAttr(basin_attr).unsqueeze(0).repeat(lg_dyn_seq.shape[0], 1, 1)
            lg_dyn = torch.sigmoid(lg_dyn_seq + lg_attr_bias)

        x_rep = x.unsqueeze(2).repeat(1, 1, self.nmul, 1).view(nt_x, ngage * self.nmul, x.shape[2])
        x_bt = x_rep.permute(1, 0, 2)
        theta = snowpara.permute(0, 2, 1).contiguous().view(ngage * self.nmul, self.nfea)

        if lg_dyn is None:
            lg_bt = None
        else:
            lg_bt = lg_dyn.permute(1, 2, 0).contiguous().view(ngage * self.nmul, lg_dyn.shape[0], 1)
        if lg_bt is not None and self.inittime > 0:
            lg_bt_main = lg_bt[:, self.inittime:, :]
        else:
            lg_bt_main = lg_bt

        if snow_frac_raw is None:
            snow_frac_rep = None
        else:
            snow_frac_rep = snow_frac_raw.unsqueeze(1).repeat(1, self.nmul, 1).view(ngage * self.nmul, 1)

        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            x_use = x[self.inittime:, :, :]
            _, warm_state = self.simhyd_analysis(
                warm_inputs,
                theta,
                lg_dyn_seq=None,
                lg_dyn_weight=self.lgdynweight,
                snow_frac_raw=snow_frac_rep,
                return_final_state=True)
            if return_diagnostics is True:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    main_inputs,
                    theta,
                    initial_state=warm_state,
                    lg_dyn_seq=lg_bt_main,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_diagnostics=True,
                    return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    main_inputs,
                    theta,
                    initial_state=warm_state,
                    lg_dyn_seq=lg_bt_main,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_regularization=True)
        else:
            if return_diagnostics is True:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    x_bt,
                    theta,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_diagnostics=True,
                    return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    x_bt,
                    theta,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_regularization=True)

        reg_total = self.reg_amp_w * reg_terms['dynamic_amplitude_loss'] \
            + self.reg_smooth_w * reg_terms['dynamic_smoothness_loss'] \
            + self.reg_part_w * reg_terms['partition_entropy_loss']
        self._last_aux_terms = reg_terms
        self._last_aux_loss = reg_total

        q_comp = q_seq.view(ngage, self.nmul, q_seq.shape[1], 1).permute(2, 0, 1, 3)
        if wts is None:
            q_mix = torch.mean(q_comp, dim=2)
        else:
            q_mix = torch.sum(q_comp * wts.unsqueeze(0).unsqueeze(-1), dim=2)

        route_mult_seq = None
        if self.routOpt is True and self.dynamic_routing_scale is True and self.comprout is False:
            if return_diagnostics is True and 'soil_moisture' in diag_comp:
                sm_mix = self._mix_component_tensor(diag_comp['soil_moisture'], ngage)
            else:
                sm_mix = torch.ones_like(q_mix) * 50.0
            smsc_mix = torch.ones_like(q_mix) * 200.0
            x_route = torch.cat([x[:, :, 0:1], sm_mix, smsc_mix, x[:, :, 3:4], x[:, :, 4:5]], dim=2)
            out, route_mult_seq = self._route_q_dynamic_scale(q_mix, routpara, x_route)
            self._last_aux_loss = self._last_aux_loss + self.reg_amp_w * torch.mean((route_mult_seq - 1.0) ** 2)
            self._last_aux_loss = self._last_aux_loss + self.reg_smooth_w * torch.mean(
                (route_mult_seq[1:, :, :] - route_mult_seq[:-1, :, :]) ** 2)
        elif self.routOpt is True:
            out = self._route_q(q_mix, routpara)
        else:
            out = q_mix

        if return_diagnostics is not True:
            return out

        diag_out = dict()
        for name, tensor_comp in diag_comp.items():
            diag_out[name] = self._mix_component_tensor(tensor_comp, ngage)
        diag_out['total_discharge'] = out
        if route_mult_seq is not None:
            diag_out['route_b_t_multiplier'] = route_mult_seq
        return out, diag_out


class MultiInv_DynamicSimHydModelSix(MultiInv_DynamicSimHydModelFive):
    """
    Model Six: Model Five with component-wise routing, dry-channel loss,
    and a learned zero-flow gate. The external loss and training loop remain unchanged.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 component_routing=True, dry_channel_loss=True, zero_flow_gate=True,
                 channel_loss_max=0.60, zero_gate_hidden=None):
        super(MultiInv_DynamicSimHydModelSix, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w)
        self.component_routing = component_routing
        self.dry_channel_loss = dry_channel_loss
        self.zero_flow_gate_enabled = zero_flow_gate
        self.channel_loss_max = channel_loss_max
        self.zero_gate_hidden = hiddeninv // 2 if zero_gate_hidden is None else zero_gate_hidden

        # Force component-wise routing parameters when requested, regardless of comprout.
        self.nroutpm = nmul * 2 if self.component_routing else (nmul * 2 if comprout else 2)
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)

        self.channelLossHead = nn.Linear(nattr, nmul)
        self.zeroFlowGate = nn.Sequential(
            nn.Linear(5, self.zero_gate_hidden),
            nn.ReLU(),
            nn.Linear(self.zero_gate_hidden, nmul)
        )

    def _component_tensor_4d(self, tensor_comp, ngage):
        # tensor_comp comes in as [B*nmul, T, 1] from the process model.
        return tensor_comp.view(ngage, self.nmul, tensor_comp.shape[1], tensor_comp.shape[2]).permute(2, 0, 1, 3)

    def _theta_to_smsc(self, theta, ngage):
        theta4 = theta.view(ngage, self.nmul, self.nfea)
        return 50.0 + theta4[:, :, 3:4] * (500.0 - 50.0)

    def _apply_channel_loss(self, q_comp, diag_comp, theta, basin_attr, ngage):
        # q_comp: [T, B, M, 1]
        if self.dry_channel_loss is not True:
            zeros = torch.zeros_like(q_comp)
            return q_comp, zeros, zeros

        if 'soil_moisture' in diag_comp:
            soil_m = self._component_tensor_4d(diag_comp['soil_moisture'], ngage)
        else:
            soil_m = torch.ones_like(q_comp) * 100.0

        smsc = self._theta_to_smsc(theta, ngage).unsqueeze(0)
        wetness = torch.clamp(soil_m / torch.clamp(smsc, min=1e-6), 0.0, 1.0)
        dryness = 1.0 - wetness

        gamma = self.channel_loss_max * torch.sigmoid(self.channelLossHead(basin_attr))
        gamma = gamma.unsqueeze(0).unsqueeze(-1)  # [1, B, M, 1]

        loss_frac = 1.0 - torch.exp(-gamma * dryness)
        loss_frac = torch.clamp(loss_frac, 0.0, 0.95)
        q_after = q_comp * (1.0 - loss_frac)
        channel_loss = q_comp - q_after
        return torch.clamp(q_after, min=0.0), channel_loss, loss_frac

    def _apply_zero_flow_gate(self, q_comp, x, diag_comp, theta, ngage):
        # q_comp: [T, B, M, 1], x: [T, B, nx]
        if self.zero_flow_gate_enabled is not True:
            ones = torch.ones_like(q_comp)
            return q_comp, ones

        T, B, M, _ = q_comp.shape
        if 'soil_moisture' in diag_comp:
            soil_m = self._component_tensor_4d(diag_comp['soil_moisture'], ngage)
            smsc = self._theta_to_smsc(theta, ngage).unsqueeze(0)
            wetness = torch.clamp(soil_m / torch.clamp(smsc, min=1e-6), 0.0, 1.0)
        else:
            wetness = torch.ones_like(q_comp) * 0.5

        p_t = x[:, :, 0:1].unsqueeze(2).repeat(1, 1, M, 1)
        if x.shape[-1] >= 5:
            sin_t = x[:, :, 3:4].unsqueeze(2).repeat(1, 1, M, 1)
            cos_t = x[:, :, 4:5].unsqueeze(2).repeat(1, 1, M, 1)
        else:
            sin_t = torch.zeros_like(q_comp)
            cos_t = torch.ones_like(q_comp)

        gate_in = torch.cat([p_t / 20.0, wetness, q_comp / 20.0, sin_t, cos_t], dim=-1)
        gate_in_flat = gate_in.view(T * B * M, 5)
        logits_all = self.zeroFlowGate(gate_in_flat)
        comp_idx = torch.arange(M, device=q_comp.device).view(1, 1, M, 1).repeat(T, B, 1, 1).view(T * B * M, 1)
        logits = logits_all.gather(1, comp_idx)
        p_flow = torch.sigmoid(logits).view(T, B, M, 1)
        return torch.clamp(p_flow * q_comp, min=0.0), p_flow

    def _mix_or_mean(self, tensor4, wts):
        if wts is None:
            return torch.mean(tensor4, dim=2)
        return torch.sum(tensor4 * wts.unsqueeze(0).unsqueeze(-1), dim=2)

    def forward(self, x, z, doDropMC=False, return_diagnostics=False):
        nt_x = x.shape[0]
        ngage = z.shape[1]
        basin_attr = z[-1, :, -self.nattr:]

        if z.shape[2] > self.nattr:
            snow_frac_raw = torch.clamp(z[-1, :, -self.nattr - 1:-self.nattr], min=0.0, max=1.0)
        else:
            snow_frac_raw = None

        staticFeat = self.staticFeat(basin_attr)
        staticParams0 = self.staticOut(staticFeat)

        cursor = 0
        static0 = staticParams0[:, cursor:cursor + self.nstaticpm].view(ngage, self.nfea, self.nmul)
        static0 = static0 + self.compStaticBias
        snowpara = torch.sigmoid(static0)
        cursor += self.nstaticpm

        routpara0 = staticParams0[:, cursor:cursor + self.nroutpm]
        if self.component_routing is True:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        elif self.comprout is False:
            routpara = torch.sigmoid(routpara0)
        else:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        cursor += self.nroutpm

        if self.nwtspm == 0:
            wts = None
        else:
            wtspara = staticParams0[:, cursor:cursor + self.nwtspm] + self.compWeightBias
            wts = F.softmax(wtspara, dim=-1)
        self._last_wts = wts

        if self.lgdyn is False:
            lg_dyn = None
        else:
            lg_dyn_seq = self.lstmdyn(z)
            lg_attr_bias = self.lgAttr(basin_attr).unsqueeze(0).repeat(lg_dyn_seq.shape[0], 1, 1)
            lg_dyn = torch.sigmoid(lg_dyn_seq + lg_attr_bias)

        x_rep = x.unsqueeze(2).repeat(1, 1, self.nmul, 1).view(nt_x, ngage * self.nmul, x.shape[2])
        x_bt = x_rep.permute(1, 0, 2)
        theta = snowpara.permute(0, 2, 1).contiguous().view(ngage * self.nmul, self.nfea)

        if lg_dyn is None:
            lg_bt = None
        else:
            lg_bt = lg_dyn.permute(1, 2, 0).contiguous().view(ngage * self.nmul, lg_dyn.shape[0], 1)
        if lg_bt is not None and self.inittime > 0:
            lg_bt_main = lg_bt[:, self.inittime:, :]
        else:
            lg_bt_main = lg_bt

        if snow_frac_raw is None:
            snow_frac_rep = None
        else:
            snow_frac_rep = snow_frac_raw.unsqueeze(1).repeat(1, self.nmul, 1).view(ngage * self.nmul, 1)

        need_process_diag = return_diagnostics or self.dry_channel_loss or self.zero_flow_gate_enabled

        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            x_use = x[self.inittime:, :, :]
            _, warm_state = self.simhyd_analysis(
                warm_inputs,
                theta,
                lg_dyn_seq=None,
                lg_dyn_weight=self.lgdynweight,
                snow_frac_raw=snow_frac_rep,
                return_final_state=True)
            if need_process_diag:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    main_inputs,
                    theta,
                    initial_state=warm_state,
                    lg_dyn_seq=lg_bt_main,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_diagnostics=True,
                    return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    main_inputs,
                    theta,
                    initial_state=warm_state,
                    lg_dyn_seq=lg_bt_main,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_regularization=True)
                diag_comp = None
        else:
            x_use = x
            if need_process_diag:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    x_bt,
                    theta,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_diagnostics=True,
                    return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    x_bt,
                    theta,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_regularization=True)
                diag_comp = None

        reg_total = self.reg_amp_w * reg_terms['dynamic_amplitude_loss'] \
            + self.reg_smooth_w * reg_terms['dynamic_smoothness_loss'] \
            + self.reg_part_w * reg_terms['partition_entropy_loss']
        self._last_aux_terms = reg_terms
        self._last_aux_loss = reg_total

        q_comp_raw = q_seq.view(ngage, self.nmul, q_seq.shape[1], 1).permute(2, 0, 1, 3)
        assert q_comp_raw.shape[1] == ngage and q_comp_raw.shape[2] == self.nmul
        q_comp_raw = torch.clamp(q_comp_raw, min=0.0)

        q_after_loss, channel_loss, channel_loss_fraction = self._apply_channel_loss(q_comp_raw, diag_comp, theta, basin_attr, ngage)
        q_after_gate, zero_flow_probability = self._apply_zero_flow_gate(q_after_loss, x_use, diag_comp, theta, ngage)
        q_comp = torch.clamp(q_after_gate, min=0.0)

        q_mix_before_routing = self._mix_or_mean(q_comp, wts)

        route_mult_seq = None
        if self.routOpt is True and self.component_routing is True:
            q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
            assert routpara.shape[0] == ngage * self.nmul and routpara.shape[1] == 2
            if self.dynamic_routing_scale is True:
                if diag_comp is not None and 'soil_moisture' in diag_comp:
                    sm_comp = self._component_tensor_4d(diag_comp['soil_moisture'], ngage)
                else:
                    sm_comp = torch.ones_like(q_comp) * 50.0
                smsc_comp = self._theta_to_smsc(theta, ngage).unsqueeze(0).repeat(q_comp.shape[0], 1, 1, 1)
                p_rep = x_use[:, :, 0:1].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                if x_use.shape[-1] >= 5:
                    sin_rep = x_use[:, :, 3:4].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                    cos_rep = x_use[:, :, 4:5].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                else:
                    sin_rep = torch.zeros_like(q_comp)
                    cos_rep = torch.ones_like(q_comp)
                x_route = torch.cat([p_rep, sm_comp, smsc_comp, sin_rep, cos_rep], dim=-1).view(q_comp.shape[0], ngage * self.nmul, 5)
                q_routed_flat, route_mult_flat = self._route_q_dynamic_scale(q_for_routing, routpara, x_route)
                route_mult_4d = route_mult_flat.view(q_comp.shape[0], ngage, self.nmul, 1)
                route_mult_seq = self._mix_or_mean(route_mult_4d, wts)
                self._last_aux_loss = self._last_aux_loss + self.reg_amp_w * torch.mean((route_mult_flat - 1.0) ** 2)
                self._last_aux_loss = self._last_aux_loss + self.reg_smooth_w * torch.mean(
                    (route_mult_flat[1:, :, :] - route_mult_flat[:-1, :, :]) ** 2)
                q_routed = q_routed_flat.view(q_comp.shape[0], ngage, self.nmul, 1)
            else:
                q_routed = self._route_q(q_for_routing, routpara).view(q_comp.shape[0], ngage, self.nmul, 1)
            out = self._mix_or_mean(q_routed, wts)
        else:
            if self.routOpt is True and self.dynamic_routing_scale is True and self.comprout is False:
                if diag_comp is not None and 'soil_moisture' in diag_comp:
                    sm_mix = self._mix_component_tensor(diag_comp['soil_moisture'], ngage)
                else:
                    sm_mix = torch.ones_like(q_mix_before_routing) * 50.0
                smsc_mix = torch.ones_like(q_mix_before_routing) * 200.0
                x_route = torch.cat([x_use[:, :, 0:1], sm_mix, smsc_mix, x_use[:, :, 3:4], x_use[:, :, 4:5]], dim=2)
                out, route_mult_seq = self._route_q_dynamic_scale(q_mix_before_routing, routpara, x_route)
                self._last_aux_loss = self._last_aux_loss + self.reg_amp_w * torch.mean((route_mult_seq - 1.0) ** 2)
                self._last_aux_loss = self._last_aux_loss + self.reg_smooth_w * torch.mean(
                    (route_mult_seq[1:, :, :] - route_mult_seq[:-1, :, :]) ** 2)
            elif self.routOpt is True:
                if self.comprout is True:
                    q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
                    q_routed = self._route_q(q_for_routing, routpara).view(q_comp.shape[0], ngage, self.nmul, 1)
                    out = self._mix_or_mean(q_routed, wts)
                else:
                    assert routpara.shape[0] == ngage and routpara.shape[1] == 2
                    out = self._route_q(q_mix_before_routing, routpara)
            else:
                out = q_mix_before_routing

        out = torch.clamp(out, min=0.0)
        if return_diagnostics is not True:
            return out

        diag_out = dict()
        if diag_comp is not None:
            for name, tensor_comp in diag_comp.items():
                diag_out[name] = self._mix_component_tensor(tensor_comp, ngage)
        diag_out['total_discharge'] = out
        diag_out['q_mix_before_routing'] = q_mix_before_routing
        diag_out['component_discharge_raw'] = self._mix_or_mean(q_comp_raw, wts)
        if self.dry_channel_loss is True:
            diag_out['channel_loss'] = self._mix_or_mean(channel_loss, wts)
            diag_out['channel_loss_fraction'] = self._mix_or_mean(channel_loss_fraction, wts)
        if self.zero_flow_gate_enabled is True:
            diag_out['zero_flow_probability'] = self._mix_or_mean(zero_flow_probability, wts)
        if route_mult_seq is not None:
            diag_out['route_b_t_multiplier'] = route_mult_seq
        return out, diag_out


class DynamicSimHydModelFiveDifferentiablePhysicalFix(DynamicSimHydModelFiveDifferentiable):
    """
    Model Five process simulator with explicit mass-conserving groundwater,
    snow, and soil bookkeeping for Model 6 physical-fix experiments.
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True, dynamic_cfmax_snow=True,
                 dynamic_routing_scale=False, dynamic_all=False, dyn_hidden=32):
        super(DynamicSimHydModelFiveDifferentiablePhysicalFix, self).__init__(
            mode=mode, theta_is_raw=theta_is_raw, smooth=smooth, eps=eps, rain_snow_gain=rain_snow_gain,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, dyn_hidden=dyn_hidden)

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH, SG_CRIT = self._expand(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SMS = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
        else:
            SMS = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]
            SNOWPACK = initial_state[:, 2:3]
            MELTWATER = initial_state[:, 3:4]

        if lg_dyn_seq is not None and (lg_dyn_seq.shape[0] != B or lg_dyn_seq.shape[1] != Tlen):
            raise ValueError('lg_dyn_seq must have shape [B, T, 1] matching inputs')

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = dict()
        if return_diagnostics is True:
            for name in [
                    'SMS', 'GW', 'SNOWPACK', 'MELTWATER',
                    'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release_to_soil',
                    'interception_storage', 'interception_evaporation', 'actual_ET', 'infiltration',
                    'surface_runoff', 'interflow', 'recharge_to_groundwater', 'soil_overflow',
                    'baseflow_raw', 'baseflow_capped', 'baseflow',
                    'groundwater_loss_raw', 'groundwater_loss_capped', 'groundwater_loss',
                    'channel_loss', 'gate_loss', 'q_raw_process', 'q_after_channel_loss', 'q_after_gate',
                    'total_discharge', 'INSC', 'COEF_t', 'SQ_t', 'SMSC', 'ETGAM_t', 'SUB_t', 'INTER_t', 'RECH_t',
                    'CRAK_t', 'K_t', 'LG_t', 'TT', 'SG_CRIT', 'CFMAX_t', 'CFR', 'CWH', 'partition_sum_error',
                    'available_before_baseflow', 'available_after_baseflow']:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SMS0 = self._min(self._pos(SMS), SMSC)
            GW0 = self._pos(GW)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            wetness = torch.clamp(SMS0 / (SMSC + 1e-8), min=0.0, max=1.5)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0,
                Tt / 20.0,
                E0t / 10.0,
                SMS0 / 300.0,
                wetness,
                GW0 / 300.0,
                SNOWPACK0 / 300.0,
                sin_t,
                cos_t], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)

            if self.dynamic_etgam is True:
                ETGAM_t = 0.25 + 3.75 * torch.sigmoid(dyn_raw[:, self.dyn_slices['etgam']])
            else:
                ETGAM_t = torch.ones_like(SQ)

            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX * m_cf_eff

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()

            RAIN = Pt * frac_rain
            SNOW = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + SNOW
            melt_pot = CFMAX_t * self._pos(Tt - TT)
            melt = self._min(melt_pot, SNOWPACK1)
            MELTWATER1 = MELTWATER0 + melt
            SNOWPACK2 = self._pos(SNOWPACK1 - melt)

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            SNOWPACK3 = SNOWPACK2 + refreezing
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)

            water_holding = CWH * SNOWPACK3
            tosoil_raw = self._pos(MELTWATER2 - water_holding)
            tosoil = self._min(tosoil_raw, MELTWATER2)
            MELTWATER_next = self._pos(MELTWATER2 - tosoil)
            SNOWPACK_next = self._pos(SNOWPACK3)
            Peff = RAIN + tosoil

            IMAX = self._min(INSC, E0t)
            INT = self._min(IMAX, Peff)
            INR = self._pos(Peff - INT)

            infil_cap = COEF * torch.exp(-SQ_t * wetness)
            RMO = self._min(infil_cap, INR)
            IRUN_excess = self._pos(INR - RMO)

            available = wetness * RMO
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_inter = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([
                torch.log(base_surface),
                torch.log(base_inter),
                torch.log(base_recharge)], dim=1)
            if self.dynamic_partition is True:
                part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']]
            else:
                part_logits = base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]

            SRUN = IRUN_excess + f_surface * available
            IFLOW = f_inter * available
            REC = f_recharge * available
            SMF = self._pos(RMO - available)

            POT = self._pos(E0t - INT)
            et_scale = torch.pow(torch.clamp(wetness, min=1e-6, max=1.0), ETGAM_t)
            et_scale = torch.clamp(et_scale, max=1.0)
            ETS_raw = POT * et_scale
            ETS = self._min(ETS_raw, SMS0 + SMF)
            ETS = self._min(ETS, POT)

            SMS_pre = SMS0 + SMF - ETS
            SOIL_EXCESS = self._pos(SMS_pre - SMSC)
            SMS_next = self._min(self._pos(SMS_pre), SMSC)
            REC_total = REC + SOIL_EXCESS

            if lg_dyn_seq is None:
                LG_t = LG
            else:
                LG_dyn_t = torch.clamp(lg_dyn_seq[:, t, :], 0.0, 1.0)
                LG_eff = (1.0 - lg_dyn_weight) * LG + lg_dyn_weight * (LG * LG_dyn_t * 1.5)
                LG_t = torch.clamp(LG_eff, 0.0, 0.2)

            available_before_baseflow = self._pos(GW0 + REC_total)
            BAS_raw = K * F.softplus(GW0 - SG_CRIT)
            BAS = self._min(BAS_raw, available_before_baseflow)
            available_after_baseflow = self._pos(available_before_baseflow - BAS)
            GWLOSS_raw = LG_t * F.softplus(SG_CRIT - GW0)
            GWLOSS = self._min(GWLOSS_raw, available_after_baseflow)
            GW_next = self._pos(available_after_baseflow - GWLOSS)
            Q = self._pos(SRUN + IFLOW + BAS)

            SMS = SMS_next
            GW = GW_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)

            if return_diagnostics is True:
                diag_vals = {
                    'SMS': SMS_next,
                    'GW': GW_next,
                    'SNOWPACK': SNOWPACK_next,
                    'MELTWATER': MELTWATER_next,
                    'precipitation': Pt,
                    'rainfall': RAIN,
                    'snowfall': SNOW,
                    'snowmelt': melt,
                    'refreezing': refreezing,
                    'snow_release_to_soil': tosoil,
                    'interception_storage': torch.zeros_like(INT),
                    'interception_evaporation': INT,
                    'actual_ET': ETS,
                    'infiltration': RMO,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'recharge_to_groundwater': REC_total,
                    'soil_overflow': SOIL_EXCESS,
                    'baseflow_raw': BAS_raw,
                    'baseflow_capped': BAS,
                    'baseflow': BAS,
                    'groundwater_loss_raw': GWLOSS_raw,
                    'groundwater_loss_capped': GWLOSS,
                    'groundwater_loss': GWLOSS,
                    'channel_loss': torch.zeros_like(Q),
                    'gate_loss': torch.zeros_like(Q),
                    'q_raw_process': Q,
                    'q_after_channel_loss': Q,
                    'q_after_gate': Q,
                    'total_discharge': Q,
                    'INSC': INSC,
                    'COEF_t': COEF,
                    'SQ_t': SQ_t,
                    'SMSC': SMSC,
                    'ETGAM_t': ETGAM_t,
                    'SUB_t': f_surface,
                    'INTER_t': f_inter,
                    'RECH_t': f_recharge,
                    'CRAK_t': f_recharge,
                    'K_t': K,
                    'LG_t': LG_t,
                    'TT': TT,
                    'SG_CRIT': SG_CRIT,
                    'CFMAX_t': CFMAX_t,
                    'CFR': CFR,
                    'CWH': CWH,
                    'partition_sum_error': torch.abs(f_surface + f_inter + f_recharge - 1.0),
                    'available_before_baseflow': available_before_baseflow,
                    'available_after_baseflow': available_after_baseflow,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SMS, GW, SNOWPACK, MELTWATER], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)

        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': reg_part
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out

        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelSix_Physical(MultiInv_DynamicSimHydModelSix):
    """
    Physically corrected Model 6 variant with explicit, bounded bookkeeping for
    groundwater loss, channel loss, and zero-flow gate loss.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 component_routing=True, dry_channel_loss=True, zero_flow_gate=True,
                 channel_loss_max=0.60, zero_gate_hidden=None,
                 gate_variant='explicit', gate_strength_max=0.30):
        super(MultiInv_DynamicSimHydModelSix_Physical, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing,
            dry_channel_loss=dry_channel_loss, zero_flow_gate=zero_flow_gate,
            channel_loss_max=channel_loss_max, zero_gate_hidden=zero_gate_hidden)
        assert gate_variant in ('explicit', 'soft')
        self.gate_variant = gate_variant
        self.gate_strength_max = gate_strength_max
        self.simhyd = DynamicSimHydModelFiveDifferentiablePhysicalFix(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiablePhysicalFix(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all)

    def _pos(self, x):
        return torch.clamp(x, min=0.0)

    def _min(self, a, b):
        return torch.minimum(a, b)

    def _apply_channel_loss(self, q_comp, diag_comp, theta, basin_attr, ngage):
        if self.dry_channel_loss is not True:
            zeros = torch.zeros_like(q_comp)
            return q_comp, zeros, zeros

        if 'SMS' in diag_comp:
            soil_m = self._component_tensor_4d(diag_comp['SMS'], ngage)
        elif 'soil_moisture' in diag_comp:
            soil_m = self._component_tensor_4d(diag_comp['soil_moisture'], ngage)
        else:
            soil_m = torch.ones_like(q_comp) * 100.0

        smsc = self._theta_to_smsc(theta, ngage).unsqueeze(0)
        wetness = torch.clamp(soil_m / torch.clamp(smsc, min=1e-6), 0.0, 1.0)
        dryness = 1.0 - wetness

        gamma = self.channel_loss_max * torch.sigmoid(self.channelLossHead(basin_attr))
        gamma = gamma.unsqueeze(0).unsqueeze(-1)
        loss_frac = 1.0 - torch.exp(-gamma * dryness)
        loss_frac = torch.clamp(loss_frac, 0.0, 0.95)
        channel_loss_raw = q_comp * loss_frac
        channel_loss = self._min(channel_loss_raw, q_comp)
        q_after = self._pos(q_comp - channel_loss)
        return q_after, channel_loss, loss_frac

    def _apply_zero_flow_gate(self, q_comp, x, diag_comp, theta, ngage):
        if self.zero_flow_gate_enabled is not True:
            ones = torch.ones_like(q_comp)
            zeros = torch.zeros_like(q_comp)
            return q_comp, ones, zeros, ones

        T, B, M, _ = q_comp.shape
        if 'SMS' in diag_comp:
            soil_m = self._component_tensor_4d(diag_comp['SMS'], ngage)
        elif 'soil_moisture' in diag_comp:
            soil_m = self._component_tensor_4d(diag_comp['soil_moisture'], ngage)
        else:
            soil_m = torch.ones_like(q_comp) * 100.0
        smsc = self._theta_to_smsc(theta, ngage).unsqueeze(0)
        wetness = torch.clamp(soil_m / torch.clamp(smsc, min=1e-6), 0.0, 1.0)

        p_t = x[:, :, 0:1].unsqueeze(2).repeat(1, 1, M, 1)
        if x.shape[-1] >= 5:
            sin_t = x[:, :, 3:4].unsqueeze(2).repeat(1, 1, M, 1)
            cos_t = x[:, :, 4:5].unsqueeze(2).repeat(1, 1, M, 1)
        else:
            sin_t = torch.zeros_like(q_comp)
            cos_t = torch.ones_like(q_comp)

        gate_in = torch.cat([p_t / 20.0, wetness, q_comp / 20.0, sin_t, cos_t], dim=-1)
        gate_in_flat = gate_in.view(T * B * M, 5)
        logits_all = self.zeroFlowGate(gate_in_flat)
        comp_idx = torch.arange(M, device=q_comp.device).view(1, 1, M, 1).repeat(T, B, 1, 1).view(T * B * M, 1)
        logits = logits_all.gather(1, comp_idx)
        p_flow = torch.sigmoid(logits).view(T, B, M, 1)

        if self.gate_variant == 'explicit':
            gate_loss_frac = 1.0 - p_flow
            keep_fraction = p_flow
        else:
            gate_loss_frac = self.gate_strength_max * (1.0 - p_flow)
            keep_fraction = 1.0 - gate_loss_frac
        gate_loss_raw = q_comp * gate_loss_frac
        gate_loss = self._min(gate_loss_raw, q_comp)
        q_after = self._pos(q_comp - gate_loss)
        return q_after, p_flow, gate_loss, keep_fraction

    def forward(self, x, z, doDropMC=False, return_diagnostics=False, return_component_diagnostics=False):
        nt_x = x.shape[0]
        ngage = z.shape[1]
        basin_attr = z[-1, :, -self.nattr:]

        if z.shape[2] > self.nattr:
            snow_frac_raw = torch.clamp(z[-1, :, -self.nattr - 1:-self.nattr], min=0.0, max=1.0)
        else:
            snow_frac_raw = None

        staticFeat = self.staticFeat(basin_attr)
        staticParams0 = self.staticOut(staticFeat)

        cursor = 0
        static0 = staticParams0[:, cursor:cursor + self.nstaticpm].view(ngage, self.nfea, self.nmul)
        static0 = static0 + self.compStaticBias
        snowpara = torch.sigmoid(static0)
        cursor += self.nstaticpm

        routpara0 = staticParams0[:, cursor:cursor + self.nroutpm]
        if self.component_routing is True:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        elif self.comprout is False:
            routpara = torch.sigmoid(routpara0)
        else:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        cursor += self.nroutpm

        if self.nwtspm == 0:
            wts = None
        else:
            wtspara = staticParams0[:, cursor:cursor + self.nwtspm] + self.compWeightBias
            wts = F.softmax(wtspara, dim=-1)
        self._last_wts = wts

        if self.lgdyn is False:
            lg_dyn = None
        else:
            lg_dyn_seq = self.lstmdyn(z)
            lg_attr_bias = self.lgAttr(basin_attr).unsqueeze(0).repeat(lg_dyn_seq.shape[0], 1, 1)
            lg_dyn = torch.sigmoid(lg_dyn_seq + lg_attr_bias)

        x_rep = x.unsqueeze(2).repeat(1, 1, self.nmul, 1).view(nt_x, ngage * self.nmul, x.shape[2])
        x_bt = x_rep.permute(1, 0, 2)
        theta = snowpara.permute(0, 2, 1).contiguous().view(ngage * self.nmul, self.nfea)

        if lg_dyn is None:
            lg_bt = None
        else:
            lg_bt = lg_dyn.permute(1, 2, 0).contiguous().view(ngage * self.nmul, lg_dyn.shape[0], 1)
        if lg_bt is not None and self.inittime > 0:
            lg_bt_main = lg_bt[:, self.inittime:, :]
        else:
            lg_bt_main = lg_bt

        if snow_frac_raw is None:
            snow_frac_rep = None
        else:
            snow_frac_rep = snow_frac_raw.unsqueeze(1).repeat(1, self.nmul, 1).view(ngage * self.nmul, 1)

        need_process_diag = return_diagnostics or return_component_diagnostics or self.dry_channel_loss or self.zero_flow_gate_enabled

        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            x_use = x[self.inittime:, :, :]
            _, warm_state = self.simhyd_analysis(
                warm_inputs,
                theta,
                lg_dyn_seq=None,
                lg_dyn_weight=self.lgdynweight,
                snow_frac_raw=snow_frac_rep,
                return_final_state=True)
            if need_process_diag:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    main_inputs,
                    theta,
                    initial_state=warm_state,
                    lg_dyn_seq=lg_bt_main,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_diagnostics=True,
                    return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    main_inputs,
                    theta,
                    initial_state=warm_state,
                    lg_dyn_seq=lg_bt_main,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_regularization=True)
                diag_comp = None
        else:
            x_use = x
            if need_process_diag:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    x_bt,
                    theta,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_diagnostics=True,
                    return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    x_bt,
                    theta,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_regularization=True)
                diag_comp = None

        reg_total = self.reg_amp_w * reg_terms['dynamic_amplitude_loss'] \
            + self.reg_smooth_w * reg_terms['dynamic_smoothness_loss'] \
            + self.reg_part_w * reg_terms['partition_entropy_loss']
        self._last_aux_terms = reg_terms
        self._last_aux_loss = reg_total

        q_comp_raw = q_seq.view(ngage, self.nmul, q_seq.shape[1], 1).permute(2, 0, 1, 3)
        q_comp_raw = torch.clamp(q_comp_raw, min=0.0)
        q_after_loss, channel_loss, channel_loss_fraction = self._apply_channel_loss(q_comp_raw, diag_comp, theta, basin_attr, ngage)
        q_after_gate, zero_flow_probability, gate_loss, zero_flow_keep_fraction = self._apply_zero_flow_gate(
            q_after_loss, x_use, diag_comp, theta, ngage)
        q_comp = torch.clamp(q_after_gate, min=0.0)

        q_mix_before_routing = self._mix_or_mean(q_comp, wts)

        route_mult_seq = None
        if self.routOpt is True and self.component_routing is True:
            q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
            assert routpara.shape[0] == ngage * self.nmul and routpara.shape[1] == 2
            if self.dynamic_routing_scale is True:
                if diag_comp is not None and 'SMS' in diag_comp:
                    sm_comp = self._component_tensor_4d(diag_comp['SMS'], ngage)
                elif diag_comp is not None and 'soil_moisture' in diag_comp:
                    sm_comp = self._component_tensor_4d(diag_comp['soil_moisture'], ngage)
                else:
                    sm_comp = torch.ones_like(q_comp) * 50.0
                smsc_comp = self._theta_to_smsc(theta, ngage).unsqueeze(0).repeat(q_comp.shape[0], 1, 1, 1)
                p_rep = x_use[:, :, 0:1].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                if x_use.shape[-1] >= 5:
                    sin_rep = x_use[:, :, 3:4].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                    cos_rep = x_use[:, :, 4:5].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                else:
                    sin_rep = torch.zeros_like(q_comp)
                    cos_rep = torch.ones_like(q_comp)
                x_route = torch.cat([p_rep, sm_comp, smsc_comp, sin_rep, cos_rep], dim=-1).view(q_comp.shape[0], ngage * self.nmul, 5)
                q_routed_flat, route_mult_flat = self._route_q_dynamic_scale(q_for_routing, routpara, x_route)
                route_mult_4d = route_mult_flat.view(q_comp.shape[0], ngage, self.nmul, 1)
                route_mult_seq = self._mix_or_mean(route_mult_4d, wts)
                self._last_aux_loss = self._last_aux_loss + self.reg_amp_w * torch.mean((route_mult_flat - 1.0) ** 2)
                self._last_aux_loss = self._last_aux_loss + self.reg_smooth_w * torch.mean(
                    (route_mult_flat[1:, :, :] - route_mult_flat[:-1, :, :]) ** 2)
                q_routed = q_routed_flat.view(q_comp.shape[0], ngage, self.nmul, 1)
            else:
                q_routed = self._route_q(q_for_routing, routpara).view(q_comp.shape[0], ngage, self.nmul, 1)
            out = self._mix_or_mean(q_routed, wts)
        else:
            q_routed = q_comp
            if self.routOpt is True and self.dynamic_routing_scale is True and self.comprout is False:
                if diag_comp is not None and 'SMS' in diag_comp:
                    sm_mix = self._mix_component_tensor(diag_comp['SMS'], ngage)
                elif diag_comp is not None and 'soil_moisture' in diag_comp:
                    sm_mix = self._mix_component_tensor(diag_comp['soil_moisture'], ngage)
                else:
                    sm_mix = torch.ones_like(q_mix_before_routing) * 50.0
                smsc_mix = torch.ones_like(q_mix_before_routing) * 200.0
                x_route = torch.cat([x_use[:, :, 0:1], sm_mix, smsc_mix, x_use[:, :, 3:4], x_use[:, :, 4:5]], dim=2)
                out, route_mult_seq = self._route_q_dynamic_scale(q_mix_before_routing, routpara, x_route)
                self._last_aux_loss = self._last_aux_loss + self.reg_amp_w * torch.mean((route_mult_seq - 1.0) ** 2)
                self._last_aux_loss = self._last_aux_loss + self.reg_smooth_w * torch.mean(
                    (route_mult_seq[1:, :, :] - route_mult_seq[:-1, :, :]) ** 2)
            elif self.routOpt is True:
                if self.comprout is True:
                    q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
                    q_routed = self._route_q(q_for_routing, routpara).view(q_comp.shape[0], ngage, self.nmul, 1)
                    out = self._mix_or_mean(q_routed, wts)
                else:
                    assert routpara.shape[0] == ngage and routpara.shape[1] == 2
                    out = self._route_q(q_mix_before_routing, routpara)
            else:
                out = q_mix_before_routing

        out = torch.clamp(out, min=0.0)
        if return_diagnostics is not True and return_component_diagnostics is not True:
            return out

        diag_out = dict()
        if diag_comp is not None:
            for name, tensor_comp in diag_comp.items():
                diag_out[name] = self._mix_component_tensor(tensor_comp, ngage)
        diag_out['total_discharge'] = out
        diag_out['q_mix_before_routing'] = q_mix_before_routing
        diag_out['component_discharge_raw'] = self._mix_or_mean(q_comp_raw, wts)
        diag_out['channel_loss'] = self._mix_or_mean(channel_loss, wts)
        diag_out['channel_loss_fraction'] = self._mix_or_mean(channel_loss_fraction, wts)
        diag_out['gate_loss'] = self._mix_or_mean(gate_loss, wts)
        diag_out['zero_flow_probability'] = self._mix_or_mean(zero_flow_probability, wts)
        diag_out['zero_flow_keep_fraction'] = self._mix_or_mean(zero_flow_keep_fraction, wts)
        if route_mult_seq is not None:
            diag_out['route_b_t_multiplier'] = route_mult_seq

        if return_component_diagnostics is True:
            if diag_comp is not None:
                for name, tensor_comp in diag_comp.items():
                    diag_out[name + '_components'] = self._component_tensor_4d(tensor_comp, ngage).squeeze(-1)
            diag_out['q_raw_process_components'] = q_comp_raw.squeeze(-1)
            diag_out['q_after_channel_loss_components'] = q_after_loss.squeeze(-1)
            diag_out['q_after_gate_components'] = q_after_gate.squeeze(-1)
            diag_out['q_routed_components'] = q_routed.squeeze(-1)
            diag_out['channel_loss_components'] = channel_loss.squeeze(-1)
            diag_out['gate_loss_components'] = gate_loss.squeeze(-1)
            diag_out['channel_loss_fraction_components'] = channel_loss_fraction.squeeze(-1)
            diag_out['zero_flow_probability_components'] = zero_flow_probability.squeeze(-1)
            diag_out['zero_flow_keep_fraction_components'] = zero_flow_keep_fraction.squeeze(-1)
            diag_out['component_weights'] = wts
            if self.component_routing is True:
                routpara_view = torch.sigmoid(routpara0).view(ngage, self.nmul, 2)
                diag_out['route_a_components'] = 0.0 + routpara_view[:, :, 0] * 2.9
                diag_out['route_b_components'] = 0.0 + routpara_view[:, :, 1] * 6.5

        return out, diag_out


class DynamicSimHydModelFiveDifferentiablePhysicalaSrzHBVReservoirRegulated(
        DynamicSimHydModelFiveDifferentiablePhysicalFix):
    """
    Active-root-zone Model 6 variant with regulated HBV-style UZ/LZ
    reservoirs, throttled recharge/percolation into LZ, and zero external
    groundwater loss.
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True, dynamic_cfmax_snow=True,
                 dynamic_routing_scale=False, dynamic_all=False, dyn_hidden=32,
                 use_lg_transfer=True):
        super(DynamicSimHydModelFiveDifferentiablePhysicalaSrzHBVReservoirRegulated, self).__init__(
            mode=mode, theta_is_raw=theta_is_raw, smooth=smooth, eps=eps, rain_snow_gain=rain_snow_gain,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, dyn_hidden=dyn_hidden)
        self.use_lg_transfer = use_lg_transfer

    def _expand_asrz_hbv_regulated(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)
        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = theta[:, 2:3] * 6.0
        SMSC_legacy = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = theta[:, 4:5]
        CRAK = theta[:, 5:6]
        K_FAST = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        LG = theta[:, 7:8] * 0.2
        TT = -2.5 + theta[:, 8:9] * 5.0
        CFMAX = 0.5 + theta[:, 9:10] * (10.0 - 0.5)
        CFR = theta[:, 10:11] * 0.1
        CWH = theta[:, 11:12] * 0.2
        SG_CRIT = theta[:, 12:13] * 300.0
        theta_ab = 0.5 + theta[:, 13:14] * 0.5
        theta_ak = 1.0 + theta[:, 14:15] * 9.0
        theta_cap = 10.0 + theta[:, 15:16] * (1500.0 - 10.0)
        theta_efmax = 0.5 + theta[:, 16:17] * 0.5
        theta_wetpoint = 0.3 + theta[:, 17:18] * 0.6
        K_SLOW = 0.001 + theta[:, 18:19] * (0.3 - 0.001)
        PERC_CAP = theta[:, 19:20] * 10.0
        K_CAP = theta[:, 20:21] * 0.05
        K_CHANNEL = 0.05 + theta[:, 21:22] * (1.0 - 0.05)
        LZ_CAP = 50.0 + theta[:, 22:23] * (1000.0 - 50.0)
        beta_slow = theta[:, 23:24] * 3.0
        return (
            INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K_FAST, LG, TT, CFMAX, CFR, CWH, SG_CRIT,
            theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint,
            K_SLOW, PERC_CAP, K_CAP, K_CHANNEL, LZ_CAP, beta_slow
        )

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        (INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K_FAST, LG, TT, CFMAX_base, CFR, CWH, SG_CRIT,
         theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint,
         K_SLOW, PERC_CAP, K_CAP, K_CHANNEL, LZ_CAP, beta_slow) = self._expand_asrz_hbv_regulated(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SA = torch.zeros(B, 1, device=device, dtype=dtype)
            UZ = torch.zeros(B, 1, device=device, dtype=dtype)
            LZ = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
            init_uz = torch.zeros(B, 1, device=device, dtype=dtype)
            init_lz = torch.zeros(B, 1, device=device, dtype=dtype)
        else:
            SA = initial_state[:, 0:1]
            UZ = initial_state[:, 1:2]
            LZ = initial_state[:, 2:3]
            SNOWPACK = initial_state[:, 3:4]
            MELTWATER = initial_state[:, 4:5]
            init_uz = initial_state[:, 1:2]
            init_lz = initial_state[:, 2:3]

        if lg_dyn_seq is not None and (lg_dyn_seq.shape[0] != B or lg_dyn_seq.shape[1] != Tlen):
            raise ValueError('lg_dyn_seq must have shape [B, T, 1] matching inputs')

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = dict()
        if return_diagnostics is True:
            for name in [
                    'Sa_prev', 'UZ_prev', 'LZ_prev', 'SNOWPACK_prev', 'MELTWATER_prev',
                    'Sa', 'UZ', 'LZ', 'SNOWPACK', 'MELTWATER',
                    'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release_to_soil',
                    'interception_storage', 'interception_evaporation', 'actual_ET',
                    'Smoist', 'theta_cap', 'alpha_t', 'P_accessible', 'P_inaccessible', 'aSrz_overflow',
                    'surface_runoff', 'interflow', 'recharge_to_groundwater', 'soil_overflow',
                    'recharge_throttle', 'REC_to_UZ', 'REC_rejected',
                    'Q_fast_raw', 'Q_fast', 'PERC_raw', 'PERC', 'rejected_perc',
                    'UZ_to_LZ_extra', 'LZ_overflow', 'Q_slow_raw', 'Q_slow',
                    'CAP', 'channel_loss', 'gate_loss',
                    'q_raw_process', 'q_after_channel_loss', 'q_after_gate', 'Q_process',
                    'partition_sum_error', 'soil_local_residual', 'uz_local_residual', 'lz_local_residual',
                    'snowpack_local_residual', 'meltwater_local_residual', 'snow_total_local_residual',
                    'process_local_residual', 'INSC', 'COEF_t', 'SQ_t', 'K_fast_t', 'K_slow_t',
                    'PERC_cap_t', 'K_cap_t', 'K_channel', 'LG_t', 'TT', 'SG_CRIT', 'CFMAX_t', 'CFR', 'CWH',
                    'LZ_cap_t', 'beta_slow_t'
            ]:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SA0 = self._min(self._pos(SA), theta_cap)
            UZ0 = self._pos(UZ)
            LZ0 = self._pos(LZ)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            Smoist0 = torch.clamp(SA0 / (theta_cap + 1e-8), min=0.0, max=1.0)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0, Tt / 20.0, E0t / 10.0,
                SA0 / 300.0, Smoist0, (UZ0 + LZ0) / 300.0, SNOWPACK0 / 300.0, sin_t, cos_t
            ], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)

            if self.dynamic_etgam is True:
                ETGAM_t = 0.25 + 3.75 * torch.sigmoid(dyn_raw[:, self.dyn_slices['etgam']])
            else:
                ETGAM_t = torch.ones_like(SQ)

            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX_base * m_cf_eff

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()

            RAIN = Pt * frac_rain
            SNOW = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + SNOW
            melt_pot = CFMAX_t * self._pos(Tt - TT)
            melt = self._min(melt_pot, SNOWPACK1)
            MELTWATER1 = MELTWATER0 + melt
            SNOWPACK2 = self._pos(SNOWPACK1 - melt)

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            SNOWPACK3 = SNOWPACK2 + refreezing
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)

            water_holding = CWH * SNOWPACK3
            tosoil_raw = self._pos(MELTWATER2 - water_holding)
            tosoil = self._min(tosoil_raw, MELTWATER2)
            MELTWATER_next = self._pos(MELTWATER2 - tosoil)
            SNOWPACK_next = self._pos(SNOWPACK3)

            PL = RAIN + tosoil
            INT = self._min(INSC, self._min(E0t, PL))
            PL_after_int = self._pos(PL - INT)
            POT = self._pos(E0t - INT)

            alpha_t = theta_ab * torch.pow(torch.clamp(1.0 - Smoist0, min=0.0), theta_ak)
            alpha_t = torch.clamp(alpha_t, 0.0, 1.0)
            P_accessible = alpha_t * PL_after_int
            P_inaccessible = (1.0 - alpha_t) * PL_after_int

            SA_pre = SA0 + P_accessible
            g = torch.clamp(Smoist0 / torch.clamp(theta_wetpoint, min=1e-6), 0.0, 1.0)
            ET_a_potential = POT * theta_efmax * g
            ET_a = self._min(ET_a_potential, SA_pre)
            SA_after_ET = self._pos(SA_pre - ET_a)
            aSrz_overflow = self._pos(SA_after_ET - theta_cap)
            SA_tmp = self._pos(SA_after_ET - aSrz_overflow)

            available_for_partition = self._pos(P_inaccessible + aSrz_overflow)
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_inter = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([torch.log(base_surface), torch.log(base_inter), torch.log(base_recharge)], dim=1)
            if self.dynamic_partition is True and dyn_raw is not None:
                part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']]
            else:
                part_logits = base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]

            SRUN_base = f_surface * available_for_partition
            IFLOW = f_inter * available_for_partition
            REC_raw = f_recharge * available_for_partition

            recharge_throttle = torch.clamp(1.0 - LZ0 / torch.clamp(LZ_CAP, min=1e-6), min=0.0, max=1.0)
            REC_to_UZ = REC_raw * recharge_throttle
            REC_rejected = self._pos(REC_raw - REC_to_UZ)
            SRUN = SRUN_base + REC_rejected

            UZ1 = UZ0 + REC_to_UZ
            Q_fast_raw = K_FAST * UZ1
            Q_fast = self._min(Q_fast_raw, UZ1)
            UZ2 = self._pos(UZ1 - Q_fast)

            PERC_raw = self._min(PERC_CAP, UZ2)
            LZ_deficit = self._pos(LZ_CAP - LZ0)
            PERC = self._min(PERC_raw, LZ_deficit)
            rejected_perc = self._pos(PERC_raw - PERC)
            UZ3 = self._pos(UZ2 - PERC_raw)
            LZ1 = LZ0 + PERC

            if lg_dyn_seq is None:
                LG_t = LG
            else:
                LG_dyn_t = torch.clamp(lg_dyn_seq[:, t, :], 0.0, 1.0)
                LG_eff = (1.0 - lg_dyn_weight) * LG + lg_dyn_weight * (LG * LG_dyn_t * 1.5)
                LG_t = torch.clamp(LG_eff, 0.0, 0.2)

            if self.use_lg_transfer:
                UZ_to_LZ_raw = LG_t * F.softplus(SG_CRIT - UZ3)
                LZ_deficit2 = self._pos(LZ_CAP - LZ1)
                UZ_to_LZ_extra = self._min(UZ_to_LZ_raw, self._min(UZ3, LZ_deficit2))
            else:
                UZ_to_LZ_extra = torch.zeros_like(UZ3)
            UZ4 = self._pos(UZ3 - UZ_to_LZ_extra)
            LZ2 = LZ1 + UZ_to_LZ_extra

            rel_lz = torch.clamp(LZ2 / torch.clamp(LZ_CAP, min=1e-6), min=0.0, max=2.0)
            Q_slow_raw = K_SLOW * LZ2 * torch.pow(rel_lz, beta_slow)
            Q_slow = self._min(Q_slow_raw, LZ2)
            LZ3 = self._pos(LZ2 - Q_slow)

            Sa_deficit = self._pos(theta_cap - SA_tmp)
            CAP_raw = K_CAP * LZ3 * Sa_deficit / torch.clamp(theta_cap, min=1e-6)
            CAP = self._min(CAP_raw, self._min(LZ3, Sa_deficit))
            LZ4 = self._pos(LZ3 - CAP)

            LZ_overflow = self._pos(LZ4 - LZ_CAP)
            LZ_next = self._pos(LZ4 - LZ_overflow)
            SA_next = self._min(SA_tmp + CAP, theta_cap)
            UZ_next = UZ4

            Q = self._pos(SRUN + IFLOW + Q_fast + rejected_perc + Q_slow + LZ_overflow)

            soil_local_residual = P_accessible + CAP - ET_a - aSrz_overflow - (SA_next - SA0)
            uz_local_residual = REC_to_UZ - Q_fast - PERC_raw - UZ_to_LZ_extra - (UZ_next - UZ0)
            lz_local_residual = PERC + UZ_to_LZ_extra - Q_slow - CAP - LZ_overflow - (LZ_next - LZ0)
            snowpack_local_residual = SNOW - melt + refreezing - (SNOWPACK_next - SNOWPACK0)
            meltwater_local_residual = melt - refreezing - tosoil - (MELTWATER_next - MELTWATER0)
            snow_total_local_residual = SNOW - tosoil - (SNOWPACK_next - SNOWPACK0) - (MELTWATER_next - MELTWATER0)
            process_local_residual = (
                Pt - INT - ET_a - Q
                - ((SA_next - SA0) + (UZ_next - UZ0) + (LZ_next - LZ0)
                   + (SNOWPACK_next - SNOWPACK0) + (MELTWATER_next - MELTWATER0))
            )

            SA = SA_next
            UZ = UZ_next
            LZ = LZ_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)

            if return_diagnostics is True:
                diag_vals = {
                    'Sa_prev': SA0,
                    'UZ_prev': UZ0,
                    'LZ_prev': LZ0,
                    'SNOWPACK_prev': SNOWPACK0,
                    'MELTWATER_prev': MELTWATER0,
                    'Sa': SA_next,
                    'UZ': UZ_next,
                    'LZ': LZ_next,
                    'SNOWPACK': SNOWPACK_next,
                    'MELTWATER': MELTWATER_next,
                    'precipitation': Pt,
                    'rainfall': RAIN,
                    'snowfall': SNOW,
                    'snowmelt': melt,
                    'refreezing': refreezing,
                    'snow_release_to_soil': tosoil,
                    'interception_storage': torch.zeros_like(INT),
                    'interception_evaporation': INT,
                    'actual_ET': ET_a,
                    'Smoist': Smoist0,
                    'theta_cap': theta_cap,
                    'alpha_t': alpha_t,
                    'P_accessible': P_accessible,
                    'P_inaccessible': P_inaccessible,
                    'aSrz_overflow': aSrz_overflow,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'recharge_to_groundwater': REC_raw,
                    'soil_overflow': aSrz_overflow,
                    'recharge_throttle': recharge_throttle,
                    'REC_to_UZ': REC_to_UZ,
                    'REC_rejected': REC_rejected,
                    'Q_fast_raw': Q_fast_raw,
                    'Q_fast': Q_fast,
                    'PERC_raw': PERC_raw,
                    'PERC': PERC,
                    'rejected_perc': rejected_perc,
                    'UZ_to_LZ_extra': UZ_to_LZ_extra,
                    'LZ_overflow': LZ_overflow,
                    'Q_slow_raw': Q_slow_raw,
                    'Q_slow': Q_slow,
                    'CAP': CAP,
                    'channel_loss': torch.zeros_like(Q),
                    'gate_loss': torch.zeros_like(Q),
                    'q_raw_process': Q,
                    'q_after_channel_loss': Q,
                    'q_after_gate': Q,
                    'Q_process': Q,
                    'partition_sum_error': torch.abs(f_surface + f_inter + f_recharge - 1.0),
                    'soil_local_residual': soil_local_residual,
                    'uz_local_residual': uz_local_residual,
                    'lz_local_residual': lz_local_residual,
                    'snowpack_local_residual': snowpack_local_residual,
                    'meltwater_local_residual': meltwater_local_residual,
                    'snow_total_local_residual': snow_total_local_residual,
                    'process_local_residual': process_local_residual,
                    'INSC': INSC,
                    'COEF_t': COEF,
                    'SQ_t': SQ_t,
                    'K_fast_t': K_FAST,
                    'K_slow_t': K_SLOW,
                    'PERC_cap_t': PERC_CAP,
                    'K_cap_t': K_CAP,
                    'K_channel': K_CHANNEL,
                    'LG_t': LG_t,
                    'TT': TT,
                    'SG_CRIT': SG_CRIT,
                    'CFMAX_t': CFMAX_t,
                    'CFR': CFR,
                    'CWH': CWH,
                    'LZ_cap_t': LZ_CAP,
                    'beta_slow_t': beta_slow,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SA, UZ, LZ, SNOWPACK, MELTWATER], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)
        sum_p = torch.clamp(P.sum(dim=1), min=1e-6)
        drift_loss = torch.mean(torch.abs(LZ - init_lz) / sum_p)
        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': reg_part,
            'storage_drift_loss': drift_loss,
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out
        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelSix_Physical_aSrzHBVReservoirRegulated(
        MultiInv_DynamicSimHydModelSix_Physical):
    """
    Copied aSrz model with regulated HBV-style UZ/LZ reservoirs, zero external
    loss, and internal channel/gate storage.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 component_routing=True, dry_channel_loss=True, zero_flow_gate=True,
                 channel_loss_max=0.60, zero_gate_hidden=None,
                 gate_variant='soft', gate_strength_max=0.30,
                 use_lg_transfer=True, drift_reg_weight=1e-2):
        super(MultiInv_DynamicSimHydModelSix_Physical_aSrzHBVReservoirRegulated, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing,
            dry_channel_loss=dry_channel_loss, zero_flow_gate=zero_flow_gate,
            channel_loss_max=channel_loss_max, zero_gate_hidden=zero_gate_hidden,
            gate_variant=gate_variant, gate_strength_max=gate_strength_max)
        self.nfea = 24
        self.nstaticpm = self.nfea * nmul
        self.drift_reg_weight = drift_reg_weight
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.simhyd = DynamicSimHydModelFiveDifferentiablePhysicalaSrzHBVReservoirRegulated(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            use_lg_transfer=use_lg_transfer)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiablePhysicalaSrzHBVReservoirRegulated(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            use_lg_transfer=use_lg_transfer)

    def _theta_to_k_channel(self, theta, ngage):
        theta4 = theta.view(ngage, self.nmul, self.nfea)
        return 0.05 + theta4[:, :, 21:22] * (1.0 - 0.05)

    def forward(self, x, z, doDropMC=False, return_diagnostics=False, return_component_diagnostics=False):
        nt_x = x.shape[0]
        ngage = z.shape[1]
        basin_attr = z[-1, :, -self.nattr:]

        if z.shape[2] > self.nattr:
            snow_frac_raw = torch.clamp(z[-1, :, -self.nattr - 1:-self.nattr], min=0.0, max=1.0)
        else:
            snow_frac_raw = None

        staticFeat = self.staticFeat(basin_attr)
        staticParams0 = self.staticOut(staticFeat)

        cursor = 0
        static0 = staticParams0[:, cursor:cursor + self.nstaticpm].view(ngage, self.nfea, self.nmul)
        static0 = static0 + self.compStaticBias
        snowpara = torch.sigmoid(static0)
        cursor += self.nstaticpm

        routpara0 = staticParams0[:, cursor:cursor + self.nroutpm]
        if self.component_routing is True:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        elif self.comprout is False:
            routpara = torch.sigmoid(routpara0)
        else:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        cursor += self.nroutpm

        if self.nwtspm == 0:
            wts = None
        else:
            wtspara = staticParams0[:, cursor:cursor + self.nwtspm] + self.compWeightBias
            wts = F.softmax(wtspara, dim=-1)
        self._last_wts = wts

        if self.lgdyn is False:
            lg_dyn = None
        else:
            lg_dyn_seq = self.lstmdyn(z)
            lg_attr_bias = self.lgAttr(basin_attr).unsqueeze(0).repeat(lg_dyn_seq.shape[0], 1, 1)
            lg_dyn = torch.sigmoid(lg_dyn_seq + lg_attr_bias)

        x_rep = x.unsqueeze(2).repeat(1, 1, self.nmul, 1).view(nt_x, ngage * self.nmul, x.shape[2])
        x_bt = x_rep.permute(1, 0, 2)
        theta = snowpara.permute(0, 2, 1).contiguous().view(ngage * self.nmul, self.nfea)

        if lg_dyn is None:
            lg_bt = None
        else:
            lg_bt = lg_dyn.permute(1, 2, 0).contiguous().view(ngage * self.nmul, lg_dyn.shape[0], 1)
        if lg_bt is not None and self.inittime > 0:
            lg_bt_main = lg_bt[:, self.inittime:, :]
        else:
            lg_bt_main = lg_bt

        if snow_frac_raw is None:
            snow_frac_rep = None
        else:
            snow_frac_rep = snow_frac_raw.unsqueeze(1).repeat(1, self.nmul, 1).view(ngage * self.nmul, 1)

        need_process_diag = return_diagnostics or return_component_diagnostics or self.dry_channel_loss or self.zero_flow_gate_enabled

        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            x_use = x[self.inittime:, :, :]
            _, warm_state = self.simhyd_analysis(
                warm_inputs, theta, lg_dyn_seq=None, lg_dyn_weight=self.lgdynweight,
                snow_frac_raw=snow_frac_rep, return_final_state=True)
            if need_process_diag:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    main_inputs, theta, initial_state=warm_state, lg_dyn_seq=lg_bt_main,
                    lg_dyn_weight=self.lgdynweight, snow_frac_raw=snow_frac_rep,
                    return_diagnostics=True, return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    main_inputs, theta, initial_state=warm_state, lg_dyn_seq=lg_bt_main,
                    lg_dyn_weight=self.lgdynweight, snow_frac_raw=snow_frac_rep,
                    return_regularization=True)
                diag_comp = None
        else:
            x_use = x
            if need_process_diag:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    x_bt, theta, lg_dyn_seq=lg_bt, lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep, return_diagnostics=True, return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    x_bt, theta, lg_dyn_seq=lg_bt, lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep, return_regularization=True)
                diag_comp = None

        reg_total = self.reg_amp_w * reg_terms['dynamic_amplitude_loss'] \
            + self.reg_smooth_w * reg_terms['dynamic_smoothness_loss'] \
            + self.reg_part_w * reg_terms['partition_entropy_loss']
        self._last_aux_terms = reg_terms

        q_comp_raw = q_seq.view(ngage, self.nmul, q_seq.shape[1], 1).permute(2, 0, 1, 3)
        q_comp_raw = torch.clamp(q_comp_raw, min=0.0)
        q_after_loss, channel_loss, channel_loss_fraction = self._apply_channel_loss(q_comp_raw, diag_comp, theta, basin_attr, ngage)
        q_after_gate_base, zero_flow_probability, gate_loss, zero_flow_keep_fraction = self._apply_zero_flow_gate(
            q_after_loss, x_use, diag_comp, theta, ngage)
        q_after_gate_base = torch.clamp(q_after_gate_base, min=0.0)

        K_channel = self._theta_to_k_channel(theta, ngage).unsqueeze(0)
        channel_store = torch.zeros_like(q_after_gate_base[0])
        channel_prev_hist = []
        channel_hist = []
        channel_release_hist = []
        q_process_hist = []
        for t in range(q_after_gate_base.shape[0]):
            ch_prev = channel_store
            ch_in = ch_prev + channel_loss[t] + gate_loss[t]
            ch_release = torch.minimum(K_channel[0] * ch_in, ch_in)
            channel_store = torch.clamp(ch_in - ch_release, min=0.0)
            q_process = torch.clamp(q_after_gate_base[t] + ch_release, min=0.0)
            channel_prev_hist.append(ch_prev)
            channel_hist.append(channel_store)
            channel_release_hist.append(ch_release)
            q_process_hist.append(q_process)
        channel_prev_seq = torch.stack(channel_prev_hist, dim=0)
        channel_seq = torch.stack(channel_hist, dim=0)
        channel_release_seq = torch.stack(channel_release_hist, dim=0)
        q_comp = torch.stack(q_process_hist, dim=0)

        p_comp = self._component_tensor_4d(diag_comp['precipitation'], ngage)
        int_comp = self._component_tensor_4d(diag_comp['interception_evaporation'], ngage)
        et_comp = self._component_tensor_4d(diag_comp['actual_ET'], ngage)
        sa_prev = self._component_tensor_4d(diag_comp['Sa_prev'], ngage)
        uz_prev = self._component_tensor_4d(diag_comp['UZ_prev'], ngage)
        lz_prev = self._component_tensor_4d(diag_comp['LZ_prev'], ngage)
        snow_prev = self._component_tensor_4d(diag_comp['SNOWPACK_prev'], ngage)
        melt_prev = self._component_tensor_4d(diag_comp['MELTWATER_prev'], ngage)
        sa_next = self._component_tensor_4d(diag_comp['Sa'], ngage)
        uz_next = self._component_tensor_4d(diag_comp['UZ'], ngage)
        lz_next = self._component_tensor_4d(diag_comp['LZ'], ngage)
        snow_next = self._component_tensor_4d(diag_comp['SNOWPACK'], ngage)
        melt_next = self._component_tensor_4d(diag_comp['MELTWATER'], ngage)
        delta_storage = (
            (sa_next - sa_prev) + (uz_next - uz_prev) + (lz_next - lz_prev)
            + (snow_next - snow_prev) + (melt_next - melt_prev)
            + (channel_seq - channel_prev_seq)
        )
        process_local_residual = p_comp - int_comp - et_comp - q_comp - delta_storage

        cum_p_comp = torch.clamp(torch.sum(p_comp, dim=0), min=1e-6)
        channel_drift_loss = torch.mean(torch.abs(channel_seq[-1] - channel_prev_seq[0]) / cum_p_comp)
        reg_total = reg_total + self.drift_reg_weight * (reg_terms.get('storage_drift_loss', 0.0) + channel_drift_loss)
        self._last_aux_loss = reg_total

        q_mix_before_routing = self._mix_or_mean(q_comp, wts)
        route_mult_seq = None
        if self.routOpt is True and self.component_routing is True:
            q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
            if self.dynamic_routing_scale is True:
                if 'Sa' in diag_comp:
                    sm_comp = self._component_tensor_4d(diag_comp['Sa'], ngage)
                else:
                    sm_comp = torch.ones_like(q_comp) * 50.0
                smsc_comp = self._theta_to_smsc(theta, ngage).unsqueeze(0).repeat(q_comp.shape[0], 1, 1, 1)
                p_rep = x_use[:, :, 0:1].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                if x_use.shape[-1] >= 5:
                    sin_rep = x_use[:, :, 3:4].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                    cos_rep = x_use[:, :, 4:5].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                else:
                    sin_rep = torch.zeros_like(q_comp)
                    cos_rep = torch.ones_like(q_comp)
                x_route = torch.cat([p_rep, sm_comp, smsc_comp, sin_rep, cos_rep], dim=-1).view(q_comp.shape[0], ngage * self.nmul, 5)
                q_routed_flat, route_mult_flat = self._route_q_dynamic_scale(q_for_routing, routpara, x_route)
                route_mult_4d = route_mult_flat.view(q_comp.shape[0], ngage, self.nmul, 1)
                route_mult_seq = self._mix_or_mean(route_mult_4d, wts)
                self._last_aux_loss = self._last_aux_loss + self.reg_amp_w * torch.mean((route_mult_flat - 1.0) ** 2)
                self._last_aux_loss = self._last_aux_loss + self.reg_smooth_w * torch.mean(
                    (route_mult_flat[1:, :, :] - route_mult_flat[:-1, :, :]) ** 2)
                q_routed = q_routed_flat.view(q_comp.shape[0], ngage, self.nmul, 1)
            else:
                q_routed = self._route_q(q_for_routing, routpara).view(q_comp.shape[0], ngage, self.nmul, 1)
            out = self._mix_or_mean(q_routed, wts)
        else:
            q_routed = q_comp
            if self.routOpt is True:
                if self.comprout is True:
                    q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
                    q_routed = self._route_q(q_for_routing, routpara).view(q_comp.shape[0], ngage, self.nmul, 1)
                    out = self._mix_or_mean(q_routed, wts)
                else:
                    out = self._route_q(q_mix_before_routing, routpara)
            else:
                out = q_mix_before_routing

        out = torch.clamp(out, min=0.0)
        if return_diagnostics is not True and return_component_diagnostics is not True:
            return out

        diag_out = dict()
        if diag_comp is not None:
            for name, tensor_comp in diag_comp.items():
                diag_out[name] = self._mix_component_tensor(tensor_comp, ngage)
        diag_out['total_discharge'] = out
        diag_out['q_mix_before_routing'] = q_mix_before_routing
        diag_out['component_discharge_raw'] = self._mix_or_mean(q_comp_raw, wts)
        diag_out['channel_loss'] = self._mix_or_mean(channel_loss, wts)
        diag_out['channel_loss_fraction'] = self._mix_or_mean(channel_loss_fraction, wts)
        diag_out['gate_loss'] = self._mix_or_mean(gate_loss, wts)
        diag_out['zero_flow_probability'] = self._mix_or_mean(zero_flow_probability, wts)
        diag_out['zero_flow_keep_fraction'] = self._mix_or_mean(zero_flow_keep_fraction, wts)
        diag_out['CHANNEL_STORE'] = self._mix_or_mean(channel_seq, wts)
        diag_out['CHANNEL_RELEASE'] = self._mix_or_mean(channel_release_seq, wts)
        diag_out['Q_process'] = q_mix_before_routing
        diag_out['water_balance_residual'] = self._mix_or_mean(process_local_residual, wts)
        if route_mult_seq is not None:
            diag_out['route_b_t_multiplier'] = route_mult_seq

        if return_component_diagnostics is True:
            if diag_comp is not None:
                for name, tensor_comp in diag_comp.items():
                    diag_out[name + '_components'] = self._component_tensor_4d(tensor_comp, ngage).squeeze(-1)
            diag_out['q_raw_process_components'] = q_comp_raw.squeeze(-1)
            diag_out['q_after_channel_loss_components'] = q_after_loss.squeeze(-1)
            diag_out['q_after_gate_base_components'] = q_after_gate_base.squeeze(-1)
            diag_out['q_after_gate_components'] = q_comp.squeeze(-1)
            diag_out['Q_process_components'] = q_comp.squeeze(-1)
            diag_out['q_routed_components'] = q_routed.squeeze(-1)
            diag_out['channel_loss_components'] = channel_loss.squeeze(-1)
            diag_out['gate_loss_components'] = gate_loss.squeeze(-1)
            diag_out['channel_loss_fraction_components'] = channel_loss_fraction.squeeze(-1)
            diag_out['zero_flow_probability_components'] = zero_flow_probability.squeeze(-1)
            diag_out['zero_flow_keep_fraction_components'] = zero_flow_keep_fraction.squeeze(-1)
            diag_out['CHANNEL_STORE_components'] = channel_seq.squeeze(-1)
            diag_out['CHANNEL_RELEASE_components'] = channel_release_seq.squeeze(-1)
            diag_out['process_local_residual_components'] = process_local_residual.squeeze(-1)
            diag_out['component_weights'] = wts
            if self.component_routing is True:
                routpara_view = torch.sigmoid(routpara0).view(ngage, self.nmul, 2)
                diag_out['route_a_components'] = 0.0 + routpara_view[:, :, 0] * 2.9
                diag_out['route_b_components'] = 0.0 + routpara_view[:, :, 1] * 6.5

        return out, diag_out


class DynamicSimHydModelFiveDifferentiablePhysicalaSrzMinimal(DynamicSimHydModelFiveDifferentiablePhysicalFix):
    """
    Minimal active-root-zone-storage copy of the retained soft-gate process model.
    It keeps the snow/interception/GW/channel/gate structure but replaces the
    old SMS-based ET formulation with an active root-zone storage state Sa.
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True, dynamic_cfmax_snow=True,
                 dynamic_routing_scale=False, dynamic_all=False, dyn_hidden=32):
        super(DynamicSimHydModelFiveDifferentiablePhysicalaSrzMinimal, self).__init__(
            mode=mode, theta_is_raw=theta_is_raw, smooth=smooth, eps=eps, rain_snow_gain=rain_snow_gain,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, dyn_hidden=dyn_hidden)

    def _expand_asrz(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)
        theta = torch.clamp(theta, 0.0, 1.0)
        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = 0.0 + theta[:, 2:3] * (6.0 - 0.0)
        SMSC_legacy = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = 0.0 + theta[:, 4:5] * (1.0 - 0.0)
        CRAK = 0.0 + theta[:, 5:6] * (1.0 - 0.0)
        K = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        LG = 0.0 + theta[:, 7:8] * (0.2 - 0.0)
        TT = -2.5 + theta[:, 8:9] * (2.5 - (-2.5))
        CFMAX = 0.5 + theta[:, 9:10] * (10.0 - 0.5)
        CFR = 0.0 + theta[:, 10:11] * (0.1 - 0.0)
        CWH = 0.0 + theta[:, 11:12] * (0.2 - 0.0)
        SG_CRIT = 0.0 + theta[:, 12:13] * (300.0 - 0.0)
        theta_ab = 0.5 + theta[:, 13:14] * (1.0 - 0.5)
        theta_ak = 1.0 + theta[:, 14:15] * (10.0 - 1.0)
        theta_cap = 10.0 + theta[:, 15:16] * (1500.0 - 10.0)
        theta_efmax = 0.5 + theta[:, 16:17] * (1.0 - 0.5)
        theta_wetpoint = 0.3 + theta[:, 17:18] * (0.9 - 0.3)
        return (INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH, SG_CRIT,
                theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint)

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        (INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH, SG_CRIT,
         theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint) = self._expand_asrz(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SA = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
        else:
            SA = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]
            SNOWPACK = initial_state[:, 2:3]
            MELTWATER = initial_state[:, 3:4]

        if lg_dyn_seq is not None and (lg_dyn_seq.shape[0] != B or lg_dyn_seq.shape[1] != Tlen):
            raise ValueError('lg_dyn_seq must have shape [B, T, 1] matching inputs')

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = dict()
        if return_diagnostics is True:
            for name in [
                    'SMS_prev', 'GW_prev', 'SNOWPACK_prev', 'MELTWATER_prev',
                    'SMS', 'GW', 'SNOWPACK', 'MELTWATER',
                    'Sa', 'Smoist', 'theta_cap', 'alpha_t', 'P_accessible', 'P_inaccessible', 'ET_a',
                    'aSrz_overflow', 'water_balance_residual',
                    'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release_to_soil',
                    'interception_storage', 'interception_evaporation', 'actual_ET', 'infiltration',
                    'surface_runoff', 'interflow', 'recharge_to_groundwater', 'soil_overflow',
                    'baseflow_raw', 'baseflow_capped', 'baseflow',
                    'groundwater_loss_raw', 'groundwater_loss_capped', 'groundwater_loss',
                    'channel_loss', 'gate_loss', 'q_raw_process', 'q_after_channel_loss', 'q_after_gate',
                    'total_discharge', 'INSC', 'COEF_t', 'SQ_t', 'SMSC', 'SMSC_legacy',
                    'ETGAM_t', 'SUB_t', 'INTER_t', 'RECH_t', 'CRAK_t', 'K_t', 'LG_t', 'TT',
                    'SG_CRIT', 'CFMAX_t', 'CFR', 'CWH', 'partition_sum_error',
                    'soil_local_residual', 'gw_local_residual', 'snowpack_local_residual',
                    'meltwater_local_residual', 'snow_total_local_residual', 'process_local_residual']:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SA0 = self._min(self._pos(SA), theta_cap)
            GW0 = self._pos(GW)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            Smoist = torch.clamp(SA0 / (theta_cap + 1e-8), min=0.0, max=1.0)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0,
                Tt / 20.0,
                E0t / 10.0,
                SA0 / 300.0,
                Smoist,
                GW0 / 300.0,
                SNOWPACK0 / 300.0,
                sin_t,
                cos_t], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)

            if self.dynamic_etgam is True:
                ETGAM_t = 0.25 + 3.75 * torch.sigmoid(dyn_raw[:, self.dyn_slices['etgam']])
            else:
                ETGAM_t = torch.ones_like(SQ)

            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX * m_cf_eff

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()

            RAIN = Pt * frac_rain
            SNOW = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + SNOW
            melt_pot = CFMAX_t * self._pos(Tt - TT)
            melt = self._min(melt_pot, SNOWPACK1)
            MELTWATER1 = MELTWATER0 + melt
            SNOWPACK2 = self._pos(SNOWPACK1 - melt)

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            SNOWPACK3 = SNOWPACK2 + refreezing
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)

            water_holding = CWH * SNOWPACK3
            tosoil_raw = self._pos(MELTWATER2 - water_holding)
            tosoil = self._min(tosoil_raw, MELTWATER2)
            MELTWATER_next = self._pos(MELTWATER2 - tosoil)
            SNOWPACK_next = self._pos(SNOWPACK3)

            PL = RAIN + tosoil
            INT = self._min(INSC, self._min(E0t, PL))
            PL_after_int = self._pos(PL - INT)
            POT = self._pos(E0t - INT)

            alpha_t = theta_ab * torch.pow(torch.clamp(1.0 - Smoist, min=0.0, max=1.0), theta_ak)
            alpha_t = torch.clamp(alpha_t, 0.0, 1.0)
            P_accessible = alpha_t * PL_after_int
            P_inaccessible = (1.0 - alpha_t) * PL_after_int

            SA_pre = SA0 + P_accessible
            g = torch.clamp(Smoist / (theta_wetpoint + 1e-8), min=0.0, max=1.0)
            ET_a_potential = POT * theta_efmax * g
            ET_a = self._min(ET_a_potential, SA_pre)

            SA_after_ET = self._pos(SA_pre - ET_a)
            aSrz_overflow = self._pos(SA_after_ET - theta_cap)
            SA_next = self._pos(SA_after_ET - aSrz_overflow)

            available_for_partition = self._pos(P_inaccessible + aSrz_overflow)
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_inter = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([
                torch.log(base_surface),
                torch.log(base_inter),
                torch.log(base_recharge)], dim=1)
            if self.dynamic_partition is True:
                part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']]
            else:
                part_logits = base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]

            SRUN = f_surface * available_for_partition
            IFLOW = f_inter * available_for_partition
            REC = f_recharge * available_for_partition
            REC_total = REC

            if lg_dyn_seq is None:
                LG_t = LG
            else:
                LG_dyn_t = torch.clamp(lg_dyn_seq[:, t, :], 0.0, 1.0)
                LG_eff = (1.0 - lg_dyn_weight) * LG + lg_dyn_weight * (LG * LG_dyn_t * 1.5)
                LG_t = torch.clamp(LG_eff, 0.0, 0.2)

            available_before_baseflow = self._pos(GW0 + REC_total)
            BAS_raw = K * F.softplus(GW0 - SG_CRIT)
            BAS = self._min(BAS_raw, available_before_baseflow)
            available_after_baseflow = self._pos(available_before_baseflow - BAS)
            GWLOSS_raw = LG_t * F.softplus(SG_CRIT - GW0)
            GWLOSS = self._min(GWLOSS_raw, available_after_baseflow)
            GW_next = self._pos(available_after_baseflow - GWLOSS)
            Q = self._pos(SRUN + IFLOW + BAS)

            soil_local_residual = P_accessible - ET_a - aSrz_overflow - (SA_next - SA0)
            gw_local_residual = REC_total - BAS - GWLOSS - (GW_next - GW0)
            snowpack_local_residual = SNOW - melt + refreezing - (SNOWPACK_next - SNOWPACK0)
            meltwater_local_residual = melt - refreezing - tosoil - (MELTWATER_next - MELTWATER0)
            snow_total_local_residual = SNOW - tosoil - (SNOWPACK_next - SNOWPACK0) - (MELTWATER_next - MELTWATER0)
            process_local_residual = (
                Pt - INT - ET_a - GWLOSS - Q
                - ((SA_next - SA0) + (GW_next - GW0) + (SNOWPACK_next - SNOWPACK0) + (MELTWATER_next - MELTWATER0))
            )

            SA = SA_next
            GW = GW_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)

            if return_diagnostics is True:
                diag_vals = {
                    'SMS_prev': SA0,
                    'GW_prev': GW0,
                    'SNOWPACK_prev': SNOWPACK0,
                    'MELTWATER_prev': MELTWATER0,
                    'SMS': SA_next,
                    'GW': GW_next,
                    'SNOWPACK': SNOWPACK_next,
                    'MELTWATER': MELTWATER_next,
                    'Sa': SA_next,
                    'Smoist': Smoist,
                    'theta_cap': theta_cap,
                    'alpha_t': alpha_t,
                    'P_accessible': P_accessible,
                    'P_inaccessible': P_inaccessible,
                    'ET_a': ET_a,
                    'aSrz_overflow': aSrz_overflow,
                    'water_balance_residual': process_local_residual,
                    'precipitation': Pt,
                    'rainfall': RAIN,
                    'snowfall': SNOW,
                    'snowmelt': melt,
                    'refreezing': refreezing,
                    'snow_release_to_soil': tosoil,
                    'interception_storage': torch.zeros_like(INT),
                    'interception_evaporation': INT,
                    'actual_ET': ET_a,
                    'infiltration': P_accessible,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'recharge_to_groundwater': REC_total,
                    'soil_overflow': aSrz_overflow,
                    'baseflow_raw': BAS_raw,
                    'baseflow_capped': BAS,
                    'baseflow': BAS,
                    'groundwater_loss_raw': GWLOSS_raw,
                    'groundwater_loss_capped': GWLOSS,
                    'groundwater_loss': GWLOSS,
                    'channel_loss': torch.zeros_like(Q),
                    'gate_loss': torch.zeros_like(Q),
                    'q_raw_process': Q,
                    'q_after_channel_loss': Q,
                    'q_after_gate': Q,
                    'total_discharge': Q,
                    'INSC': INSC,
                    'COEF_t': COEF,
                    'SQ_t': SQ_t,
                    'SMSC': theta_cap,
                    'SMSC_legacy': SMSC_legacy,
                    'ETGAM_t': ETGAM_t,
                    'SUB_t': f_surface,
                    'INTER_t': f_inter,
                    'RECH_t': f_recharge,
                    'CRAK_t': f_recharge,
                    'K_t': K,
                    'LG_t': LG_t,
                    'TT': TT,
                    'SG_CRIT': SG_CRIT,
                    'CFMAX_t': CFMAX_t,
                    'CFR': CFR,
                    'CWH': CWH,
                    'partition_sum_error': torch.abs(f_surface + f_inter + f_recharge - 1.0),
                    'soil_local_residual': soil_local_residual,
                    'gw_local_residual': gw_local_residual,
                    'snowpack_local_residual': snowpack_local_residual,
                    'meltwater_local_residual': meltwater_local_residual,
                    'snow_total_local_residual': snow_total_local_residual,
                    'process_local_residual': process_local_residual,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SA, GW, SNOWPACK, MELTWATER], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)

        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': reg_part
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out

        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelSix_Physical_aSrzMinimal(MultiInv_DynamicSimHydModelSix_Physical):
    """
    Copied physical-fix Model 6 where the soil state becomes an active
    root-zone storage Sa with minimal changes elsewhere.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 component_routing=True, dry_channel_loss=True, zero_flow_gate=True,
                 channel_loss_max=0.60, zero_gate_hidden=None,
                 gate_variant='soft', gate_strength_max=0.30):
        super(MultiInv_DynamicSimHydModelSix_Physical_aSrzMinimal, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing,
            dry_channel_loss=dry_channel_loss, zero_flow_gate=zero_flow_gate,
            channel_loss_max=channel_loss_max, zero_gate_hidden=zero_gate_hidden,
            gate_variant=gate_variant, gate_strength_max=gate_strength_max)
        self.nfea = 18
        self.nstaticpm = self.nfea * nmul
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.simhyd = DynamicSimHydModelFiveDifferentiablePhysicalaSrzMinimal(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiablePhysicalaSrzMinimal(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all)

    def _theta_to_smsc(self, theta, ngage):
        theta4 = theta.view(ngage, self.nmul, self.nfea)
        return 10.0 + theta4[:, :, 15:16] * (1500.0 - 10.0)


class DynamicSimHydModelFiveDifferentiablePhysicalaSrzDeepReturn(DynamicSimHydModelFiveDifferentiablePhysicalFix):
    """
    Active-root-zone variant where the old groundwater-loss sink is converted
    into internal deep groundwater storage with slow return, capillary rise,
    and only a tiny optional true leakage term.
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True, dynamic_cfmax_snow=True,
                 dynamic_routing_scale=False, dynamic_all=False, dyn_hidden=32,
                 k_true_leak_max=0.0):
        super(DynamicSimHydModelFiveDifferentiablePhysicalaSrzDeepReturn, self).__init__(
            mode=mode, theta_is_raw=theta_is_raw, smooth=smooth, eps=eps, rain_snow_gain=rain_snow_gain,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, dyn_hidden=dyn_hidden)
        self.k_true_leak_max = k_true_leak_max

    def _expand_asrz_deepreturn(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)
        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = theta[:, 2:3] * 6.0
        SMSC_legacy = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = theta[:, 4:5]
        CRAK = theta[:, 5:6]
        K = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        LG = theta[:, 7:8] * 0.2
        TT = -2.5 + theta[:, 8:9] * 5.0
        CFMAX = 0.5 + theta[:, 9:10] * (10.0 - 0.5)
        CFR = theta[:, 10:11] * 0.1
        CWH = theta[:, 11:12] * 0.2
        SG_CRIT = theta[:, 12:13] * 300.0
        theta_ab = 0.5 + theta[:, 13:14] * 0.5
        theta_ak = 1.0 + theta[:, 14:15] * 9.0
        theta_cap = 10.0 + theta[:, 15:16] * (1500.0 - 10.0)
        theta_efmax = 0.5 + theta[:, 16:17] * 0.5
        theta_wetpoint = 0.3 + theta[:, 17:18] * 0.6
        K_DEEP_RETURN = 0.001 + theta[:, 18:19] * (0.01 - 0.001)
        CAP_COEF = theta[:, 19:20] * 0.05
        K_TRUE_LEAK = theta[:, 20:21] * self.k_true_leak_max
        K_CHANNEL = 0.05 + theta[:, 21:22] * (1.0 - 0.05)
        return (
            INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH, SG_CRIT,
            theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint,
            K_DEEP_RETURN, CAP_COEF, K_TRUE_LEAK, K_CHANNEL
        )

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        (INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K, LG, TT, CFMAX_base, CFR, CWH, SG_CRIT,
         theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint,
         K_DEEP_RETURN, CAP_COEF, K_TRUE_LEAK, K_CHANNEL) = self._expand_asrz_deepreturn(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SA = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
            DEEPGW = torch.zeros(B, 1, device=device, dtype=dtype)
        else:
            SA = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]
            SNOWPACK = initial_state[:, 2:3]
            MELTWATER = initial_state[:, 3:4]
            if initial_state.shape[1] >= 5:
                DEEPGW = initial_state[:, 4:5]
            else:
                DEEPGW = torch.zeros_like(GW)

        if lg_dyn_seq is not None and (lg_dyn_seq.shape[0] != B or lg_dyn_seq.shape[1] != Tlen):
            raise ValueError('lg_dyn_seq must have shape [B, T, 1] matching inputs')

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = dict()
        if return_diagnostics is True:
            for name in [
                    'Sa_prev', 'GW_prev', 'DEEP_GW_prev', 'SNOWPACK_prev', 'MELTWATER_prev',
                    'Sa', 'GW', 'DEEP_GW', 'SNOWPACK', 'MELTWATER',
                    'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release_to_soil',
                    'interception_storage', 'interception_evaporation', 'actual_ET',
                    'Smoist', 'theta_cap', 'alpha_t', 'P_accessible', 'P_inaccessible', 'aSrz_overflow',
                    'surface_runoff', 'interflow', 'recharge_to_groundwater', 'soil_overflow',
                    'baseflow_raw', 'baseflow_capped', 'baseflow',
                    'groundwater_loss_raw', 'groundwater_loss_capped', 'groundwater_loss',
                    'GW_to_deep', 'DEEP_RETURN', 'CAP', 'TRUE_LEAK', 'K_DEEP_RETURN', 'K_TRUE_LEAK', 'K_CHANNEL',
                    'q_raw_process', 'q_after_channel_loss', 'q_after_gate', 'Q_process',
                    'partition_sum_error', 'soil_local_residual', 'gw_local_residual',
                    'deep_local_residual', 'snowpack_local_residual', 'meltwater_local_residual',
                    'snow_total_local_residual', 'process_local_residual'
            ]:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SA0 = self._min(self._pos(SA), theta_cap)
            GW0 = self._pos(GW)
            DEEPGW0 = self._pos(DEEPGW)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            Smoist0 = torch.clamp(SA0 / (theta_cap + 1e-8), min=0.0, max=1.0)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0, Tt / 20.0, E0t / 10.0,
                SA0 / 300.0, Smoist0, GW0 / 300.0, SNOWPACK0 / 300.0, sin_t, cos_t
            ], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)

            if self.dynamic_etgam is True:
                ETGAM_t = 0.25 + 3.75 * torch.sigmoid(dyn_raw[:, self.dyn_slices['etgam']])
            else:
                ETGAM_t = torch.ones_like(SQ)

            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX_base * m_cf_eff

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()

            RAIN = Pt * frac_rain
            SNOW = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + SNOW
            melt_pot = CFMAX_t * self._pos(Tt - TT)
            melt = self._min(melt_pot, SNOWPACK1)
            MELTWATER1 = MELTWATER0 + melt
            SNOWPACK2 = self._pos(SNOWPACK1 - melt)

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            SNOWPACK3 = SNOWPACK2 + refreezing
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)

            water_holding = CWH * SNOWPACK3
            tosoil_raw = self._pos(MELTWATER2 - water_holding)
            tosoil = self._min(tosoil_raw, MELTWATER2)
            MELTWATER_next = self._pos(MELTWATER2 - tosoil)
            SNOWPACK_next = self._pos(SNOWPACK3)

            PL = RAIN + tosoil
            INT = self._min(INSC, self._min(E0t, PL))
            PL_after_int = self._pos(PL - INT)
            POT = self._pos(E0t - INT)

            Sa_deficit0 = self._pos(theta_cap - SA0)
            deep_wet = DEEPGW0 / (DEEPGW0 + theta_cap + 1e-6)
            CAP_raw = CAP_COEF * Sa_deficit0 * deep_wet
            CAP = self._min(CAP_raw, self._min(DEEPGW0, Sa_deficit0))
            SA_cap = self._min(SA0 + CAP, theta_cap)
            DEEPGW_cap = self._pos(DEEPGW0 - CAP)

            Smoist_cap = torch.clamp(SA_cap / (theta_cap + 1e-8), min=0.0, max=1.0)
            alpha_t = theta_ab * torch.pow(torch.clamp(1.0 - Smoist_cap, min=0.0), theta_ak)
            alpha_t = torch.clamp(alpha_t, 0.0, 1.0)
            P_accessible = alpha_t * PL_after_int
            P_inaccessible = (1.0 - alpha_t) * PL_after_int

            SA_pre = SA_cap + P_accessible
            g = torch.clamp(Smoist_cap / torch.clamp(theta_wetpoint, min=1e-6), 0.0, 1.0)
            ET_a_potential = POT * theta_efmax * g
            ET_a = self._min(ET_a_potential, SA_pre)
            SA_after_ET = self._pos(SA_pre - ET_a)
            aSrz_overflow = self._pos(SA_after_ET - theta_cap)
            SA_next = self._pos(SA_after_ET - aSrz_overflow)

            available_for_partition = self._pos(P_inaccessible + aSrz_overflow)
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_inter = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([torch.log(base_surface), torch.log(base_inter), torch.log(base_recharge)], dim=1)
            if self.dynamic_partition is True and dyn_raw is not None:
                part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']]
            else:
                part_logits = base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]
            SRUN = f_surface * available_for_partition
            IFLOW = f_inter * available_for_partition
            REC_total = f_recharge * available_for_partition

            if lg_dyn_seq is None:
                LG_t = LG
            else:
                LG_dyn_t = torch.clamp(lg_dyn_seq[:, t, :], 0.0, 1.0)
                LG_eff = (1.0 - lg_dyn_weight) * LG + lg_dyn_weight * (LG * LG_dyn_t * 1.5)
                LG_t = torch.clamp(LG_eff, 0.0, 0.2)

            GW_avail_before = self._pos(GW0 + REC_total)
            BAS_raw = K * F.softplus(GW0 - SG_CRIT)
            BAS = self._min(BAS_raw, GW_avail_before)
            GW_avail_after = self._pos(GW_avail_before - BAS)

            GWLOSS_raw = LG_t * F.softplus(SG_CRIT - GW0)
            GW_to_deep = self._min(GWLOSS_raw, GW_avail_after)
            DEEP_pool = self._pos(DEEPGW_cap + GW_to_deep)
            TRUE_LEAK = self._min(K_TRUE_LEAK * DEEP_pool, DEEP_pool)
            DEEP_after_leak = self._pos(DEEP_pool - TRUE_LEAK)
            DEEP_RETURN = self._min(K_DEEP_RETURN * DEEP_after_leak, DEEP_after_leak)
            DEEPGW_next = self._pos(DEEP_after_leak - DEEP_RETURN)
            GW_next = self._pos(GW_avail_after - GW_to_deep + DEEP_RETURN)

            Q = self._pos(SRUN + IFLOW + BAS)

            soil_local_residual = P_accessible + CAP - ET_a - aSrz_overflow - (SA_next - SA0)
            gw_local_residual = REC_total - BAS - GW_to_deep + DEEP_RETURN - (GW_next - GW0)
            deep_local_residual = GW_to_deep - CAP - DEEP_RETURN - TRUE_LEAK - (DEEPGW_next - DEEPGW0)
            snowpack_local_residual = SNOW - melt + refreezing - (SNOWPACK_next - SNOWPACK0)
            meltwater_local_residual = melt - refreezing - tosoil - (MELTWATER_next - MELTWATER0)
            snow_total_local_residual = SNOW - tosoil - (SNOWPACK_next - SNOWPACK0) - (MELTWATER_next - MELTWATER0)
            process_local_residual = (
                Pt - INT - ET_a - TRUE_LEAK - Q
                - ((SA_next - SA0) + (GW_next - GW0) + (DEEPGW_next - DEEPGW0)
                   + (SNOWPACK_next - SNOWPACK0) + (MELTWATER_next - MELTWATER0))
            )

            SA = SA_next
            GW = GW_next
            DEEPGW = DEEPGW_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)

            if return_diagnostics is True:
                diag_vals = {
                    'Sa_prev': SA0,
                    'GW_prev': GW0,
                    'DEEP_GW_prev': DEEPGW0,
                    'SNOWPACK_prev': SNOWPACK0,
                    'MELTWATER_prev': MELTWATER0,
                    'Sa': SA_next,
                    'GW': GW_next,
                    'DEEP_GW': DEEPGW_next,
                    'SNOWPACK': SNOWPACK_next,
                    'MELTWATER': MELTWATER_next,
                    'precipitation': Pt,
                    'rainfall': RAIN,
                    'snowfall': SNOW,
                    'snowmelt': melt,
                    'refreezing': refreezing,
                    'snow_release_to_soil': tosoil,
                    'interception_storage': torch.zeros_like(INT),
                    'interception_evaporation': INT,
                    'actual_ET': ET_a,
                    'Smoist': Smoist_cap,
                    'theta_cap': theta_cap,
                    'alpha_t': alpha_t,
                    'P_accessible': P_accessible,
                    'P_inaccessible': P_inaccessible,
                    'aSrz_overflow': aSrz_overflow,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'recharge_to_groundwater': REC_total,
                    'soil_overflow': aSrz_overflow,
                    'baseflow_raw': BAS_raw,
                    'baseflow_capped': BAS,
                    'baseflow': BAS,
                    'groundwater_loss_raw': GWLOSS_raw,
                    'groundwater_loss_capped': TRUE_LEAK,
                    'groundwater_loss': TRUE_LEAK,
                    'GW_to_deep': GW_to_deep,
                    'DEEP_RETURN': DEEP_RETURN,
                    'CAP': CAP,
                    'TRUE_LEAK': TRUE_LEAK,
                    'K_DEEP_RETURN': K_DEEP_RETURN,
                    'K_TRUE_LEAK': K_TRUE_LEAK,
                    'K_CHANNEL': K_CHANNEL,
                    'q_raw_process': Q,
                    'q_after_channel_loss': Q,
                    'q_after_gate': Q,
                    'Q_process': Q,
                    'partition_sum_error': torch.abs(f_surface + f_inter + f_recharge - 1.0),
                    'soil_local_residual': soil_local_residual,
                    'gw_local_residual': gw_local_residual,
                    'deep_local_residual': deep_local_residual,
                    'snowpack_local_residual': snowpack_local_residual,
                    'meltwater_local_residual': meltwater_local_residual,
                    'snow_total_local_residual': snow_total_local_residual,
                    'process_local_residual': process_local_residual,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SA, GW, SNOWPACK, MELTWATER, DEEPGW], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)

        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': reg_part
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out
        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelSix_Physical_aSrzDeepReturn(MultiInv_DynamicSimHydModelSix_Physical):
    """
    Copied aSrz physical model where groundwater loss, channel loss, and gate
    loss are internalized into deep and channel stores instead of discarded.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 component_routing=True, dry_channel_loss=True, zero_flow_gate=True,
                 channel_loss_max=0.60, zero_gate_hidden=None,
                 gate_variant='soft', gate_strength_max=0.30, k_true_leak_max=0.0):
        super(MultiInv_DynamicSimHydModelSix_Physical_aSrzDeepReturn, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing,
            dry_channel_loss=dry_channel_loss, zero_flow_gate=zero_flow_gate,
            channel_loss_max=channel_loss_max, zero_gate_hidden=zero_gate_hidden,
            gate_variant=gate_variant, gate_strength_max=gate_strength_max)
        self.nfea = 22
        self.nstaticpm = self.nfea * nmul
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.simhyd = DynamicSimHydModelFiveDifferentiablePhysicalaSrzDeepReturn(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            k_true_leak_max=k_true_leak_max)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiablePhysicalaSrzDeepReturn(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            k_true_leak_max=k_true_leak_max)

    def _theta_to_smsc(self, theta, ngage):
        theta4 = theta.view(ngage, self.nmul, self.nfea)
        return 10.0 + theta4[:, :, 15:16] * (1500.0 - 10.0)

    def _theta_to_k_channel(self, theta, ngage):
        theta4 = theta.view(ngage, self.nmul, self.nfea)
        return 0.05 + theta4[:, :, 21:22] * (1.0 - 0.05)

    def forward(self, x, z, doDropMC=False, return_diagnostics=False, return_component_diagnostics=False):
        nt_x = x.shape[0]
        ngage = z.shape[1]
        basin_attr = z[-1, :, -self.nattr:]

        if z.shape[2] > self.nattr:
            snow_frac_raw = torch.clamp(z[-1, :, -self.nattr - 1:-self.nattr], min=0.0, max=1.0)
        else:
            snow_frac_raw = None

        staticFeat = self.staticFeat(basin_attr)
        staticParams0 = self.staticOut(staticFeat)

        cursor = 0
        static0 = staticParams0[:, cursor:cursor + self.nstaticpm].view(ngage, self.nfea, self.nmul)
        static0 = static0 + self.compStaticBias
        snowpara = torch.sigmoid(static0)
        cursor += self.nstaticpm

        routpara0 = staticParams0[:, cursor:cursor + self.nroutpm]
        if self.component_routing is True:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        elif self.comprout is False:
            routpara = torch.sigmoid(routpara0)
        else:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        cursor += self.nroutpm

        if self.nwtspm == 0:
            wts = None
        else:
            wtspara = staticParams0[:, cursor:cursor + self.nwtspm] + self.compWeightBias
            wts = F.softmax(wtspara, dim=-1)
        self._last_wts = wts

        if self.lgdyn is False:
            lg_dyn = None
        else:
            lg_dyn_seq = self.lstmdyn(z)
            lg_attr_bias = self.lgAttr(basin_attr).unsqueeze(0).repeat(lg_dyn_seq.shape[0], 1, 1)
            lg_dyn = torch.sigmoid(lg_dyn_seq + lg_attr_bias)

        x_rep = x.unsqueeze(2).repeat(1, 1, self.nmul, 1).view(nt_x, ngage * self.nmul, x.shape[2])
        x_bt = x_rep.permute(1, 0, 2)
        theta = snowpara.permute(0, 2, 1).contiguous().view(ngage * self.nmul, self.nfea)

        if lg_dyn is None:
            lg_bt = None
        else:
            lg_bt = lg_dyn.permute(1, 2, 0).contiguous().view(ngage * self.nmul, lg_dyn.shape[0], 1)
        if lg_bt is not None and self.inittime > 0:
            lg_bt_main = lg_bt[:, self.inittime:, :]
        else:
            lg_bt_main = lg_bt

        if snow_frac_raw is None:
            snow_frac_rep = None
        else:
            snow_frac_rep = snow_frac_raw.unsqueeze(1).repeat(1, self.nmul, 1).view(ngage * self.nmul, 1)

        need_process_diag = return_diagnostics or return_component_diagnostics or self.dry_channel_loss or self.zero_flow_gate_enabled

        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            x_use = x[self.inittime:, :, :]
            _, warm_state = self.simhyd_analysis(
                warm_inputs,
                theta,
                lg_dyn_seq=None,
                lg_dyn_weight=self.lgdynweight,
                snow_frac_raw=snow_frac_rep,
                return_final_state=True)
            if need_process_diag:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    main_inputs,
                    theta,
                    initial_state=warm_state,
                    lg_dyn_seq=lg_bt_main,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_diagnostics=True,
                    return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    main_inputs,
                    theta,
                    initial_state=warm_state,
                    lg_dyn_seq=lg_bt_main,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_regularization=True)
                diag_comp = None
        else:
            x_use = x
            if need_process_diag:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    x_bt, theta, lg_dyn_seq=lg_bt, lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep, return_diagnostics=True, return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    x_bt, theta, lg_dyn_seq=lg_bt, lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep, return_regularization=True)
                diag_comp = None

        reg_total = self.reg_amp_w * reg_terms['dynamic_amplitude_loss'] \
            + self.reg_smooth_w * reg_terms['dynamic_smoothness_loss'] \
            + self.reg_part_w * reg_terms['partition_entropy_loss']
        self._last_aux_terms = reg_terms
        self._last_aux_loss = reg_total

        q_comp_raw = q_seq.view(ngage, self.nmul, q_seq.shape[1], 1).permute(2, 0, 1, 3)
        q_comp_raw = torch.clamp(q_comp_raw, min=0.0)
        q_after_loss, channel_loss, channel_loss_fraction = self._apply_channel_loss(q_comp_raw, diag_comp, theta, basin_attr, ngage)
        q_after_gate_base, zero_flow_probability, gate_loss, zero_flow_keep_fraction = self._apply_zero_flow_gate(
            q_after_loss, x_use, diag_comp, theta, ngage)
        q_after_gate_base = torch.clamp(q_after_gate_base, min=0.0)

        K_channel = self._theta_to_k_channel(theta, ngage).unsqueeze(0)  # [1, B, M, 1]
        Tmain = q_after_gate_base.shape[0]
        channel_store = torch.zeros_like(q_after_gate_base[0])
        channel_store_prev_hist = []
        channel_store_hist = []
        channel_release_hist = []
        q_process_hist = []
        for t in range(Tmain):
            channel_store_prev = channel_store
            channel_store_in = channel_store_prev + channel_loss[t] + gate_loss[t]
            channel_release = torch.minimum(K_channel[0] * channel_store_in, channel_store_in)
            channel_store = torch.clamp(channel_store_in - channel_release, min=0.0)
            q_process = torch.clamp(q_after_gate_base[t] + channel_release, min=0.0)
            channel_store_prev_hist.append(channel_store_prev)
            channel_store_hist.append(channel_store)
            channel_release_hist.append(channel_release)
            q_process_hist.append(q_process)

        channel_store_prev_seq = torch.stack(channel_store_prev_hist, dim=0)
        channel_store_seq = torch.stack(channel_store_hist, dim=0)
        channel_release_seq = torch.stack(channel_release_hist, dim=0)
        q_comp = torch.stack(q_process_hist, dim=0)

        p_comp = self._component_tensor_4d(diag_comp['precipitation'], ngage)
        int_comp = self._component_tensor_4d(diag_comp['interception_evaporation'], ngage)
        et_comp = self._component_tensor_4d(diag_comp['actual_ET'], ngage)
        true_leak_comp = self._component_tensor_4d(diag_comp['TRUE_LEAK'], ngage)
        sa_prev = self._component_tensor_4d(diag_comp['Sa_prev'], ngage)
        gw_prev = self._component_tensor_4d(diag_comp['GW_prev'], ngage)
        deep_prev = self._component_tensor_4d(diag_comp['DEEP_GW_prev'], ngage)
        snow_prev = self._component_tensor_4d(diag_comp['SNOWPACK_prev'], ngage)
        melt_prev = self._component_tensor_4d(diag_comp['MELTWATER_prev'], ngage)
        sa_next = self._component_tensor_4d(diag_comp['Sa'], ngage)
        gw_next = self._component_tensor_4d(diag_comp['GW'], ngage)
        deep_next = self._component_tensor_4d(diag_comp['DEEP_GW'], ngage)
        snow_next = self._component_tensor_4d(diag_comp['SNOWPACK'], ngage)
        melt_next = self._component_tensor_4d(diag_comp['MELTWATER'], ngage)

        delta_storage = (
            (sa_next - sa_prev) + (gw_next - gw_prev) + (deep_next - deep_prev)
            + (snow_next - snow_prev) + (melt_next - melt_prev)
            + (channel_store_seq - channel_store_prev_seq)
        )
        process_local_residual = p_comp - int_comp - et_comp - true_leak_comp - q_comp - delta_storage

        q_mix_before_routing = self._mix_or_mean(q_comp, wts)
        route_mult_seq = None
        if self.routOpt is True and self.component_routing is True:
            q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
            if self.dynamic_routing_scale is True:
                if 'Sa' in diag_comp:
                    sm_comp = self._component_tensor_4d(diag_comp['Sa'], ngage)
                else:
                    sm_comp = torch.ones_like(q_comp) * 50.0
                smsc_comp = self._theta_to_smsc(theta, ngage).unsqueeze(0).repeat(q_comp.shape[0], 1, 1, 1)
                p_rep = x_use[:, :, 0:1].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                if x_use.shape[-1] >= 5:
                    sin_rep = x_use[:, :, 3:4].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                    cos_rep = x_use[:, :, 4:5].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                else:
                    sin_rep = torch.zeros_like(q_comp)
                    cos_rep = torch.ones_like(q_comp)
                x_route = torch.cat([p_rep, sm_comp, smsc_comp, sin_rep, cos_rep], dim=-1).view(q_comp.shape[0], ngage * self.nmul, 5)
                q_routed_flat, route_mult_flat = self._route_q_dynamic_scale(q_for_routing, routpara, x_route)
                route_mult_4d = route_mult_flat.view(q_comp.shape[0], ngage, self.nmul, 1)
                route_mult_seq = self._mix_or_mean(route_mult_4d, wts)
                self._last_aux_loss = self._last_aux_loss + self.reg_amp_w * torch.mean((route_mult_flat - 1.0) ** 2)
                self._last_aux_loss = self._last_aux_loss + self.reg_smooth_w * torch.mean(
                    (route_mult_flat[1:, :, :] - route_mult_flat[:-1, :, :]) ** 2)
                q_routed = q_routed_flat.view(q_comp.shape[0], ngage, self.nmul, 1)
            else:
                q_routed = self._route_q(q_for_routing, routpara).view(q_comp.shape[0], ngage, self.nmul, 1)
            out = self._mix_or_mean(q_routed, wts)
        else:
            q_routed = q_comp
            if self.routOpt is True:
                if self.comprout is True:
                    q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
                    q_routed = self._route_q(q_for_routing, routpara).view(q_comp.shape[0], ngage, self.nmul, 1)
                    out = self._mix_or_mean(q_routed, wts)
                else:
                    out = self._route_q(q_mix_before_routing, routpara)
            else:
                out = q_mix_before_routing

        out = torch.clamp(out, min=0.0)
        if return_diagnostics is not True and return_component_diagnostics is not True:
            return out

        diag_out = dict()
        if diag_comp is not None:
            for name, tensor_comp in diag_comp.items():
                diag_out[name] = self._mix_component_tensor(tensor_comp, ngage)
        diag_out['total_discharge'] = out
        diag_out['q_mix_before_routing'] = q_mix_before_routing
        diag_out['component_discharge_raw'] = self._mix_or_mean(q_comp_raw, wts)
        diag_out['channel_loss'] = self._mix_or_mean(channel_loss, wts)
        diag_out['channel_loss_fraction'] = self._mix_or_mean(channel_loss_fraction, wts)
        diag_out['gate_loss'] = self._mix_or_mean(gate_loss, wts)
        diag_out['zero_flow_probability'] = self._mix_or_mean(zero_flow_probability, wts)
        diag_out['zero_flow_keep_fraction'] = self._mix_or_mean(zero_flow_keep_fraction, wts)
        diag_out['CHANNEL_STORE'] = self._mix_or_mean(channel_store_seq, wts)
        diag_out['CHANNEL_RELEASE'] = self._mix_or_mean(channel_release_seq, wts)
        diag_out['Q_process'] = q_mix_before_routing
        diag_out['water_balance_residual'] = self._mix_or_mean(process_local_residual, wts)
        if route_mult_seq is not None:
            diag_out['route_b_t_multiplier'] = route_mult_seq

        if return_component_diagnostics is True:
            if diag_comp is not None:
                for name, tensor_comp in diag_comp.items():
                    diag_out[name + '_components'] = self._component_tensor_4d(tensor_comp, ngage).squeeze(-1)
            diag_out['q_raw_process_components'] = q_comp_raw.squeeze(-1)
            diag_out['q_after_channel_loss_components'] = q_after_loss.squeeze(-1)
            diag_out['q_after_gate_base_components'] = q_after_gate_base.squeeze(-1)
            diag_out['q_after_gate_components'] = q_comp.squeeze(-1)
            diag_out['Q_process_components'] = q_comp.squeeze(-1)
            diag_out['q_routed_components'] = q_routed.squeeze(-1)
            diag_out['channel_loss_components'] = channel_loss.squeeze(-1)
            diag_out['gate_loss_components'] = gate_loss.squeeze(-1)
            diag_out['channel_loss_fraction_components'] = channel_loss_fraction.squeeze(-1)
            diag_out['zero_flow_probability_components'] = zero_flow_probability.squeeze(-1)
            diag_out['zero_flow_keep_fraction_components'] = zero_flow_keep_fraction.squeeze(-1)
            diag_out['CHANNEL_STORE_components'] = channel_store_seq.squeeze(-1)
            diag_out['CHANNEL_RELEASE_components'] = channel_release_seq.squeeze(-1)
            diag_out['process_local_residual_components'] = process_local_residual.squeeze(-1)
            diag_out['component_weights'] = wts
            if self.component_routing is True:
                routpara_view = torch.sigmoid(routpara0).view(ngage, self.nmul, 2)
                diag_out['route_a_components'] = 0.0 + routpara_view[:, :, 0] * 2.9
                diag_out['route_b_components'] = 0.0 + routpara_view[:, :, 1] * 6.5

        return out, diag_out


class DynamicSimHydModelFiveDifferentiableClosedSIMHYDaSrzHBV(
        DynamicSimHydModelFiveDifferentiablePhysicalFix):
    """
    Closed, mass-conserving Model 6 variant with HBV snow, aSrz active root
    zone, SIMHYD-style partitioning, HBV-style UZ/LZ groundwater, and channel
    storage delay. No explicit discarded water sink is allowed.
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True, dynamic_cfmax_snow=True,
                 dynamic_routing_scale=False, dynamic_all=False, dyn_hidden=32):
        super(DynamicSimHydModelFiveDifferentiableClosedSIMHYDaSrzHBV, self).__init__(
            mode=mode, theta_is_raw=theta_is_raw, smooth=smooth, eps=eps, rain_snow_gain=rain_snow_gain,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, dyn_hidden=dyn_hidden)

    def _expand_closed_simhyd_asrz_hbv(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)
        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = theta[:, 2:3] * 6.0
        SMSC_legacy = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = theta[:, 4:5]
        CRAK = theta[:, 5:6]
        K_fast = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        LG = theta[:, 7:8] * 0.2
        TT = -2.5 + theta[:, 8:9] * 5.0
        CFMAX = 0.5 + theta[:, 9:10] * (10.0 - 0.5)
        CFR = theta[:, 10:11] * 0.1
        CWH = theta[:, 11:12] * 0.2
        SG_CRIT = theta[:, 12:13] * 300.0
        theta_ab = 0.5 + theta[:, 13:14] * 0.5
        theta_ak = 1.0 + theta[:, 14:15] * 9.0
        theta_cap = 10.0 + theta[:, 15:16] * (1500.0 - 10.0)
        theta_efmax = 0.5 + theta[:, 16:17] * 0.5
        theta_wetpoint = 0.3 + theta[:, 17:18] * 0.6
        K_slow = 0.001 + theta[:, 18:19] * (0.3 - 0.001)
        PERC_cap = theta[:, 19:20] * 10.0
        K_cap = theta[:, 20:21] * 0.05
        K_channel = 0.05 + theta[:, 21:22] * (1.0 - 0.05)
        return (
            INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K_fast, LG, TT, CFMAX, CFR, CWH, SG_CRIT,
            theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint,
            K_slow, PERC_cap, K_cap, K_channel
        )

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        (INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K_fast, LG, TT, CFMAX_base, CFR, CWH, SG_CRIT,
         theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint,
         K_slow, PERC_cap, K_cap, K_channel) = self._expand_closed_simhyd_asrz_hbv(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SA = torch.zeros(B, 1, device=device, dtype=dtype)
            UZ = torch.zeros(B, 1, device=device, dtype=dtype)
            LZ = torch.zeros(B, 1, device=device, dtype=dtype)
            CHANNEL = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
            init_sa = SA
            init_uz = UZ
            init_lz = LZ
            init_channel = CHANNEL
        else:
            SA = initial_state[:, 0:1]
            UZ = initial_state[:, 1:2]
            LZ = initial_state[:, 2:3]
            CHANNEL = initial_state[:, 3:4]
            SNOWPACK = initial_state[:, 4:5]
            MELTWATER = initial_state[:, 5:6]
            init_sa = initial_state[:, 0:1]
            init_uz = initial_state[:, 1:2]
            init_lz = initial_state[:, 2:3]
            init_channel = initial_state[:, 3:4]

        if lg_dyn_seq is not None and (lg_dyn_seq.shape[0] != B or lg_dyn_seq.shape[1] != Tlen):
            raise ValueError('lg_dyn_seq must have shape [B, T, 1] matching inputs')

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = dict()
        if return_diagnostics is True:
            for name in [
                    'Sa_prev', 'UZ_prev', 'LZ_prev', 'CHANNEL_prev', 'SNOWPACK_prev', 'MELTWATER_prev',
                    'Sa', 'UZ', 'LZ', 'CHANNEL', 'SNOWPACK', 'MELTWATER',
                    'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release_to_soil',
                    'PL', 'interception_storage', 'interception_evaporation', 'actual_ET',
                    'Smoist', 'theta_cap', 'alpha_t', 'P_accessible', 'P_inaccessible', 'aSrz_overflow',
                    'surface_runoff', 'interflow', 'recharge_to_groundwater', 'soil_overflow',
                    'Qfast', 'PERC', 'Qslow', 'CAP', 'Q_generated', 'Q_process',
                    'channel_loss', 'gate_loss', 'groundwater_loss',
                    'partition_sum_error', 'soil_local_residual', 'uz_local_residual', 'lz_local_residual',
                    'channel_local_residual', 'snowpack_local_residual', 'meltwater_local_residual',
                    'snow_total_local_residual', 'process_local_residual',
                    'INSC', 'COEF_t', 'SQ_t', 'K_fast_t', 'K_slow_t', 'PERC_cap_t', 'K_cap_t',
                    'K_channel_t', 'LG_t', 'TT', 'SG_CRIT', 'CFMAX_t', 'CFR', 'CWH'
            ]:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SA0 = self._min(self._pos(SA), theta_cap)
            UZ0 = self._pos(UZ)
            LZ0 = self._pos(LZ)
            CHANNEL0 = self._pos(CHANNEL)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            Smoist0 = torch.clamp(SA0 / (theta_cap + 1e-8), min=0.0, max=1.0)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0, Tt / 20.0, E0t / 10.0,
                SA0 / 300.0, Smoist0, (UZ0 + LZ0) / 300.0, SNOWPACK0 / 300.0, sin_t, cos_t
            ], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)

            if self.dynamic_etgam is True:
                ETGAM_t = 0.25 + 3.75 * torch.sigmoid(dyn_raw[:, self.dyn_slices['etgam']])
            else:
                ETGAM_t = torch.ones_like(SQ)

            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX_base * m_cf_eff

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()

            RAIN = Pt * frac_rain
            SNOW = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + SNOW
            melt_pot = CFMAX_t * self._pos(Tt - TT)
            melt = self._min(melt_pot, SNOWPACK1)
            MELTWATER1 = MELTWATER0 + melt
            SNOWPACK2 = self._pos(SNOWPACK1 - melt)

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            SNOWPACK3 = SNOWPACK2 + refreezing
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)

            water_holding = CWH * SNOWPACK3
            tosoil_raw = self._pos(MELTWATER2 - water_holding)
            tosoil = self._min(tosoil_raw, MELTWATER2)
            MELTWATER_next = self._pos(MELTWATER2 - tosoil)
            SNOWPACK_next = self._pos(SNOWPACK3)

            PL = RAIN + tosoil
            INT = self._min(INSC, self._min(E0t, PL))
            PL_after_int = self._pos(PL - INT)
            POT = self._pos(E0t - INT)

            alpha_t = theta_ab * torch.pow(torch.clamp(1.0 - Smoist0, min=0.0), theta_ak)
            alpha_t = torch.clamp(alpha_t, 0.0, 1.0)
            P_accessible = alpha_t * PL_after_int
            P_inaccessible = (1.0 - alpha_t) * PL_after_int

            SA_pre = SA0 + P_accessible
            ET_stress = torch.clamp(Smoist0 / torch.clamp(theta_wetpoint, min=1e-6), 0.0, 1.0)
            ET_a_potential = POT * theta_efmax * ET_stress
            ET_a = self._min(ET_a_potential, SA_pre)
            SA_after = self._pos(SA_pre - ET_a)
            Sa_overflow = self._pos(SA_after - theta_cap)
            SA1 = self._pos(SA_after - Sa_overflow)

            water_for_partition = self._pos(P_inaccessible + Sa_overflow)
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_inter = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([torch.log(base_surface), torch.log(base_inter), torch.log(base_recharge)], dim=1)
            if self.dynamic_partition is True and dyn_raw is not None:
                part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']]
            else:
                part_logits = base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]

            SRUN = f_surface * water_for_partition
            IFLOW = f_inter * water_for_partition
            REC = f_recharge * water_for_partition

            UZ1 = UZ0 + REC
            Qfast_raw = K_fast * UZ1
            Qfast = self._min(Qfast_raw, UZ1)
            UZ2 = self._pos(UZ1 - Qfast)

            PERC = self._min(PERC_cap, UZ2)
            UZ3 = self._pos(UZ2 - PERC)
            LZ1 = LZ0 + PERC

            Qslow_raw = K_slow * LZ1
            Qslow = self._min(Qslow_raw, LZ1)
            LZ2 = self._pos(LZ1 - Qslow)

            Sa_deficit = self._pos(theta_cap - SA1)
            CAP_raw = K_cap * LZ2 * Sa_deficit / torch.clamp(theta_cap, min=1e-6)
            CAP = self._min(CAP_raw, self._min(LZ2, Sa_deficit))
            LZ_next = self._pos(LZ2 - CAP)
            SA_next = self._min(SA1 + CAP, theta_cap)
            UZ_next = UZ3

            Q_generated = self._pos(SRUN + IFLOW + Qfast + Qslow)
            CHANNEL1 = CHANNEL0 + Q_generated
            Q_process = self._min(K_channel * CHANNEL1, CHANNEL1)
            CHANNEL_next = self._pos(CHANNEL1 - Q_process)

            soil_local_residual = P_accessible + CAP - ET_a - Sa_overflow - (SA_next - SA0)
            uz_local_residual = REC - Qfast - PERC - (UZ_next - UZ0)
            lz_local_residual = PERC - Qslow - CAP - (LZ_next - LZ0)
            channel_local_residual = Q_generated - Q_process - (CHANNEL_next - CHANNEL0)
            snowpack_local_residual = SNOW - melt + refreezing - (SNOWPACK_next - SNOWPACK0)
            meltwater_local_residual = melt - refreezing - tosoil - (MELTWATER_next - MELTWATER0)
            snow_total_local_residual = SNOW - tosoil - (SNOWPACK_next - SNOWPACK0) - (MELTWATER_next - MELTWATER0)
            process_local_residual = (
                Pt - INT - ET_a - Q_process
                - ((SA_next - SA0) + (UZ_next - UZ0) + (LZ_next - LZ0)
                   + (CHANNEL_next - CHANNEL0) + (SNOWPACK_next - SNOWPACK0) + (MELTWATER_next - MELTWATER0))
            )

            SA = SA_next
            UZ = UZ_next
            LZ = LZ_next
            CHANNEL = CHANNEL_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q_process)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)

            if return_diagnostics is True:
                diag_vals = {
                    'Sa_prev': SA0,
                    'UZ_prev': UZ0,
                    'LZ_prev': LZ0,
                    'CHANNEL_prev': CHANNEL0,
                    'SNOWPACK_prev': SNOWPACK0,
                    'MELTWATER_prev': MELTWATER0,
                    'Sa': SA_next,
                    'UZ': UZ_next,
                    'LZ': LZ_next,
                    'CHANNEL': CHANNEL_next,
                    'SNOWPACK': SNOWPACK_next,
                    'MELTWATER': MELTWATER_next,
                    'precipitation': Pt,
                    'rainfall': RAIN,
                    'snowfall': SNOW,
                    'snowmelt': melt,
                    'refreezing': refreezing,
                    'snow_release_to_soil': tosoil,
                    'PL': PL,
                    'interception_storage': torch.zeros_like(INT),
                    'interception_evaporation': INT,
                    'actual_ET': ET_a,
                    'Smoist': Smoist0,
                    'theta_cap': theta_cap,
                    'alpha_t': alpha_t,
                    'P_accessible': P_accessible,
                    'P_inaccessible': P_inaccessible,
                    'aSrz_overflow': Sa_overflow,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'recharge_to_groundwater': REC,
                    'soil_overflow': Sa_overflow,
                    'Qfast': Qfast,
                    'PERC': PERC,
                    'Qslow': Qslow,
                    'CAP': CAP,
                    'Q_generated': Q_generated,
                    'Q_process': Q_process,
                    'channel_loss': torch.zeros_like(Q_process),
                    'gate_loss': torch.zeros_like(Q_process),
                    'groundwater_loss': torch.zeros_like(Q_process),
                    'partition_sum_error': torch.abs(f_surface + f_inter + f_recharge - 1.0),
                    'soil_local_residual': soil_local_residual,
                    'uz_local_residual': uz_local_residual,
                    'lz_local_residual': lz_local_residual,
                    'channel_local_residual': channel_local_residual,
                    'snowpack_local_residual': snowpack_local_residual,
                    'meltwater_local_residual': meltwater_local_residual,
                    'snow_total_local_residual': snow_total_local_residual,
                    'process_local_residual': process_local_residual,
                    'INSC': INSC,
                    'COEF_t': COEF,
                    'SQ_t': SQ_t,
                    'K_fast_t': K_fast,
                    'K_slow_t': K_slow,
                    'PERC_cap_t': PERC_cap,
                    'K_cap_t': K_cap,
                    'K_channel_t': K_channel,
                    'LG_t': LG,
                    'TT': TT,
                    'SG_CRIT': SG_CRIT,
                    'CFMAX_t': CFMAX_t,
                    'CFR': CFR,
                    'CWH': CWH,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SA, UZ, LZ, CHANNEL, SNOWPACK, MELTWATER], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)
        sum_p = torch.clamp(P.sum(dim=1), min=1e-6)
        drift_loss = (
            torch.mean(torch.abs(SA - init_sa) / sum_p)
            + torch.mean(torch.abs(UZ - init_uz) / sum_p)
            + torch.mean(torch.abs(LZ - init_lz) / sum_p)
            + torch.mean(torch.abs(CHANNEL - init_channel) / sum_p)
        )
        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': reg_part,
            'storage_drift_loss': drift_loss,
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out
        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelSix_Physical_ClosedSIMHYDaSrzHBV(MultiInv_DynamicSimHydModelSix_Physical):
    """
    Closed, mass-conserving Model 6 variant with HBV snow, aSrz, SIMHYD
    partitioning, HBV UZ/LZ reservoirs, and internal channel storage delay.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 component_routing=True):
        super(MultiInv_DynamicSimHydModelSix_Physical_ClosedSIMHYDaSrzHBV, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing,
            dry_channel_loss=False, zero_flow_gate=False,
            channel_loss_max=0.0, gate_variant='soft', gate_strength_max=0.0)
        self.nfea = 22
        self.nstaticpm = self.nfea * nmul
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.simhyd = DynamicSimHydModelFiveDifferentiableClosedSIMHYDaSrzHBV(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiableClosedSIMHYDaSrzHBV(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all)

    def _theta_to_smsc(self, theta, ngage):
        theta4 = theta.view(ngage, self.nmul, self.nfea)
        return 10.0 + theta4[:, :, 15:16] * (1500.0 - 10.0)

    def forward(self, x, z, doDropMC=False, return_diagnostics=False, return_component_diagnostics=False):
        nt_x = x.shape[0]
        ngage = z.shape[1]
        basin_attr = z[-1, :, -self.nattr:]

        if z.shape[2] > self.nattr:
            snow_frac_raw = torch.clamp(z[-1, :, -self.nattr - 1:-self.nattr], min=0.0, max=1.0)
        else:
            snow_frac_raw = None

        staticFeat = self.staticFeat(basin_attr)
        staticParams0 = self.staticOut(staticFeat)

        cursor = 0
        static0 = staticParams0[:, cursor:cursor + self.nstaticpm].view(ngage, self.nfea, self.nmul)
        static0 = static0 + self.compStaticBias
        snowpara = torch.sigmoid(static0)
        cursor += self.nstaticpm

        routpara0 = staticParams0[:, cursor:cursor + self.nroutpm]
        if self.component_routing is True:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        elif self.comprout is False:
            routpara = torch.sigmoid(routpara0)
        else:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        cursor += self.nroutpm

        if self.nwtspm == 0:
            wts = None
        else:
            wtspara = staticParams0[:, cursor:cursor + self.nwtspm] + self.compWeightBias
            wts = F.softmax(wtspara, dim=-1)
        self._last_wts = wts

        if self.lgdyn is False:
            lg_dyn = None
        else:
            lg_dyn_seq = self.lstmdyn(z)
            lg_attr_bias = self.lgAttr(basin_attr).unsqueeze(0).repeat(lg_dyn_seq.shape[0], 1, 1)
            lg_dyn = torch.sigmoid(lg_dyn_seq + lg_attr_bias)

        x_rep = x.unsqueeze(2).repeat(1, 1, self.nmul, 1).view(nt_x, ngage * self.nmul, x.shape[2])
        x_bt = x_rep.permute(1, 0, 2)
        theta = snowpara.permute(0, 2, 1).contiguous().view(ngage * self.nmul, self.nfea)

        if lg_dyn is None:
            lg_bt = None
        else:
            lg_bt = lg_dyn.permute(1, 2, 0).contiguous().view(ngage * self.nmul, lg_dyn.shape[0], 1)
        if lg_bt is not None and self.inittime > 0:
            lg_bt_main = lg_bt[:, self.inittime:, :]
        else:
            lg_bt_main = lg_bt

        if snow_frac_raw is None:
            snow_frac_rep = None
        else:
            snow_frac_rep = snow_frac_raw.unsqueeze(1).repeat(1, self.nmul, 1).view(ngage * self.nmul, 1)

        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            _, warm_state = self.simhyd_analysis(
                warm_inputs, theta, lg_dyn_seq=None, lg_dyn_weight=self.lgdynweight,
                snow_frac_raw=snow_frac_rep, return_final_state=True)
            q_seq, diag_comp, reg_terms = self.simhyd(
                main_inputs, theta, initial_state=warm_state, lg_dyn_seq=lg_bt_main,
                lg_dyn_weight=self.lgdynweight, snow_frac_raw=snow_frac_rep,
                return_diagnostics=True, return_regularization=True)
        else:
            q_seq, diag_comp, reg_terms = self.simhyd(
                x_bt, theta, lg_dyn_seq=lg_bt, lg_dyn_weight=self.lgdynweight,
                snow_frac_raw=snow_frac_rep, return_diagnostics=True, return_regularization=True)

        reg_total = self.reg_amp_w * reg_terms['dynamic_amplitude_loss'] \
            + self.reg_smooth_w * reg_terms['dynamic_smoothness_loss'] \
            + self.reg_part_w * reg_terms['partition_entropy_loss'] \
            + 1e-3 * reg_terms.get('storage_drift_loss', 0.0)
        self._last_aux_terms = reg_terms
        self._last_aux_loss = reg_total

        q_comp = q_seq.view(ngage, self.nmul, q_seq.shape[1], 1).permute(2, 0, 1, 3)
        q_comp = torch.clamp(q_comp, min=0.0)
        q_mix_before_routing = self._mix_or_mean(q_comp, wts)
        route_mult_seq = None

        if self.routOpt is True and self.component_routing is True:
            q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
            if self.dynamic_routing_scale is True:
                sm_comp = self._component_tensor_4d(diag_comp['Sa'], ngage)
                smsc_comp = self._theta_to_smsc(theta, ngage).unsqueeze(0).repeat(q_comp.shape[0], 1, 1, 1)
                p_rep = x[self.inittime:, :, 0:1].unsqueeze(2).repeat(1, 1, self.nmul, 1) if self.inittime > 0 else x[:, :, 0:1].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                x_use = x[self.inittime:, :, :] if self.inittime > 0 else x
                if x_use.shape[-1] >= 5:
                    sin_rep = x_use[:, :, 3:4].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                    cos_rep = x_use[:, :, 4:5].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                else:
                    sin_rep = torch.zeros_like(q_comp)
                    cos_rep = torch.ones_like(q_comp)
                x_route = torch.cat([p_rep, sm_comp, smsc_comp, sin_rep, cos_rep], dim=-1).view(q_comp.shape[0], ngage * self.nmul, 5)
                q_routed_flat, route_mult_flat = self._route_q_dynamic_scale(q_for_routing, routpara, x_route)
                route_mult_4d = route_mult_flat.view(q_comp.shape[0], ngage, self.nmul, 1)
                route_mult_seq = self._mix_or_mean(route_mult_4d, wts)
                self._last_aux_loss = self._last_aux_loss + self.reg_amp_w * torch.mean((route_mult_flat - 1.0) ** 2)
                self._last_aux_loss = self._last_aux_loss + self.reg_smooth_w * torch.mean(
                    (route_mult_flat[1:, :, :] - route_mult_flat[:-1, :, :]) ** 2)
                q_routed = q_routed_flat.view(q_comp.shape[0], ngage, self.nmul, 1)
            else:
                q_routed = self._route_q(q_for_routing, routpara).view(q_comp.shape[0], ngage, self.nmul, 1)
            out = self._mix_or_mean(q_routed, wts)
        else:
            q_routed = q_comp
            if self.routOpt is True:
                if self.comprout is True:
                    q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
                    q_routed = self._route_q(q_for_routing, routpara).view(q_comp.shape[0], ngage, self.nmul, 1)
                    out = self._mix_or_mean(q_routed, wts)
                else:
                    out = self._route_q(q_mix_before_routing, routpara)
            else:
                out = q_mix_before_routing

        out = torch.clamp(out, min=0.0)
        if return_diagnostics is not True and return_component_diagnostics is not True:
            return out

        diag_out = dict()
        for name, tensor_comp in diag_comp.items():
            diag_out[name] = self._mix_component_tensor(tensor_comp, ngage)
        diag_out['total_discharge'] = out
        diag_out['q_mix_before_routing'] = q_mix_before_routing
        diag_out['component_discharge_raw'] = self._mix_or_mean(q_comp, wts)
        diag_out['channel_loss'] = torch.zeros_like(q_mix_before_routing)
        diag_out['gate_loss'] = torch.zeros_like(q_mix_before_routing)
        diag_out['groundwater_loss'] = torch.zeros_like(q_mix_before_routing)
        diag_out['Q_process'] = q_mix_before_routing
        diag_out['water_balance_residual'] = self._mix_component_tensor(diag_comp['process_local_residual'], ngage)
        if route_mult_seq is not None:
            diag_out['route_b_t_multiplier'] = route_mult_seq

        if return_component_diagnostics is True:
            for name, tensor_comp in diag_comp.items():
                diag_out[name + '_components'] = self._component_tensor_4d(tensor_comp, ngage).squeeze(-1)
            diag_out['q_raw_process_components'] = q_comp.squeeze(-1)
            diag_out['q_after_channel_loss_components'] = q_comp.squeeze(-1)
            diag_out['q_after_gate_components'] = q_comp.squeeze(-1)
            diag_out['Q_process_components'] = q_comp.squeeze(-1)
            diag_out['q_routed_components'] = q_routed.squeeze(-1)
            diag_out['channel_loss_components'] = torch.zeros_like(q_comp.squeeze(-1))
            diag_out['gate_loss_components'] = torch.zeros_like(q_comp.squeeze(-1))
            diag_out['process_local_residual_components'] = self._component_tensor_4d(diag_comp['process_local_residual'], ngage).squeeze(-1)
            diag_out['component_weights'] = wts
            if self.component_routing is True:
                routpara_view = torch.sigmoid(routpara0).view(ngage, self.nmul, 2)
                diag_out['route_a_components'] = 0.0 + routpara_view[:, :, 0] * 2.9
                diag_out['route_b_components'] = 0.0 + routpara_view[:, :, 1] * 6.5

        return out, diag_out


class DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimple(
        DynamicSimHydModelFiveDifferentiablePhysicalFix):
    """
    Very simple closed hydrology model: HBV snow, aSrz active root zone,
    SIMHYD-style partitioning, one groundwater store, and no discarded losses.
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True, dynamic_cfmax_snow=True,
                 dynamic_routing_scale=False, dynamic_all=False, dyn_hidden=32):
        super(DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimple, self).__init__(
            mode=mode, theta_is_raw=theta_is_raw, smooth=smooth, eps=eps, rain_snow_gain=rain_snow_gain,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, dyn_hidden=dyn_hidden)

    def _expand_simple(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)
        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = theta[:, 2:3] * 6.0
        SMSC_legacy = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = theta[:, 4:5]
        CRAK = theta[:, 5:6]
        K = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        TT = -2.5 + theta[:, 8:9] * 5.0
        CFMAX = 0.5 + theta[:, 9:10] * (10.0 - 0.5)
        CFR = theta[:, 10:11] * 0.1
        CWH = theta[:, 11:12] * 0.2
        theta_ab = 0.5 + theta[:, 13:14] * 0.5
        theta_ak = 1.0 + theta[:, 14:15] * 9.0
        theta_cap = 10.0 + theta[:, 15:16] * (1500.0 - 10.0)
        theta_efmax = 0.5 + theta[:, 16:17] * 0.5
        theta_wetpoint = 0.3 + theta[:, 17:18] * 0.6
        return (
            INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K, TT, CFMAX, CFR, CWH,
            theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint
        )

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        (INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K, TT, CFMAX_base, CFR, CWH,
         theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint) = self._expand_simple(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SA = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
            init_sa = SA
            init_gw = GW
            init_snow = SNOWPACK
            init_melt = MELTWATER
        else:
            SA = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]
            SNOWPACK = initial_state[:, 2:3]
            MELTWATER = initial_state[:, 3:4]
            init_sa = initial_state[:, 0:1]
            init_gw = initial_state[:, 1:2]
            init_snow = initial_state[:, 2:3]
            init_melt = initial_state[:, 3:4]

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = {}
        if return_diagnostics is True:
            for name in [
                    'SNOWPACK_prev', 'MELTWATER_prev', 'Sa_prev', 'GW_prev',
                    'SNOWPACK', 'MELTWATER', 'Sa', 'GW',
                    'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release',
                    'PL', 'interception_evaporation', 'actual_ET', 'LAI_t', 'LAI_scalar',
                    'Smoist', 'theta_cap', 'alpha', 'P_accessible', 'P_inaccessible', 'Sa_overflow',
                    'surface_runoff', 'interflow', 'recharge_to_groundwater',
                    'baseflow', 'Q_process', 'groundwater_loss', 'channel_loss', 'gate_loss',
                    'partition_sum_error', 'soil_local_residual', 'gw_local_residual',
                    'snowpack_local_residual', 'meltwater_local_residual', 'snow_total_local_residual',
                    'process_local_residual',
                    'INSC', 'COEF_t', 'SQ_t', 'K_t', 'TT', 'CFMAX_t', 'CFR', 'CWH'
            ]:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SA0 = self._min(self._pos(SA), theta_cap)
            GW0 = self._pos(GW)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            Smoist0 = torch.clamp(SA0 / (theta_cap + 1e-8), 0.0, 1.0)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0, Tt / 20.0, E0t / 10.0,
                SA0 / 300.0, Smoist0, GW0 / 300.0, SNOWPACK0 / 300.0, sin_t, cos_t
            ], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)

            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX_base * m_cf_eff

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()
            rainfall = Pt * frac_rain
            snowfall = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + snowfall
            snowmelt_pot = CFMAX_t * self._pos(Tt - TT)
            snowmelt = self._min(snowmelt_pot, SNOWPACK1)
            SNOWPACK2 = self._pos(SNOWPACK1 - snowmelt)
            MELTWATER1 = MELTWATER0 + snowmelt

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)
            SNOWPACK3 = SNOWPACK2 + refreezing

            snow_holding = CWH * SNOWPACK3
            snow_release_raw = self._pos(MELTWATER2 - snow_holding)
            snow_release = self._min(snow_release_raw, MELTWATER2)
            MELTWATER_next = self._pos(MELTWATER2 - snow_release)
            SNOWPACK_next = self._pos(SNOWPACK3)

            PL = rainfall + snow_release
            INT = self._min(INSC, self._min(E0t, PL))
            PL_after_int = self._pos(PL - INT)
            POT = self._pos(E0t - INT)
            if inputs.shape[-1] >= 6:
                LAI_t = self._pos(inputs[:, t, 5:6])
                LAI_scalar = 0.25 + 0.75 * torch.clamp(LAI_t / 5.0, 0.0, 1.0)
            else:
                LAI_t = torch.zeros_like(Pt)
                LAI_scalar = torch.ones_like(Pt)

            alpha = theta_ab * torch.pow(torch.clamp(1.0 - Smoist0, min=0.0), theta_ak)
            alpha = torch.clamp(alpha, 0.0, 1.0)
            P_accessible = alpha * PL_after_int
            P_inaccessible = (1.0 - alpha) * PL_after_int

            SA_pre = SA0 + P_accessible
            ET_stress = torch.clamp(Smoist0 / torch.clamp(theta_wetpoint, min=1e-6), 0.0, 1.0)
            ET_a_pot = POT * theta_efmax * ET_stress * LAI_scalar
            ET_a = self._min(ET_a_pot, SA_pre)
            SA_after_ET = self._pos(SA_pre - ET_a)
            SA_overflow = self._pos(SA_after_ET - theta_cap)
            SA_next = self._pos(SA_after_ET - SA_overflow)

            water_for_partition = self._pos(P_inaccessible + SA_overflow)
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_interflow = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([
                torch.log(base_surface),
                torch.log(base_interflow),
                torch.log(base_recharge)
            ], dim=1)
            if self.dynamic_partition is True:
                part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']]
            else:
                part_logits = base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]

            SRUN = f_surface * water_for_partition
            IFLOW = f_inter * water_for_partition
            REC = f_recharge * water_for_partition

            GW1 = GW0 + REC
            BAS_raw = K * GW1
            BAS = self._min(BAS_raw, GW1)
            GW_next = self._pos(GW1 - BAS)

            Q_process = self._pos(SRUN + IFLOW + BAS)
            process_local_residual = (
                Pt - INT - ET_a - Q_process
                - ((SNOWPACK_next - SNOWPACK0) + (MELTWATER_next - MELTWATER0) + (SA_next - SA0) + (GW_next - GW0))
            )
            soil_local_residual = P_accessible - ET_a - SA_overflow - (SA_next - SA0)
            gw_local_residual = REC - BAS - (GW_next - GW0)
            snowpack_local_residual = snowfall - snowmelt + refreezing - (SNOWPACK_next - SNOWPACK0)
            meltwater_local_residual = snowmelt - refreezing - snow_release - (MELTWATER_next - MELTWATER0)
            snow_total_local_residual = snowfall - snow_release - (SNOWPACK_next - SNOWPACK0) - (MELTWATER_next - MELTWATER0)

            SA = SA_next
            GW = GW_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q_process)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)

            if return_diagnostics is True:
                diag_vals = {
                    'SNOWPACK_prev': SNOWPACK0,
                    'MELTWATER_prev': MELTWATER0,
                    'Sa_prev': SA0,
                    'GW_prev': GW0,
                    'SNOWPACK': SNOWPACK_next,
                    'MELTWATER': MELTWATER_next,
                    'Sa': SA_next,
                    'GW': GW_next,
                    'precipitation': Pt,
                    'rainfall': rainfall,
                    'snowfall': snowfall,
                    'snowmelt': snowmelt,
                    'refreezing': refreezing,
                    'snow_release': snow_release,
                    'PL': PL,
                    'interception_evaporation': INT,
                    'actual_ET': ET_a,
                    'LAI_t': LAI_t,
                    'LAI_scalar': LAI_scalar,
                    'Smoist': Smoist0,
                    'theta_cap': theta_cap,
                    'alpha': alpha,
                    'P_accessible': P_accessible,
                    'P_inaccessible': P_inaccessible,
                    'Sa_overflow': SA_overflow,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'recharge_to_groundwater': REC,
                    'baseflow': BAS,
                    'Q_process': Q_process,
                    'groundwater_loss': torch.zeros_like(Q_process),
                    'channel_loss': torch.zeros_like(Q_process),
                    'gate_loss': torch.zeros_like(Q_process),
                    'partition_sum_error': torch.abs(f_surface + f_inter + f_recharge - 1.0),
                    'soil_local_residual': soil_local_residual,
                    'gw_local_residual': gw_local_residual,
                    'snowpack_local_residual': snowpack_local_residual,
                    'meltwater_local_residual': meltwater_local_residual,
                    'snow_total_local_residual': snow_total_local_residual,
                    'process_local_residual': process_local_residual,
                    'INSC': INSC,
                    'COEF_t': COEF,
                    'SQ_t': SQ_t,
                    'K_t': K,
                    'TT': TT,
                    'CFMAX_t': CFMAX_t,
                    'CFR': CFR,
                    'CWH': CWH,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SA, GW, SNOWPACK, MELTWATER], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)
        sum_p = torch.clamp(P.sum(dim=1), min=1e-6)
        drift_loss = (
            torch.mean(torch.abs(SA - init_sa) / sum_p)
            + torch.mean(torch.abs(GW - init_gw) / sum_p)
            + torch.mean(torch.abs(SNOWPACK - init_snow) / sum_p)
            + torch.mean(torch.abs(MELTWATER - init_melt) / sum_p)
        )
        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': reg_part,
            'storage_drift_loss': drift_loss,
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out
        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleLAIEco(
        DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimple):
    """
    Like the simple closed Model 6 copy, but when daily LAI is supplied it is
    used explicitly in interception, active-root-zone accessibility, and ET.
    The storage and runoff structure is otherwise unchanged from the legacy
    closed simple model.
    """

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        (INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K, TT, CFMAX_base, CFR, CWH,
         theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint) = self._expand_simple(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SA = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
            init_sa = SA
            init_gw = GW
            init_snow = SNOWPACK
            init_melt = MELTWATER
        else:
            SA = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]
            SNOWPACK = initial_state[:, 2:3]
            MELTWATER = initial_state[:, 3:4]
            init_sa = initial_state[:, 0:1]
            init_gw = initial_state[:, 1:2]
            init_snow = initial_state[:, 2:3]
            init_melt = initial_state[:, 3:4]

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = {}
        if return_diagnostics is True:
            for name in [
                    'SNOWPACK_prev', 'MELTWATER_prev', 'Sa_prev', 'GW_prev',
                    'SNOWPACK', 'MELTWATER', 'Sa', 'GW',
                    'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release',
                    'PL', 'interception_evaporation', 'actual_ET', 'LAI_t', 'LAI_rel',
                    'LAI_interception_scalar', 'LAI_et_scalar', 'alpha_base', 'alpha_veg_scalar',
                    'Smoist', 'theta_cap', 'alpha', 'P_accessible', 'P_inaccessible', 'Sa_overflow',
                    'surface_runoff', 'interflow', 'recharge_to_groundwater',
                    'baseflow', 'Q_process', 'groundwater_loss', 'channel_loss', 'gate_loss',
                    'partition_sum_error', 'soil_local_residual', 'gw_local_residual',
                    'snowpack_local_residual', 'meltwater_local_residual', 'snow_total_local_residual',
                    'process_local_residual',
                    'INSC', 'INSC_t', 'COEF_t', 'SQ_t', 'K_t', 'TT', 'CFMAX_t', 'CFR', 'CWH'
            ]:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SA0 = self._min(self._pos(SA), theta_cap)
            GW0 = self._pos(GW)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            Smoist0 = torch.clamp(SA0 / (theta_cap + 1e-8), 0.0, 1.0)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0, Tt / 20.0, E0t / 10.0,
                SA0 / 300.0, Smoist0, GW0 / 300.0, SNOWPACK0 / 300.0, sin_t, cos_t
            ], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)

            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX_base * m_cf_eff

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()
            rainfall = Pt * frac_rain
            snowfall = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + snowfall
            snowmelt_pot = CFMAX_t * self._pos(Tt - TT)
            snowmelt = self._min(snowmelt_pot, SNOWPACK1)
            SNOWPACK2 = self._pos(SNOWPACK1 - snowmelt)
            MELTWATER1 = MELTWATER0 + snowmelt

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)
            SNOWPACK3 = SNOWPACK2 + refreezing

            snow_holding = CWH * SNOWPACK3
            snow_release_raw = self._pos(MELTWATER2 - snow_holding)
            snow_release = self._min(snow_release_raw, MELTWATER2)
            MELTWATER_next = self._pos(MELTWATER2 - snow_release)
            SNOWPACK_next = self._pos(SNOWPACK3)

            PL = rainfall + snow_release
            if inputs.shape[-1] >= 6:
                LAI_t = self._pos(inputs[:, t, 5:6])
                LAI_rel = torch.clamp(LAI_t / 6.0, 0.0, 1.0)
                # Keep the old closed-model behavior as the reference state.
                # LAI is used here mainly to refine ET and interception rather
                # than to overhaul the rainfall-accessibility partition.
                LAI_anom = LAI_rel - 0.5
                LAI_interception_scalar = torch.clamp(1.0 + 0.25 * LAI_anom, 0.90, 1.10)
                LAI_et_scalar = torch.clamp(1.15 + 0.50 * LAI_anom, 0.95, 1.35)
                alpha_veg_scalar = torch.ones_like(Pt)
            else:
                LAI_t = torch.zeros_like(Pt)
                LAI_rel = torch.zeros_like(Pt)
                LAI_interception_scalar = torch.ones_like(Pt)
                LAI_et_scalar = torch.ones_like(Pt)
                alpha_veg_scalar = torch.ones_like(Pt)

            INSC_t = INSC * LAI_interception_scalar
            INT = self._min(INSC_t, self._min(E0t, PL))
            PL_after_int = self._pos(PL - INT)
            POT = self._pos(E0t - INT)

            alpha_base = theta_ab * torch.pow(torch.clamp(1.0 - Smoist0, min=0.0), theta_ak)
            alpha = torch.clamp(alpha_base * alpha_veg_scalar, 0.0, 1.0)
            P_accessible = alpha * PL_after_int
            P_inaccessible = (1.0 - alpha) * PL_after_int

            SA_pre = SA0 + P_accessible
            ET_stress = torch.clamp(Smoist0 / torch.clamp(theta_wetpoint, min=1e-6), 0.0, 1.0)
            ET_a_pot = POT * theta_efmax * ET_stress * LAI_et_scalar
            ET_a = self._min(ET_a_pot, SA_pre)
            SA_after_ET = self._pos(SA_pre - ET_a)
            SA_overflow = self._pos(SA_after_ET - theta_cap)
            SA_next = self._pos(SA_after_ET - SA_overflow)

            water_for_partition = self._pos(P_inaccessible + SA_overflow)
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_interflow = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([
                torch.log(base_surface),
                torch.log(base_interflow),
                torch.log(base_recharge)
            ], dim=1)
            if self.dynamic_partition is True:
                part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']]
            else:
                part_logits = base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]

            SRUN = f_surface * water_for_partition
            IFLOW = f_inter * water_for_partition
            REC = f_recharge * water_for_partition

            GW1 = GW0 + REC
            BAS_raw = K * GW1
            BAS = self._min(BAS_raw, GW1)
            GW_next = self._pos(GW1 - BAS)

            Q_process = self._pos(SRUN + IFLOW + BAS)
            process_local_residual = (
                Pt - INT - ET_a - Q_process
                - ((SNOWPACK_next - SNOWPACK0) + (MELTWATER_next - MELTWATER0) + (SA_next - SA0) + (GW_next - GW0))
            )
            soil_local_residual = P_accessible - ET_a - SA_overflow - (SA_next - SA0)
            gw_local_residual = REC - BAS - (GW_next - GW0)
            snowpack_local_residual = snowfall - snowmelt + refreezing - (SNOWPACK_next - SNOWPACK0)
            meltwater_local_residual = snowmelt - refreezing - snow_release - (MELTWATER_next - MELTWATER0)
            snow_total_local_residual = snowfall - snow_release - (SNOWPACK_next - SNOWPACK0) - (MELTWATER_next - MELTWATER0)

            SA = SA_next
            GW = GW_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q_process)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)

            if return_diagnostics is True:
                diag_vals = {
                    'SNOWPACK_prev': SNOWPACK0,
                    'MELTWATER_prev': MELTWATER0,
                    'Sa_prev': SA0,
                    'GW_prev': GW0,
                    'SNOWPACK': SNOWPACK_next,
                    'MELTWATER': MELTWATER_next,
                    'Sa': SA_next,
                    'GW': GW_next,
                    'precipitation': Pt,
                    'rainfall': rainfall,
                    'snowfall': snowfall,
                    'snowmelt': snowmelt,
                    'refreezing': refreezing,
                    'snow_release': snow_release,
                    'PL': PL,
                    'interception_evaporation': INT,
                    'actual_ET': ET_a,
                    'LAI_t': LAI_t,
                    'LAI_rel': LAI_rel,
                    'LAI_interception_scalar': LAI_interception_scalar,
                    'LAI_et_scalar': LAI_et_scalar,
                    'alpha_base': alpha_base,
                    'alpha_veg_scalar': alpha_veg_scalar,
                    'Smoist': Smoist0,
                    'theta_cap': theta_cap,
                    'alpha': alpha,
                    'P_accessible': P_accessible,
                    'P_inaccessible': P_inaccessible,
                    'Sa_overflow': SA_overflow,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'recharge_to_groundwater': REC,
                    'baseflow': BAS,
                    'Q_process': Q_process,
                    'groundwater_loss': torch.zeros_like(Q_process),
                    'channel_loss': torch.zeros_like(Q_process),
                    'gate_loss': torch.zeros_like(Q_process),
                    'partition_sum_error': torch.abs(f_surface + f_inter + f_recharge - 1.0),
                    'soil_local_residual': soil_local_residual,
                    'gw_local_residual': gw_local_residual,
                    'snowpack_local_residual': snowpack_local_residual,
                    'meltwater_local_residual': meltwater_local_residual,
                    'snow_total_local_residual': snow_total_local_residual,
                    'process_local_residual': process_local_residual,
                    'INSC': INSC,
                    'INSC_t': INSC_t,
                    'COEF_t': COEF,
                    'SQ_t': SQ_t,
                    'K_t': K,
                    'TT': TT,
                    'CFMAX_t': CFMAX_t,
                    'CFR': CFR,
                    'CWH': CWH,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SA, GW, SNOWPACK, MELTWATER], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)
        sum_p = torch.clamp(P.sum(dim=1), min=1e-6)
        drift_loss = (
            torch.mean(torch.abs(SA - init_sa) / sum_p)
            + torch.mean(torch.abs(GW - init_gw) / sum_p)
            + torch.mean(torch.abs(SNOWPACK - init_snow) / sum_p)
            + torch.mean(torch.abs(MELTWATER - init_melt) / sum_p)
        )
        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': reg_part,
            'storage_drift_loss': drift_loss,
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out
        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple(
        MultiInv_DynamicSimHydModelSix_Physical):
    """
    Very simple closed Model 6 copy with HBV snow, aSrz, SIMHYD partition,
    one groundwater store, and optional routing after closure.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 component_routing=True):
        super(MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing,
            dry_channel_loss=False, zero_flow_gate=False,
            channel_loss_max=0.0, gate_variant='soft', gate_strength_max=0.0)
        self.nfea = 18
        self.nstaticpm = self.nfea * nmul
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.simhyd = DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimple(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimple(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all)

    def _theta_to_smsc(self, theta, ngage):
        theta4 = theta.view(ngage, self.nmul, self.nfea)
        return 10.0 + theta4[:, :, 15:16] * (1500.0 - 10.0)


class MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleLAIEco(
        MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple):
    """
    Closed simple Model 6 copy with explicit daily-LAI controls for
    interception, ET, and active-root-zone accessibility, while preserving the
    legacy one-groundwater-store structure.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 component_routing=True):
        super(MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleLAIEco, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing)
        self.simhyd = DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleLAIEco(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleLAIEco(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all)


class DynamicSimHydModelSevenDifferentiableClosedRootDrainRTD(
        DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimple):
    """
    Closed Model 7 variant that adds continuous root-zone drainage and either
    a one-store or RTD-style three-store groundwater system without external
    losses. All groundwater recharge must eventually return as baseflow.
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True, dynamic_cfmax_snow=True,
                 dynamic_routing_scale=False, dynamic_all=False, dyn_hidden=32,
                 use_rtd=True, use_equilibrium_gw_init=True, gw_init_recharge_frac=0.08,
                 gw_init_max_store=800.0):
        super(DynamicSimHydModelSevenDifferentiableClosedRootDrainRTD, self).__init__(
            mode=mode, theta_is_raw=theta_is_raw, smooth=smooth, eps=eps, rain_snow_gain=rain_snow_gain,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, dyn_hidden=dyn_hidden)
        self.use_rtd = use_rtd
        self.use_equilibrium_gw_init = use_equilibrium_gw_init
        self.gw_init_recharge_frac = gw_init_recharge_frac
        self.gw_init_max_store = gw_init_max_store

    def _expand_rootdrain_rtd(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)
        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = theta[:, 2:3] * 6.0
        SMSC_legacy = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = theta[:, 4:5]
        CRAK = theta[:, 5:6]
        K = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        TT = -2.5 + theta[:, 8:9] * 5.0
        CFMAX = 0.5 + theta[:, 9:10] * (10.0 - 0.5)
        CFR = theta[:, 10:11] * 0.1
        CWH = theta[:, 11:12] * 0.2
        theta_ab = 0.5 + theta[:, 13:14] * 0.5
        theta_ak = 1.0 + theta[:, 14:15] * 9.0
        theta_cap = 10.0 + theta[:, 15:16] * (1500.0 - 10.0)
        theta_efmax = 0.5 + theta[:, 16:17] * 0.5
        theta_wetpoint = 0.3 + theta[:, 17:18] * 0.6
        KDRAIN = 1e-4 + theta[:, 18:19] * (0.25 - 1e-4)
        DRAIN_EXP = 0.5 + theta[:, 19:20] * (8.0 - 0.5)
        DRAIN_FAST_FRAC = theta[:, 20:21] * 0.8
        gw_weight_logits = theta[:, 21:24]
        tau_fast = 2.0 + 28.0 * theta[:, 24:25]
        tau_mid = 30.0 + 220.0 * theta[:, 25:26]
        tau_slow = 250.0 + 1250.0 * theta[:, 26:27]
        return (
            INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K, TT, CFMAX, CFR, CWH,
            theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint,
            KDRAIN, DRAIN_EXP, DRAIN_FAST_FRAC, gw_weight_logits,
            tau_fast, tau_mid, tau_slow
        )

    def _init_groundwater_states(self, P, K, tau_fast, tau_mid, tau_slow, gw_weights, device, dtype):
        B = P.shape[0]
        if self.use_equilibrium_gw_init is not True:
            zeros = torch.zeros(B, 1, device=device, dtype=dtype)
            return zeros, zeros, zeros
        mean_p = torch.mean(P, dim=1)
        rec_proxy = torch.clamp(mean_p * self.gw_init_recharge_frac, min=0.0).detach()
        if self.use_rtd:
            k_fast = 1.0 - torch.exp(-1.0 / tau_fast.detach())
            k_mid = 1.0 - torch.exp(-1.0 / tau_mid.detach())
            k_slow = 1.0 - torch.exp(-1.0 / tau_slow.detach())
            rec_fast = gw_weights[:, 0:1].detach() * rec_proxy
            rec_mid = gw_weights[:, 1:2].detach() * rec_proxy
            rec_slow = gw_weights[:, 2:3].detach() * rec_proxy
            gw_fast = torch.clamp(rec_fast / torch.clamp(k_fast, min=1e-6), 0.0, self.gw_init_max_store)
            gw_mid = torch.clamp(rec_mid / torch.clamp(k_mid, min=1e-6), 0.0, self.gw_init_max_store)
            gw_slow = torch.clamp(rec_slow / torch.clamp(k_slow, min=1e-6), 0.0, self.gw_init_max_store)
            return gw_fast, gw_mid, gw_slow
        gw_fast = torch.clamp(rec_proxy / torch.clamp(K.detach(), min=1e-6), 0.0, self.gw_init_max_store)
        zeros = torch.zeros(B, 1, device=device, dtype=dtype)
        return gw_fast, zeros, zeros

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        (INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K, TT, CFMAX_base, CFR, CWH,
         theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint,
         KDRAIN, DRAIN_EXP, DRAIN_FAST_FRAC, gw_weight_logits,
         tau_fast, tau_mid, tau_slow) = self._expand_rootdrain_rtd(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        gw_weights_static = F.softmax(gw_weight_logits, dim=1)
        if initial_state is None:
            SA = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
            GW_FAST, GW_MID, GW_SLOW = self._init_groundwater_states(
                P, K, tau_fast, tau_mid, tau_slow, gw_weights_static, device, dtype)
            init_sa = SA.clone()
            init_gw_fast = GW_FAST.clone()
            init_gw_mid = GW_MID.clone()
            init_gw_slow = GW_SLOW.clone()
            init_snow = SNOWPACK.clone()
            init_melt = MELTWATER.clone()
        else:
            SA = initial_state[:, 0:1]
            GW_FAST = initial_state[:, 1:2]
            GW_MID = initial_state[:, 2:3]
            GW_SLOW = initial_state[:, 3:4]
            SNOWPACK = initial_state[:, 4:5]
            MELTWATER = initial_state[:, 5:6]
            init_sa = SA.clone()
            init_gw_fast = GW_FAST.clone()
            init_gw_mid = GW_MID.clone()
            init_gw_slow = GW_SLOW.clone()
            init_snow = SNOWPACK.clone()
            init_melt = MELTWATER.clone()

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = {}
        if return_diagnostics is True:
            for name in [
                    'SNOWPACK_prev', 'MELTWATER_prev', 'Sa_prev', 'GW_prev',
                    'GW_FAST_prev', 'GW_MID_prev', 'GW_SLOW_prev',
                    'SNOWPACK', 'MELTWATER', 'Sa', 'GW',
                    'GW_FAST', 'GW_MID', 'GW_SLOW',
                    'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release',
                    'PL', 'interception_evaporation', 'actual_ET', 'LAI_t', 'LAI_scalar',
                    'Smoist', 'theta_cap', 'alpha', 'P_accessible', 'P_inaccessible',
                    'ROOT_DRAIN', 'ROOT_IFLOW', 'ROOT_RECHARGE', 'Sa_overflow',
                    'surface_runoff', 'interflow', 'recharge_to_groundwater',
                    'REC_PARTITION', 'REC_TOTAL', 'REC_FAST', 'REC_MID', 'REC_SLOW',
                    'BAS_FAST', 'BAS_MID', 'BAS_SLOW', 'BAS_TOTAL',
                    'GW_WEIGHT_FAST', 'GW_WEIGHT_MID', 'GW_WEIGHT_SLOW',
                    'TAU_FAST', 'TAU_MID', 'TAU_SLOW',
                    'KDRAIN', 'DRAIN_EXP', 'DRAIN_FAST_FRAC',
                    'Q_process', 'groundwater_loss', 'channel_loss', 'gate_loss',
                    'partition_sum_error', 'soil_local_residual', 'gw_local_residual',
                    'snowpack_local_residual', 'meltwater_local_residual', 'snow_total_local_residual',
                    'process_local_residual', 'mean_recharge_minus_baseflow', 'groundwater_storage_drift',
                    'INSC', 'COEF_t', 'SQ_t', 'K_t', 'TT', 'CFMAX_t', 'CFR', 'CWH'
            ]:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []
        gw_stationarity_hist = []
        gw_drift_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SA0 = self._min(self._pos(SA), theta_cap)
            GW_FAST0 = self._pos(GW_FAST)
            GW_MID0 = self._pos(GW_MID)
            GW_SLOW0 = self._pos(GW_SLOW)
            GW0 = GW_FAST0 + GW_MID0 + GW_SLOW0
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            Smoist0 = torch.clamp(SA0 / (theta_cap + 1e-8), 0.0, 1.0)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0, Tt / 20.0, E0t / 10.0,
                SA0 / 300.0, Smoist0, GW0 / 300.0, SNOWPACK0 / 300.0, sin_t, cos_t
            ], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)

            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX_base * m_cf_eff

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()
            rainfall = Pt * frac_rain
            snowfall = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + snowfall
            snowmelt_pot = CFMAX_t * self._pos(Tt - TT)
            snowmelt = self._min(snowmelt_pot, SNOWPACK1)
            SNOWPACK2 = self._pos(SNOWPACK1 - snowmelt)
            MELTWATER1 = MELTWATER0 + snowmelt

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)
            SNOWPACK3 = SNOWPACK2 + refreezing

            snow_holding = CWH * SNOWPACK3
            snow_release_raw = self._pos(MELTWATER2 - snow_holding)
            snow_release = self._min(snow_release_raw, MELTWATER2)
            MELTWATER_next = self._pos(MELTWATER2 - snow_release)
            SNOWPACK_next = self._pos(SNOWPACK3)

            PL = rainfall + snow_release
            INT = self._min(INSC, self._min(E0t, PL))
            PL_after_int = self._pos(PL - INT)
            POT = self._pos(E0t - INT)
            if inputs.shape[-1] >= 6:
                LAI_t = self._pos(inputs[:, t, 5:6])
                LAI_scalar = 0.25 + 0.75 * torch.clamp(LAI_t / 5.0, 0.0, 1.0)
            else:
                LAI_t = torch.zeros_like(Pt)
                LAI_scalar = torch.ones_like(Pt)

            alpha = theta_ab * torch.pow(torch.clamp(1.0 - Smoist0, min=0.0), theta_ak)
            alpha = torch.clamp(alpha, 0.0, 1.0)
            P_accessible = alpha * PL_after_int
            P_inaccessible = (1.0 - alpha) * PL_after_int

            SA_pre = SA0 + P_accessible
            ET_stress = torch.clamp(Smoist0 / torch.clamp(theta_wetpoint, min=1e-6), 0.0, 1.0)
            ET_a_pot = POT * theta_efmax * ET_stress * LAI_scalar
            ET_a = self._min(ET_a_pot, SA_pre)
            SA_after_ET = self._pos(SA_pre - ET_a)

            s_rel = torch.clamp(SA_after_ET / torch.clamp(theta_cap, min=1e-6), 0.0, 1.0)
            drain_rate = KDRAIN * (self.eps + torch.pow(s_rel, DRAIN_EXP))
            ROOT_DRAIN = SA_after_ET * (1.0 - torch.exp(-drain_rate))
            ROOT_DRAIN = self._min(ROOT_DRAIN, SA_after_ET)
            ROOT_DRAIN = torch.clamp(ROOT_DRAIN, min=0.0)
            ROOT_IFLOW = DRAIN_FAST_FRAC * ROOT_DRAIN
            ROOT_RECHARGE = (1.0 - DRAIN_FAST_FRAC) * ROOT_DRAIN

            SA_after_drain = self._pos(SA_after_ET - ROOT_DRAIN)
            SA_overflow = self._pos(SA_after_drain - theta_cap)
            SA_next = self._pos(SA_after_drain - SA_overflow)

            water_for_partition = self._pos(P_inaccessible + SA_overflow)
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_interflow = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([
                torch.log(base_surface),
                torch.log(base_interflow),
                torch.log(base_recharge)
            ], dim=1)
            if self.dynamic_partition is True:
                part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']]
            else:
                part_logits = base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]

            SRUN = f_surface * water_for_partition
            IFLOW_PARTITION = f_inter * water_for_partition
            REC_PARTITION = f_recharge * water_for_partition

            IFLOW = IFLOW_PARTITION + ROOT_IFLOW
            REC_TOTAL = REC_PARTITION + ROOT_RECHARGE

            if self.use_rtd:
                gw_weights = gw_weights_static
                REC_FAST = gw_weights[:, 0:1] * REC_TOTAL
                REC_MID = gw_weights[:, 1:2] * REC_TOTAL
                REC_SLOW = gw_weights[:, 2:3] * REC_TOTAL

                k_fast = 1.0 - torch.exp(-1.0 / tau_fast)
                k_mid = 1.0 - torch.exp(-1.0 / tau_mid)
                k_slow = 1.0 - torch.exp(-1.0 / tau_slow)

                GW_FAST_PRE = GW_FAST0 + REC_FAST
                GW_MID_PRE = GW_MID0 + REC_MID
                GW_SLOW_PRE = GW_SLOW0 + REC_SLOW

                BAS_FAST = self._min(k_fast * GW_FAST_PRE, GW_FAST_PRE)
                BAS_MID = self._min(k_mid * GW_MID_PRE, GW_MID_PRE)
                BAS_SLOW = self._min(k_slow * GW_SLOW_PRE, GW_SLOW_PRE)

                GW_FAST_next = self._pos(GW_FAST_PRE - BAS_FAST)
                GW_MID_next = self._pos(GW_MID_PRE - BAS_MID)
                GW_SLOW_next = self._pos(GW_SLOW_PRE - BAS_SLOW)
            else:
                gw_weights = torch.cat([
                    torch.ones(B, 1, device=device, dtype=dtype),
                    torch.zeros(B, 1, device=device, dtype=dtype),
                    torch.zeros(B, 1, device=device, dtype=dtype)
                ], dim=1)
                REC_FAST = REC_TOTAL
                REC_MID = torch.zeros_like(REC_TOTAL)
                REC_SLOW = torch.zeros_like(REC_TOTAL)
                GW_FAST_PRE = GW_FAST0 + REC_FAST
                GW_MID_PRE = GW_MID0
                GW_SLOW_PRE = GW_SLOW0
                BAS_FAST = self._min(K * GW_FAST_PRE, GW_FAST_PRE)
                BAS_MID = torch.zeros_like(BAS_FAST)
                BAS_SLOW = torch.zeros_like(BAS_FAST)
                GW_FAST_next = self._pos(GW_FAST_PRE - BAS_FAST)
                GW_MID_next = GW_MID0
                GW_SLOW_next = GW_SLOW0

            BAS = BAS_FAST + BAS_MID + BAS_SLOW
            GW_next = GW_FAST_next + GW_MID_next + GW_SLOW_next

            Q_process = self._pos(SRUN + IFLOW + BAS)
            process_local_residual = (
                Pt - INT - ET_a - Q_process
                - ((SNOWPACK_next - SNOWPACK0) + (MELTWATER_next - MELTWATER0) + (SA_next - SA0)
                   + (GW_FAST_next - GW_FAST0) + (GW_MID_next - GW_MID0) + (GW_SLOW_next - GW_SLOW0))
            )
            soil_local_residual = P_accessible - ET_a - ROOT_DRAIN - SA_overflow - (SA_next - SA0)
            gw_local_residual = REC_TOTAL - BAS - (GW_next - GW0)
            snowpack_local_residual = snowfall - snowmelt + refreezing - (SNOWPACK_next - SNOWPACK0)
            meltwater_local_residual = snowmelt - refreezing - snow_release - (MELTWATER_next - MELTWATER0)
            snow_total_local_residual = snowfall - snow_release - (SNOWPACK_next - SNOWPACK0) - (MELTWATER_next - MELTWATER0)

            SA = SA_next
            GW_FAST = GW_FAST_next
            GW_MID = GW_MID_next
            GW_SLOW = GW_SLOW_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q_process)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)
            gw_stationarity_hist.append(torch.abs(REC_TOTAL - BAS))
            gw_drift_hist.append(GW_next)

            if return_diagnostics is True:
                diag_vals = {
                    'SNOWPACK_prev': SNOWPACK0,
                    'MELTWATER_prev': MELTWATER0,
                    'Sa_prev': SA0,
                    'GW_prev': GW0,
                    'GW_FAST_prev': GW_FAST0,
                    'GW_MID_prev': GW_MID0,
                    'GW_SLOW_prev': GW_SLOW0,
                    'SNOWPACK': SNOWPACK_next,
                    'MELTWATER': MELTWATER_next,
                    'Sa': SA_next,
                    'GW': GW_next,
                    'GW_FAST': GW_FAST_next,
                    'GW_MID': GW_MID_next,
                    'GW_SLOW': GW_SLOW_next,
                    'precipitation': Pt,
                    'rainfall': rainfall,
                    'snowfall': snowfall,
                    'snowmelt': snowmelt,
                    'refreezing': refreezing,
                    'snow_release': snow_release,
                    'PL': PL,
                    'interception_evaporation': INT,
                    'actual_ET': ET_a,
                    'LAI_t': LAI_t,
                    'LAI_scalar': LAI_scalar,
                    'Smoist': Smoist0,
                    'theta_cap': theta_cap,
                    'alpha': alpha,
                    'P_accessible': P_accessible,
                    'P_inaccessible': P_inaccessible,
                    'ROOT_DRAIN': ROOT_DRAIN,
                    'ROOT_IFLOW': ROOT_IFLOW,
                    'ROOT_RECHARGE': ROOT_RECHARGE,
                    'Sa_overflow': SA_overflow,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'recharge_to_groundwater': REC_TOTAL,
                    'REC_PARTITION': REC_PARTITION,
                    'REC_TOTAL': REC_TOTAL,
                    'REC_FAST': REC_FAST,
                    'REC_MID': REC_MID,
                    'REC_SLOW': REC_SLOW,
                    'BAS_FAST': BAS_FAST,
                    'BAS_MID': BAS_MID,
                    'BAS_SLOW': BAS_SLOW,
                    'BAS_TOTAL': BAS,
                    'GW_WEIGHT_FAST': gw_weights[:, 0:1],
                    'GW_WEIGHT_MID': gw_weights[:, 1:2],
                    'GW_WEIGHT_SLOW': gw_weights[:, 2:3],
                    'TAU_FAST': tau_fast,
                    'TAU_MID': tau_mid,
                    'TAU_SLOW': tau_slow,
                    'KDRAIN': KDRAIN,
                    'DRAIN_EXP': DRAIN_EXP,
                    'DRAIN_FAST_FRAC': DRAIN_FAST_FRAC,
                    'Q_process': Q_process,
                    'groundwater_loss': torch.zeros_like(Q_process),
                    'channel_loss': torch.zeros_like(Q_process),
                    'gate_loss': torch.zeros_like(Q_process),
                    'partition_sum_error': torch.abs(f_surface + f_inter + f_recharge - 1.0),
                    'soil_local_residual': soil_local_residual,
                    'gw_local_residual': gw_local_residual,
                    'snowpack_local_residual': snowpack_local_residual,
                    'meltwater_local_residual': meltwater_local_residual,
                    'snow_total_local_residual': snow_total_local_residual,
                    'process_local_residual': process_local_residual,
                    'mean_recharge_minus_baseflow': REC_TOTAL - BAS,
                    'groundwater_storage_drift': GW_next - GW0,
                    'INSC': INSC,
                    'COEF_t': COEF,
                    'SQ_t': SQ_t,
                    'K_t': K,
                    'TT': TT,
                    'CFMAX_t': CFMAX_t,
                    'CFR': CFR,
                    'CWH': CWH,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SA, GW_FAST, GW_MID, GW_SLOW, SNOWPACK, MELTWATER], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)
        sum_p = torch.clamp(P.sum(dim=1), min=1e-6)
        drift_loss = (
            torch.mean(torch.abs(SA - init_sa) / sum_p)
            + torch.mean(torch.abs(GW_FAST - init_gw_fast) / sum_p)
            + torch.mean(torch.abs(GW_MID - init_gw_mid) / sum_p)
            + torch.mean(torch.abs(GW_SLOW - init_gw_slow) / sum_p)
            + torch.mean(torch.abs(SNOWPACK - init_snow) / sum_p)
            + torch.mean(torch.abs(MELTWATER - init_melt) / sum_p)
        )
        gw_drift_loss = torch.mean(torch.abs((GW_FAST + GW_MID + GW_SLOW) - (init_gw_fast + init_gw_mid + init_gw_slow)) / sum_p)
        sa_drift_loss = torch.mean(torch.abs(SA - init_sa) / sum_p)
        mean_p = torch.clamp(torch.mean(P), min=1e-6)
        if len(gw_stationarity_hist) > 0:
            gw_stationarity_loss = torch.mean(torch.stack(gw_stationarity_hist, dim=1)) / mean_p
        else:
            gw_stationarity_loss = torch.tensor(0.0, device=device, dtype=dtype)
        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': reg_part,
            'storage_drift_loss': drift_loss,
            'gw_drift_loss': gw_drift_loss,
            'sa_drift_loss': sa_drift_loss,
            'gw_stationarity_loss': gw_stationarity_loss,
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out
        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelSeven_Physical_ClosedRootDrainRTD(
        MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple):
    """
    Wrapper for Model 7 that preserves the existing regionalization, routing,
    and component-mixture machinery while swapping in the new groundwater
    physics.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 reg_storage_drift_w=1e-3, reg_gw_drift_w=1e-3, reg_sa_drift_w=1e-3,
                 reg_gw_stationarity_w=1e-2, component_routing=True,
                 use_rtd=True, use_equilibrium_gw_init=True):
        super(MultiInv_DynamicSimHydModelSeven_Physical_ClosedRootDrainRTD, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing)
        self.reg_storage_drift_w = reg_storage_drift_w
        self.reg_gw_drift_w = reg_gw_drift_w
        self.reg_sa_drift_w = reg_sa_drift_w
        self.reg_gw_stationarity_w = reg_gw_stationarity_w
        self.use_rtd = use_rtd
        self.nfea = 27
        self.nstaticpm = self.nfea * nmul
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.simhyd = DynamicSimHydModelSevenDifferentiableClosedRootDrainRTD(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            use_rtd=use_rtd, use_equilibrium_gw_init=use_equilibrium_gw_init)
        self.simhyd_analysis = DynamicSimHydModelSevenDifferentiableClosedRootDrainRTD(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            use_rtd=use_rtd, use_equilibrium_gw_init=use_equilibrium_gw_init)

    def _theta_to_smsc(self, theta, ngage):
        theta4 = theta.view(ngage, self.nmul, self.nfea)
        return 10.0 + theta4[:, :, 15:16] * (1500.0 - 10.0)

    def get_auxiliary_loss(self):
        aux = super(MultiInv_DynamicSimHydModelSeven_Physical_ClosedRootDrainRTD, self).get_auxiliary_loss()
        if aux is None:
            return None
        if self._last_aux_terms is None:
            return aux
        return (
            aux
            + self.reg_storage_drift_w * self._last_aux_terms.get('storage_drift_loss', 0.0)
            + self.reg_gw_drift_w * self._last_aux_terms.get('gw_drift_loss', 0.0)
            + self.reg_sa_drift_w * self._last_aux_terms.get('sa_drift_loss', 0.0)
            + self.reg_gw_stationarity_w * self._last_aux_terms.get('gw_stationarity_loss', 0.0)
        )


class DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicK(
        DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimple):
    """
    Closed simple snow+aSrz+SIMHYD copy with optional dynamic groundwater
    recession coefficient K_t. All other physics remain unchanged.
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True, dynamic_cfmax_snow=True,
                 dynamic_routing_scale=False, dynamic_all=False, dyn_hidden=32,
                 dynamic_k=True):
        super(DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicK, self).__init__(
            mode=mode, theta_is_raw=theta_is_raw, smooth=smooth, eps=eps, rain_snow_gain=rain_snow_gain,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, dyn_hidden=dyn_hidden)
        self.dynamic_k = dynamic_k or dynamic_all
        if self.dynamic_k is True:
            self.dyn_slices['k'] = slice(self.dyn_out_dim, self.dyn_out_dim + 1)
            self.dyn_out_dim += 1
            self.dynHead = nn.Sequential(
                nn.Linear(9, dyn_hidden),
                nn.ReLU(),
                nn.Linear(dyn_hidden, dyn_hidden),
                nn.ReLU(),
                nn.Linear(dyn_hidden, self.dyn_out_dim))
            self._dynhead_force_cpu_fallback = False

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        (INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K, TT, CFMAX_base, CFR, CWH,
         theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint) = self._expand_simple(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SA = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
            init_sa = SA
            init_gw = GW
            init_snow = SNOWPACK
            init_melt = MELTWATER
        else:
            SA = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]
            SNOWPACK = initial_state[:, 2:3]
            MELTWATER = initial_state[:, 3:4]
            init_sa = initial_state[:, 0:1]
            init_gw = initial_state[:, 1:2]
            init_snow = initial_state[:, 2:3]
            init_melt = initial_state[:, 3:4]

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = {}
        if return_diagnostics is True:
            for name in [
                    'SNOWPACK_prev', 'MELTWATER_prev', 'Sa_prev', 'GW_prev',
                    'SNOWPACK', 'MELTWATER', 'Sa', 'GW',
                    'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release',
                    'PL', 'interception_evaporation', 'actual_ET', 'LAI_t', 'LAI_scalar',
                    'Smoist', 'theta_cap', 'alpha', 'P_accessible', 'P_inaccessible', 'Sa_overflow',
                    'surface_runoff', 'interflow', 'recharge_to_groundwater',
                    'baseflow', 'Q_process', 'groundwater_loss', 'channel_loss', 'gate_loss',
                    'partition_sum_error', 'soil_local_residual', 'gw_local_residual',
                    'snowpack_local_residual', 'meltwater_local_residual', 'snow_total_local_residual',
                    'process_local_residual',
                    'INSC', 'COEF_t', 'SQ_t', 'K_t', 'K_mult_t', 'TT', 'CFMAX_t', 'CFR', 'CWH'
            ]:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []
        k_t_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SA0 = self._min(self._pos(SA), theta_cap)
            GW0 = self._pos(GW)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            Smoist0 = torch.clamp(SA0 / (theta_cap + 1e-8), 0.0, 1.0)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0, Tt / 20.0, E0t / 10.0,
                SA0 / 300.0, Smoist0, GW0 / 300.0, SNOWPACK0 / 300.0, sin_t, cos_t
            ], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)

            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX_base * m_cf_eff

            if self.dynamic_k is True and dyn_raw is not None:
                K_mult_t = 0.25 + 2.75 * torch.sigmoid(dyn_raw[:, self.dyn_slices['k']])
                K_t = torch.clamp(K * K_mult_t, 0.001, 0.7)
            else:
                K_mult_t = torch.ones_like(K)
                K_t = K

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()
            rainfall = Pt * frac_rain
            snowfall = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + snowfall
            snowmelt_pot = CFMAX_t * self._pos(Tt - TT)
            snowmelt = self._min(snowmelt_pot, SNOWPACK1)
            SNOWPACK2 = self._pos(SNOWPACK1 - snowmelt)
            MELTWATER1 = MELTWATER0 + snowmelt

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)
            SNOWPACK3 = SNOWPACK2 + refreezing

            snow_holding = CWH * SNOWPACK3
            snow_release_raw = self._pos(MELTWATER2 - snow_holding)
            snow_release = self._min(snow_release_raw, MELTWATER2)
            MELTWATER_next = self._pos(MELTWATER2 - snow_release)
            SNOWPACK_next = self._pos(SNOWPACK3)

            PL = rainfall + snow_release
            INT = self._min(INSC, self._min(E0t, PL))
            PL_after_int = self._pos(PL - INT)
            POT = self._pos(E0t - INT)
            if inputs.shape[-1] >= 6:
                LAI_t = self._pos(inputs[:, t, 5:6])
                LAI_scalar = 0.25 + 0.75 * torch.clamp(LAI_t / 5.0, 0.0, 1.0)
            else:
                LAI_t = torch.zeros_like(Pt)
                LAI_scalar = torch.ones_like(Pt)

            alpha = theta_ab * torch.pow(torch.clamp(1.0 - Smoist0, min=0.0), theta_ak)
            alpha = torch.clamp(alpha, 0.0, 1.0)
            P_accessible = alpha * PL_after_int
            P_inaccessible = (1.0 - alpha) * PL_after_int

            SA_pre = SA0 + P_accessible
            ET_stress = torch.clamp(Smoist0 / torch.clamp(theta_wetpoint, min=1e-6), 0.0, 1.0)
            ET_a_pot = POT * theta_efmax * ET_stress * LAI_scalar
            ET_a = self._min(ET_a_pot, SA_pre)
            SA_after_ET = self._pos(SA_pre - ET_a)
            SA_overflow = self._pos(SA_after_ET - theta_cap)
            SA_next = self._pos(SA_after_ET - SA_overflow)

            water_for_partition = self._pos(P_inaccessible + SA_overflow)
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_interflow = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([
                torch.log(base_surface),
                torch.log(base_interflow),
                torch.log(base_recharge)
            ], dim=1)
            if self.dynamic_partition is True:
                part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']]
            else:
                part_logits = base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]

            SRUN = f_surface * water_for_partition
            IFLOW = f_inter * water_for_partition
            REC = f_recharge * water_for_partition

            GW1 = GW0 + REC
            BAS_raw = K_t * GW1
            BAS = self._min(BAS_raw, GW1)
            GW_next = self._pos(GW1 - BAS)

            Q_process = self._pos(SRUN + IFLOW + BAS)
            process_local_residual = (
                Pt - INT - ET_a - Q_process
                - ((SNOWPACK_next - SNOWPACK0) + (MELTWATER_next - MELTWATER0) + (SA_next - SA0) + (GW_next - GW0))
            )
            soil_local_residual = P_accessible - ET_a - SA_overflow - (SA_next - SA0)
            gw_local_residual = REC - BAS - (GW_next - GW0)
            snowpack_local_residual = snowfall - snowmelt + refreezing - (SNOWPACK_next - SNOWPACK0)
            meltwater_local_residual = snowmelt - refreezing - snow_release - (MELTWATER_next - MELTWATER0)
            snow_total_local_residual = snowfall - snow_release - (SNOWPACK_next - SNOWPACK0) - (MELTWATER_next - MELTWATER0)

            SA = SA_next
            GW = GW_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q_process)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)
            k_t_hist.append(K_t)

            if return_diagnostics is True:
                diag_vals = {
                    'SNOWPACK_prev': SNOWPACK0,
                    'MELTWATER_prev': MELTWATER0,
                    'Sa_prev': SA0,
                    'GW_prev': GW0,
                    'SNOWPACK': SNOWPACK_next,
                    'MELTWATER': MELTWATER_next,
                    'Sa': SA_next,
                    'GW': GW_next,
                    'precipitation': Pt,
                    'rainfall': rainfall,
                    'snowfall': snowfall,
                    'snowmelt': snowmelt,
                    'refreezing': refreezing,
                    'snow_release': snow_release,
                    'PL': PL,
                    'interception_evaporation': INT,
                    'actual_ET': ET_a,
                    'LAI_t': LAI_t,
                    'LAI_scalar': LAI_scalar,
                    'Smoist': Smoist0,
                    'theta_cap': theta_cap,
                    'alpha': alpha,
                    'P_accessible': P_accessible,
                    'P_inaccessible': P_inaccessible,
                    'Sa_overflow': SA_overflow,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'recharge_to_groundwater': REC,
                    'baseflow': BAS,
                    'Q_process': Q_process,
                    'groundwater_loss': torch.zeros_like(Q_process),
                    'channel_loss': torch.zeros_like(Q_process),
                    'gate_loss': torch.zeros_like(Q_process),
                    'partition_sum_error': torch.abs(f_surface + f_inter + f_recharge - 1.0),
                    'soil_local_residual': soil_local_residual,
                    'gw_local_residual': gw_local_residual,
                    'snowpack_local_residual': snowpack_local_residual,
                    'meltwater_local_residual': meltwater_local_residual,
                    'snow_total_local_residual': snow_total_local_residual,
                    'process_local_residual': process_local_residual,
                    'INSC': INSC,
                    'COEF_t': COEF,
                    'SQ_t': SQ_t,
                    'K_t': K_t,
                    'K_mult_t': K_mult_t,
                    'TT': TT,
                    'CFMAX_t': CFMAX_t,
                    'CFR': CFR,
                    'CWH': CWH,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SA, GW, SNOWPACK, MELTWATER], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        reg_k_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)
        if len(k_t_hist) > 1:
            k_seq = torch.stack(k_t_hist, dim=1)
            reg_k_smooth = torch.mean((k_seq[:, 1:, :] - k_seq[:, :-1, :]) ** 2)
        sum_p = torch.clamp(P.sum(dim=1), min=1e-6)
        drift_loss = (
            torch.mean(torch.abs(SA - init_sa) / sum_p)
            + torch.mean(torch.abs(GW - init_gw) / sum_p)
            + torch.mean(torch.abs(SNOWPACK - init_snow) / sum_p)
            + torch.mean(torch.abs(MELTWATER - init_melt) / sum_p)
        )
        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': reg_part,
            'storage_drift_loss': drift_loss,
            'k_smooth_loss': reg_k_smooth,
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out
        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleDynamicK(
        MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple):
    """
    Copied closed simple Model 6 variant with optional dynamic groundwater
    recession coefficient K_t and smoothness regularization.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 reg_k_smooth_w=1e-4, component_routing=True, dynamic_k=True):
        super(MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleDynamicK, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing)
        self.reg_k_smooth_w = reg_k_smooth_w
        self.simhyd = DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicK(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            dynamic_k=dynamic_k)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicK(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            dynamic_k=dynamic_k)

    def get_auxiliary_loss(self):
        aux = super(MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleDynamicK, self).get_auxiliary_loss()
        if aux is None:
            return None
        if self._last_aux_terms is None:
            return aux
        return aux + self.reg_k_smooth_w * self._last_aux_terms.get('k_smooth_loss', 0.0)


class DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicKLAIEco(
        DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicK):
    """
    Copied closed simple dynamic-K variant where daily LAI is used explicitly
    in interception, active-root-zone ET, and accessible root-zone partitioning.
    All other physics remain unchanged.
    """

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        (INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K, TT, CFMAX_base, CFR, CWH,
         theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint) = self._expand_simple(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SA = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
            init_sa = SA
            init_gw = GW
            init_snow = SNOWPACK
            init_melt = MELTWATER
        else:
            SA = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]
            SNOWPACK = initial_state[:, 2:3]
            MELTWATER = initial_state[:, 3:4]
            init_sa = initial_state[:, 0:1]
            init_gw = initial_state[:, 1:2]
            init_snow = initial_state[:, 2:3]
            init_melt = initial_state[:, 3:4]

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = {}
        if return_diagnostics is True:
            for name in [
                    'SNOWPACK_prev', 'MELTWATER_prev', 'Sa_prev', 'GW_prev',
                    'SNOWPACK', 'MELTWATER', 'Sa', 'GW',
                    'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release',
                    'PL', 'interception_evaporation', 'actual_ET', 'LAI_t', 'LAI_rel',
                    'LAI_interception_scalar', 'LAI_et_scalar', 'alpha_base', 'alpha_veg_scalar',
                    'Smoist', 'theta_cap', 'alpha', 'P_accessible', 'P_inaccessible', 'Sa_overflow',
                    'surface_runoff', 'interflow', 'recharge_to_groundwater',
                    'baseflow', 'Q_process', 'groundwater_loss', 'channel_loss', 'gate_loss',
                    'partition_sum_error', 'soil_local_residual', 'gw_local_residual',
                    'snowpack_local_residual', 'meltwater_local_residual', 'snow_total_local_residual',
                    'process_local_residual',
                    'INSC', 'INSC_t', 'COEF_t', 'SQ_t', 'K_t', 'K_mult_t', 'TT', 'CFMAX_t', 'CFR', 'CWH'
            ]:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []
        k_t_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SA0 = self._min(self._pos(SA), theta_cap)
            GW0 = self._pos(GW)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            Smoist0 = torch.clamp(SA0 / (theta_cap + 1e-8), 0.0, 1.0)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0, Tt / 20.0, E0t / 10.0,
                SA0 / 300.0, Smoist0, GW0 / 300.0, SNOWPACK0 / 300.0, sin_t, cos_t
            ], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)

            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX_base * m_cf_eff

            if self.dynamic_k is True and dyn_raw is not None:
                K_mult_t = 0.25 + 2.75 * torch.sigmoid(dyn_raw[:, self.dyn_slices['k']])
                K_t = torch.clamp(K * K_mult_t, 0.001, 0.7)
            else:
                K_mult_t = torch.ones_like(K)
                K_t = K

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()
            rainfall = Pt * frac_rain
            snowfall = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + snowfall
            snowmelt_pot = CFMAX_t * self._pos(Tt - TT)
            snowmelt = self._min(snowmelt_pot, SNOWPACK1)
            SNOWPACK2 = self._pos(SNOWPACK1 - snowmelt)
            MELTWATER1 = MELTWATER0 + snowmelt

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)
            SNOWPACK3 = SNOWPACK2 + refreezing

            snow_holding = CWH * SNOWPACK3
            snow_release_raw = self._pos(MELTWATER2 - snow_holding)
            snow_release = self._min(snow_release_raw, MELTWATER2)
            MELTWATER_next = self._pos(MELTWATER2 - snow_release)
            SNOWPACK_next = self._pos(SNOWPACK3)

            PL = rainfall + snow_release
            if inputs.shape[-1] >= 6:
                LAI_t = self._pos(inputs[:, t, 5:6])
                LAI_rel = torch.clamp(LAI_t / 6.0, 0.0, 1.0)
                LAI_interception_scalar = 0.15 + 0.85 * LAI_rel
                LAI_et_scalar = 0.20 + 0.80 * LAI_rel
                alpha_veg_scalar = 0.70 + 0.60 * LAI_rel
            else:
                LAI_t = torch.zeros_like(Pt)
                LAI_rel = torch.zeros_like(Pt)
                LAI_interception_scalar = torch.ones_like(Pt)
                LAI_et_scalar = torch.ones_like(Pt)
                alpha_veg_scalar = torch.ones_like(Pt)

            INSC_t = INSC * LAI_interception_scalar
            INT = self._min(INSC_t, self._min(E0t, PL))
            PL_after_int = self._pos(PL - INT)
            POT = self._pos(E0t - INT)

            alpha_base = theta_ab * torch.pow(torch.clamp(1.0 - Smoist0, min=0.0), theta_ak)
            alpha = torch.clamp(alpha_base * alpha_veg_scalar, 0.0, 1.0)
            P_accessible = alpha * PL_after_int
            P_inaccessible = (1.0 - alpha) * PL_after_int

            SA_pre = SA0 + P_accessible
            ET_stress = torch.clamp(Smoist0 / torch.clamp(theta_wetpoint, min=1e-6), 0.0, 1.0)
            ET_a_pot = POT * theta_efmax * ET_stress * LAI_et_scalar
            ET_a = self._min(ET_a_pot, SA_pre)
            SA_after_ET = self._pos(SA_pre - ET_a)
            SA_overflow = self._pos(SA_after_ET - theta_cap)
            SA_next = self._pos(SA_after_ET - SA_overflow)

            water_for_partition = self._pos(P_inaccessible + SA_overflow)
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_interflow = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([
                torch.log(base_surface),
                torch.log(base_interflow),
                torch.log(base_recharge)
            ], dim=1)
            if self.dynamic_partition is True:
                part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']]
            else:
                part_logits = base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]

            SRUN = f_surface * water_for_partition
            IFLOW = f_inter * water_for_partition
            REC = f_recharge * water_for_partition

            GW1 = GW0 + REC
            BAS_raw = K_t * GW1
            BAS = self._min(BAS_raw, GW1)
            GW_next = self._pos(GW1 - BAS)

            Q_process = self._pos(SRUN + IFLOW + BAS)
            process_local_residual = (
                Pt - INT - ET_a - Q_process
                - ((SNOWPACK_next - SNOWPACK0) + (MELTWATER_next - MELTWATER0) + (SA_next - SA0) + (GW_next - GW0))
            )
            soil_local_residual = P_accessible - ET_a - SA_overflow - (SA_next - SA0)
            gw_local_residual = REC - BAS - (GW_next - GW0)
            snowpack_local_residual = snowfall - snowmelt + refreezing - (SNOWPACK_next - SNOWPACK0)
            meltwater_local_residual = snowmelt - refreezing - snow_release - (MELTWATER_next - MELTWATER0)
            snow_total_local_residual = snowfall - snow_release - (SNOWPACK_next - SNOWPACK0) - (MELTWATER_next - MELTWATER0)

            SA = SA_next
            GW = GW_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q_process)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)
            k_t_hist.append(K_t)

            if return_diagnostics is True:
                diag_vals = {
                    'SNOWPACK_prev': SNOWPACK0,
                    'MELTWATER_prev': MELTWATER0,
                    'Sa_prev': SA0,
                    'GW_prev': GW0,
                    'SNOWPACK': SNOWPACK_next,
                    'MELTWATER': MELTWATER_next,
                    'Sa': SA_next,
                    'GW': GW_next,
                    'precipitation': Pt,
                    'rainfall': rainfall,
                    'snowfall': snowfall,
                    'snowmelt': snowmelt,
                    'refreezing': refreezing,
                    'snow_release': snow_release,
                    'PL': PL,
                    'interception_evaporation': INT,
                    'actual_ET': ET_a,
                    'LAI_t': LAI_t,
                    'LAI_rel': LAI_rel,
                    'LAI_interception_scalar': LAI_interception_scalar,
                    'LAI_et_scalar': LAI_et_scalar,
                    'alpha_base': alpha_base,
                    'alpha_veg_scalar': alpha_veg_scalar,
                    'Smoist': Smoist0,
                    'theta_cap': theta_cap,
                    'alpha': alpha,
                    'P_accessible': P_accessible,
                    'P_inaccessible': P_inaccessible,
                    'Sa_overflow': SA_overflow,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'recharge_to_groundwater': REC,
                    'baseflow': BAS,
                    'Q_process': Q_process,
                    'groundwater_loss': torch.zeros_like(Q_process),
                    'channel_loss': torch.zeros_like(Q_process),
                    'gate_loss': torch.zeros_like(Q_process),
                    'partition_sum_error': torch.abs(f_surface + f_inter + f_recharge - 1.0),
                    'soil_local_residual': soil_local_residual,
                    'gw_local_residual': gw_local_residual,
                    'snowpack_local_residual': snowpack_local_residual,
                    'meltwater_local_residual': meltwater_local_residual,
                    'snow_total_local_residual': snow_total_local_residual,
                    'process_local_residual': process_local_residual,
                    'INSC': INSC,
                    'INSC_t': INSC_t,
                    'COEF_t': COEF,
                    'SQ_t': SQ_t,
                    'K_t': K_t,
                    'K_mult_t': K_mult_t,
                    'TT': TT,
                    'CFMAX_t': CFMAX_t,
                    'CFR': CFR,
                    'CWH': CWH,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SA, GW, SNOWPACK, MELTWATER], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        reg_k_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)
        if len(k_t_hist) > 1:
            k_seq = torch.stack(k_t_hist, dim=1)
            reg_k_smooth = torch.mean((k_seq[:, 1:, :] - k_seq[:, :-1, :]) ** 2)
        sum_p = torch.clamp(P.sum(dim=1), min=1e-6)
        drift_loss = (
            torch.mean(torch.abs(SA - init_sa) / sum_p)
            + torch.mean(torch.abs(GW - init_gw) / sum_p)
            + torch.mean(torch.abs(SNOWPACK - init_snow) / sum_p)
            + torch.mean(torch.abs(MELTWATER - init_melt) / sum_p)
        )
        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': reg_part,
            'storage_drift_loss': drift_loss,
            'k_smooth_loss': reg_k_smooth,
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out
        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleDynamicKLAIEco(
        MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleDynamicK):
    """
    Copied dynamic-K closed Model 6 variant with daily LAI used explicitly in
    interception, ET, and active-root-zone accessibility.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 reg_k_smooth_w=1e-4, component_routing=True, dynamic_k=True):
        super(MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleDynamicKLAIEco, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, reg_k_smooth_w=reg_k_smooth_w,
            component_routing=component_routing, dynamic_k=dynamic_k)
        self.simhyd = DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicKLAIEco(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            dynamic_k=dynamic_k)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicKLAIEco(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            dynamic_k=dynamic_k)


class DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicKStaticBetaGW(
        DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicK):
    """
    Closed simple dynamic-K copy with one additional static groundwater
    recession nonlinearity parameter beta_gw per component.
    """

    def _expand_simple(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)
        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = theta[:, 2:3] * 6.0
        SMSC_legacy = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = theta[:, 4:5]
        CRAK = theta[:, 5:6]
        K = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        TT = -2.5 + theta[:, 8:9] * 5.0
        CFMAX = 0.5 + theta[:, 9:10] * (10.0 - 0.5)
        CFR = theta[:, 10:11] * 0.1
        CWH = theta[:, 11:12] * 0.2
        theta_ab = 0.5 + theta[:, 13:14] * 0.5
        theta_ak = 1.0 + theta[:, 14:15] * 9.0
        theta_cap = 10.0 + theta[:, 15:16] * (1500.0 - 10.0)
        theta_efmax = 0.5 + theta[:, 16:17] * 0.5
        theta_wetpoint = 0.3 + theta[:, 17:18] * 0.6
        beta_gw = 1.0 + 2.0 * theta[:, 18:19]
        return (
            INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K, TT, CFMAX, CFR, CWH,
            theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint, beta_gw
        )

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        (INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K, TT, CFMAX_base, CFR, CWH,
         theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint, beta_gw) = self._expand_simple(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SA = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
            init_sa = SA
            init_gw = GW
            init_snow = SNOWPACK
            init_melt = MELTWATER
        else:
            SA = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]
            SNOWPACK = initial_state[:, 2:3]
            MELTWATER = initial_state[:, 3:4]
            init_sa = initial_state[:, 0:1]
            init_gw = initial_state[:, 1:2]
            init_snow = initial_state[:, 2:3]
            init_melt = initial_state[:, 3:4]

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = {}
        if return_diagnostics is True:
            for name in [
                    'SNOWPACK_prev', 'MELTWATER_prev', 'Sa_prev', 'GW_prev',
                    'SNOWPACK', 'MELTWATER', 'Sa', 'GW',
                    'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release',
                    'PL', 'interception_evaporation', 'actual_ET', 'LAI_t', 'LAI_rel',
                    'LAI_interception_scalar', 'LAI_et_scalar',
                    'Smoist', 'theta_cap', 'alpha', 'P_accessible', 'P_inaccessible', 'Sa_overflow',
                    'surface_runoff', 'interflow', 'recharge_to_groundwater',
                    'baseflow', 'Q_process', 'groundwater_loss', 'channel_loss', 'gate_loss',
                    'partition_sum_error', 'soil_local_residual', 'gw_local_residual',
                    'snowpack_local_residual', 'meltwater_local_residual', 'snow_total_local_residual',
                    'process_local_residual',
                    'INSC', 'INSC_t', 'COEF_t', 'SQ_t', 'K_t', 'K_mult_t', 'GW_norm', 'beta_gw', 'TT', 'CFMAX_t', 'CFR', 'CWH'
            ]:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []
        k_t_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SA0 = self._min(self._pos(SA), theta_cap)
            GW0 = self._pos(GW)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            Smoist0 = torch.clamp(SA0 / (theta_cap + 1e-8), 0.0, 1.0)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0, Tt / 20.0, E0t / 10.0,
                SA0 / 300.0, Smoist0, GW0 / 300.0, SNOWPACK0 / 300.0, sin_t, cos_t
            ], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)

            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX_base * m_cf_eff

            if self.dynamic_k is True and dyn_raw is not None:
                K_mult_t = 0.25 + 2.75 * torch.sigmoid(dyn_raw[:, self.dyn_slices['k']])
                K_t = torch.clamp(K * K_mult_t, 0.001, 0.7)
            else:
                K_mult_t = torch.ones_like(K)
                K_t = K

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()
            rainfall = Pt * frac_rain
            snowfall = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + snowfall
            snowmelt_pot = CFMAX_t * self._pos(Tt - TT)
            snowmelt = self._min(snowmelt_pot, SNOWPACK1)
            SNOWPACK2 = self._pos(SNOWPACK1 - snowmelt)
            MELTWATER1 = MELTWATER0 + snowmelt

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)
            SNOWPACK3 = SNOWPACK2 + refreezing

            snow_holding = CWH * SNOWPACK3
            snow_release_raw = self._pos(MELTWATER2 - snow_holding)
            snow_release = self._min(snow_release_raw, MELTWATER2)
            MELTWATER_next = self._pos(MELTWATER2 - snow_release)
            SNOWPACK_next = self._pos(SNOWPACK3)

            PL = rainfall + snow_release
            if inputs.shape[-1] >= 6:
                LAI_t = self._pos(inputs[:, t, 5:6])
                LAI_rel = torch.clamp(LAI_t / 6.0, 0.0, 1.0)
                LAI_anom = LAI_rel - 0.5
                LAI_interception_scalar = torch.clamp(1.0 + 0.25 * LAI_anom, 0.90, 1.10)
                LAI_et_scalar = torch.clamp(1.15 + 0.50 * LAI_anom, 0.95, 1.35)
            else:
                LAI_t = torch.zeros_like(Pt)
                LAI_rel = torch.zeros_like(Pt)
                LAI_interception_scalar = torch.ones_like(Pt)
                LAI_et_scalar = torch.ones_like(Pt)

            INSC_t = INSC * LAI_interception_scalar
            INT = self._min(INSC_t, self._min(E0t, PL))
            PL_after_int = self._pos(PL - INT)
            POT = self._pos(E0t - INT)

            alpha = theta_ab * torch.pow(torch.clamp(1.0 - Smoist0, min=0.0), theta_ak)
            alpha = torch.clamp(alpha, 0.0, 1.0)
            P_accessible = alpha * PL_after_int
            P_inaccessible = (1.0 - alpha) * PL_after_int

            SA_pre = SA0 + P_accessible
            ET_stress = torch.clamp(Smoist0 / torch.clamp(theta_wetpoint, min=1e-6), 0.0, 1.0)
            ET_a_pot = POT * theta_efmax * ET_stress * LAI_et_scalar
            ET_a = self._min(ET_a_pot, SA_pre)
            SA_after_ET = self._pos(SA_pre - ET_a)
            SA_overflow = self._pos(SA_after_ET - theta_cap)
            SA_next = self._pos(SA_after_ET - SA_overflow)

            water_for_partition = self._pos(P_inaccessible + SA_overflow)
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_interflow = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([
                torch.log(base_surface),
                torch.log(base_interflow),
                torch.log(base_recharge)
            ], dim=1)
            if self.dynamic_partition is True:
                part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']]
            else:
                part_logits = base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]

            SRUN = f_surface * water_for_partition
            IFLOW = f_inter * water_for_partition
            REC = f_recharge * water_for_partition

            GW1 = GW0 + REC
            GW_ref = theta_cap
            GW_norm = torch.clamp(GW1 / (GW_ref + 1e-6), 1e-4, 50.0)
            BAS_raw = K_t * GW1 * torch.pow(GW_norm, beta_gw - 1.0)
            BAS = self._min(BAS_raw, GW1)
            GW_next = self._pos(GW1 - BAS)

            Q_process = self._pos(SRUN + IFLOW + BAS)
            process_local_residual = (
                Pt - INT - ET_a - Q_process
                - ((SNOWPACK_next - SNOWPACK0) + (MELTWATER_next - MELTWATER0) + (SA_next - SA0) + (GW_next - GW0))
            )
            soil_local_residual = P_accessible - ET_a - SA_overflow - (SA_next - SA0)
            gw_local_residual = REC - BAS - (GW_next - GW0)
            snowpack_local_residual = snowfall - snowmelt + refreezing - (SNOWPACK_next - SNOWPACK0)
            meltwater_local_residual = snowmelt - refreezing - snow_release - (MELTWATER_next - MELTWATER0)
            snow_total_local_residual = snowfall - snow_release - (SNOWPACK_next - SNOWPACK0) - (MELTWATER_next - MELTWATER0)

            SA = SA_next
            GW = GW_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q_process)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)
            k_t_hist.append(K_t)

            if return_diagnostics is True:
                diag_vals = {
                    'SNOWPACK_prev': SNOWPACK0,
                    'MELTWATER_prev': MELTWATER0,
                    'Sa_prev': SA0,
                    'GW_prev': GW0,
                    'SNOWPACK': SNOWPACK_next,
                    'MELTWATER': MELTWATER_next,
                    'Sa': SA_next,
                    'GW': GW_next,
                    'precipitation': Pt,
                    'rainfall': rainfall,
                    'snowfall': snowfall,
                    'snowmelt': snowmelt,
                    'refreezing': refreezing,
                    'snow_release': snow_release,
                    'PL': PL,
                    'interception_evaporation': INT,
                    'actual_ET': ET_a,
                    'LAI_t': LAI_t,
                    'LAI_rel': LAI_rel,
                    'LAI_interception_scalar': LAI_interception_scalar,
                    'LAI_et_scalar': LAI_et_scalar,
                    'Smoist': Smoist0,
                    'theta_cap': theta_cap,
                    'alpha': alpha,
                    'P_accessible': P_accessible,
                    'P_inaccessible': P_inaccessible,
                    'Sa_overflow': SA_overflow,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'recharge_to_groundwater': REC,
                    'baseflow': BAS,
                    'Q_process': Q_process,
                    'groundwater_loss': torch.zeros_like(Q_process),
                    'channel_loss': torch.zeros_like(Q_process),
                    'gate_loss': torch.zeros_like(Q_process),
                    'partition_sum_error': torch.abs(f_surface + f_inter + f_recharge - 1.0),
                    'soil_local_residual': soil_local_residual,
                    'gw_local_residual': gw_local_residual,
                    'snowpack_local_residual': snowpack_local_residual,
                    'meltwater_local_residual': meltwater_local_residual,
                    'snow_total_local_residual': snow_total_local_residual,
                    'process_local_residual': process_local_residual,
                    'INSC': INSC,
                    'INSC_t': INSC_t,
                    'COEF_t': COEF,
                    'SQ_t': SQ_t,
                    'K_t': K_t,
                    'K_mult_t': K_mult_t,
                    'GW_norm': GW_norm,
                    'beta_gw': beta_gw,
                    'TT': TT,
                    'CFMAX_t': CFMAX_t,
                    'CFR': CFR,
                    'CWH': CWH,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SA, GW, SNOWPACK, MELTWATER], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        reg_k_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_beta = torch.mean((beta_gw - 1.0) ** 2)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)
        if len(k_t_hist) > 1:
            k_seq = torch.stack(k_t_hist, dim=1)
            reg_k_smooth = torch.mean((k_seq[:, 1:, :] - k_seq[:, :-1, :]) ** 2)
        sum_p = torch.clamp(P.sum(dim=1), min=1e-6)
        drift_loss = (
            torch.mean(torch.abs(SA - init_sa) / sum_p)
            + torch.mean(torch.abs(GW - init_gw) / sum_p)
            + torch.mean(torch.abs(SNOWPACK - init_snow) / sum_p)
            + torch.mean(torch.abs(MELTWATER - init_melt) / sum_p)
        )
        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': reg_part,
            'storage_drift_loss': drift_loss,
            'k_smooth_loss': reg_k_smooth,
            'beta_reg_loss': reg_beta,
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out
        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleDynamicKStaticBetaGW(
        MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple):
    """
    Copied closed simple Model 6 variant with dynamic K_t and one additional
    static groundwater recession nonlinearity parameter beta_gw.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 reg_k_smooth_w=1e-4, reg_beta_w=1e-5, component_routing=True, dynamic_k=True):
        super(MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleDynamicKStaticBetaGW, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing)
        self.reg_k_smooth_w = reg_k_smooth_w
        self.reg_beta_w = reg_beta_w
        self.nfea = 19
        self.nstaticpm = self.nfea * nmul
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.simhyd = DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicKStaticBetaGW(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            dynamic_k=dynamic_k)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicKStaticBetaGW(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            dynamic_k=dynamic_k)

    def _theta_to_smsc(self, theta, ngage):
        theta4 = theta.view(ngage, self.nmul, self.nfea)
        return 10.0 + theta4[:, :, 15:16] * (1500.0 - 10.0)

    def get_auxiliary_loss(self):
        aux = super(MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleDynamicKStaticBetaGW, self).get_auxiliary_loss()
        if aux is None:
            return None
        if self._last_aux_terms is None:
            return aux
        return (
            aux
            + self.reg_k_smooth_w * self._last_aux_terms.get('k_smooth_loss', 0.0)
            + self.reg_beta_w * self._last_aux_terms.get('beta_reg_loss', 0.0)
        )


class DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicKPowerGW(
        DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicK):
    """
    Closed simple dynamic-K copy with one additional static groundwater
    nonlinearity parameter beta_gw and a power-law dynamic groundwater
    recession:

        GW1 = GW + REC
        K_t = clamp(K * m_k, 0.001, 0.5)
        GW_scaled = max(GW1, eps) / 300.0
        BAS_raw = K_t * 300.0 * (GW_scaled ** beta_gw)
        BAS = min(BAS_raw, GW1)
        GW_next = GW1 - BAS
    """

    def _expand_simple(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)
        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = theta[:, 2:3] * 6.0
        SMSC_legacy = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = theta[:, 4:5]
        CRAK = theta[:, 5:6]
        K = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        TT = -2.5 + theta[:, 8:9] * 5.0
        CFMAX = 0.5 + theta[:, 9:10] * (10.0 - 0.5)
        CFR = theta[:, 10:11] * 0.1
        CWH = theta[:, 11:12] * 0.2
        theta_ab = 0.5 + theta[:, 13:14] * 0.5
        theta_ak = 1.0 + theta[:, 14:15] * 9.0
        theta_cap = 10.0 + theta[:, 15:16] * (1500.0 - 10.0)
        theta_efmax = 0.5 + theta[:, 16:17] * 0.5
        theta_wetpoint = 0.3 + theta[:, 17:18] * 0.6
        beta_gw = self._beta_from_theta(theta[:, 18:19])
        return (
            INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K, TT, CFMAX, CFR, CWH,
            theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint, beta_gw
        )

    def _beta_from_theta(self, theta_beta):
        return 0.5 + 2.5 * theta_beta

    def _compute_k_terms(self, K, dyn_raw):
        if self.dynamic_k is True and dyn_raw is not None:
            m_k = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['k']])
            K_t = torch.clamp(K * m_k, 0.001, 0.5)
        else:
            m_k = torch.ones_like(K)
            K_t = torch.clamp(K, 0.001, 0.5)
        return m_k, K_t

    def _compute_groundwater_recession(self, GW1, K_t, beta_gw):
        GW_scaled = torch.clamp(GW1 / 300.0, min=1e-6)
        BAS_raw = K_t * 300.0 * torch.pow(GW_scaled, beta_gw)
        return BAS_raw, GW_scaled

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        (INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K, TT, CFMAX_base, CFR, CWH,
         theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint, beta_gw) = self._expand_simple(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SA = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
            init_sa = SA
            init_gw = GW
            init_snow = SNOWPACK
            init_melt = MELTWATER
        else:
            SA = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]
            SNOWPACK = initial_state[:, 2:3]
            MELTWATER = initial_state[:, 3:4]
            init_sa = initial_state[:, 0:1]
            init_gw = initial_state[:, 1:2]
            init_snow = initial_state[:, 2:3]
            init_melt = initial_state[:, 3:4]

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = {}
        if return_diagnostics is True:
            for name in [
                    'SNOWPACK_prev', 'MELTWATER_prev', 'Sa_prev', 'GW_prev',
                    'SNOWPACK', 'MELTWATER', 'Sa', 'GW',
                    'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release',
                    'PL', 'interception_evaporation', 'actual_ET',
                    'Smoist', 'theta_cap', 'alpha', 'P_accessible', 'P_inaccessible', 'Sa_overflow',
                    'surface_runoff', 'interflow', 'recharge_to_groundwater',
                    'baseflow', 'Q_process', 'groundwater_loss', 'channel_loss', 'gate_loss',
                    'partition_sum_error', 'soil_local_residual', 'gw_local_residual',
                    'snowpack_local_residual', 'meltwater_local_residual', 'snow_total_local_residual',
                    'process_local_residual',
                    'INSC', 'COEF_t', 'SQ_t', 'K_t', 'm_k', 'beta_gw', 'GW_scaled', 'TT', 'CFMAX_t', 'CFR', 'CWH'
            ]:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []
        m_k_hist = []
        k_t_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SA0 = self._min(self._pos(SA), theta_cap)
            GW0 = self._pos(GW)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            Smoist0 = torch.clamp(SA0 / (theta_cap + 1e-8), 0.0, 1.0)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0, Tt / 20.0, E0t / 10.0,
                SA0 / 300.0, Smoist0, GW0 / 300.0, SNOWPACK0 / 300.0, sin_t, cos_t
            ], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)

            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX_base * m_cf_eff

            m_k, K_t = self._compute_k_terms(K, dyn_raw)

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()
            rainfall = Pt * frac_rain
            snowfall = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + snowfall
            snowmelt_pot = CFMAX_t * self._pos(Tt - TT)
            snowmelt = self._min(snowmelt_pot, SNOWPACK1)
            SNOWPACK2 = self._pos(SNOWPACK1 - snowmelt)
            MELTWATER1 = MELTWATER0 + snowmelt

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)
            SNOWPACK3 = SNOWPACK2 + refreezing

            snow_holding = CWH * SNOWPACK3
            snow_release_raw = self._pos(MELTWATER2 - snow_holding)
            snow_release = self._min(snow_release_raw, MELTWATER2)
            MELTWATER_next = self._pos(MELTWATER2 - snow_release)
            SNOWPACK_next = self._pos(SNOWPACK3)

            PL = rainfall + snow_release
            INT = self._min(INSC, self._min(E0t, PL))
            PL_after_int = self._pos(PL - INT)
            POT = self._pos(E0t - INT)

            alpha = theta_ab * torch.pow(torch.clamp(1.0 - Smoist0, min=0.0), theta_ak)
            alpha = torch.clamp(alpha, 0.0, 1.0)
            P_accessible = alpha * PL_after_int
            P_inaccessible = (1.0 - alpha) * PL_after_int

            SA_pre = SA0 + P_accessible
            ET_stress = torch.clamp(Smoist0 / torch.clamp(theta_wetpoint, min=1e-6), 0.0, 1.0)
            ET_a_pot = POT * theta_efmax * ET_stress
            ET_a = self._min(ET_a_pot, SA_pre)
            SA_after_ET = self._pos(SA_pre - ET_a)
            SA_overflow = self._pos(SA_after_ET - theta_cap)
            SA_next = self._pos(SA_after_ET - SA_overflow)

            water_for_partition = self._pos(P_inaccessible + SA_overflow)
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_interflow = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([
                torch.log(base_surface),
                torch.log(base_interflow),
                torch.log(base_recharge)
            ], dim=1)
            if self.dynamic_partition is True:
                part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']]
            else:
                part_logits = base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]

            SRUN = f_surface * water_for_partition
            IFLOW = f_inter * water_for_partition
            REC = f_recharge * water_for_partition

            GW1 = GW0 + REC
            BAS_raw, GW_scaled = self._compute_groundwater_recession(GW1, K_t, beta_gw)
            if torch.isnan(BAS_raw).any() or torch.isnan(K_t).any() or torch.isnan(beta_gw).any():
                raise RuntimeError("NaN detected in DynamicKPowerGW groundwater block")
            BAS = torch.minimum(BAS_raw, GW1)
            if not torch.all(BAS <= GW1 + 1e-6):
                raise RuntimeError("DynamicKPowerGW violated BAS <= GW1")
            GW_next = GW1 - BAS
            if not torch.all(GW_next >= -1e-6):
                raise RuntimeError("DynamicKPowerGW violated GW_next >= -1e-6")
            GW_next = self._pos(GW_next)

            Q_process = self._pos(SRUN + IFLOW + BAS)
            process_local_residual = (
                Pt - INT - ET_a - Q_process
                - ((SNOWPACK_next - SNOWPACK0) + (MELTWATER_next - MELTWATER0) + (SA_next - SA0) + (GW_next - GW0))
            )
            soil_local_residual = P_accessible - ET_a - SA_overflow - (SA_next - SA0)
            gw_local_residual = REC - BAS - (GW_next - GW0)
            snowpack_local_residual = snowfall - snowmelt + refreezing - (SNOWPACK_next - SNOWPACK0)
            meltwater_local_residual = snowmelt - refreezing - snow_release - (MELTWATER_next - MELTWATER0)
            snow_total_local_residual = snowfall - snow_release - (SNOWPACK_next - SNOWPACK0) - (MELTWATER_next - MELTWATER0)

            SA = SA_next
            GW = GW_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q_process)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)
            m_k_hist.append(m_k)
            k_t_hist.append(K_t)

            if return_diagnostics is True:
                diag_vals = {
                    'SNOWPACK_prev': SNOWPACK0,
                    'MELTWATER_prev': MELTWATER0,
                    'Sa_prev': SA0,
                    'GW_prev': GW0,
                    'SNOWPACK': SNOWPACK_next,
                    'MELTWATER': MELTWATER_next,
                    'Sa': SA_next,
                    'GW': GW_next,
                    'precipitation': Pt,
                    'rainfall': rainfall,
                    'snowfall': snowfall,
                    'snowmelt': snowmelt,
                    'refreezing': refreezing,
                    'snow_release': snow_release,
                    'PL': PL,
                    'interception_evaporation': INT,
                    'actual_ET': ET_a,
                    'Smoist': Smoist0,
                    'theta_cap': theta_cap,
                    'alpha': alpha,
                    'P_accessible': P_accessible,
                    'P_inaccessible': P_inaccessible,
                    'Sa_overflow': SA_overflow,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'recharge_to_groundwater': REC,
                    'baseflow': BAS,
                    'Q_process': Q_process,
                    'groundwater_loss': torch.zeros_like(Q_process),
                    'channel_loss': torch.zeros_like(Q_process),
                    'gate_loss': torch.zeros_like(Q_process),
                    'partition_sum_error': torch.abs(f_surface + f_inter + f_recharge - 1.0),
                    'soil_local_residual': soil_local_residual,
                    'gw_local_residual': gw_local_residual,
                    'snowpack_local_residual': snowpack_local_residual,
                    'meltwater_local_residual': meltwater_local_residual,
                    'snow_total_local_residual': snow_total_local_residual,
                    'process_local_residual': process_local_residual,
                    'INSC': INSC,
                    'COEF_t': COEF,
                    'SQ_t': SQ_t,
                    'K_t': K_t,
                    'm_k': m_k,
                    'beta_gw': beta_gw,
                    'GW_scaled': GW_scaled,
                    'TT': TT,
                    'CFMAX_t': CFMAX_t,
                    'CFR': CFR,
                    'CWH': CWH,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SA, GW, SNOWPACK, MELTWATER], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        reg_dk_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_dk_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_beta_center = torch.mean((beta_gw - 1.0) ** 2)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)
        if len(m_k_hist) > 0:
            m_k_seq = torch.stack(m_k_hist, dim=1)
            reg_dk_amp = torch.mean((m_k_seq - 1.0) ** 2)
        if len(m_k_hist) > 1:
            m_k_seq = torch.stack(m_k_hist, dim=1)
            reg_dk_smooth = torch.mean((m_k_seq[:, 1:, :] - m_k_seq[:, :-1, :]) ** 2)
        sum_p = torch.clamp(P.sum(dim=1), min=1e-6)
        s_start = init_sa + init_gw + init_snow + init_melt
        s_end = SA + GW + SNOWPACK + MELTWATER
        drift_loss = torch.mean(((s_end - s_start) / sum_p) ** 2)
        gw_drift_loss = torch.mean(((GW - init_gw) / sum_p) ** 2)
        sa_drift_loss = torch.mean(((SA - init_sa) / sum_p) ** 2)
        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': reg_part,
            'storage_drift_loss': drift_loss,
            'gw_drift_loss': gw_drift_loss,
            'sa_drift_loss': sa_drift_loss,
            'dynamic_k_amplitude_loss': reg_dk_amp,
            'dynamic_k_smoothness_loss': reg_dk_smooth,
            'beta_center_loss': reg_beta_center,
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out
        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleDynamicKPowerGW(
        MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple):
    """
    Copied closed simple Model 6 variant with dynamic K_t, static beta_gw,
    and nonlinear power-law groundwater recession.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 reg_dynamic_k_amp_w=1e-4, reg_dynamic_k_smooth_w=1e-4,
                 reg_beta_w=1e-5, component_routing=True, dynamic_k=True):
        super(MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleDynamicKPowerGW, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing)
        self.reg_dynamic_k_amp_w = reg_dynamic_k_amp_w
        self.reg_dynamic_k_smooth_w = reg_dynamic_k_smooth_w
        self.reg_beta_w = reg_beta_w
        self.nfea = 19
        self.nstaticpm = self.nfea * nmul
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.simhyd = DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicKPowerGW(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            dynamic_k=dynamic_k)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicKPowerGW(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            dynamic_k=dynamic_k)

    def _theta_to_smsc(self, theta, ngage):
        theta4 = theta.view(ngage, self.nmul, self.nfea)
        return 10.0 + theta4[:, :, 15:16] * (1500.0 - 10.0)

    def get_auxiliary_loss(self):
        aux = super(MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleDynamicKPowerGW, self).get_auxiliary_loss()
        if aux is None:
            return None
        if self._last_aux_terms is None:
            return aux
        return (
            aux
            + self.reg_dynamic_k_amp_w * self._last_aux_terms.get('dynamic_k_amplitude_loss', 0.0)
            + self.reg_dynamic_k_smooth_w * self._last_aux_terms.get('dynamic_k_smoothness_loss', 0.0)
            + self.reg_beta_w * self._last_aux_terms.get('beta_center_loss', 0.0)
        )


class DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicKPowerGWConstrained(
        DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicKPowerGW):
    """
    Constrained power-law groundwater recession version with narrower
    beta_gw and m_k ranges to reduce storage drift.
    """

    def _beta_from_theta(self, theta_beta):
        return 1.0 + 0.5 * theta_beta

    def _compute_k_terms(self, K, dyn_raw):
        if self.dynamic_k is True and dyn_raw is not None:
            m_k = 0.75 + 0.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['k']])
            K_t = torch.clamp(K * m_k, 0.003, 0.3)
        else:
            m_k = torch.ones_like(K)
            K_t = torch.clamp(K, 0.003, 0.3)
        return m_k, K_t

    def _compute_groundwater_recession(self, GW1, K_t, beta_gw):
        GW_scaled = torch.clamp(GW1 / 300.0, min=1e-6)
        log_scaled = torch.log(GW_scaled)
        curv = torch.exp((beta_gw - 1.0) * torch.tanh(log_scaled))
        BAS_raw = K_t * GW1 * curv
        return BAS_raw, curv


class MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleDynamicKPowerGWConstrained(
        MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple):
    """
    Constrained closed simple Model 6 variant with dynamic K_t, static beta_gw,
    nonlinear power-law groundwater recession, and stronger drift regularization.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 reg_storage_drift_w=1e-2, reg_gw_drift_w=5e-3, reg_sa_drift_w=5e-3,
                 reg_dynamic_k_amp_w=1e-4, reg_dynamic_k_smooth_w=1e-4,
                 reg_beta_w=1e-4, component_routing=True, dynamic_k=True):
        super(MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleDynamicKPowerGWConstrained, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing)
        self.reg_storage_drift_w = reg_storage_drift_w
        self.reg_gw_drift_w = reg_gw_drift_w
        self.reg_sa_drift_w = reg_sa_drift_w
        self.reg_dynamic_k_amp_w = reg_dynamic_k_amp_w
        self.reg_dynamic_k_smooth_w = reg_dynamic_k_smooth_w
        self.reg_beta_w = reg_beta_w
        self.nfea = 19
        self.nstaticpm = self.nfea * nmul
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.simhyd = DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicKPowerGWConstrained(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            dynamic_k=dynamic_k)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicKPowerGWConstrained(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            dynamic_k=dynamic_k)

    def _theta_to_smsc(self, theta, ngage):
        theta4 = theta.view(ngage, self.nmul, self.nfea)
        return 10.0 + theta4[:, :, 15:16] * (1500.0 - 10.0)

    def get_auxiliary_loss(self):
        aux = super(MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleDynamicKPowerGWConstrained, self).get_auxiliary_loss()
        if aux is None:
            return None
        if self._last_aux_terms is None:
            return aux
        return (
            aux
            + self.reg_storage_drift_w * self._last_aux_terms.get('storage_drift_loss', 0.0)
            + self.reg_gw_drift_w * self._last_aux_terms.get('gw_drift_loss', 0.0)
            + self.reg_sa_drift_w * self._last_aux_terms.get('sa_drift_loss', 0.0)
            + self.reg_dynamic_k_amp_w * self._last_aux_terms.get('dynamic_k_amplitude_loss', 0.0)
            + self.reg_dynamic_k_smooth_w * self._last_aux_terms.get('dynamic_k_smoothness_loss', 0.0)
            + self.reg_beta_w * self._last_aux_terms.get('beta_center_loss', 0.0)
        )


class DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicKPowerGWConstrainedLegacy(
        DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicKPowerGW):
    """
    Legacy constrained power-law groundwater recession version used before the
    bounded-beta curvature formulation.
    """

    def _beta_from_theta(self, theta_beta):
        return 0.75 + 1.0 * theta_beta

    def _compute_k_terms(self, K, dyn_raw):
        if self.dynamic_k is True and dyn_raw is not None:
            m_k = 0.75 + 0.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['k']])
            K_t = torch.clamp(K * m_k, 0.003, 0.3)
        else:
            m_k = torch.ones_like(K)
            K_t = torch.clamp(K, 0.003, 0.3)
        return m_k, K_t


class MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleDynamicKPowerGWConstrainedLegacy(
        MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple):
    """
    Legacy constrained closed simple Model 6 variant with dynamic K_t and the
    earlier constrained power-law groundwater recession.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 reg_storage_drift_w=1e-2, reg_gw_drift_w=5e-3, reg_sa_drift_w=5e-3,
                 reg_dynamic_k_amp_w=1e-4, reg_dynamic_k_smooth_w=1e-4,
                 reg_beta_w=1e-4, component_routing=True, dynamic_k=True):
        super(MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleDynamicKPowerGWConstrainedLegacy, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing)
        self.reg_storage_drift_w = reg_storage_drift_w
        self.reg_gw_drift_w = reg_gw_drift_w
        self.reg_sa_drift_w = reg_sa_drift_w
        self.reg_dynamic_k_amp_w = reg_dynamic_k_amp_w
        self.reg_dynamic_k_smooth_w = reg_dynamic_k_smooth_w
        self.reg_beta_w = reg_beta_w
        self.nfea = 19
        self.nstaticpm = self.nfea * nmul
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.simhyd = DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicKPowerGWConstrainedLegacy(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            dynamic_k=dynamic_k)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleDynamicKPowerGWConstrainedLegacy(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            dynamic_k=dynamic_k)

    def _theta_to_smsc(self, theta, ngage):
        theta4 = theta.view(ngage, self.nmul, self.nfea)
        return 10.0 + theta4[:, :, 15:16] * (1500.0 - 10.0)

    def get_auxiliary_loss(self):
        aux = super(MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleDynamicKPowerGWConstrainedLegacy, self).get_auxiliary_loss()
        if aux is None:
            return None
        if self._last_aux_terms is None:
            return aux
        return (
            aux
            + self.reg_storage_drift_w * self._last_aux_terms.get('storage_drift_loss', 0.0)
            + self.reg_gw_drift_w * self._last_aux_terms.get('gw_drift_loss', 0.0)
            + self.reg_sa_drift_w * self._last_aux_terms.get('sa_drift_loss', 0.0)
            + self.reg_dynamic_k_amp_w * self._last_aux_terms.get('dynamic_k_amplitude_loss', 0.0)
            + self.reg_dynamic_k_smooth_w * self._last_aux_terms.get('dynamic_k_smoothness_loss', 0.0)
            + self.reg_beta_w * self._last_aux_terms.get('beta_center_loss', 0.0)
        )


class DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDNeuralResponseSpectrum(
        DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimple):
    """
    Closed simple Model 6 copy where the SIMHYD/GW response block is replaced
    by a fixed-timescale neural response spectrum. Snow, interception, aSrz,
    and ET_a remain unchanged.
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True, dynamic_cfmax_snow=True,
                 dynamic_routing_scale=False, dynamic_all=False, dyn_hidden=32,
                 response_hidden=32, response_variant='A', response_tau_days=None):
        super(DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDNeuralResponseSpectrum, self).__init__(
            mode=mode, theta_is_raw=theta_is_raw, smooth=smooth, eps=eps, rain_snow_gain=rain_snow_gain,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, dyn_hidden=dyn_hidden)
        tau_days = response_tau_days
        if tau_days is None:
            tau_days = [1.0, 3.0, 7.0, 15.0, 30.0, 60.0, 120.0, 240.0, 480.0]
        tau_tensor = torch.tensor(tau_days, dtype=torch.float32)
        k_tensor = 1.0 - torch.exp(-1.0 / tau_tensor)
        self.register_buffer('response_tau_days', tau_tensor)
        self.register_buffer('response_k', k_tensor)
        self.n_response = int(tau_tensor.numel())
        self.response_variant = str(response_variant).upper()
        if self.response_variant not in ('A', 'B'):
            raise ValueError("response_variant must be 'A' or 'B'")
        self.responseHead = nn.Sequential(
            nn.Linear(11, response_hidden),
            nn.ReLU(),
            nn.Linear(response_hidden, response_hidden),
            nn.ReLU(),
            nn.Linear(response_hidden, self.n_response))

    def _response_features(self, Pt, Tt, E0t, PL_after_int, W_response, Smoist0,
                           SNOWPACK0, MELTWATER0, RESP_total0, sin_t, cos_t):
        return torch.cat([
            Pt / 20.0,
            Tt / 20.0,
            E0t / 10.0,
            PL_after_int / 20.0,
            W_response / 20.0,
            Smoist0,
            SNOWPACK0 / 300.0,
            MELTWATER0 / 300.0,
            RESP_total0 / 300.0,
            sin_t,
            cos_t,
        ], dim=1)

    def _response_step(self, RESP0, W_response, features):
        logits = self.responseHead(features)
        weights = F.softmax(logits, dim=1)
        inputs_i = weights * W_response
        k_i = self.response_k.to(device=RESP0.device, dtype=RESP0.dtype).view(1, self.n_response)
        q_raw_i = k_i * RESP0
        q_i = torch.minimum(q_raw_i, RESP0 + inputs_i)
        RESP_next = self._pos(RESP0 + inputs_i - q_i)
        q_total = torch.sum(q_i, dim=1, keepdim=True)
        resp_total = torch.sum(RESP_next, dim=1, keepdim=True)
        tau_eff = torch.sum(weights * self.response_tau_days.to(device=RESP0.device, dtype=RESP0.dtype).view(1, self.n_response), dim=1, keepdim=True)
        return weights, inputs_i, q_i, RESP_next, q_total, resp_total, tau_eff

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        (INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K, TT, CFMAX_base, CFR, CWH,
         theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint) = self._expand_simple(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SA = torch.zeros(B, 1, device=device, dtype=dtype)
            RESP = torch.zeros(B, self.n_response, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
            init_sa = SA
            init_resp = RESP
            init_snow = SNOWPACK
            init_melt = MELTWATER
        else:
            SA = initial_state[:, 0:1]
            RESP = initial_state[:, 1:1 + self.n_response]
            SNOWPACK = initial_state[:, 1 + self.n_response:2 + self.n_response]
            MELTWATER = initial_state[:, 2 + self.n_response:3 + self.n_response]
            init_sa = SA
            init_resp = RESP
            init_snow = SNOWPACK
            init_melt = MELTWATER

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = {}
        tau_names = [str(int(x)) for x in self.response_tau_days.detach().cpu().tolist()]
        if return_diagnostics is True:
            diag_names = [
                'SNOWPACK_prev', 'MELTWATER_prev', 'Sa_prev', 'GW_prev',
                'SNOWPACK', 'MELTWATER', 'Sa', 'GW',
                'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release',
                'PL', 'interception_evaporation', 'actual_ET',
                'Smoist', 'theta_cap', 'alpha', 'P_accessible', 'P_inaccessible', 'Sa_overflow',
                'surface_runoff', 'interflow', 'recharge_to_groundwater',
                'baseflow', 'Q_process', 'groundwater_loss', 'channel_loss', 'gate_loss',
                'partition_sum_error', 'soil_local_residual', 'gw_local_residual',
                'snowpack_local_residual', 'meltwater_local_residual', 'snow_total_local_residual',
                'process_local_residual', 'response_storage_total', 'response_discharge_total',
                'response_input_total', 'tau_eff', 'INSC', 'COEF_t', 'SQ_t', 'K_t', 'TT', 'CFMAX_t', 'CFR', 'CWH'
            ]
            for tau_name in tau_names:
                diag_names.extend([
                    f'response_weight_tau_{tau_name}',
                    f'response_input_tau_{tau_name}',
                    f'response_store_tau_{tau_name}',
                    f'response_discharge_tau_{tau_name}',
                ])
            for name in diag_names:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []
        response_weight_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SA0 = self._min(self._pos(SA), theta_cap)
            RESP0 = self._pos(RESP)
            RESP_total0 = torch.sum(RESP0, dim=1, keepdim=True)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            Smoist0 = torch.clamp(SA0 / (theta_cap + 1e-8), 0.0, 1.0)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0, Tt / 20.0, E0t / 10.0,
                SA0 / 300.0, Smoist0, RESP_total0 / 300.0, SNOWPACK0 / 300.0, sin_t, cos_t
            ], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)

            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX_base * m_cf_eff

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()
            rainfall = Pt * frac_rain
            snowfall = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + snowfall
            snowmelt_pot = CFMAX_t * self._pos(Tt - TT)
            snowmelt = self._min(snowmelt_pot, SNOWPACK1)
            SNOWPACK2 = self._pos(SNOWPACK1 - snowmelt)
            MELTWATER1 = MELTWATER0 + snowmelt

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)
            SNOWPACK3 = SNOWPACK2 + refreezing

            snow_holding = CWH * SNOWPACK3
            snow_release_raw = self._pos(MELTWATER2 - snow_holding)
            snow_release = self._min(snow_release_raw, MELTWATER2)
            MELTWATER_next = self._pos(MELTWATER2 - snow_release)
            SNOWPACK_next = self._pos(SNOWPACK3)

            PL = rainfall + snow_release
            INT = self._min(INSC, self._min(E0t, PL))
            PL_after_int = self._pos(PL - INT)
            POT = self._pos(E0t - INT)

            alpha = theta_ab * torch.pow(torch.clamp(1.0 - Smoist0, min=0.0), theta_ak)
            alpha = torch.clamp(alpha, 0.0, 1.0)
            P_accessible = alpha * PL_after_int
            P_inaccessible = (1.0 - alpha) * PL_after_int

            SA_pre = SA0 + P_accessible
            ET_stress = torch.clamp(Smoist0 / torch.clamp(theta_wetpoint, min=1e-6), 0.0, 1.0)
            ET_a_pot = POT * theta_efmax * ET_stress
            ET_a = self._min(ET_a_pot, SA_pre)
            SA_after_ET = self._pos(SA_pre - ET_a)
            SA_overflow = self._pos(SA_after_ET - theta_cap)
            SA_next = self._pos(SA_after_ET - SA_overflow)

            water_for_partition = self._pos(P_inaccessible + SA_overflow)
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_interflow = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([
                torch.log(base_surface),
                torch.log(base_interflow),
                torch.log(base_recharge)
            ], dim=1)
            if self.dynamic_partition is True:
                part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']]
            else:
                part_logits = base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]

            if self.response_variant == 'A':
                SRUN = f_surface * water_for_partition
                IFLOW = f_inter * water_for_partition
                REC = f_recharge * water_for_partition
                response_input_total = REC
                response_partition_sum_error = torch.abs(f_surface + f_inter + f_recharge - 1.0)
                recharge_frac_hist.append(f_recharge)
            else:
                SRUN = torch.zeros_like(water_for_partition)
                IFLOW = torch.zeros_like(water_for_partition)
                REC = water_for_partition
                response_input_total = water_for_partition
                response_partition_sum_error = torch.zeros_like(water_for_partition)

            response_features = self._response_features(
                Pt, Tt, E0t, PL_after_int, response_input_total, Smoist0,
                SNOWPACK0, MELTWATER0, RESP_total0, sin_t, cos_t)
            response_w, response_inputs_i, response_q_i, RESP_next, BAS, RESP_total_next, tau_eff = self._response_step(
                RESP0, response_input_total, response_features)
            if torch.isnan(response_q_i).any() or torch.isnan(RESP_next).any():
                raise RuntimeError("NaN detected in NeuralResponseSpectrum response block")
            if not torch.all(response_q_i <= RESP0 + response_inputs_i + 1e-6):
                raise RuntimeError("NeuralResponseSpectrum violated Qi <= Si + Inputi")
            if not torch.all(RESP_next >= -1e-6):
                raise RuntimeError("NeuralResponseSpectrum violated RESP_next >= -1e-6")

            Q_process = self._pos(SRUN + IFLOW + BAS)
            process_local_residual = (
                Pt - INT - ET_a - Q_process
                - ((SNOWPACK_next - SNOWPACK0) + (MELTWATER_next - MELTWATER0) + (SA_next - SA0) + (RESP_total_next - RESP_total0))
            )
            soil_local_residual = P_accessible - ET_a - SA_overflow - (SA_next - SA0)
            gw_local_residual = response_input_total - BAS - (RESP_total_next - RESP_total0)
            snowpack_local_residual = snowfall - snowmelt + refreezing - (SNOWPACK_next - SNOWPACK0)
            meltwater_local_residual = snowmelt - refreezing - snow_release - (MELTWATER_next - MELTWATER0)
            snow_total_local_residual = snowfall - snow_release - (SNOWPACK_next - SNOWPACK0) - (MELTWATER_next - MELTWATER0)

            SA = SA_next
            RESP = RESP_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q_process)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            response_weight_hist.append(response_w)

            if return_diagnostics is True:
                diag_vals = {
                    'SNOWPACK_prev': SNOWPACK0,
                    'MELTWATER_prev': MELTWATER0,
                    'Sa_prev': SA0,
                    'GW_prev': RESP_total0,
                    'SNOWPACK': SNOWPACK_next,
                    'MELTWATER': MELTWATER_next,
                    'Sa': SA_next,
                    'GW': RESP_total_next,
                    'precipitation': Pt,
                    'rainfall': rainfall,
                    'snowfall': snowfall,
                    'snowmelt': snowmelt,
                    'refreezing': refreezing,
                    'snow_release': snow_release,
                    'PL': PL,
                    'interception_evaporation': INT,
                    'actual_ET': ET_a,
                    'Smoist': Smoist0,
                    'theta_cap': theta_cap,
                    'alpha': alpha,
                    'P_accessible': P_accessible,
                    'P_inaccessible': P_inaccessible,
                    'Sa_overflow': SA_overflow,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'recharge_to_groundwater': REC,
                    'baseflow': BAS,
                    'Q_process': Q_process,
                    'groundwater_loss': torch.zeros_like(Q_process),
                    'channel_loss': torch.zeros_like(Q_process),
                    'gate_loss': torch.zeros_like(Q_process),
                    'partition_sum_error': response_partition_sum_error,
                    'soil_local_residual': soil_local_residual,
                    'gw_local_residual': gw_local_residual,
                    'snowpack_local_residual': snowpack_local_residual,
                    'meltwater_local_residual': meltwater_local_residual,
                    'snow_total_local_residual': snow_total_local_residual,
                    'process_local_residual': process_local_residual,
                    'response_storage_total': RESP_total_next,
                    'response_discharge_total': BAS,
                    'response_input_total': response_input_total,
                    'tau_eff': tau_eff,
                    'INSC': INSC,
                    'COEF_t': COEF,
                    'SQ_t': SQ_t,
                    'K_t': K,
                    'TT': TT,
                    'CFMAX_t': CFMAX_t,
                    'CFR': CFR,
                    'CWH': CWH,
                }
                for tau_i, tau_name in enumerate(tau_names):
                    diag_vals[f'response_weight_tau_{tau_name}'] = response_w[:, tau_i:tau_i + 1]
                    diag_vals[f'response_input_tau_{tau_name}'] = response_inputs_i[:, tau_i:tau_i + 1]
                    diag_vals[f'response_store_tau_{tau_name}'] = RESP_next[:, tau_i:tau_i + 1]
                    diag_vals[f'response_discharge_tau_{tau_name}'] = response_q_i[:, tau_i:tau_i + 1]
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SA, RESP, SNOWPACK, MELTWATER], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        reg_response_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)
        if len(response_weight_hist) > 1:
            w_seq = torch.stack(response_weight_hist, dim=1)
            reg_response_smooth = torch.mean((w_seq[:, 1:, :] - w_seq[:, :-1, :]) ** 2)
        sum_p = torch.clamp(P.sum(dim=1), min=1e-6)
        resp_end = torch.sum(RESP, dim=1, keepdim=True)
        resp_start = torch.sum(init_resp, dim=1, keepdim=True)
        storage_drift_loss = (
            torch.mean(torch.abs(SA - init_sa) / sum_p)
            + torch.mean(torch.abs(resp_end - resp_start) / sum_p)
            + torch.mean(torch.abs(SNOWPACK - init_snow) / sum_p)
            + torch.mean(torch.abs(MELTWATER - init_melt) / sum_p)
        )
        gw_drift_loss = torch.mean(torch.abs(resp_end - resp_start) / sum_p)
        sa_drift_loss = torch.mean(torch.abs(SA - init_sa) / sum_p)
        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': reg_part,
            'storage_drift_loss': storage_drift_loss,
            'gw_drift_loss': gw_drift_loss,
            'sa_drift_loss': sa_drift_loss,
            'response_weight_smooth_loss': reg_response_smooth,
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out
        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDNeuralResponseSpectrum(
        MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple):
    """
    Closed simple Model 6 wrapper with a mass-conserving neural response
    spectrum. Variant A keeps SRUN/IFLOW and only replaces the REC->BAS bank;
    Variant B routes all response water through the spectrum.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 reg_storage_drift_w=1e-2, reg_gw_drift_w=5e-3, reg_sa_drift_w=5e-3,
                 reg_response_weight_smooth_w=1e-4, component_routing=True,
                 response_hidden=32, response_variant='A', response_tau_days=None):
        super(MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDNeuralResponseSpectrum, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing)
        self.reg_storage_drift_w = reg_storage_drift_w
        self.reg_gw_drift_w = reg_gw_drift_w
        self.reg_sa_drift_w = reg_sa_drift_w
        self.reg_response_weight_smooth_w = reg_response_weight_smooth_w
        self.response_variant = str(response_variant).upper()
        self.simhyd = DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDNeuralResponseSpectrum(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            response_hidden=response_hidden, response_variant=self.response_variant,
            response_tau_days=response_tau_days)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDNeuralResponseSpectrum(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            response_hidden=response_hidden, response_variant=self.response_variant,
            response_tau_days=response_tau_days)

    def get_auxiliary_loss(self):
        aux = super(MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDNeuralResponseSpectrum, self).get_auxiliary_loss()
        if aux is None:
            return None
        if self._last_aux_terms is None:
            return aux
        return (
            aux
            + self.reg_storage_drift_w * self._last_aux_terms.get('storage_drift_loss', 0.0)
            + self.reg_gw_drift_w * self._last_aux_terms.get('gw_drift_loss', 0.0)
            + self.reg_sa_drift_w * self._last_aux_terms.get('sa_drift_loss', 0.0)
            + self.reg_response_weight_smooth_w * self._last_aux_terms.get('response_weight_smooth_loss', 0.0)
        )


class TC_MODEL(DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimple):
    """
    Hydrology-first Tethys-Chloris-inspired catchment model.
    Keeps differentiable catchment-scale structure and mass closure, while
    replacing the simple aSrz+single-GW internal water organization with:
    canopy, snow, meltwater, surface, root, deep, and groundwater stores.
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True, dynamic_cfmax_snow=True,
                 dynamic_routing_scale=False, dynamic_all=False, dyn_hidden=32,
                 use_capillary_rise=True):
        super(TC_MODEL, self).__init__(
            mode=mode, theta_is_raw=theta_is_raw, smooth=smooth, eps=eps, rain_snow_gain=rain_snow_gain,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, dyn_hidden=dyn_hidden)
        self.use_capillary_rise = use_capillary_rise

    def _expand_tc(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)
        C_canopy_base = 0.1 + theta[:, 0:1] * (2.0 - 0.1)
        k_lai = 0.2 + theta[:, 1:2] * (1.2 - 0.2)
        canopy_evap_coef = 0.2 + theta[:, 2:3] * (1.2 - 0.2)
        TT = -2.5 + theta[:, 3:4] * 5.0
        CFMAX = 0.5 + theta[:, 4:5] * (10.0 - 0.5)
        CFR = theta[:, 5:6] * 0.1
        CWH = theta[:, 6:7] * 0.2
        RAD_MELT_COEF = theta[:, 7:8] * 2.0
        SUBLIM_COEF = theta[:, 8:9] * 0.2
        SURF_CAP = 5.0 + theta[:, 9:10] * (250.0 - 5.0)
        I_MAX = 1.0 + theta[:, 10:11] * (120.0 - 1.0)
        I_EXP = 0.3 + theta[:, 11:12] * (4.0 - 0.3)
        SOIL_EVAP_FRAC = 0.1 + theta[:, 12:13] * (0.9 - 0.1)
        ROOT_CAP = 20.0 + theta[:, 13:14] * (800.0 - 20.0)
        ROOT_WETPOINT = 0.2 + theta[:, 14:15] * (0.9 - 0.2)
        K_SR = 0.01 + theta[:, 15:16] * (1.0 - 0.01)
        K_SR_EXP = 0.3 + theta[:, 16:17] * (4.0 - 0.3)
        TRANSP_MAX = 0.2 + theta[:, 17:18] * (1.2 - 0.2)
        VEG_FACTOR = 0.05 + theta[:, 18:19] * (1.0 - 0.05)
        DEEP_CAP = 20.0 + theta[:, 19:20] * (1500.0 - 20.0)
        DEEP_ACCESS_FRAC = theta[:, 20:21]
        DEEP_WETPOINT = 0.2 + theta[:, 21:22] * (0.9 - 0.2)
        K_PERC = 0.001 + theta[:, 22:23] * (0.3 - 0.001)
        PERC_EXP = 0.5 + theta[:, 23:24] * (4.0 - 0.5)
        K_RECH = 0.001 + theta[:, 24:25] * (0.2 - 0.001)
        RECH_EXP = 0.5 + theta[:, 25:26] * (4.0 - 0.5)
        SAT_THRESHOLD = 0.2 + theta[:, 26:27] * (1.1 - 0.2)
        SAT_SHARPNESS = 1.0 + theta[:, 27:28] * (15.0 - 1.0)
        K_LAT = 0.001 + theta[:, 28:29] * (0.5 - 0.001)
        LAT_EXP = 0.5 + theta[:, 29:30] * (4.0 - 0.5)
        K_GW = 0.001 + theta[:, 30:31] * (0.3 - 0.001)
        GW_EXP = 0.5 + theta[:, 31:32] * (3.0 - 0.5)
        GW_REF = 10.0 + theta[:, 32:33] * (800.0 - 10.0)
        K_CAP = theta[:, 33:34] * 0.1
        SURF_MOB_FRAC = theta[:, 34:35] * 0.5
        return (
            C_canopy_base, k_lai, canopy_evap_coef,
            TT, CFMAX, CFR, CWH, RAD_MELT_COEF, SUBLIM_COEF,
            SURF_CAP, I_MAX, I_EXP, SOIL_EVAP_FRAC,
            ROOT_CAP, ROOT_WETPOINT, K_SR, K_SR_EXP, TRANSP_MAX, VEG_FACTOR,
            DEEP_CAP, DEEP_ACCESS_FRAC, DEEP_WETPOINT, K_PERC, PERC_EXP, K_RECH, RECH_EXP,
            SAT_THRESHOLD, SAT_SHARPNESS, K_LAT, LAT_EXP,
            K_GW, GW_EXP, GW_REF, K_CAP, SURF_MOB_FRAC
        )

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        (
            C_canopy_base, k_lai, canopy_evap_coef,
            TT, CFMAX_base, CFR, CWH, RAD_MELT_COEF, SUBLIM_COEF,
            SURF_CAP, I_MAX, I_EXP, SOIL_EVAP_FRAC,
            ROOT_CAP, ROOT_WETPOINT, K_SR, K_SR_EXP, TRANSP_MAX, VEG_FACTOR,
            DEEP_CAP, DEEP_ACCESS_FRAC, DEEP_WETPOINT, K_PERC, PERC_EXP, K_RECH, RECH_EXP,
            SAT_THRESHOLD, SAT_SHARPNESS, K_LAT, LAT_EXP,
            K_GW, GW_EXP, GW_REF, K_CAP, SURF_MOB_FRAC
        ) = self._expand_tc(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        PET = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            CANOPY = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
            SURF = torch.zeros(B, 1, device=device, dtype=dtype)
            ROOT = torch.zeros(B, 1, device=device, dtype=dtype)
            DEEP = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            init_canopy = CANOPY
            init_snow = SNOWPACK
            init_melt = MELTWATER
            init_surf = SURF
            init_root = ROOT
            init_deep = DEEP
            init_gw = GW
        else:
            CANOPY = initial_state[:, 0:1]
            SNOWPACK = initial_state[:, 1:2]
            MELTWATER = initial_state[:, 2:3]
            SURF = initial_state[:, 3:4]
            ROOT = initial_state[:, 4:5]
            DEEP = initial_state[:, 5:6]
            GW = initial_state[:, 6:7]
            init_canopy = CANOPY
            init_snow = SNOWPACK
            init_melt = MELTWATER
            init_surf = SURF
            init_root = ROOT
            init_deep = DEEP
            init_gw = GW

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = {}
        if return_diagnostics is True:
            diag_names = [
                'CANOPY_prev', 'SNOWPACK_prev', 'MELTWATER_prev', 'SURF_prev', 'ROOT_prev', 'DEEP_prev', 'GW_prev',
                'CANOPY', 'SNOWPACK', 'MELTWATER', 'SURF', 'ROOT', 'DEEP', 'GW',
                'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release', 'snow_sublimation',
                'throughfall_rain', 'throughfall_total', 'canopy_evaporation', 'soil_evaporation',
                'transpiration_root', 'transpiration_deep', 'actual_ET', 'interception_evaporation',
                'PL', 'INF', 'Q_infil_excess', 'SURF_to_ROOT', 'SURF_overflow', 'Q_sat', 'Q_lateral',
                'PERC_DEEP', 'recharge_to_groundwater', 'capillary_rise', 'baseflow', 'Q_process',
                'surface_runoff', 'interflow', 'P_accessible', 'P_inaccessible', 'alpha',
                'surf_rel', 'root_rel', 'deep_rel', 'root_stress', 'deep_stress',
                'process_local_residual', 'soil_local_residual', 'gw_local_residual',
                'snowpack_local_residual', 'meltwater_local_residual', 'snow_total_local_residual',
                'water_balance_error', 'partition_sum_error',
                'C_canopy_base', 'k_lai_param', 'canopy_evap_coef',
                'TT', 'CFMAX_t', 'CFR', 'CWH', 'RAD_MELT_COEF', 'SUBLIM_COEF',
                'SURF_CAP', 'I_MAX', 'I_EXP', 'SOIL_EVAP_FRAC',
                'ROOT_CAP', 'ROOT_WETPOINT', 'K_SR', 'K_SR_EXP', 'TRANSP_MAX', 'VEG_FACTOR',
                'DEEP_CAP', 'DEEP_ACCESS_FRAC', 'DEEP_WETPOINT', 'K_PERC', 'PERC_EXP', 'K_RECH', 'RECH_EXP',
                'SAT_THRESHOLD', 'SAT_SHARPNESS', 'K_LAT', 'LAT_EXP',
                'K_GW', 'GW_EXP', 'GW_REF', 'K_CAP', 'SURF_MOB_FRAC'
            ]
            for name in diag_names:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        drift_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            PETt = PET[:, t, :]
            if inputs.shape[-1] >= 6:
                LAI_t = self._pos(inputs[:, t, 5:6])
            else:
                LAI_t = VEG_FACTOR * 2.0
            if inputs.shape[-1] >= 7:
                RAD_t = self._pos(inputs[:, t, 6:7])
            else:
                RAD_t = torch.zeros_like(Pt)

            CANOPY0 = self._pos(CANOPY)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            SURF0 = self._min(self._pos(SURF), SURF_CAP)
            ROOT0 = self._min(self._pos(ROOT), ROOT_CAP)
            DEEP0 = self._min(self._pos(DEEP), DEEP_CAP)
            GW0 = self._pos(GW)

            surf_rel0 = torch.clamp(SURF0 / (SURF_CAP + 1e-8), 0.0, 1.0)
            root_rel0 = torch.clamp(ROOT0 / (ROOT_CAP + 1e-8), 0.0, 1.0)
            deep_rel0 = torch.clamp(DEEP0 / (DEEP_CAP + 1e-8), 0.0, 1.0)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0, Tt / 20.0, PETt / 10.0,
                ROOT0 / 300.0, root_rel0, GW0 / 300.0, SNOWPACK0 / 300.0, sin_t, cos_t
            ], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(CFMAX_base)
            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf_eff = torch.ones_like(CFMAX_base)
            CFMAX_t = CFMAX_base * m_cf_eff

            veg_activity = torch.clamp(1.0 - torch.exp(-k_lai * torch.clamp(LAI_t, min=0.0)), 0.0, 1.0)
            veg_activity = torch.maximum(veg_activity, VEG_FACTOR)
            P_rain = Pt * (Tt >= TT).float() if not self.smooth else Pt * torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            P_snow = Pt - P_rain

            Cmax = C_canopy_base * torch.clamp(LAI_t, min=0.1)
            CANOPY_space = self._pos(Cmax - CANOPY0)
            INT_in = self._min(P_rain, CANOPY_space)
            throughfall_rain = self._pos(P_rain - INT_in)
            CANOPY_pre = CANOPY0 + INT_in
            wetness = torch.clamp(CANOPY_pre / (Cmax + 1e-8), 0.0, 1.0)
            E_canopy_pot = PETt * canopy_evap_coef * wetness
            E_canopy = self._min(CANOPY_pre, E_canopy_pot)
            CANOPY_next = self._pos(CANOPY_pre - E_canopy)

            SNOWPACK1 = SNOWPACK0 + P_snow
            melt_temp = CFMAX_t * self._pos(Tt - TT)
            melt_rad = RAD_MELT_COEF * self._pos(RAD_t)
            snowmelt_pot = melt_temp + melt_rad
            snowmelt = self._min(snowmelt_pot, SNOWPACK1)
            SNOWPACK2 = self._pos(SNOWPACK1 - snowmelt)
            MELTWATER1 = MELTWATER0 + snowmelt

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)
            SNOWPACK3 = SNOWPACK2 + refreezing

            snow_holding = CWH * SNOWPACK3
            snow_release_raw = self._pos(MELTWATER2 - snow_holding)
            snow_release = self._min(snow_release_raw, MELTWATER2)
            MELTWATER3 = self._pos(MELTWATER2 - snow_release)

            snow_exposure_factor = torch.clamp(1.0 - veg_activity, 0.0, 1.0)
            S_sublim_pot = PETt * SUBLIM_COEF * snow_exposure_factor
            S_sublim = self._min(SNOWPACK3, S_sublim_pot)
            SNOWPACK_next = self._pos(SNOWPACK3 - S_sublim)
            MELTWATER_next = self._pos(MELTWATER3)

            PET_remaining = self._pos(PETt - E_canopy - S_sublim)
            PL = throughfall_rain + snow_release

            surf_rel = torch.clamp(SURF0 / (SURF_CAP + 1e-8), 0.0, 1.0)
            Icap = I_MAX * torch.pow(torch.clamp(1.0 - surf_rel, min=0.0), I_EXP)
            INF = self._min(PL, Icap)
            Q_infil_excess = self._pos(PL - INF)
            SURF_pre = SURF0 + INF

            vegetation_cover = veg_activity
            E_soil_pot = PET_remaining * SOIL_EVAP_FRAC * torch.clamp(1.0 - vegetation_cover, 0.0, 1.0)
            E_soil = self._min(SURF_pre, E_soil_pot * torch.clamp(SURF_pre / (SURF_CAP + 1e-8), 0.0, 1.0))
            SURF_after_E = self._pos(SURF_pre - E_soil)

            root_rel = torch.clamp(ROOT0 / (ROOT_CAP + 1e-8), 0.0, 1.0)
            SURF_TO_ROOT_pot = K_SR * torch.pow(torch.clamp(1.0 - root_rel, min=0.0), K_SR_EXP) * SURF_after_E
            SURF_TO_ROOT = self._min(SURF_after_E, SURF_TO_ROOT_pot)
            SURF_after_root = self._pos(SURF_after_E - SURF_TO_ROOT)
            ROOT_pre = ROOT0 + SURF_TO_ROOT

            SURF_overflow = self._pos(SURF_after_root - SURF_CAP)
            SURF_hold = self._pos(SURF_after_root - SURF_overflow)

            root_rel_pre = torch.clamp(ROOT_pre / (ROOT_CAP + 1e-8), 0.0, 1.0)
            deep_rel = torch.clamp(DEEP0 / (DEEP_CAP + 1e-8), 0.0, 1.0)
            root_stress = torch.clamp(root_rel_pre / torch.clamp(ROOT_WETPOINT, min=1e-6), 0.0, 1.0)
            deep_stress = torch.clamp(deep_rel / torch.clamp(DEEP_WETPOINT, min=1e-6), 0.0, 1.0)

            PET_remaining2 = self._pos(PET_remaining - E_soil)
            T_pot = PET_remaining2 * TRANSP_MAX * veg_activity
            T_root = self._min(ROOT_pre, T_pot * root_stress)
            T_rem = self._pos(T_pot - T_root)
            T_deep = self._min(DEEP0, T_rem * DEEP_ACCESS_FRAC * deep_stress)
            ROOT_after_T = self._pos(ROOT_pre - T_root)
            DEEP_after_T = self._pos(DEEP0 - T_deep)

            sat_index = 0.5 * torch.clamp(SURF_hold / (SURF_CAP + 1e-8), 0.0, 1.0) + 0.5 * torch.clamp(ROOT_after_T / (ROOT_CAP + 1e-8), 0.0, 1.0)
            contributing_area = torch.sigmoid(SAT_SHARPNESS * (sat_index - SAT_THRESHOLD))
            Q_sat_extra_cap = SURF_MOB_FRAC * SURF_hold
            Q_sat_extra = self._min(SURF_hold, contributing_area * Q_sat_extra_cap)
            Q_sat = SURF_overflow + Q_sat_extra
            SURF_next = self._pos(SURF_hold - Q_sat_extra)

            Q_lateral_pot = K_LAT * torch.pow(torch.clamp(ROOT_after_T / (ROOT_CAP + 1e-8), min=0.0), LAT_EXP) * ROOT_after_T
            Q_lateral = self._min(Q_lateral_pot, ROOT_after_T)
            ROOT_after_lat = self._pos(ROOT_after_T - Q_lateral)

            PERC_DEEP_pot = K_PERC * torch.pow(torch.clamp(ROOT_after_lat / (ROOT_CAP + 1e-8), min=0.0), PERC_EXP) * ROOT_after_lat
            PERC_DEEP = self._min(PERC_DEEP_pot, ROOT_after_lat)
            ROOT_after_perc = self._pos(ROOT_after_lat - PERC_DEEP)
            DEEP_pre = DEEP_after_T + PERC_DEEP

            RECHARGE_pot = K_RECH * torch.pow(torch.clamp(DEEP_pre / (DEEP_CAP + 1e-8), min=0.0), RECH_EXP) * DEEP_pre
            RECHARGE = self._min(RECHARGE_pot, DEEP_pre)
            DEEP_after_recharge = self._pos(DEEP_pre - RECHARGE)
            DEEP_overflow = self._pos(DEEP_after_recharge - DEEP_CAP)
            DEEP_next = self._pos(DEEP_after_recharge - DEEP_overflow)

            if self.use_capillary_rise is True:
                water_stress = 1.0 - root_stress
                CAP_pot = K_CAP * water_stress * GW0
                CAP1 = self._min(CAP_pot, GW0)
                CAP_RISE = self._min(CAP1, self._pos(ROOT_CAP - ROOT_after_perc))
            else:
                CAP_RISE = torch.zeros_like(GW0)
            GW_after_cap = self._pos(GW0 - CAP_RISE)
            ROOT_next = ROOT_after_perc + CAP_RISE

            GW_pre = GW_after_cap + RECHARGE + DEEP_overflow
            GW_ratio = torch.clamp(GW_pre / (GW_REF + 1e-8), min=0.0)
            BAS_pot = K_GW * torch.pow(GW_ratio, GW_EXP) * GW_pre
            BAS = self._min(BAS_pot, GW_pre)
            GW_next = self._pos(GW_pre - BAS)

            surface_runoff = Q_infil_excess + Q_sat
            interflow = Q_lateral
            Q_process = self._pos(surface_runoff + interflow + BAS)
            ET_total = E_canopy + E_soil + T_root + T_deep + S_sublim

            S_before = CANOPY0 + SNOWPACK0 + MELTWATER0 + SURF0 + ROOT0 + DEEP0 + GW0
            S_after = CANOPY_next + SNOWPACK_next + MELTWATER_next + SURF_next + ROOT_next + DEEP_next + GW_next
            process_local_residual = Pt - ET_total - Q_process - (S_after - S_before)
            soil_local_residual = INF - E_soil - SURF_TO_ROOT - Q_sat - Q_infil_excess - (SURF_next - SURF0)
            gw_local_residual = RECHARGE + DEEP_overflow - CAP_RISE - BAS - (GW_next - GW0)
            snowpack_local_residual = P_snow - snowmelt + refreezing - S_sublim - (SNOWPACK_next - SNOWPACK0)
            meltwater_local_residual = snowmelt - refreezing - snow_release - (MELTWATER_next - MELTWATER0)
            snow_total_local_residual = P_snow - snow_release - S_sublim - (SNOWPACK_next - SNOWPACK0) - (MELTWATER_next - MELTWATER0)
            water_balance_error = torch.abs(process_local_residual)

            CANOPY = CANOPY_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next
            SURF = SURF_next
            ROOT = ROOT_next
            DEEP = DEEP_next
            GW = GW_next

            q_hist.append(Q_process)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            drift_hist.append(torch.abs(S_after - S_before))

            if return_diagnostics is True:
                diag_vals = {
                    'CANOPY_prev': CANOPY0,
                    'SNOWPACK_prev': SNOWPACK0,
                    'MELTWATER_prev': MELTWATER0,
                    'SURF_prev': SURF0,
                    'ROOT_prev': ROOT0,
                    'DEEP_prev': DEEP0,
                    'GW_prev': GW0,
                    'CANOPY': CANOPY_next,
                    'SNOWPACK': SNOWPACK_next,
                    'MELTWATER': MELTWATER_next,
                    'SURF': SURF_next,
                    'ROOT': ROOT_next,
                    'DEEP': DEEP_next,
                    'GW': GW_next,
                    'precipitation': Pt,
                    'rainfall': P_rain,
                    'snowfall': P_snow,
                    'snowmelt': snowmelt,
                    'refreezing': refreezing,
                    'snow_release': snow_release,
                    'snow_sublimation': S_sublim,
                    'throughfall_rain': throughfall_rain,
                    'throughfall_total': PL,
                    'canopy_evaporation': E_canopy,
                    'soil_evaporation': E_soil,
                    'transpiration_root': T_root,
                    'transpiration_deep': T_deep,
                    'actual_ET': E_soil + T_root + T_deep,
                    'interception_evaporation': E_canopy + S_sublim,
                    'PL': PL,
                    'INF': INF,
                    'Q_infil_excess': Q_infil_excess,
                    'SURF_to_ROOT': SURF_TO_ROOT,
                    'SURF_overflow': SURF_overflow,
                    'Q_sat': Q_sat,
                    'Q_lateral': Q_lateral,
                    'PERC_DEEP': PERC_DEEP,
                    'recharge_to_groundwater': RECHARGE,
                    'capillary_rise': CAP_RISE,
                    'baseflow': BAS,
                    'Q_process': Q_process,
                    'surface_runoff': surface_runoff,
                    'interflow': interflow,
                    'P_accessible': SURF_TO_ROOT,
                    'P_inaccessible': Q_infil_excess + Q_sat,
                    'alpha': veg_activity,
                    'surf_rel': surf_rel,
                    'root_rel': root_rel_pre,
                    'deep_rel': deep_rel,
                    'root_stress': root_stress,
                    'deep_stress': deep_stress,
                    'process_local_residual': process_local_residual,
                    'soil_local_residual': soil_local_residual,
                    'gw_local_residual': gw_local_residual,
                    'snowpack_local_residual': snowpack_local_residual,
                    'meltwater_local_residual': meltwater_local_residual,
                    'snow_total_local_residual': snow_total_local_residual,
                    'water_balance_error': water_balance_error,
                    'partition_sum_error': torch.zeros_like(Q_process),
                    'C_canopy_base': C_canopy_base,
                    'k_lai_param': k_lai,
                    'canopy_evap_coef': canopy_evap_coef,
                    'TT': TT,
                    'CFMAX_t': CFMAX_t,
                    'CFR': CFR,
                    'CWH': CWH,
                    'RAD_MELT_COEF': RAD_MELT_COEF,
                    'SUBLIM_COEF': SUBLIM_COEF,
                    'SURF_CAP': SURF_CAP,
                    'I_MAX': I_MAX,
                    'I_EXP': I_EXP,
                    'SOIL_EVAP_FRAC': SOIL_EVAP_FRAC,
                    'ROOT_CAP': ROOT_CAP,
                    'ROOT_WETPOINT': ROOT_WETPOINT,
                    'K_SR': K_SR,
                    'K_SR_EXP': K_SR_EXP,
                    'TRANSP_MAX': TRANSP_MAX,
                    'VEG_FACTOR': VEG_FACTOR,
                    'DEEP_CAP': DEEP_CAP,
                    'DEEP_ACCESS_FRAC': DEEP_ACCESS_FRAC,
                    'DEEP_WETPOINT': DEEP_WETPOINT,
                    'K_PERC': K_PERC,
                    'PERC_EXP': PERC_EXP,
                    'K_RECH': K_RECH,
                    'RECH_EXP': RECH_EXP,
                    'SAT_THRESHOLD': SAT_THRESHOLD,
                    'SAT_SHARPNESS': SAT_SHARPNESS,
                    'K_LAT': K_LAT,
                    'LAT_EXP': LAT_EXP,
                    'K_GW': K_GW,
                    'GW_EXP': GW_EXP,
                    'GW_REF': GW_REF,
                    'K_CAP': K_CAP,
                    'SURF_MOB_FRAC': SURF_MOB_FRAC,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([CANOPY, SNOWPACK, MELTWATER, SURF, ROOT, DEEP, GW], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        sum_p = torch.clamp(P.sum(dim=1), min=1e-6)
        s_start = init_canopy + init_snow + init_melt + init_surf + init_root + init_deep + init_gw
        s_end = CANOPY + SNOWPACK + MELTWATER + SURF + ROOT + DEEP + GW
        storage_drift_loss = torch.mean(((s_end - s_start) / sum_p) ** 2)
        gw_drift_loss = torch.mean(((GW - init_gw) / sum_p) ** 2)
        root_drift_loss = torch.mean(((ROOT - init_root) / sum_p) ** 2)
        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': torch.tensor(0.0, device=device, dtype=dtype),
            'storage_drift_loss': storage_drift_loss,
            'gw_drift_loss': gw_drift_loss,
            'sa_drift_loss': root_drift_loss,
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out
        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_TC_MODEL(MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple):
    """
    Regionalized wrapper for TC_MODEL that preserves the existing component,
    routing, and mixture framework.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 reg_storage_drift_w=1e-2, reg_gw_drift_w=5e-3, reg_root_drift_w=5e-3,
                 component_routing=True, use_capillary_rise=True):
        super(MultiInv_TC_MODEL, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing)
        self.reg_storage_drift_w = reg_storage_drift_w
        self.reg_gw_drift_w = reg_gw_drift_w
        self.reg_root_drift_w = reg_root_drift_w
        self.nfea = 35
        self.nstaticpm = self.nfea * nmul
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.simhyd = TC_MODEL(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            use_capillary_rise=use_capillary_rise)
        self.simhyd_analysis = TC_MODEL(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            use_capillary_rise=use_capillary_rise)

    def _theta_to_smsc(self, theta, ngage):
        theta4 = theta.view(ngage, self.nmul, self.nfea)
        return 20.0 + theta4[:, :, 13:14] * (800.0 - 20.0)

    def get_auxiliary_loss(self):
        aux = super(MultiInv_TC_MODEL, self).get_auxiliary_loss()
        if aux is None:
            return None
        if self._last_aux_terms is None:
            return aux
        return (
            aux
            + self.reg_storage_drift_w * self._last_aux_terms.get('storage_drift_loss', 0.0)
            + self.reg_gw_drift_w * self._last_aux_terms.get('gw_drift_loss', 0.0)
            + self.reg_root_drift_w * self._last_aux_terms.get('sa_drift_loss', 0.0)
        )


class DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleRegulated(
        DynamicSimHydModelFiveDifferentiablePhysicalFix):
    """
    Simple closed snow+aSrz+SIMHYD copy with minimal drift controls:
    active root-zone drainage, groundwater capacity/nonlinear baseflow,
    and recharge throttling, while keeping zero external loss.
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True, dynamic_cfmax_snow=True,
                 dynamic_routing_scale=False, dynamic_all=False, dyn_hidden=32, theta_cap_max=500.0):
        super(DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleRegulated, self).__init__(
            mode=mode, theta_is_raw=theta_is_raw, smooth=smooth, eps=eps, rain_snow_gain=rain_snow_gain,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, dyn_hidden=dyn_hidden)
        self.theta_cap_max = theta_cap_max

    def _expand_regulated(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)
        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = theta[:, 2:3] * 6.0
        SMSC_legacy = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = theta[:, 4:5]
        CRAK = theta[:, 5:6]
        K = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        TT = -2.5 + theta[:, 8:9] * 5.0
        CFMAX = 0.5 + theta[:, 9:10] * (10.0 - 0.5)
        CFR = theta[:, 10:11] * 0.1
        CWH = theta[:, 11:12] * 0.2
        theta_ab = 0.5 + theta[:, 13:14] * 0.5
        theta_ak = 1.0 + theta[:, 14:15] * 9.0
        theta_cap = 10.0 + theta[:, 15:16] * (self.theta_cap_max - 10.0)
        theta_efmax = 0.5 + theta[:, 16:17] * 0.5
        theta_wetpoint = 0.3 + theta[:, 17:18] * 0.6
        K_sa_drain = 0.001 + theta[:, 18:19] * (0.2 - 0.001)
        beta_sa = theta[:, 19:20] * 3.0
        GW_cap = 50.0 + theta[:, 20:21] * (1000.0 - 50.0)
        beta_gw = theta[:, 21:22] * 3.0
        return (
            INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K, TT, CFMAX, CFR, CWH,
            theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint,
            K_sa_drain, beta_sa, GW_cap, beta_gw
        )

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        (INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K, TT, CFMAX_base, CFR, CWH,
         theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint,
         K_sa_drain, beta_sa, GW_cap, beta_gw) = self._expand_regulated(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SA = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
            init_sa = SA
            init_gw = GW
            init_snow = SNOWPACK
            init_melt = MELTWATER
        else:
            SA = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]
            SNOWPACK = initial_state[:, 2:3]
            MELTWATER = initial_state[:, 3:4]
            init_sa = initial_state[:, 0:1]
            init_gw = initial_state[:, 1:2]
            init_snow = initial_state[:, 2:3]
            init_melt = initial_state[:, 3:4]

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = {}
        if return_diagnostics is True:
            for name in [
                    'SNOWPACK_prev', 'MELTWATER_prev', 'Sa_prev', 'GW_prev',
                    'SNOWPACK', 'MELTWATER', 'Sa', 'GW',
                    'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release',
                    'PL', 'interception_evaporation', 'actual_ET',
                    'Smoist', 'theta_cap', 'alpha', 'P_accessible', 'P_inaccessible',
                    'Sa_drain', 'Sa_overflow', 'recharge_throttle', 'recharge_rejected',
                    'surface_runoff', 'interflow', 'recharge_to_groundwater',
                    'baseflow', 'Q_process', 'groundwater_loss', 'channel_loss', 'gate_loss',
                    'GW_cap', 'K_sa_drain', 'beta_sa', 'beta_gw',
                    'partition_sum_error', 'soil_local_residual', 'gw_local_residual',
                    'snowpack_local_residual', 'meltwater_local_residual', 'snow_total_local_residual',
                    'process_local_residual',
                    'INSC', 'COEF_t', 'SQ_t', 'K_t', 'TT', 'CFMAX_t', 'CFR', 'CWH'
            ]:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SA0 = self._pos(SA)
            GW0 = self._pos(GW)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            Smoist0 = torch.clamp(SA0 / (theta_cap + 1e-8), 0.0, 1.0)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0, Tt / 20.0, E0t / 10.0,
                SA0 / 300.0, Smoist0, GW0 / 300.0, SNOWPACK0 / 300.0, sin_t, cos_t
            ], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)

            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX_base * m_cf_eff

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()
            rainfall = Pt * frac_rain
            snowfall = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + snowfall
            snowmelt_pot = CFMAX_t * self._pos(Tt - TT)
            snowmelt = self._min(snowmelt_pot, SNOWPACK1)
            SNOWPACK2 = self._pos(SNOWPACK1 - snowmelt)
            MELTWATER1 = MELTWATER0 + snowmelt

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)
            SNOWPACK3 = SNOWPACK2 + refreezing

            snow_holding = CWH * SNOWPACK3
            snow_release_raw = self._pos(MELTWATER2 - snow_holding)
            snow_release = self._min(snow_release_raw, MELTWATER2)
            MELTWATER_next = self._pos(MELTWATER2 - snow_release)
            SNOWPACK_next = self._pos(SNOWPACK3)

            PL = rainfall + snow_release
            INT = self._min(INSC, self._min(E0t, PL))
            PL_after_int = self._pos(PL - INT)
            POT = self._pos(E0t - INT)

            alpha = theta_ab * torch.pow(torch.clamp(1.0 - Smoist0, min=0.0), theta_ak)
            alpha = torch.clamp(alpha, 0.0, 1.0)
            P_accessible = alpha * PL_after_int
            P_inaccessible = (1.0 - alpha) * PL_after_int

            SA_pre = SA0 + P_accessible
            ET_stress = torch.clamp(Smoist0 / torch.clamp(theta_wetpoint, min=1e-6), 0.0, 1.0)
            ET_a_pot = POT * theta_efmax * ET_stress
            ET_a = self._min(ET_a_pot, SA_pre)
            SA_after_ET = self._pos(SA_pre - ET_a)

            rel_sa = torch.clamp(SA_after_ET / (theta_cap + 1e-8), 0.0, 1.5)
            SA_drain_raw = K_sa_drain * SA_after_ET * torch.pow(torch.clamp(rel_sa, min=0.0), beta_sa)
            SA_drain = self._min(SA_drain_raw, SA_after_ET)
            SA_after_drain = self._pos(SA_after_ET - SA_drain)

            SA_overflow = self._pos(SA_after_drain - theta_cap)
            SA_next = self._pos(SA_after_drain - SA_overflow)

            water_for_partition = self._pos(P_inaccessible + SA_drain + SA_overflow)
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_interflow = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([
                torch.log(base_surface),
                torch.log(base_interflow),
                torch.log(base_recharge)
            ], dim=1)
            if self.dynamic_partition is True:
                part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']]
            else:
                part_logits = base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]

            SRUN0 = f_surface * water_for_partition
            IFLOW = f_inter * water_for_partition
            REC_raw = f_recharge * water_for_partition

            gw_deficit = self._pos(GW_cap - GW0)
            rech_throttle = torch.clamp(gw_deficit / (GW_cap + 1e-8), 0.0, 1.0)
            REC_to_GW = REC_raw * rech_throttle
            REC_rejected = self._pos(REC_raw - REC_to_GW)
            SRUN = SRUN0 + REC_rejected

            GW1 = GW0 + REC_to_GW
            rel_gw = torch.clamp(GW1 / (GW_cap + 1e-8), 0.0, 2.0)
            BAS_raw = K * GW1 * torch.pow(torch.clamp(rel_gw, min=0.0), beta_gw)
            BAS = self._min(BAS_raw, GW1)
            GW_next = self._pos(GW1 - BAS)

            Q_process = self._pos(SRUN + IFLOW + BAS)
            process_local_residual = (
                Pt - INT - ET_a - Q_process
                - ((SNOWPACK_next - SNOWPACK0) + (MELTWATER_next - MELTWATER0) + (SA_next - SA0) + (GW_next - GW0))
            )
            soil_local_residual = P_accessible - ET_a - SA_drain - SA_overflow - (SA_next - SA0)
            gw_local_residual = REC_to_GW - BAS - (GW_next - GW0)
            snowpack_local_residual = snowfall - snowmelt + refreezing - (SNOWPACK_next - SNOWPACK0)
            meltwater_local_residual = snowmelt - refreezing - snow_release - (MELTWATER_next - MELTWATER0)
            snow_total_local_residual = snowfall - snow_release - (SNOWPACK_next - SNOWPACK0) - (MELTWATER_next - MELTWATER0)

            SA = SA_next
            GW = GW_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q_process)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)

            if return_diagnostics is True:
                diag_vals = {
                    'SNOWPACK_prev': SNOWPACK0,
                    'MELTWATER_prev': MELTWATER0,
                    'Sa_prev': SA0,
                    'GW_prev': GW0,
                    'SNOWPACK': SNOWPACK_next,
                    'MELTWATER': MELTWATER_next,
                    'Sa': SA_next,
                    'GW': GW_next,
                    'precipitation': Pt,
                    'rainfall': rainfall,
                    'snowfall': snowfall,
                    'snowmelt': snowmelt,
                    'refreezing': refreezing,
                    'snow_release': snow_release,
                    'PL': PL,
                    'interception_evaporation': INT,
                    'actual_ET': ET_a,
                    'Smoist': Smoist0,
                    'theta_cap': theta_cap,
                    'alpha': alpha,
                    'P_accessible': P_accessible,
                    'P_inaccessible': P_inaccessible,
                    'Sa_drain': SA_drain,
                    'Sa_overflow': SA_overflow,
                    'recharge_throttle': rech_throttle,
                    'recharge_rejected': REC_rejected,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'recharge_to_groundwater': REC_to_GW,
                    'baseflow': BAS,
                    'Q_process': Q_process,
                    'groundwater_loss': torch.zeros_like(Q_process),
                    'channel_loss': torch.zeros_like(Q_process),
                    'gate_loss': torch.zeros_like(Q_process),
                    'GW_cap': GW_cap,
                    'K_sa_drain': K_sa_drain,
                    'beta_sa': beta_sa,
                    'beta_gw': beta_gw,
                    'partition_sum_error': torch.abs(f_surface + f_inter + f_recharge - 1.0),
                    'soil_local_residual': soil_local_residual,
                    'gw_local_residual': gw_local_residual,
                    'snowpack_local_residual': snowpack_local_residual,
                    'meltwater_local_residual': meltwater_local_residual,
                    'snow_total_local_residual': snow_total_local_residual,
                    'process_local_residual': process_local_residual,
                    'INSC': INSC,
                    'COEF_t': COEF,
                    'SQ_t': SQ_t,
                    'K_t': K,
                    'TT': TT,
                    'CFMAX_t': CFMAX_t,
                    'CFR': CFR,
                    'CWH': CWH,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SA, GW, SNOWPACK, MELTWATER], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)
        sum_p = torch.clamp(P.sum(dim=1), min=1e-6)
        drift_loss = (
            torch.mean(torch.abs(SA - init_sa) / sum_p)
            + torch.mean(torch.abs(GW - init_gw) / sum_p)
        )
        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': reg_part,
            'storage_drift_loss': drift_loss,
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out
        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleRegulated(
        MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple):
    """
    Regulated copy of the simple closed snow+aSrz+SIMHYD model with
    root-zone drainage, groundwater throttling, and an explicit storage-drift
    regularizer. The original simple model is left untouched.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 component_routing=True, theta_cap_max=500.0, drift_reg_weight=0.01):
        super(MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleRegulated, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing)
        self.theta_cap_max = theta_cap_max
        self.drift_reg_weight = drift_reg_weight
        self.nfea = 22
        self.nstaticpm = self.nfea * nmul
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.simhyd = DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleRegulated(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            theta_cap_max=theta_cap_max)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimpleRegulated(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            theta_cap_max=theta_cap_max)

    def _theta_to_smsc(self, theta, ngage):
        theta4 = theta.view(ngage, self.nmul, self.nfea)
        return 10.0 + theta4[:, :, 15:16] * (self.theta_cap_max - 10.0)

    def get_auxiliary_loss(self):
        aux = super(MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleRegulated, self).get_auxiliary_loss()
        drift_term = None
        if getattr(self, "_last_aux_terms", None) is not None:
            drift_term = self._last_aux_terms.get('storage_drift_loss', None)
        if drift_term is None:
            return aux
        drift_term = self.drift_reg_weight * drift_term
        if aux is None:
            return drift_term
        return aux + drift_term


class DynamicSimHydModelFiveDifferentiablePhysicalaSrzHBVReservoir(DynamicSimHydModelFiveDifferentiablePhysicalFix):
    """
    Active-root-zone Model 6 variant with HBV-style bounded upper/lower
    groundwater reservoirs and no discarded external groundwater sink.
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True, dynamic_cfmax_snow=True,
                 dynamic_routing_scale=False, dynamic_all=False, dyn_hidden=32,
                 use_lg_transfer=True):
        super(DynamicSimHydModelFiveDifferentiablePhysicalaSrzHBVReservoir, self).__init__(
            mode=mode, theta_is_raw=theta_is_raw, smooth=smooth, eps=eps, rain_snow_gain=rain_snow_gain,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, dyn_hidden=dyn_hidden)
        self.use_lg_transfer = use_lg_transfer

    def _expand_asrz_hbv(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)
        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = theta[:, 2:3] * 6.0
        SMSC_legacy = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = theta[:, 4:5]
        CRAK = theta[:, 5:6]
        K_FAST = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        LG = theta[:, 7:8] * 0.2
        TT = -2.5 + theta[:, 8:9] * 5.0
        CFMAX = 0.5 + theta[:, 9:10] * (10.0 - 0.5)
        CFR = theta[:, 10:11] * 0.1
        CWH = theta[:, 11:12] * 0.2
        SG_CRIT = theta[:, 12:13] * 300.0
        theta_ab = 0.5 + theta[:, 13:14] * 0.5
        theta_ak = 1.0 + theta[:, 14:15] * 9.0
        theta_cap = 10.0 + theta[:, 15:16] * (1500.0 - 10.0)
        theta_efmax = 0.5 + theta[:, 16:17] * 0.5
        theta_wetpoint = 0.3 + theta[:, 17:18] * 0.6
        K_SLOW = 0.0005 + theta[:, 18:19] * (0.15 - 0.0005)
        PERC_CAP = theta[:, 19:20] * 10.0
        K_CAP = theta[:, 20:21] * 0.05
        K_CHANNEL = 0.05 + theta[:, 21:22] * (1.0 - 0.05)
        return (
            INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K_FAST, LG, TT, CFMAX, CFR, CWH, SG_CRIT,
            theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint,
            K_SLOW, PERC_CAP, K_CAP, K_CHANNEL
        )

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        (INSC, COEF, SQ, SMSC_legacy, SUB, CRAK, K_FAST, LG, TT, CFMAX_base, CFR, CWH, SG_CRIT,
         theta_ab, theta_ak, theta_cap, theta_efmax, theta_wetpoint,
         K_SLOW, PERC_CAP, K_CAP, K_CHANNEL) = self._expand_asrz_hbv(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SA = torch.zeros(B, 1, device=device, dtype=dtype)
            UZ = torch.zeros(B, 1, device=device, dtype=dtype)
            LZ = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
        else:
            SA = initial_state[:, 0:1]
            UZ = initial_state[:, 1:2]
            LZ = initial_state[:, 2:3]
            SNOWPACK = initial_state[:, 3:4]
            MELTWATER = initial_state[:, 4:5]

        if lg_dyn_seq is not None and (lg_dyn_seq.shape[0] != B or lg_dyn_seq.shape[1] != Tlen):
            raise ValueError('lg_dyn_seq must have shape [B, T, 1] matching inputs')

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = dict()
        if return_diagnostics is True:
            for name in [
                    'Sa_prev', 'UZ_prev', 'LZ_prev', 'SNOWPACK_prev', 'MELTWATER_prev',
                    'Sa', 'UZ', 'LZ', 'SNOWPACK', 'MELTWATER',
                    'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release_to_soil',
                    'interception_storage', 'interception_evaporation', 'actual_ET',
                    'Smoist', 'theta_cap', 'alpha_t', 'P_accessible', 'P_inaccessible', 'aSrz_overflow',
                    'surface_runoff', 'interflow', 'recharge_to_groundwater', 'soil_overflow',
                    'Q_fast_raw', 'Q_fast', 'PERC', 'Q_slow_raw', 'Q_slow',
                    'UZ_to_LZ_extra', 'CAP', 'channel_loss', 'gate_loss',
                    'q_raw_process', 'q_after_channel_loss', 'q_after_gate', 'Q_process',
                    'partition_sum_error', 'soil_local_residual', 'uz_local_residual', 'lz_local_residual',
                    'snowpack_local_residual', 'meltwater_local_residual', 'snow_total_local_residual',
                    'process_local_residual', 'INSC', 'COEF_t', 'SQ_t', 'K_fast_t', 'K_slow_t',
                    'PERC_cap_t', 'K_cap_t', 'K_channel', 'LG_t', 'TT', 'SG_CRIT', 'CFMAX_t', 'CFR', 'CWH'
            ]:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SA0 = self._min(self._pos(SA), theta_cap)
            UZ0 = self._pos(UZ)
            LZ0 = self._pos(LZ)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            Smoist0 = torch.clamp(SA0 / (theta_cap + 1e-8), min=0.0, max=1.0)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0, Tt / 20.0, E0t / 10.0,
                SA0 / 300.0, Smoist0, (UZ0 + LZ0) / 300.0, SNOWPACK0 / 300.0, sin_t, cos_t
            ], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)

            if self.dynamic_etgam is True:
                ETGAM_t = 0.25 + 3.75 * torch.sigmoid(dyn_raw[:, self.dyn_slices['etgam']])
            else:
                ETGAM_t = torch.ones_like(SQ)

            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX_base * m_cf_eff

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()

            RAIN = Pt * frac_rain
            SNOW = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + SNOW
            melt_pot = CFMAX_t * self._pos(Tt - TT)
            melt = self._min(melt_pot, SNOWPACK1)
            MELTWATER1 = MELTWATER0 + melt
            SNOWPACK2 = self._pos(SNOWPACK1 - melt)

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            SNOWPACK3 = SNOWPACK2 + refreezing
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)

            water_holding = CWH * SNOWPACK3
            tosoil_raw = self._pos(MELTWATER2 - water_holding)
            tosoil = self._min(tosoil_raw, MELTWATER2)
            MELTWATER_next = self._pos(MELTWATER2 - tosoil)
            SNOWPACK_next = self._pos(SNOWPACK3)

            PL = RAIN + tosoil
            INT = self._min(INSC, self._min(E0t, PL))
            PL_after_int = self._pos(PL - INT)
            POT = self._pos(E0t - INT)

            alpha_t = theta_ab * torch.pow(torch.clamp(1.0 - Smoist0, min=0.0), theta_ak)
            alpha_t = torch.clamp(alpha_t, 0.0, 1.0)
            P_accessible = alpha_t * PL_after_int
            P_inaccessible = (1.0 - alpha_t) * PL_after_int

            SA_pre = SA0 + P_accessible
            g = torch.clamp(Smoist0 / torch.clamp(theta_wetpoint, min=1e-6), 0.0, 1.0)
            ET_a_potential = POT * theta_efmax * g
            ET_a = self._min(ET_a_potential, SA_pre)
            SA_after_ET = self._pos(SA_pre - ET_a)
            aSrz_overflow = self._pos(SA_after_ET - theta_cap)
            SA_tmp = self._pos(SA_after_ET - aSrz_overflow)

            available_for_partition = self._pos(P_inaccessible + aSrz_overflow)
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_inter = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([torch.log(base_surface), torch.log(base_inter), torch.log(base_recharge)], dim=1)
            if self.dynamic_partition is True and dyn_raw is not None:
                part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']]
            else:
                part_logits = base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]

            SRUN = f_surface * available_for_partition
            IFLOW = f_inter * available_for_partition
            REC = f_recharge * available_for_partition

            UZ1 = UZ0 + REC
            Q_fast_raw = K_FAST * UZ1
            Q_fast = self._min(Q_fast_raw, UZ1)
            UZ2 = self._pos(UZ1 - Q_fast)

            PERC = self._min(PERC_CAP, UZ2)
            UZ3 = self._pos(UZ2 - PERC)
            LZ1 = LZ0 + PERC

            Q_slow_raw = K_SLOW * LZ1
            Q_slow = self._min(Q_slow_raw, LZ1)
            LZ2 = self._pos(LZ1 - Q_slow)

            if lg_dyn_seq is None:
                LG_t = LG
            else:
                LG_dyn_t = torch.clamp(lg_dyn_seq[:, t, :], 0.0, 1.0)
                LG_eff = (1.0 - lg_dyn_weight) * LG + lg_dyn_weight * (LG * LG_dyn_t * 1.5)
                LG_t = torch.clamp(LG_eff, 0.0, 0.2)

            if self.use_lg_transfer:
                UZ_to_LZ_raw = LG_t * F.softplus(SG_CRIT - UZ3)
                UZ_to_LZ_extra = self._min(UZ_to_LZ_raw, UZ3)
            else:
                UZ_to_LZ_extra = torch.zeros_like(UZ3)
            UZ4 = self._pos(UZ3 - UZ_to_LZ_extra)
            LZ3 = LZ2 + UZ_to_LZ_extra

            Sa_deficit = self._pos(theta_cap - SA_tmp)
            CAP_raw = K_CAP * LZ3 * Sa_deficit / torch.clamp(theta_cap, min=1e-6)
            CAP = self._min(CAP_raw, self._min(LZ3, Sa_deficit))
            LZ_next = self._pos(LZ3 - CAP)
            SA_next = self._min(SA_tmp + CAP, theta_cap)
            UZ_next = UZ4

            Q = self._pos(SRUN + IFLOW + Q_fast + Q_slow)

            soil_local_residual = P_accessible + CAP - ET_a - aSrz_overflow - (SA_next - SA0)
            uz_local_residual = REC - Q_fast - PERC - UZ_to_LZ_extra - (UZ_next - UZ0)
            lz_local_residual = PERC + UZ_to_LZ_extra - Q_slow - CAP - (LZ_next - LZ0)
            snowpack_local_residual = SNOW - melt + refreezing - (SNOWPACK_next - SNOWPACK0)
            meltwater_local_residual = melt - refreezing - tosoil - (MELTWATER_next - MELTWATER0)
            snow_total_local_residual = SNOW - tosoil - (SNOWPACK_next - SNOWPACK0) - (MELTWATER_next - MELTWATER0)
            process_local_residual = (
                Pt - INT - ET_a - Q
                - ((SA_next - SA0) + (UZ_next - UZ0) + (LZ_next - LZ0)
                   + (SNOWPACK_next - SNOWPACK0) + (MELTWATER_next - MELTWATER0))
            )

            SA = SA_next
            UZ = UZ_next
            LZ = LZ_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)

            if return_diagnostics is True:
                diag_vals = {
                    'Sa_prev': SA0,
                    'UZ_prev': UZ0,
                    'LZ_prev': LZ0,
                    'SNOWPACK_prev': SNOWPACK0,
                    'MELTWATER_prev': MELTWATER0,
                    'Sa': SA_next,
                    'UZ': UZ_next,
                    'LZ': LZ_next,
                    'SNOWPACK': SNOWPACK_next,
                    'MELTWATER': MELTWATER_next,
                    'precipitation': Pt,
                    'rainfall': RAIN,
                    'snowfall': SNOW,
                    'snowmelt': melt,
                    'refreezing': refreezing,
                    'snow_release_to_soil': tosoil,
                    'interception_storage': torch.zeros_like(INT),
                    'interception_evaporation': INT,
                    'actual_ET': ET_a,
                    'Smoist': Smoist0,
                    'theta_cap': theta_cap,
                    'alpha_t': alpha_t,
                    'P_accessible': P_accessible,
                    'P_inaccessible': P_inaccessible,
                    'aSrz_overflow': aSrz_overflow,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'recharge_to_groundwater': REC,
                    'soil_overflow': aSrz_overflow,
                    'Q_fast_raw': Q_fast_raw,
                    'Q_fast': Q_fast,
                    'PERC': PERC,
                    'Q_slow_raw': Q_slow_raw,
                    'Q_slow': Q_slow,
                    'UZ_to_LZ_extra': UZ_to_LZ_extra,
                    'CAP': CAP,
                    'channel_loss': torch.zeros_like(Q),
                    'gate_loss': torch.zeros_like(Q),
                    'q_raw_process': Q,
                    'q_after_channel_loss': Q,
                    'q_after_gate': Q,
                    'Q_process': Q,
                    'partition_sum_error': torch.abs(f_surface + f_inter + f_recharge - 1.0),
                    'soil_local_residual': soil_local_residual,
                    'uz_local_residual': uz_local_residual,
                    'lz_local_residual': lz_local_residual,
                    'snowpack_local_residual': snowpack_local_residual,
                    'meltwater_local_residual': meltwater_local_residual,
                    'snow_total_local_residual': snow_total_local_residual,
                    'process_local_residual': process_local_residual,
                    'INSC': INSC,
                    'COEF_t': COEF,
                    'SQ_t': SQ_t,
                    'K_fast_t': K_FAST,
                    'K_slow_t': K_SLOW,
                    'PERC_cap_t': PERC_CAP,
                    'K_cap_t': K_CAP,
                    'K_channel': K_CHANNEL,
                    'LG_t': LG_t,
                    'TT': TT,
                    'SG_CRIT': SG_CRIT,
                    'CFMAX_t': CFMAX_t,
                    'CFR': CFR,
                    'CWH': CWH,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SA, UZ, LZ, SNOWPACK, MELTWATER], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)
        sum_p = torch.clamp(P.sum(dim=1), min=1e-6)
        drift_loss = torch.mean(torch.abs(UZ - initial_state[:, 1:2] if initial_state is not None else UZ) / sum_p) \
            + torch.mean(torch.abs(LZ - initial_state[:, 2:3] if initial_state is not None else LZ) / sum_p)
        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': reg_part,
            'storage_drift_loss': drift_loss,
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out
        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelSix_Physical_aSrzHBVReservoir(MultiInv_DynamicSimHydModelSix_Physical):
    """
    Copied aSrz model with bounded HBV-style UZ/LZ groundwater reservoirs and
    internal channel/gate storage instead of discarded losses.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 component_routing=True, dry_channel_loss=True, zero_flow_gate=True,
                 channel_loss_max=0.60, zero_gate_hidden=None,
                 gate_variant='soft', gate_strength_max=0.30,
                 use_lg_transfer=True, drift_reg_weight=1e-3):
        super(MultiInv_DynamicSimHydModelSix_Physical_aSrzHBVReservoir, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing,
            dry_channel_loss=dry_channel_loss, zero_flow_gate=zero_flow_gate,
            channel_loss_max=channel_loss_max, zero_gate_hidden=zero_gate_hidden,
            gate_variant=gate_variant, gate_strength_max=gate_strength_max)
        self.nfea = 22
        self.nstaticpm = self.nfea * nmul
        self.drift_reg_weight = drift_reg_weight
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.simhyd = DynamicSimHydModelFiveDifferentiablePhysicalaSrzHBVReservoir(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            use_lg_transfer=use_lg_transfer)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiablePhysicalaSrzHBVReservoir(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            use_lg_transfer=use_lg_transfer)

    def _theta_to_smsc(self, theta, ngage):
        theta4 = theta.view(ngage, self.nmul, self.nfea)
        return 10.0 + theta4[:, :, 15:16] * (1500.0 - 10.0)

    def _theta_to_k_channel(self, theta, ngage):
        theta4 = theta.view(ngage, self.nmul, self.nfea)
        return 0.05 + theta4[:, :, 21:22] * (1.0 - 0.05)

    def forward(self, x, z, doDropMC=False, return_diagnostics=False, return_component_diagnostics=False):
        nt_x = x.shape[0]
        ngage = z.shape[1]
        basin_attr = z[-1, :, -self.nattr:]

        if z.shape[2] > self.nattr:
            snow_frac_raw = torch.clamp(z[-1, :, -self.nattr - 1:-self.nattr], min=0.0, max=1.0)
        else:
            snow_frac_raw = None

        staticFeat = self.staticFeat(basin_attr)
        staticParams0 = self.staticOut(staticFeat)

        cursor = 0
        static0 = staticParams0[:, cursor:cursor + self.nstaticpm].view(ngage, self.nfea, self.nmul)
        static0 = static0 + self.compStaticBias
        snowpara = torch.sigmoid(static0)
        cursor += self.nstaticpm

        routpara0 = staticParams0[:, cursor:cursor + self.nroutpm]
        if self.component_routing is True:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        elif self.comprout is False:
            routpara = torch.sigmoid(routpara0)
        else:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        cursor += self.nroutpm

        if self.nwtspm == 0:
            wts = None
        else:
            wtspara = staticParams0[:, cursor:cursor + self.nwtspm] + self.compWeightBias
            wts = F.softmax(wtspara, dim=-1)
        self._last_wts = wts

        if self.lgdyn is False:
            lg_dyn = None
        else:
            lg_dyn_seq = self.lstmdyn(z)
            lg_attr_bias = self.lgAttr(basin_attr).unsqueeze(0).repeat(lg_dyn_seq.shape[0], 1, 1)
            lg_dyn = torch.sigmoid(lg_dyn_seq + lg_attr_bias)

        x_rep = x.unsqueeze(2).repeat(1, 1, self.nmul, 1).view(nt_x, ngage * self.nmul, x.shape[2])
        x_bt = x_rep.permute(1, 0, 2)
        theta = snowpara.permute(0, 2, 1).contiguous().view(ngage * self.nmul, self.nfea)

        if lg_dyn is None:
            lg_bt = None
        else:
            lg_bt = lg_dyn.permute(1, 2, 0).contiguous().view(ngage * self.nmul, lg_dyn.shape[0], 1)
        if lg_bt is not None and self.inittime > 0:
            lg_bt_main = lg_bt[:, self.inittime:, :]
        else:
            lg_bt_main = lg_bt

        if snow_frac_raw is None:
            snow_frac_rep = None
        else:
            snow_frac_rep = snow_frac_raw.unsqueeze(1).repeat(1, self.nmul, 1).view(ngage * self.nmul, 1)

        need_process_diag = return_diagnostics or return_component_diagnostics or self.dry_channel_loss or self.zero_flow_gate_enabled

        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            x_use = x[self.inittime:, :, :]
            _, warm_state = self.simhyd_analysis(
                warm_inputs, theta, lg_dyn_seq=None, lg_dyn_weight=self.lgdynweight,
                snow_frac_raw=snow_frac_rep, return_final_state=True)
            if need_process_diag:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    main_inputs, theta, initial_state=warm_state, lg_dyn_seq=lg_bt_main,
                    lg_dyn_weight=self.lgdynweight, snow_frac_raw=snow_frac_rep,
                    return_diagnostics=True, return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    main_inputs, theta, initial_state=warm_state, lg_dyn_seq=lg_bt_main,
                    lg_dyn_weight=self.lgdynweight, snow_frac_raw=snow_frac_rep,
                    return_regularization=True)
                diag_comp = None
        else:
            x_use = x
            if need_process_diag:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    x_bt, theta, lg_dyn_seq=lg_bt, lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep, return_diagnostics=True, return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    x_bt, theta, lg_dyn_seq=lg_bt, lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep, return_regularization=True)
                diag_comp = None

        reg_total = self.reg_amp_w * reg_terms['dynamic_amplitude_loss'] \
            + self.reg_smooth_w * reg_terms['dynamic_smoothness_loss'] \
            + self.reg_part_w * reg_terms['partition_entropy_loss']
        self._last_aux_terms = reg_terms

        q_comp_raw = q_seq.view(ngage, self.nmul, q_seq.shape[1], 1).permute(2, 0, 1, 3)
        q_comp_raw = torch.clamp(q_comp_raw, min=0.0)
        q_after_loss, channel_loss, channel_loss_fraction = self._apply_channel_loss(q_comp_raw, diag_comp, theta, basin_attr, ngage)
        q_after_gate_base, zero_flow_probability, gate_loss, zero_flow_keep_fraction = self._apply_zero_flow_gate(
            q_after_loss, x_use, diag_comp, theta, ngage)
        q_after_gate_base = torch.clamp(q_after_gate_base, min=0.0)

        K_channel = self._theta_to_k_channel(theta, ngage).unsqueeze(0)
        channel_store = torch.zeros_like(q_after_gate_base[0])
        channel_prev_hist = []
        channel_hist = []
        channel_release_hist = []
        q_process_hist = []
        for t in range(q_after_gate_base.shape[0]):
            ch_prev = channel_store
            ch_in = ch_prev + channel_loss[t] + gate_loss[t]
            ch_release = torch.minimum(K_channel[0] * ch_in, ch_in)
            channel_store = torch.clamp(ch_in - ch_release, min=0.0)
            q_process = torch.clamp(q_after_gate_base[t] + ch_release, min=0.0)
            channel_prev_hist.append(ch_prev)
            channel_hist.append(channel_store)
            channel_release_hist.append(ch_release)
            q_process_hist.append(q_process)
        channel_prev_seq = torch.stack(channel_prev_hist, dim=0)
        channel_seq = torch.stack(channel_hist, dim=0)
        channel_release_seq = torch.stack(channel_release_hist, dim=0)
        q_comp = torch.stack(q_process_hist, dim=0)

        p_comp = self._component_tensor_4d(diag_comp['precipitation'], ngage)
        int_comp = self._component_tensor_4d(diag_comp['interception_evaporation'], ngage)
        et_comp = self._component_tensor_4d(diag_comp['actual_ET'], ngage)
        sa_prev = self._component_tensor_4d(diag_comp['Sa_prev'], ngage)
        uz_prev = self._component_tensor_4d(diag_comp['UZ_prev'], ngage)
        lz_prev = self._component_tensor_4d(diag_comp['LZ_prev'], ngage)
        snow_prev = self._component_tensor_4d(diag_comp['SNOWPACK_prev'], ngage)
        melt_prev = self._component_tensor_4d(diag_comp['MELTWATER_prev'], ngage)
        sa_next = self._component_tensor_4d(diag_comp['Sa'], ngage)
        uz_next = self._component_tensor_4d(diag_comp['UZ'], ngage)
        lz_next = self._component_tensor_4d(diag_comp['LZ'], ngage)
        snow_next = self._component_tensor_4d(diag_comp['SNOWPACK'], ngage)
        melt_next = self._component_tensor_4d(diag_comp['MELTWATER'], ngage)
        delta_storage = (
            (sa_next - sa_prev) + (uz_next - uz_prev) + (lz_next - lz_prev)
            + (snow_next - snow_prev) + (melt_next - melt_prev)
            + (channel_seq - channel_prev_seq)
        )
        process_local_residual = p_comp - int_comp - et_comp - q_comp - delta_storage

        cum_p_comp = torch.clamp(torch.sum(p_comp, dim=0), min=1e-6)
        channel_drift_loss = torch.mean(torch.abs(channel_seq[-1] - channel_prev_seq[0]) / cum_p_comp)
        reg_total = reg_total + self.drift_reg_weight * (reg_terms.get('storage_drift_loss', 0.0) + channel_drift_loss)
        self._last_aux_loss = reg_total

        q_mix_before_routing = self._mix_or_mean(q_comp, wts)
        route_mult_seq = None
        if self.routOpt is True and self.component_routing is True:
            q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
            q_routed = self._route_q(q_for_routing, routpara).view(q_comp.shape[0], ngage, self.nmul, 1)
            out = self._mix_or_mean(q_routed, wts)
        else:
            q_routed = q_comp
            if self.routOpt is True:
                if self.comprout is True:
                    q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
                    q_routed = self._route_q(q_for_routing, routpara).view(q_comp.shape[0], ngage, self.nmul, 1)
                    out = self._mix_or_mean(q_routed, wts)
                else:
                    out = self._route_q(q_mix_before_routing, routpara)
            else:
                out = q_mix_before_routing

        out = torch.clamp(out, min=0.0)
        if return_diagnostics is not True and return_component_diagnostics is not True:
            return out

        diag_out = dict()
        if diag_comp is not None:
            for name, tensor_comp in diag_comp.items():
                diag_out[name] = self._mix_component_tensor(tensor_comp, ngage)
        diag_out['total_discharge'] = out
        diag_out['q_mix_before_routing'] = q_mix_before_routing
        diag_out['component_discharge_raw'] = self._mix_or_mean(q_comp_raw, wts)
        diag_out['channel_loss'] = self._mix_or_mean(channel_loss, wts)
        diag_out['channel_loss_fraction'] = self._mix_or_mean(channel_loss_fraction, wts)
        diag_out['gate_loss'] = self._mix_or_mean(gate_loss, wts)
        diag_out['zero_flow_probability'] = self._mix_or_mean(zero_flow_probability, wts)
        diag_out['zero_flow_keep_fraction'] = self._mix_or_mean(zero_flow_keep_fraction, wts)
        diag_out['CHANNEL_STORE'] = self._mix_or_mean(channel_seq, wts)
        diag_out['CHANNEL_RELEASE'] = self._mix_or_mean(channel_release_seq, wts)
        diag_out['Q_process'] = q_mix_before_routing
        diag_out['water_balance_residual'] = self._mix_or_mean(process_local_residual, wts)
        if route_mult_seq is not None:
            diag_out['route_b_t_multiplier'] = route_mult_seq

        if return_component_diagnostics is True:
            if diag_comp is not None:
                for name, tensor_comp in diag_comp.items():
                    diag_out[name + '_components'] = self._component_tensor_4d(tensor_comp, ngage).squeeze(-1)
            diag_out['q_raw_process_components'] = q_comp_raw.squeeze(-1)
            diag_out['q_after_channel_loss_components'] = q_after_loss.squeeze(-1)
            diag_out['q_after_gate_base_components'] = q_after_gate_base.squeeze(-1)
            diag_out['q_after_gate_components'] = q_comp.squeeze(-1)
            diag_out['Q_process_components'] = q_comp.squeeze(-1)
            diag_out['q_routed_components'] = q_routed.squeeze(-1)
            diag_out['channel_loss_components'] = channel_loss.squeeze(-1)
            diag_out['gate_loss_components'] = gate_loss.squeeze(-1)
            diag_out['channel_loss_fraction_components'] = channel_loss_fraction.squeeze(-1)
            diag_out['zero_flow_probability_components'] = zero_flow_probability.squeeze(-1)
            diag_out['zero_flow_keep_fraction_components'] = zero_flow_keep_fraction.squeeze(-1)
            diag_out['CHANNEL_STORE_components'] = channel_seq.squeeze(-1)
            diag_out['CHANNEL_RELEASE_components'] = channel_release_seq.squeeze(-1)
            diag_out['process_local_residual_components'] = process_local_residual.squeeze(-1)
            diag_out['component_weights'] = wts
            if self.component_routing is True:
                routpara_view = torch.sigmoid(routpara0).view(ngage, self.nmul, 2)
                diag_out['route_a_components'] = 0.0 + routpara_view[:, :, 0] * 2.9
                diag_out['route_b_components'] = 0.0 + routpara_view[:, :, 1] * 6.5

        return out, diag_out


class DynamicECOHBVaSrzHBV11b(DynamicSimHydModelFiveDifferentiablePhysicalFix):
    """
    Ecohydrology-first active-root-zone storage + HBV1.1b-style runoff core.

    The accessible storage follows the requested aSrz formulation driven by
    net radiation and LAI, while all water that is inaccessible to the active
    root zone is passed into a closed HBV-style UZ/LZ response system.
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0,
                 theta_cap_mode='stocker', veg_function='exp', cap_upper=1000.0, min_theta_cap=1.0):
        super(DynamicECOHBVaSrzHBV11b, self).__init__(
            mode=mode, theta_is_raw=theta_is_raw, smooth=smooth, eps=eps, rain_snow_gain=rain_snow_gain,
            dynamic_sq=False, dynamic_etgam=False, dynamic_partition=False, dynamic_cfmax_snow=False,
            dynamic_routing_scale=False, dynamic_all=False)
        assert theta_cap_mode in ('stocker', 'direct')
        assert veg_function in ('exp', 'michaelis')
        self.theta_cap_mode = theta_cap_mode
        self.veg_function = veg_function
        self.cap_upper = cap_upper
        self.min_theta_cap = min_theta_cap

    def _expand_ecohbv(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)

        TT = -2.5 + theta[:, 0:1] * 5.0
        CFMAX = 0.5 + theta[:, 1:2] * 9.5
        CWH = theta[:, 2:3] * 0.2
        CFR = theta[:, 3:4] * 0.1
        theta_ab = 0.5 + theta[:, 4:5] * 0.5
        theta_ak = 1.0 + theta[:, 5:6] * 9.0
        theta_capb = theta[:, 6:7] * self.cap_upper
        theta_efmax = 0.5 + theta[:, 7:8] * 0.5
        theta_wetpoint = 0.3 + theta[:, 8:9] * 0.6
        theta_veg = 0.05 + theta[:, 9:10] * 0.9
        K0 = 0.01 + theta[:, 10:11] * 0.79
        K1 = 0.001 + theta[:, 11:12] * 0.299
        K2 = 0.0001 + theta[:, 12:13] * 0.1499
        UZL = theta[:, 13:14] * 150.0
        PERC_MAX = theta[:, 14:15] * 10.0
        cap_max = theta[:, 15:16] * 8.0
        cap_shape = 0.1 + theta[:, 16:17] * 4.9
        return (
            TT, CFMAX, CWH, CFR,
            theta_ab, theta_ak, theta_capb, theta_efmax, theta_wetpoint, theta_veg,
            K0, K1, K2, UZL, PERC_MAX, cap_max, cap_shape
        )

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, theta_cap_scale=None, return_diagnostics=False,
                return_final_state=False, return_regularization=False):
        del lg_dyn_seq, lg_dyn_weight

        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        (TT, CFMAX, CWH, CFR,
         theta_ab, theta_ak, theta_capb, theta_efmax, theta_wetpoint, theta_veg,
         K0, K1, K2, UZL, PERC_MAX, cap_max, cap_shape) = self._expand_ecohbv(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        if inputs.shape[-1] >= 6:
            RN = inputs[:, :, 5:6]
        else:
            RN = inputs[:, :, 2:3]
        if inputs.shape[-1] >= 7:
            LAI = self._pos(inputs[:, :, 6:7])
        else:
            LAI = torch.zeros(B, Tlen, 1, device=device, dtype=dtype)

        if theta_cap_scale is None:
            theta_cap_scale = torch.ones_like(theta_capb)
        theta_cap_direct = torch.clamp(theta_capb, min=self.min_theta_cap, max=self.cap_upper)
        if self.theta_cap_mode == 'stocker':
            theta_cap = torch.clamp(theta_capb * theta_cap_scale, min=self.min_theta_cap, max=self.cap_upper)
        else:
            theta_cap = theta_cap_direct

        if initial_state is None:
            SA = torch.zeros(B, 1, device=device, dtype=dtype)
            UZ = torch.zeros(B, 1, device=device, dtype=dtype)
            LZ = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
            init_sa = SA
            init_uz = UZ
            init_lz = LZ
            init_snow = SNOWPACK
            init_melt = MELTWATER
        else:
            SA = initial_state[:, 0:1]
            UZ = initial_state[:, 1:2]
            LZ = initial_state[:, 2:3]
            SNOWPACK = initial_state[:, 3:4]
            MELTWATER = initial_state[:, 4:5]
            init_sa = initial_state[:, 0:1]
            init_uz = initial_state[:, 1:2]
            init_lz = initial_state[:, 2:3]
            init_snow = initial_state[:, 3:4]
            init_melt = initial_state[:, 4:5]

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = {}
        if return_diagnostics is True:
            for name in [
                    'Sa_prev', 'UZ_prev', 'LZ_prev', 'SNOWPACK_prev', 'MELTWATER_prev',
                    'Sa', 'UZ', 'LZ', 'SNOWPACK', 'MELTWATER',
                    'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release_to_soil',
                    'liquid_input', 'actual_ET', 'ET_potential', 'net_radiation_MJ_m2_d', 'LAI_t',
                    'Smoist', 'theta_cap', 'theta_cap_direct', 'theta_cap_scale',
                    'alpha', 'Peff', 'Pinacc', 'Sa_overflow',
                    'Q0', 'Q1', 'Q2', 'PERC', 'CAP', 'stress_deficit', 'g_stress', 'h_veg',
                    'Q_process', 'SWE_sim', 'TWS_sim', 'process_local_residual',
                    'TT', 'CFMAX_t', 'CFR', 'CWH', 'K0_t', 'K1_t', 'K2_t', 'UZL_t',
                    'PERC_max_t', 'cap_max_t', 'cap_shape_t',
            ]:
                diag_hist[name] = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            RNt = self._pos(RN[:, t, :])
            LAIt = self._pos(LAI[:, t, :])

            SA0 = self._min(self._pos(SA), theta_cap)
            UZ0 = self._pos(UZ)
            LZ0 = self._pos(LZ)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            Smoist0 = torch.clamp(SA0 / (theta_cap + 1e-8), min=0.0, max=1.0)

            CFMAX_t = CFMAX * snow_mask + CFMAX * (1.0 - snow_mask)

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()
            rainfall = Pt * frac_rain
            snowfall = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + snowfall
            snowmelt_pot = CFMAX_t * self._pos(Tt - TT)
            snowmelt = self._min(snowmelt_pot, SNOWPACK1)
            SNOWPACK2 = self._pos(SNOWPACK1 - snowmelt)
            MELTWATER1 = MELTWATER0 + snowmelt

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)
            SNOWPACK3 = SNOWPACK2 + refreezing

            snow_holding = CWH * SNOWPACK3
            snow_release_raw = self._pos(MELTWATER2 - snow_holding)
            snow_release = self._min(snow_release_raw, MELTWATER2)
            MELTWATER_next = self._pos(MELTWATER2 - snow_release)
            SNOWPACK_next = self._pos(SNOWPACK3)

            PL = rainfall + snow_release
            alpha = theta_ab * torch.pow(torch.clamp(1.0 - Smoist0, min=0.0), theta_ak)
            alpha = torch.clamp(alpha, 0.0, 1.0)
            Peff = self._pos(alpha * PL)
            Pinacc = self._pos((1.0 - alpha) * PL)

            SA_pre = SA0 + Peff
            g_stress = torch.clamp(Smoist0 / torch.clamp(theta_wetpoint, min=1e-6), 0.0, 1.0)
            if self.veg_function == 'michaelis':
                theta_veg_scaled = 0.2 + 5.8 * theta_veg
                h_veg = LAIt / (LAIt + theta_veg_scaled + 1e-6)
            else:
                h_veg = 1.0 - torch.exp(-theta_veg * LAIt)
            h_veg = torch.clamp(h_veg, 0.0, 1.0)

            ET_potential = (RNt / 2.45) * theta_efmax * g_stress * h_veg
            ET_potential = self._pos(ET_potential)
            ET_a = self._min(ET_potential, SA_pre)
            SA_after_ET = self._pos(SA_pre - ET_a)
            Sa_overflow = self._pos(SA_after_ET - theta_cap)
            SA_tmp = self._pos(SA_after_ET - Sa_overflow)

            UZ1 = UZ0 + Pinacc + Sa_overflow
            Q0_raw = self._pos(K0 * self._pos(UZ1 - UZL))
            Q0 = self._min(Q0_raw, UZ1)
            UZ2 = self._pos(UZ1 - Q0)

            Q1_raw = self._pos(K1 * UZ2)
            Q1 = self._min(Q1_raw, UZ2)
            UZ3 = self._pos(UZ2 - Q1)

            PERC = self._min(PERC_MAX, UZ3)
            UZ4 = self._pos(UZ3 - PERC)
            LZ1 = LZ0 + PERC

            Q2_raw = self._pos(K2 * LZ1)
            Q2 = self._min(Q2_raw, LZ1)
            LZ2 = self._pos(LZ1 - Q2)

            stress_deficit = 1.0 - torch.clamp(SA_tmp / (theta_cap + 1e-8), min=0.0, max=1.0)
            lz_rel = LZ2 / (LZ2 + theta_cap + 1e-8)
            lz_smooth = 1.0 - torch.exp(-cap_shape * lz_rel * 5.0)
            CAP_raw = self._pos(cap_max * stress_deficit * lz_smooth)
            CAP = self._min(CAP_raw, LZ2)
            LZ_next = self._pos(LZ2 - CAP)
            SA_next = self._min(SA_tmp + CAP, theta_cap)
            UZ_next = UZ4

            SWE_sim = SNOWPACK_next + MELTWATER_next
            TWS_sim = SWE_sim + SA_next + UZ_next + LZ_next
            Q_process = self._pos(Q0 + Q1 + Q2)
            process_local_residual = (
                Pt - ET_a - Q_process
                - ((SNOWPACK_next - SNOWPACK0) + (MELTWATER_next - MELTWATER0)
                   + (SA_next - SA0) + (UZ_next - UZ0) + (LZ_next - LZ0))
            )

            SA = SA_next
            UZ = UZ_next
            LZ = LZ_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q_process)

            if return_diagnostics is True:
                diag_vals = {
                    'Sa_prev': SA0,
                    'UZ_prev': UZ0,
                    'LZ_prev': LZ0,
                    'SNOWPACK_prev': SNOWPACK0,
                    'MELTWATER_prev': MELTWATER0,
                    'Sa': SA_next,
                    'UZ': UZ_next,
                    'LZ': LZ_next,
                    'SNOWPACK': SNOWPACK_next,
                    'MELTWATER': MELTWATER_next,
                    'precipitation': Pt,
                    'rainfall': rainfall,
                    'snowfall': snowfall,
                    'snowmelt': snowmelt,
                    'refreezing': refreezing,
                    'snow_release_to_soil': snow_release,
                    'liquid_input': PL,
                    'actual_ET': ET_a,
                    'ET_potential': ET_potential,
                    'net_radiation_MJ_m2_d': RNt,
                    'LAI_t': LAIt,
                    'Smoist': Smoist0,
                    'theta_cap': theta_cap,
                    'theta_cap_direct': theta_cap_direct,
                    'theta_cap_scale': theta_cap_scale,
                    'alpha': alpha,
                    'Peff': Peff,
                    'Pinacc': Pinacc,
                    'Sa_overflow': Sa_overflow,
                    'Q0': Q0,
                    'Q1': Q1,
                    'Q2': Q2,
                    'PERC': PERC,
                    'CAP': CAP,
                    'stress_deficit': stress_deficit,
                    'g_stress': g_stress,
                    'h_veg': h_veg,
                    'Q_process': Q_process,
                    'SWE_sim': SWE_sim,
                    'TWS_sim': TWS_sim,
                    'process_local_residual': process_local_residual,
                    'TT': TT,
                    'CFMAX_t': CFMAX_t,
                    'CFR': CFR,
                    'CWH': CWH,
                    'K0_t': K0,
                    'K1_t': K1,
                    'K2_t': K2,
                    'UZL_t': UZL,
                    'PERC_max_t': PERC_MAX,
                    'cap_max_t': cap_max,
                    'cap_shape_t': cap_shape,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SA, UZ, LZ, SNOWPACK, MELTWATER], dim=1)

        sum_p = torch.clamp(P.sum(dim=1), min=1e-6)
        drift_loss = (
            torch.mean(torch.abs(SA - init_sa) / sum_p)
            + torch.mean(torch.abs(UZ - init_uz) / sum_p)
            + torch.mean(torch.abs(LZ - init_lz) / sum_p)
            + torch.mean(torch.abs(SNOWPACK - init_snow) / sum_p)
            + torch.mean(torch.abs(MELTWATER - init_melt) / sum_p)
        )
        reg_terms = {
            'dynamic_amplitude_loss': torch.tensor(0.0, device=device, dtype=dtype),
            'dynamic_smoothness_loss': torch.tensor(0.0, device=device, dtype=dtype),
            'partition_entropy_loss': torch.tensor(0.0, device=device, dtype=dtype),
            'storage_drift_loss': drift_loss,
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out

        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicECOHBVaSrz_HBV11b(MultiInv_DynamicSimHydModelSix_Physical):
    """
    Multi-component wrapper for the ecohydrology-first ECOHBV aSrz + HBV11b core.

    Expected `z` layout:
    - first dynamic normalized forcings
    - optional raw extras: snow_frac, mean_lai, slope
    - final `nattr` static normalized attributes
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=False, lgdynweight=0.0,
                 reg_amp_w=0.0, reg_smooth_w=0.0, reg_part_w=0.0,
                 component_routing=True, dry_channel_loss=False, zero_flow_gate=False,
                 theta_cap_mode='stocker', veg_function='exp', theta_cap_upper=1000.0,
                 drift_reg_weight=1e-3):
        super(MultiInv_DynamicECOHBVaSrz_HBV11b, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=False, dynamic_etgam=False, dynamic_partition=False,
            dynamic_cfmax_snow=False, dynamic_routing_scale=False, dynamic_all=False,
            reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w, reg_part_w=reg_part_w,
            component_routing=component_routing, dry_channel_loss=dry_channel_loss,
            zero_flow_gate=zero_flow_gate, channel_loss_max=0.0)
        self.nfea = 17
        self.nstaticpm = self.nfea * nmul
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        comp_bias = torch.linspace(-0.05, 0.05, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.theta_cap_upper = theta_cap_upper
        self.theta_cap_mode = theta_cap_mode
        self.veg_function = veg_function
        self.drift_reg_weight = drift_reg_weight
        self.simhyd = DynamicECOHBVaSrzHBV11b(
            mode='normal', theta_is_raw=False, smooth=True,
            theta_cap_mode=theta_cap_mode, veg_function=veg_function, cap_upper=theta_cap_upper)
        self.simhyd_analysis = DynamicECOHBVaSrzHBV11b(
            mode='analysis', theta_is_raw=False, smooth=True,
            theta_cap_mode=theta_cap_mode, veg_function=veg_function, cap_upper=theta_cap_upper)

    def _theta_to_smsc(self, theta, ngage):
        theta4 = theta.view(ngage, self.nmul, self.nfea)
        theta_capb = theta4[:, :, 6:7] * self.theta_cap_upper
        return torch.clamp(theta_capb, min=1.0, max=self.theta_cap_upper)

    def _theta_to_k_channel(self, theta, ngage):
        del theta, ngage
        return None

    def _build_theta_cap_scale(self, mean_lai_raw, slope_raw, ngage):
        lai = torch.clamp(mean_lai_raw, min=0.0)
        slope = torch.clamp(slope_raw, min=0.0)
        lai_scale = 0.5 + 0.5 * lai / (lai + 2.0)
        slope_norm = slope / (slope + 10.0)
        slope_scale = torch.exp(-1.5 * slope_norm)
        scale = torch.clamp(lai_scale * slope_scale, min=0.05, max=2.0)
        return scale.unsqueeze(1).repeat(1, self.nmul, 1).view(ngage * self.nmul, 1)

    def get_auxiliary_loss(self):
        aux = super(MultiInv_DynamicECOHBVaSrz_HBV11b, self).get_auxiliary_loss()
        if aux is None:
            return None
        if getattr(self, "_last_aux_terms", None) is None:
            return aux
        return aux + self.drift_reg_weight * self._last_aux_terms.get('storage_drift_loss', 0.0)

    def forward(self, x, z, doDropMC=False, return_diagnostics=False, return_component_diagnostics=False):
        del doDropMC
        nt_x = x.shape[0]
        ngage = z.shape[1]
        basin_attr = z[-1, :, -self.nattr:]

        snow_frac_raw = None
        mean_lai_raw = None
        slope_raw = None
        extra_cols = z.shape[2] - self.nattr
        if extra_cols >= 3:
            snow_frac_raw = torch.clamp(z[-1, :, -self.nattr - 3:-self.nattr - 2], min=0.0, max=1.0)
            mean_lai_raw = torch.clamp(z[-1, :, -self.nattr - 2:-self.nattr - 1], min=0.0)
            slope_raw = torch.clamp(z[-1, :, -self.nattr - 1:-self.nattr], min=0.0)

        staticFeat = self.staticFeat(basin_attr)
        staticParams0 = self.staticOut(staticFeat)

        cursor = 0
        static0 = staticParams0[:, cursor:cursor + self.nstaticpm].view(ngage, self.nfea, self.nmul)
        static0 = static0 + self.compStaticBias
        snowpara = torch.sigmoid(static0)
        cursor += self.nstaticpm

        routpara0 = staticParams0[:, cursor:cursor + self.nroutpm]
        if self.component_routing is True:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        elif self.comprout is False:
            routpara = torch.sigmoid(routpara0)
        else:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        cursor += self.nroutpm

        if self.nwtspm == 0:
            wts = None
        else:
            wtspara = staticParams0[:, cursor:cursor + self.nwtspm] + self.compWeightBias
            wts = F.softmax(wtspara, dim=-1)
        self._last_wts = wts

        x_rep = x.unsqueeze(2).repeat(1, 1, self.nmul, 1).view(nt_x, ngage * self.nmul, x.shape[2])
        x_bt = x_rep.permute(1, 0, 2)
        theta = snowpara.permute(0, 2, 1).contiguous().view(ngage * self.nmul, self.nfea)

        if snow_frac_raw is None:
            snow_frac_rep = None
        else:
            snow_frac_rep = snow_frac_raw.unsqueeze(1).repeat(1, self.nmul, 1).view(ngage * self.nmul, 1)
        if mean_lai_raw is None or slope_raw is None:
            theta_cap_scale = None
        else:
            theta_cap_scale = self._build_theta_cap_scale(mean_lai_raw, slope_raw, ngage)

        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            x_use = x[self.inittime:, :, :]
            _, warm_state = self.simhyd_analysis(
                warm_inputs,
                theta,
                snow_frac_raw=snow_frac_rep,
                theta_cap_scale=theta_cap_scale,
                return_final_state=True)
            q_seq, diag_comp, reg_terms = self.simhyd(
                main_inputs,
                theta,
                initial_state=warm_state,
                snow_frac_raw=snow_frac_rep,
                theta_cap_scale=theta_cap_scale,
                return_diagnostics=True,
                return_regularization=True)
        else:
            x_use = x
            q_seq, diag_comp, reg_terms = self.simhyd(
                x_bt,
                theta,
                snow_frac_raw=snow_frac_rep,
                theta_cap_scale=theta_cap_scale,
                return_diagnostics=True,
                return_regularization=True)

        reg_total = self.reg_amp_w * reg_terms['dynamic_amplitude_loss'] \
            + self.reg_smooth_w * reg_terms['dynamic_smoothness_loss'] \
            + self.reg_part_w * reg_terms['partition_entropy_loss']
        self._last_aux_terms = reg_terms
        self._last_aux_loss = reg_total

        q_comp_raw = q_seq.view(ngage, self.nmul, q_seq.shape[1], 1).permute(2, 0, 1, 3)
        q_comp_raw = torch.clamp(q_comp_raw, min=0.0)

        q_comp = q_comp_raw
        q_mix_before_routing = self._mix_or_mean(q_comp, wts)

        if self.routOpt is True and self.component_routing is True:
            q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
            q_routed = self._route_q(q_for_routing, routpara).view(q_comp.shape[0], ngage, self.nmul, 1)
            out = self._mix_or_mean(q_routed, wts)
        elif self.routOpt is True:
            out = self._route_q(q_mix_before_routing, routpara)
            q_routed = q_comp
        else:
            out = q_mix_before_routing
            q_routed = q_comp

        out = torch.clamp(out, min=0.0)
        if return_diagnostics is not True and return_component_diagnostics is not True:
            return out

        diag_out = {}
        for name, tensor_comp in diag_comp.items():
            diag_out[name] = self._mix_component_tensor(tensor_comp, ngage)
        diag_out['total_discharge'] = out
        diag_out['q_mix_before_routing'] = q_mix_before_routing
        diag_out['component_discharge_raw'] = self._mix_or_mean(q_comp_raw, wts)

        if return_component_diagnostics is True:
            for name, tensor_comp in diag_comp.items():
                diag_out[name + '_components'] = self._component_tensor_4d(tensor_comp, ngage).squeeze(-1)
            diag_out['q_raw_process_components'] = q_comp_raw.squeeze(-1)
            diag_out['q_routed_components'] = q_routed.squeeze(-1)
            diag_out['component_weights'] = wts
            if self.component_routing is True:
                routpara_view = torch.sigmoid(routpara0).view(ngage, self.nmul, 2)
                diag_out['route_a_components'] = 0.0 + routpara_view[:, :, 0] * 2.9
                diag_out['route_b_components'] = 0.0 + routpara_view[:, :, 1] * 6.5

        return out, diag_out


class MultiInv_DynamicSimHydModelSix_Physical_DLearnedRouting(MultiInv_DynamicSimHydModelSix_Physical):
    """
    Copy of the retained soft-gate physical-fix model where only the routing
    function is replaced by a learned, mass-conserving causal kernel.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 component_routing=True, dry_channel_loss=True, zero_flow_gate=True,
                 channel_loss_max=0.60, zero_gate_hidden=None,
                 gate_variant='soft', gate_strength_max=0.30, route_kernel_len=30):
        super(MultiInv_DynamicSimHydModelSix_Physical_DLearnedRouting, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing,
            dry_channel_loss=dry_channel_loss, zero_flow_gate=zero_flow_gate,
            channel_loss_max=channel_loss_max, zero_gate_hidden=zero_gate_hidden,
            gate_variant=gate_variant, gate_strength_max=gate_strength_max)
        self.route_kernel_len = route_kernel_len
        self.route_kernel_head = nn.Linear(hiddeninv, nmul * route_kernel_len)

    def _route_q_learned_kernel(self, qin, kernels):
        """
        qin: [T, N, 1]
        kernels: [N, L], nonnegative and sums to 1
        returns: [T, N, 1]
        """
        Tlen, ngrid, _ = qin.shape
        q_ch = qin.permute(1, 2, 0)  # [N, 1, T]
        q_group = q_ch.permute(1, 0, 2)  # [1, N, T]
        weight = kernels.unsqueeze(1)  # [N, 1, L]
        q_pad = F.pad(q_group, (self.route_kernel_len - 1, 0))
        routed = F.conv1d(q_pad, weight, bias=None, stride=1, padding=0, groups=ngrid)
        routed = routed[:, :, :Tlen]  # [1, N, T]
        return routed.permute(2, 1, 0)  # [T, N, 1]

    def forward(self, x, z, doDropMC=False, return_diagnostics=False, return_component_diagnostics=False):
        nt_x = x.shape[0]
        ngage = z.shape[1]
        basin_attr = z[-1, :, -self.nattr:]

        if z.shape[2] > self.nattr:
            snow_frac_raw = torch.clamp(z[-1, :, -self.nattr - 1:-self.nattr], min=0.0, max=1.0)
        else:
            snow_frac_raw = None

        staticFeat = self.staticFeat(basin_attr)
        staticParams0 = self.staticOut(staticFeat)

        cursor = 0
        static0 = staticParams0[:, cursor:cursor + self.nstaticpm].view(ngage, self.nfea, self.nmul)
        static0 = static0 + self.compStaticBias
        snowpara = torch.sigmoid(static0)
        cursor += self.nstaticpm

        routpara0 = staticParams0[:, cursor:cursor + self.nroutpm]
        cursor += self.nroutpm

        if self.nwtspm == 0:
            wts = None
        else:
            wtspara = staticParams0[:, cursor:cursor + self.nwtspm] + self.compWeightBias
            wts = F.softmax(wtspara, dim=-1)
        self._last_wts = wts

        route_logits = self.route_kernel_head(staticFeat).view(ngage, self.nmul, self.route_kernel_len)
        route_kernel_h = torch.softmax(route_logits, dim=-1)
        lag_idx = torch.arange(self.route_kernel_len, device=route_kernel_h.device, dtype=route_kernel_h.dtype).view(1, 1, self.route_kernel_len)
        mean_route_lag = torch.sum(route_kernel_h * lag_idx, dim=-1)

        if self.lgdyn is False:
            lg_dyn = None
        else:
            lg_dyn_seq = self.lstmdyn(z)
            lg_attr_bias = self.lgAttr(basin_attr).unsqueeze(0).repeat(lg_dyn_seq.shape[0], 1, 1)
            lg_dyn = torch.sigmoid(lg_dyn_seq + lg_attr_bias)

        x_rep = x.unsqueeze(2).repeat(1, 1, self.nmul, 1).view(nt_x, ngage * self.nmul, x.shape[2])
        x_bt = x_rep.permute(1, 0, 2)
        theta = snowpara.permute(0, 2, 1).contiguous().view(ngage * self.nmul, self.nfea)

        if lg_dyn is None:
            lg_bt = None
        else:
            lg_bt = lg_dyn.permute(1, 2, 0).contiguous().view(ngage * self.nmul, lg_dyn.shape[0], 1)
        if lg_bt is not None and self.inittime > 0:
            lg_bt_main = lg_bt[:, self.inittime:, :]
        else:
            lg_bt_main = lg_bt

        if snow_frac_raw is None:
            snow_frac_rep = None
        else:
            snow_frac_rep = snow_frac_raw.unsqueeze(1).repeat(1, self.nmul, 1).view(ngage * self.nmul, 1)

        need_process_diag = return_diagnostics or return_component_diagnostics or self.dry_channel_loss or self.zero_flow_gate_enabled

        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            x_use = x[self.inittime:, :, :]
            _, warm_state = self.simhyd_analysis(
                warm_inputs,
                theta,
                lg_dyn_seq=None,
                lg_dyn_weight=self.lgdynweight,
                snow_frac_raw=snow_frac_rep,
                return_final_state=True)
            if need_process_diag:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    main_inputs,
                    theta,
                    initial_state=warm_state,
                    lg_dyn_seq=lg_bt_main,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_diagnostics=True,
                    return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    main_inputs,
                    theta,
                    initial_state=warm_state,
                    lg_dyn_seq=lg_bt_main,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_regularization=True)
                diag_comp = None
        else:
            x_use = x
            if need_process_diag:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    x_bt,
                    theta,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_diagnostics=True,
                    return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    x_bt,
                    theta,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_regularization=True)
                diag_comp = None

        reg_total = self.reg_amp_w * reg_terms['dynamic_amplitude_loss'] \
            + self.reg_smooth_w * reg_terms['dynamic_smoothness_loss'] \
            + self.reg_part_w * reg_terms['partition_entropy_loss']
        self._last_aux_terms = reg_terms
        self._last_aux_loss = reg_total

        q_comp_raw = q_seq.view(ngage, self.nmul, q_seq.shape[1], 1).permute(2, 0, 1, 3)
        q_comp_raw = torch.clamp(q_comp_raw, min=0.0)
        q_after_loss, channel_loss, channel_loss_fraction = self._apply_channel_loss(q_comp_raw, diag_comp, theta, basin_attr, ngage)
        q_after_gate, zero_flow_probability, gate_loss, zero_flow_keep_fraction = self._apply_zero_flow_gate(
            q_after_loss, x_use, diag_comp, theta, ngage)
        q_comp = torch.clamp(q_after_gate, min=0.0)

        q_mix_before_routing = self._mix_or_mean(q_comp, wts)
        route_mult_seq = None

        if self.routOpt is True and self.component_routing is True:
            q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
            kernel_flat = route_kernel_h.view(ngage * self.nmul, self.route_kernel_len)
            q_routed_flat = self._route_q_learned_kernel(q_for_routing, kernel_flat)
            q_routed = q_routed_flat.view(q_comp.shape[0], ngage, self.nmul, 1)
            out = self._mix_or_mean(q_routed, wts)
        else:
            out = q_mix_before_routing
            q_routed = q_comp

        out = torch.clamp(out, min=0.0)
        if return_diagnostics is not True and return_component_diagnostics is not True:
            return out

        diag_out = dict()
        if diag_comp is not None:
            for name, tensor_comp in diag_comp.items():
                diag_out[name] = self._mix_component_tensor(tensor_comp, ngage)
        diag_out['total_discharge'] = out
        diag_out['q_mix_before_routing'] = q_mix_before_routing
        diag_out['component_discharge_raw'] = self._mix_or_mean(q_comp_raw, wts)
        diag_out['channel_loss'] = self._mix_or_mean(channel_loss, wts)
        diag_out['channel_loss_fraction'] = self._mix_or_mean(channel_loss_fraction, wts)
        diag_out['gate_loss'] = self._mix_or_mean(gate_loss, wts)
        diag_out['zero_flow_probability'] = self._mix_or_mean(zero_flow_probability, wts)
        diag_out['zero_flow_keep_fraction'] = self._mix_or_mean(zero_flow_keep_fraction, wts)
        diag_out['mean_route_lag'] = torch.sum(mean_route_lag * (wts if wts is not None else torch.ones_like(mean_route_lag) / self.nmul), dim=1).unsqueeze(0).unsqueeze(-1).repeat(q_comp.shape[0], 1, 1)
        diag_out['route_kernel_h'] = route_kernel_h

        if return_component_diagnostics is True:
            if diag_comp is not None:
                for name, tensor_comp in diag_comp.items():
                    diag_out[name + '_components'] = self._component_tensor_4d(tensor_comp, ngage).squeeze(-1)
            diag_out['q_raw_process_components'] = q_comp_raw.squeeze(-1)
            diag_out['q_after_channel_loss_components'] = q_after_loss.squeeze(-1)
            diag_out['q_after_gate_components'] = q_after_gate.squeeze(-1)
            diag_out['q_routed_components'] = q_routed.squeeze(-1)
            diag_out['routed_component_Q'] = q_routed.squeeze(-1)
            diag_out['channel_loss_components'] = channel_loss.squeeze(-1)
            diag_out['gate_loss_components'] = gate_loss.squeeze(-1)
            diag_out['channel_loss_fraction_components'] = channel_loss_fraction.squeeze(-1)
            diag_out['zero_flow_probability_components'] = zero_flow_probability.squeeze(-1)
            diag_out['zero_flow_keep_fraction_components'] = zero_flow_keep_fraction.squeeze(-1)
            diag_out['component_weights'] = wts
            diag_out['route_kernel_h_components'] = route_kernel_h
            diag_out['mean_route_lag_components'] = mean_route_lag
            if self.component_routing is True:
                diag_out['route_a_components'] = 0.0 + torch.zeros(ngage, self.nmul, device=route_kernel_h.device, dtype=route_kernel_h.dtype)
                diag_out['route_b_components'] = 0.0 + torch.zeros(ngage, self.nmul, device=route_kernel_h.device, dtype=route_kernel_h.dtype)

        return out, diag_out


class DynamicSimHydModelFiveDifferentiablePhysicalFixDeepGWResidual(DynamicSimHydModelFiveDifferentiablePhysicalFix):
    """
    Conservative deep-groundwater residual variant:
    keep original shallow groundwater/baseflow logic and only redirect the old
    groundwater-loss sink into a capped deep groundwater store with slow return.
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True, dynamic_cfmax_snow=True,
                 dynamic_routing_scale=False, dynamic_all=False, dyn_hidden=32,
                 deep_rech_cap_frac=0.05):
        super(DynamicSimHydModelFiveDifferentiablePhysicalFixDeepGWResidual, self).__init__(
            mode=mode, theta_is_raw=theta_is_raw, smooth=smooth, eps=eps, rain_snow_gain=rain_snow_gain,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, dyn_hidden=dyn_hidden)
        self.deep_rech_cap_frac = deep_rech_cap_frac

    def _expand_residual(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)
        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = 0.0 + theta[:, 2:3] * (6.0 - 0.0)
        SMSC = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = 0.0 + theta[:, 4:5] * (1.0 - 0.0)
        CRAK = 0.0 + theta[:, 5:6] * (1.0 - 0.0)
        K = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        LG = 0.0 + theta[:, 7:8] * (0.2 - 0.0)
        TT = -2.5 + theta[:, 8:9] * (2.5 - (-2.5))
        CFMAX = 0.5 + theta[:, 9:10] * (10.0 - 0.5)
        CFR = 0.0 + theta[:, 10:11] * (0.1 - 0.0)
        CWH = 0.0 + theta[:, 11:12] * (0.2 - 0.0)
        SG_CRIT = 0.0 + theta[:, 12:13] * (300.0 - 0.0)
        K_DEEP = 0.001 + theta[:, 13:14] * (0.01 - 0.001)
        return INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH, SG_CRIT, K_DEEP

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH, SG_CRIT, K_DEEP = self._expand_residual(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SMS = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
            DEEPGW = torch.zeros(B, 1, device=device, dtype=dtype)
        else:
            SMS = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]
            SNOWPACK = initial_state[:, 2:3]
            MELTWATER = initial_state[:, 3:4]
            DEEPGW = initial_state[:, 4:5]

        if lg_dyn_seq is not None and (lg_dyn_seq.shape[0] != B or lg_dyn_seq.shape[1] != Tlen):
            raise ValueError('lg_dyn_seq must have shape [B, T, 1] matching inputs')

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = dict()
        if return_diagnostics is True:
            for name in [
                    'SMS_prev', 'GW_prev', 'DEEPGW_prev', 'SNOWPACK_prev', 'MELTWATER_prev',
                    'SMS', 'GW', 'DEEPGW', 'SNOWPACK', 'MELTWATER',
                    'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release_to_soil',
                    'interception_storage', 'interception_evaporation', 'actual_ET', 'infiltration', 'soil_mobile_water',
                    'surface_runoff', 'interflow', 'recharge_to_groundwater', 'soil_overflow',
                    'baseflow_raw', 'baseflow_capped', 'baseflow', 'deep_baseflow',
                    'groundwater_loss_raw', 'groundwater_loss_capped', 'groundwater_loss', 'GWLOSS_external',
                    'DEEP_RECH', 'deep_recharge', 'Q_DEEP', 'DEEPLOSS',
                    'channel_loss', 'gate_loss', 'q_raw_process', 'q_after_channel_loss', 'q_after_gate',
                    'total_discharge', 'INSC', 'COEF_t', 'SQ_t', 'SMSC', 'ETGAM_t', 'SUB_t', 'INTER_t', 'RECH_t',
                    'CRAK_t', 'K_t', 'LG_t', 'K_DEEP', 'TT', 'SG_CRIT', 'CFMAX_t', 'CFR', 'CWH',
                    'partition_sum_error', 'GW_avail_before', 'GW_avail_after',
                    'deep_local_residual', 'gw_local_residual', 'soil_local_residual',
                    'snowpack_local_residual', 'meltwater_local_residual', 'snow_total_local_residual']:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SMS0 = self._min(self._pos(SMS), SMSC)
            GW0 = self._pos(GW)
            DEEPGW0 = self._pos(DEEPGW)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            wetness = torch.clamp(SMS0 / (SMSC + 1e-8), min=0.0, max=1.5)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0,
                Tt / 20.0,
                E0t / 10.0,
                SMS0 / 300.0,
                wetness,
                GW0 / 300.0,
                SNOWPACK0 / 300.0,
                sin_t,
                cos_t], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)

            if self.dynamic_etgam is True:
                ETGAM_t = 0.25 + 3.75 * torch.sigmoid(dyn_raw[:, self.dyn_slices['etgam']])
            else:
                ETGAM_t = torch.ones_like(SQ)

            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX * m_cf_eff

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()

            RAIN = Pt * frac_rain
            SNOW = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + SNOW
            melt_pot = CFMAX_t * self._pos(Tt - TT)
            melt = self._min(melt_pot, SNOWPACK1)
            MELTWATER1 = MELTWATER0 + melt
            SNOWPACK2 = self._pos(SNOWPACK1 - melt)

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            SNOWPACK3 = SNOWPACK2 + refreezing
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)

            water_holding = CWH * SNOWPACK3
            tosoil_raw = self._pos(MELTWATER2 - water_holding)
            tosoil = self._min(tosoil_raw, MELTWATER2)
            MELTWATER_next = self._pos(MELTWATER2 - tosoil)
            SNOWPACK_next = self._pos(SNOWPACK3)
            Peff = RAIN + tosoil

            IMAX = self._min(INSC, E0t)
            INT = self._min(IMAX, Peff)
            INR = self._pos(Peff - INT)

            infil_cap = COEF * torch.exp(-SQ_t * wetness)
            RMO = self._min(infil_cap, INR)
            IRUN_excess = self._pos(INR - RMO)

            available = wetness * RMO
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_inter = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([
                torch.log(base_surface),
                torch.log(base_inter),
                torch.log(base_recharge)], dim=1)
            if self.dynamic_partition is True:
                part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']]
            else:
                part_logits = base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]

            SRUN = IRUN_excess + f_surface * available
            IFLOW = f_inter * available
            REC = f_recharge * available
            SMF = self._pos(RMO - available)

            POT = self._pos(E0t - INT)
            et_scale = torch.pow(torch.clamp(wetness, min=1e-6, max=1.0), ETGAM_t)
            et_scale = torch.clamp(et_scale, max=1.0)
            ETS_raw = POT * et_scale
            ETS = self._min(ETS_raw, SMS0 + SMF)
            ETS = self._min(ETS, POT)

            SMS_pre = SMS0 + SMF - ETS
            SOIL_EXCESS = self._pos(SMS_pre - SMSC)
            SMS_next = self._min(self._pos(SMS_pre), SMSC)
            REC_total = REC + SOIL_EXCESS

            if lg_dyn_seq is None:
                LG_t = LG
            else:
                LG_dyn_t = torch.clamp(lg_dyn_seq[:, t, :], 0.0, 1.0)
                LG_eff = (1.0 - lg_dyn_weight) * LG + lg_dyn_weight * (LG * LG_dyn_t * 1.5)
                LG_t = torch.clamp(LG_eff, 0.0, 0.2)

            GW_avail_before = self._pos(GW0 + REC_total)
            BAS_raw = K * F.softplus(GW0 - SG_CRIT)
            BAS = self._min(BAS_raw, GW_avail_before)
            GW_avail_after = self._pos(GW_avail_before - BAS)

            DEEP_RECH_raw = LG_t * F.softplus(SG_CRIT - GW0)
            DEEP_RECH = self._min(DEEP_RECH_raw, GW_avail_after)
            DEEP_RECH = self._min(DEEP_RECH, self.deep_rech_cap_frac * GW_avail_after)

            GW_next = self._pos(GW_avail_after - DEEP_RECH)
            Q_DEEP = self._min(K_DEEP * DEEPGW0, DEEPGW0)
            DEEPGW_next = self._pos(DEEPGW0 + DEEP_RECH - Q_DEEP)
            DEEPLOSS = torch.zeros_like(Q_DEEP)
            GWLOSS_external = DEEPLOSS
            Q = self._pos(SRUN + IFLOW + BAS + Q_DEEP)

            deep_local_residual = DEEP_RECH - Q_DEEP - DEEPLOSS - (DEEPGW_next - DEEPGW0)
            gw_local_residual = REC_total - BAS - DEEP_RECH - (GW_next - GW0)
            soil_local_residual = SMF - ETS - SOIL_EXCESS - (SMS_next - SMS0)
            snowpack_local_residual = SNOW - melt + refreezing - (SNOWPACK_next - SNOWPACK0)
            meltwater_local_residual = melt - refreezing - tosoil - (MELTWATER_next - MELTWATER0)
            snow_total_local_residual = (
                SNOW - tosoil
                - (SNOWPACK_next - SNOWPACK0)
                - (MELTWATER_next - MELTWATER0)
            )

            SMS = SMS_next
            GW = GW_next
            DEEPGW = DEEPGW_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)

            if return_diagnostics is True:
                diag_vals = {
                    'SMS_prev': SMS0,
                    'GW_prev': GW0,
                    'DEEPGW_prev': DEEPGW0,
                    'SNOWPACK_prev': SNOWPACK0,
                    'MELTWATER_prev': MELTWATER0,
                    'SMS': SMS_next,
                    'GW': GW_next,
                    'DEEPGW': DEEPGW_next,
                    'SNOWPACK': SNOWPACK_next,
                    'MELTWATER': MELTWATER_next,
                    'precipitation': Pt,
                    'rainfall': RAIN,
                    'snowfall': SNOW,
                    'snowmelt': melt,
                    'refreezing': refreezing,
                    'snow_release_to_soil': tosoil,
                    'interception_storage': torch.zeros_like(INT),
                    'interception_evaporation': INT,
                    'actual_ET': ETS,
                    'infiltration': RMO,
                    'soil_mobile_water': SMF,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'recharge_to_groundwater': REC_total,
                    'soil_overflow': SOIL_EXCESS,
                    'baseflow_raw': BAS_raw,
                    'baseflow_capped': BAS,
                    'baseflow': BAS,
                    'deep_baseflow': Q_DEEP,
                    'groundwater_loss_raw': DEEP_RECH_raw,
                    'groundwater_loss_capped': GWLOSS_external,
                    'groundwater_loss': GWLOSS_external,
                    'GWLOSS_external': GWLOSS_external,
                    'DEEP_RECH': DEEP_RECH,
                    'deep_recharge': DEEP_RECH,
                    'Q_DEEP': Q_DEEP,
                    'DEEPLOSS': DEEPLOSS,
                    'channel_loss': torch.zeros_like(Q),
                    'gate_loss': torch.zeros_like(Q),
                    'q_raw_process': Q,
                    'q_after_channel_loss': Q,
                    'q_after_gate': Q,
                    'total_discharge': Q,
                    'INSC': INSC,
                    'COEF_t': COEF,
                    'SQ_t': SQ_t,
                    'SMSC': SMSC,
                    'ETGAM_t': ETGAM_t,
                    'SUB_t': f_surface,
                    'INTER_t': f_inter,
                    'RECH_t': f_recharge,
                    'CRAK_t': f_recharge,
                    'K_t': K,
                    'LG_t': LG_t,
                    'K_DEEP': K_DEEP,
                    'TT': TT,
                    'SG_CRIT': SG_CRIT,
                    'CFMAX_t': CFMAX_t,
                    'CFR': CFR,
                    'CWH': CWH,
                    'partition_sum_error': torch.abs(f_surface + f_inter + f_recharge - 1.0),
                    'GW_avail_before': GW_avail_before,
                    'GW_avail_after': GW_avail_after,
                    'deep_local_residual': deep_local_residual,
                    'gw_local_residual': gw_local_residual,
                    'soil_local_residual': soil_local_residual,
                    'snowpack_local_residual': snowpack_local_residual,
                    'meltwater_local_residual': meltwater_local_residual,
                    'snow_total_local_residual': snow_total_local_residual,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SMS, GW, SNOWPACK, MELTWATER, DEEPGW], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)

        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': reg_part
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out

        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelSix_Physical_C2DeepGWResidual(MultiInv_DynamicSimHydModelSix_Physical):
    """
    Copy of the retained soft-gate physical-fix model where groundwater loss
    becomes capped recharge to a long-memory deep groundwater store.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 component_routing=True, dry_channel_loss=True, zero_flow_gate=True,
                 channel_loss_max=0.60, zero_gate_hidden=None,
                 gate_variant='soft', gate_strength_max=0.30, deep_rech_cap_frac=0.05):
        super(MultiInv_DynamicSimHydModelSix_Physical_C2DeepGWResidual, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing,
            dry_channel_loss=dry_channel_loss, zero_flow_gate=zero_flow_gate,
            channel_loss_max=channel_loss_max, zero_gate_hidden=zero_gate_hidden,
            gate_variant=gate_variant, gate_strength_max=gate_strength_max)
        self.nfea = 14
        self.nstaticpm = self.nfea * nmul
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.simhyd = DynamicSimHydModelFiveDifferentiablePhysicalFixDeepGWResidual(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            deep_rech_cap_frac=deep_rech_cap_frac)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiablePhysicalFixDeepGWResidual(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            deep_rech_cap_frac=deep_rech_cap_frac)


class DynamicSimHydModelFiveDifferentiablePhysicalFixDeepGWResidualFinetune(DynamicSimHydModelFiveDifferentiablePhysicalFix):
    """
    C2b variant: keep the original shallow-GW/baseflow logic from the retained
    soft-gate model, reinterpret groundwater loss as capped recharge to a deep
    groundwater store, and add a learnable alpha_deep scaling factor.
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True, dynamic_cfmax_snow=True,
                 dynamic_routing_scale=False, dynamic_all=False, dyn_hidden=32,
                 deep_rech_cap_frac=0.05):
        super(DynamicSimHydModelFiveDifferentiablePhysicalFixDeepGWResidualFinetune, self).__init__(
            mode=mode, theta_is_raw=theta_is_raw, smooth=smooth, eps=eps, rain_snow_gain=rain_snow_gain,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, dyn_hidden=dyn_hidden)
        self.deep_rech_cap_frac = deep_rech_cap_frac

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.0,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        P = inputs[:, :, 0:1]
        TEMP = inputs[:, :, 1:2]
        E0 = inputs[:, :, 2:3]
        device = inputs.device
        dtype = inputs.dtype
        B, Tlen, _ = inputs.shape

        if initial_state is None:
            SMS = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
            DEEPGW = torch.zeros(B, 1, device=device, dtype=dtype)
        else:
            nstate = initial_state.shape[1]
            SMS = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]
            SNOWPACK = initial_state[:, 2:3]
            MELTWATER = initial_state[:, 3:4]
            if nstate >= 5:
                DEEPGW = initial_state[:, 4:5]
            else:
                DEEPGW = torch.zeros_like(GW)

        if self.theta_is_raw:
            theta = torch.sigmoid(theta)
        theta = torch.clamp(theta, 0.0, 1.0)
        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = theta[:, 2:3] * 6.0
        SMSC = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = theta[:, 4:5]
        CRAK = theta[:, 5:6]
        K = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        LG = theta[:, 7:8] * 0.2
        TT = -2.5 + theta[:, 8:9] * 5.0
        CFMAX_base = 0.5 + theta[:, 9:10] * (10.0 - 0.5)
        CFR = theta[:, 10:11] * 0.1
        CWH = theta[:, 11:12] * 0.2
        SG_CRIT = theta[:, 12:13] * 300.0
        K_DEEP = 0.001 + theta[:, 13:14] * (0.01 - 0.001)
        alpha_deep = theta[:, 14:15]

        q_hist = []
        diag_hist = dict()
        if return_diagnostics is True:
            for name in [
                    'SMS_prev', 'GW_prev', 'DEEPGW_prev', 'SNOWPACK_prev', 'MELTWATER_prev',
                    'SMS', 'GW', 'DEEPGW', 'SNOWPACK', 'MELTWATER',
                    'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release_to_soil',
                    'interception_storage', 'interception_evaporation', 'actual_ET', 'infiltration', 'soil_mobile_water',
                    'surface_runoff', 'interflow', 'recharge_to_groundwater', 'soil_overflow',
                    'baseflow_raw', 'baseflow_capped', 'baseflow', 'deep_baseflow',
                    'groundwater_loss_raw', 'groundwater_loss_capped', 'groundwater_loss', 'GWLOSS_external',
                    'DEEP_RECH', 'deep_recharge', 'Q_DEEP', 'DEEPLOSS', 'alpha_deep',
                    'channel_loss', 'gate_loss', 'q_raw_process', 'q_after_channel_loss', 'q_after_gate',
                    'total_discharge', 'INSC', 'COEF_t', 'SQ_t', 'SMSC', 'ETGAM_t', 'SUB_t', 'INTER_t', 'RECH_t',
                    'CRAK_t', 'K_t', 'LG_t', 'K_DEEP', 'TT', 'SG_CRIT', 'CFMAX_t', 'CFR', 'CWH',
                    'partition_sum_error', 'GW_avail_before', 'GW_avail_after',
                    'deep_local_residual', 'gw_local_residual', 'soil_local_residual',
                    'snowpack_local_residual', 'meltwater_local_residual', 'snow_total_local_residual',
                    'process_local_residual']:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SMS0 = self._min(self._pos(SMS), SMSC)
            GW0 = self._pos(GW)
            DEEPGW0 = self._pos(DEEPGW)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            wetness = torch.clamp(SMS0 / (SMSC + 1e-8), min=0.0, max=1.5)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0,
                Tt / 20.0,
                E0t / 10.0,
                SMS0 / 300.0,
                wetness,
                GW0 / 300.0,
                SNOWPACK0 / 300.0,
                sin_t,
                cos_t
            ], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)

            if self.dynamic_etgam is True:
                ETGAM_t = 0.25 + 3.75 * torch.sigmoid(dyn_raw[:, self.dyn_slices['etgam']])
            else:
                ETGAM_t = torch.ones_like(SQ)

            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX_base * m_cf_eff

            rain_frac = torch.sigmoid((Tt - TT) * self.rain_snow_gain)
            rain_frac = snow_mask * rain_frac + (1.0 - snow_mask)
            RAIN = Pt * rain_frac
            SNOW = Pt - RAIN

            melt_pot = CFMAX_t * self._pos(Tt - TT)
            melt = self._min(SNOWPACK0, melt_pot)
            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(MELTWATER0, refreeze_pot)
            SNOWPACK_next = self._pos(SNOWPACK0 + SNOW - melt + refreezing)
            MELTWATER_pre = self._pos(MELTWATER0 + melt - refreezing)
            tosoil = self._pos(MELTWATER_pre - CWH * SNOWPACK_next)
            tosoil = self._min(tosoil, MELTWATER_pre)
            MELTWATER_next = self._pos(MELTWATER_pre - tosoil)

            P_eff = RAIN + tosoil

            IMAX = self._min(INSC, E0t)
            INT = self._min(IMAX, P_eff)
            INR = self._pos(P_eff - INT)
            infil_cap = COEF * torch.exp(-SQ_t * wetness)
            RMO = self._min(infil_cap, INR)
            IRUN_excess = self._pos(INR - RMO)

            available = wetness * RMO
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_inter = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([torch.log(base_surface), torch.log(base_inter), torch.log(base_recharge)], dim=1)
            if self.dynamic_partition is True and dyn_raw is not None:
                part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']]
            else:
                part_logits = base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]

            SRUN = IRUN_excess + f_surface * available
            IFLOW = f_inter * available
            REC = f_recharge * available
            SMF = self._pos(RMO - available)

            POT = self._pos(E0t - INT)
            et_scale = torch.pow(torch.clamp(wetness, min=1e-6, max=1.0), ETGAM_t)
            et_scale = torch.clamp(et_scale, max=1.0)
            ETS_raw = POT * et_scale
            ETS = self._min(ETS_raw, SMS0 + SMF)
            ETS = self._min(ETS, POT)

            SMS_pre = SMS0 + SMF - ETS
            SOIL_EXCESS = self._pos(SMS_pre - SMSC)
            SMS_next = self._min(self._pos(SMS_pre), SMSC)
            REC_total = REC + SOIL_EXCESS

            if lg_dyn_seq is None:
                LG_t = LG
            else:
                LG_dyn_t = torch.clamp(lg_dyn_seq[:, t, :], 0.0, 1.0)
                LG_eff = (1.0 - lg_dyn_weight) * LG + lg_dyn_weight * (LG * LG_dyn_t * 1.5)
                LG_t = torch.clamp(LG_eff, 0.0, 0.2)

            GW_avail_before = self._pos(GW0 + REC_total)
            BAS_raw = K * F.softplus(GW0 - SG_CRIT)
            BAS = self._min(BAS_raw, GW_avail_before)
            GW_avail_after = self._pos(GW_avail_before - BAS)

            DEEP_RECH_raw = LG_t * F.softplus(SG_CRIT - GW0)
            DEEP_RECH = alpha_deep * DEEP_RECH_raw
            DEEP_RECH = self._min(DEEP_RECH, GW_avail_after)
            DEEP_RECH = self._min(DEEP_RECH, self.deep_rech_cap_frac * GW_avail_after)

            GW_next = self._pos(GW_avail_after - DEEP_RECH)
            Q_DEEP = self._min(K_DEEP * DEEPGW0, DEEPGW0)
            DEEPGW_next = self._pos(DEEPGW0 + DEEP_RECH - Q_DEEP)
            DEEPLOSS = torch.zeros_like(Q_DEEP)
            GWLOSS_external = DEEPLOSS
            Q = self._pos(SRUN + IFLOW + BAS + Q_DEEP)

            deep_local_residual = DEEP_RECH - Q_DEEP - DEEPLOSS - (DEEPGW_next - DEEPGW0)
            gw_local_residual = REC_total - BAS - DEEP_RECH - (GW_next - GW0)
            soil_local_residual = SMF - ETS - SOIL_EXCESS - (SMS_next - SMS0)
            snowpack_local_residual = SNOW - melt + refreezing - (SNOWPACK_next - SNOWPACK0)
            meltwater_local_residual = melt - refreezing - tosoil - (MELTWATER_next - MELTWATER0)
            snow_total_local_residual = SNOW - tosoil - (SNOWPACK_next - SNOWPACK0) - (MELTWATER_next - MELTWATER0)
            process_local_residual = (
                Pt - INT - ETS - DEEPLOSS - Q
                - ((SMS_next - SMS0) + (GW_next - GW0) + (DEEPGW_next - DEEPGW0)
                   + (SNOWPACK_next - SNOWPACK0) + (MELTWATER_next - MELTWATER0))
            )

            SMS = SMS_next
            GW = GW_next
            DEEPGW = DEEPGW_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)

            if return_diagnostics is True:
                diag_vals = {
                    'SMS_prev': SMS0,
                    'GW_prev': GW0,
                    'DEEPGW_prev': DEEPGW0,
                    'SNOWPACK_prev': SNOWPACK0,
                    'MELTWATER_prev': MELTWATER0,
                    'SMS': SMS_next,
                    'GW': GW_next,
                    'DEEPGW': DEEPGW_next,
                    'SNOWPACK': SNOWPACK_next,
                    'MELTWATER': MELTWATER_next,
                    'precipitation': Pt,
                    'rainfall': RAIN,
                    'snowfall': SNOW,
                    'snowmelt': melt,
                    'refreezing': refreezing,
                    'snow_release_to_soil': tosoil,
                    'interception_storage': torch.zeros_like(INT),
                    'interception_evaporation': INT,
                    'actual_ET': ETS,
                    'infiltration': RMO,
                    'soil_mobile_water': SMF,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'recharge_to_groundwater': REC_total,
                    'soil_overflow': SOIL_EXCESS,
                    'baseflow_raw': BAS_raw,
                    'baseflow_capped': BAS,
                    'baseflow': BAS,
                    'deep_baseflow': Q_DEEP,
                    'groundwater_loss_raw': DEEP_RECH_raw,
                    'groundwater_loss_capped': GWLOSS_external,
                    'groundwater_loss': GWLOSS_external,
                    'GWLOSS_external': GWLOSS_external,
                    'DEEP_RECH': DEEP_RECH,
                    'deep_recharge': DEEP_RECH,
                    'Q_DEEP': Q_DEEP,
                    'DEEPLOSS': DEEPLOSS,
                    'alpha_deep': alpha_deep,
                    'channel_loss': torch.zeros_like(Q),
                    'gate_loss': torch.zeros_like(Q),
                    'q_raw_process': Q,
                    'q_after_channel_loss': Q,
                    'q_after_gate': Q,
                    'total_discharge': Q,
                    'INSC': INSC,
                    'COEF_t': COEF,
                    'SQ_t': SQ_t,
                    'SMSC': SMSC,
                    'ETGAM_t': ETGAM_t,
                    'SUB_t': f_surface,
                    'INTER_t': f_inter,
                    'RECH_t': f_recharge,
                    'CRAK_t': f_recharge,
                    'K_t': K,
                    'LG_t': LG_t,
                    'K_DEEP': K_DEEP,
                    'TT': TT,
                    'SG_CRIT': SG_CRIT,
                    'CFMAX_t': CFMAX_t,
                    'CFR': CFR,
                    'CWH': CWH,
                    'partition_sum_error': torch.abs(f_surface + f_inter + f_recharge - 1.0),
                    'GW_avail_before': GW_avail_before,
                    'GW_avail_after': GW_avail_after,
                    'deep_local_residual': deep_local_residual,
                    'gw_local_residual': gw_local_residual,
                    'soil_local_residual': soil_local_residual,
                    'snowpack_local_residual': snowpack_local_residual,
                    'meltwater_local_residual': meltwater_local_residual,
                    'snow_total_local_residual': snow_total_local_residual,
                    'process_local_residual': process_local_residual,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SMS, GW, SNOWPACK, MELTWATER, DEEPGW], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)
        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': reg_part
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out
        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelSix_Physical_C2bDeepGWResidualFinetune(MultiInv_DynamicSimHydModelSix_Physical):
    """
    C2b copy of the retained soft-gate model: keep original shallow GW logic,
    convert GWLOSS into internal deep recharge, and fine-tune only K_DEEP and
    alpha_deep first before unfreezing the full model.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 component_routing=True, dry_channel_loss=True, zero_flow_gate=True,
                 channel_loss_max=0.60, zero_gate_hidden=None,
                 gate_variant='soft', gate_strength_max=0.30, deep_rech_cap_frac=0.05):
        super(MultiInv_DynamicSimHydModelSix_Physical_C2bDeepGWResidualFinetune, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing,
            dry_channel_loss=dry_channel_loss, zero_flow_gate=zero_flow_gate,
            channel_loss_max=channel_loss_max, zero_gate_hidden=zero_gate_hidden,
            gate_variant=gate_variant, gate_strength_max=gate_strength_max)
        self.nfea = 15
        self.nstaticpm = self.nfea * nmul
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.simhyd = DynamicSimHydModelFiveDifferentiablePhysicalFixDeepGWResidualFinetune(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            deep_rech_cap_frac=deep_rech_cap_frac)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiablePhysicalFixDeepGWResidualFinetune(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            deep_rech_cap_frac=deep_rech_cap_frac)


class DynamicSimHydModelFiveDifferentiablePhysicalFixDeepGWSlowReturn(DynamicSimHydModelFiveDifferentiablePhysicalFix):
    """
    Physical-fix variant that converts the old groundwater loss sink into
    internal deep-groundwater recharge with slow return to discharge.
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True, dynamic_cfmax_snow=True,
                 dynamic_routing_scale=False, dynamic_all=False, dyn_hidden=32,
                 k_deep_min=0.0001, k_deep_max=0.01, deep_q_to_stream_frac=1.0, deep_recharge_frac=1.0):
        super(DynamicSimHydModelFiveDifferentiablePhysicalFixDeepGWSlowReturn, self).__init__(
            mode=mode, theta_is_raw=theta_is_raw, smooth=smooth, eps=eps, rain_snow_gain=rain_snow_gain,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, dyn_hidden=dyn_hidden)
        self.k_deep_min = k_deep_min
        self.k_deep_max = k_deep_max
        self.deep_q_to_stream_frac = deep_q_to_stream_frac
        self.deep_recharge_frac = deep_recharge_frac

    def _expand_deep(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)
        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = 0.0 + theta[:, 2:3] * (6.0 - 0.0)
        SMSC = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = 0.0 + theta[:, 4:5] * (1.0 - 0.0)
        CRAK = 0.0 + theta[:, 5:6] * (1.0 - 0.0)
        K = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        LG = 0.0 + theta[:, 7:8] * (0.2 - 0.0)
        TT = -2.5 + theta[:, 8:9] * (2.5 - (-2.5))
        CFMAX = 0.5 + theta[:, 9:10] * (10.0 - 0.5)
        CFR = 0.0 + theta[:, 10:11] * (0.1 - 0.0)
        CWH = 0.0 + theta[:, 11:12] * (0.2 - 0.0)
        SG_CRIT = 0.0 + theta[:, 12:13] * (300.0 - 0.0)
        K_DEEP = self.k_deep_min + theta[:, 13:14] * (self.k_deep_max - self.k_deep_min)
        return INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH, SG_CRIT, K_DEEP

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH, SG_CRIT, K_DEEP = self._expand_deep(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SMS = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
            DEEPGW = torch.zeros(B, 1, device=device, dtype=dtype)
        else:
            SMS = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]
            SNOWPACK = initial_state[:, 2:3]
            MELTWATER = initial_state[:, 3:4]
            DEEPGW = initial_state[:, 4:5]

        if lg_dyn_seq is not None and (lg_dyn_seq.shape[0] != B or lg_dyn_seq.shape[1] != Tlen):
            raise ValueError('lg_dyn_seq must have shape [B, T, 1] matching inputs')

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = dict()
        if return_diagnostics is True:
            for name in [
                    'SMS', 'GW', 'DEEPGW', 'SNOWPACK', 'MELTWATER',
                    'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release_to_soil',
                    'interception_storage', 'interception_evaporation', 'actual_ET', 'infiltration',
                    'surface_runoff', 'interflow', 'recharge_to_groundwater', 'soil_overflow',
                    'baseflow_raw', 'baseflow_capped', 'baseflow',
                    'groundwater_loss_raw', 'groundwater_loss_capped', 'groundwater_loss',
                    'DEEP_RECH', 'DEEP_RECH_FULL', 'deep_recharge', 'Q_DEEP', 'Q_DEEP_FULL', 'Q_DEEP_TO_GW', 'DEEPLOSS',
                    'channel_loss', 'gate_loss', 'q_raw_process', 'q_after_channel_loss', 'q_after_gate',
                    'total_discharge', 'INSC', 'COEF_t', 'SQ_t', 'SMSC', 'ETGAM_t', 'SUB_t', 'INTER_t', 'RECH_t',
                    'CRAK_t', 'K_t', 'LG_t', 'K_DEEP', 'TT', 'SG_CRIT', 'CFMAX_t', 'CFR', 'CWH',
                    'partition_sum_error', 'available_before_baseflow', 'available_after_baseflow']:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SMS0 = self._min(self._pos(SMS), SMSC)
            GW0 = self._pos(GW)
            DEEPGW0 = self._pos(DEEPGW)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            wetness = torch.clamp(SMS0 / (SMSC + 1e-8), min=0.0, max=1.5)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0,
                Tt / 20.0,
                E0t / 10.0,
                SMS0 / 300.0,
                wetness,
                GW0 / 300.0,
                SNOWPACK0 / 300.0,
                sin_t,
                cos_t], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)

            if self.dynamic_etgam is True:
                ETGAM_t = 0.25 + 3.75 * torch.sigmoid(dyn_raw[:, self.dyn_slices['etgam']])
            else:
                ETGAM_t = torch.ones_like(SQ)

            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX * m_cf_eff

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()

            RAIN = Pt * frac_rain
            SNOW = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + SNOW
            melt_pot = CFMAX_t * self._pos(Tt - TT)
            melt = self._min(melt_pot, SNOWPACK1)
            MELTWATER1 = MELTWATER0 + melt
            SNOWPACK2 = self._pos(SNOWPACK1 - melt)

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            SNOWPACK3 = SNOWPACK2 + refreezing
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)

            water_holding = CWH * SNOWPACK3
            tosoil_raw = self._pos(MELTWATER2 - water_holding)
            tosoil = self._min(tosoil_raw, MELTWATER2)
            MELTWATER_next = self._pos(MELTWATER2 - tosoil)
            SNOWPACK_next = self._pos(SNOWPACK3)
            Peff = RAIN + tosoil

            IMAX = self._min(INSC, E0t)
            INT = self._min(IMAX, Peff)
            INR = self._pos(Peff - INT)

            infil_cap = COEF * torch.exp(-SQ_t * wetness)
            RMO = self._min(infil_cap, INR)
            IRUN_excess = self._pos(INR - RMO)

            available = wetness * RMO
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_inter = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([
                torch.log(base_surface),
                torch.log(base_inter),
                torch.log(base_recharge)], dim=1)
            if self.dynamic_partition is True:
                part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']]
            else:
                part_logits = base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]

            SRUN = IRUN_excess + f_surface * available
            IFLOW = f_inter * available
            REC = f_recharge * available
            SMF = self._pos(RMO - available)

            POT = self._pos(E0t - INT)
            et_scale = torch.pow(torch.clamp(wetness, min=1e-6, max=1.0), ETGAM_t)
            et_scale = torch.clamp(et_scale, max=1.0)
            ETS_raw = POT * et_scale
            ETS = self._min(ETS_raw, SMS0 + SMF)
            ETS = self._min(ETS, POT)

            SMS_pre = SMS0 + SMF - ETS
            SOIL_EXCESS = self._pos(SMS_pre - SMSC)
            SMS_next = self._min(self._pos(SMS_pre), SMSC)
            REC_total = REC + SOIL_EXCESS

            if lg_dyn_seq is None:
                LG_t = LG
            else:
                LG_dyn_t = torch.clamp(lg_dyn_seq[:, t, :], 0.0, 1.0)
                LG_eff = (1.0 - lg_dyn_weight) * LG + lg_dyn_weight * (LG * LG_dyn_t * 1.5)
                LG_t = torch.clamp(LG_eff, 0.0, 0.2)

            available_before_baseflow = self._pos(GW0 + REC_total)
            BAS_raw = K * F.softplus(GW0 - SG_CRIT)
            BAS = self._min(BAS_raw, available_before_baseflow)
            available_after_baseflow = self._pos(available_before_baseflow - BAS)

            DEEP_RECH_raw = LG_t * F.softplus(SG_CRIT - GW0)
            DEEP_RECH_full = self._min(DEEP_RECH_raw, available_after_baseflow)
            DEEP_RECH = self.deep_recharge_frac * DEEP_RECH_full
            GW_retained = available_after_baseflow - DEEP_RECH

            Q_DEEP_raw = K_DEEP * DEEPGW0
            Q_DEEP_FULL = self._min(Q_DEEP_raw, DEEPGW0)
            Q_DEEP = self.deep_q_to_stream_frac * Q_DEEP_FULL
            Q_DEEP_TO_GW = self._pos(Q_DEEP_FULL - Q_DEEP)
            GW_next = self._pos(GW_retained + Q_DEEP_TO_GW)
            DEEPGW_next = self._pos(DEEPGW0 + DEEP_RECH - Q_DEEP_FULL)
            DEEPLOSS = torch.zeros_like(Q_DEEP)
            Q = self._pos(SRUN + IFLOW + BAS + Q_DEEP)

            SMS = SMS_next
            GW = GW_next
            DEEPGW = DEEPGW_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)

            if return_diagnostics is True:
                diag_vals = {
                    'SMS': SMS_next,
                    'GW': GW_next,
                    'DEEPGW': DEEPGW_next,
                    'SNOWPACK': SNOWPACK_next,
                    'MELTWATER': MELTWATER_next,
                    'precipitation': Pt,
                    'rainfall': RAIN,
                    'snowfall': SNOW,
                    'snowmelt': melt,
                    'refreezing': refreezing,
                    'snow_release_to_soil': tosoil,
                    'interception_storage': torch.zeros_like(INT),
                    'interception_evaporation': INT,
                    'actual_ET': ETS,
                    'infiltration': RMO,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'recharge_to_groundwater': REC_total,
                    'soil_overflow': SOIL_EXCESS,
                    'baseflow_raw': BAS_raw,
                    'baseflow_capped': BAS,
                    'baseflow': BAS + Q_DEEP,
                    'groundwater_loss_raw': torch.zeros_like(DEEPLOSS),
                    'groundwater_loss_capped': DEEPLOSS,
                    'groundwater_loss': DEEPLOSS,
                    'DEEP_RECH': DEEP_RECH,
                    'deep_recharge': DEEP_RECH,
                    'Q_DEEP': Q_DEEP,
                    'Q_DEEP_FULL': Q_DEEP_FULL,
                    'Q_DEEP_TO_GW': Q_DEEP_TO_GW,
                    'DEEP_RECH_FULL': DEEP_RECH_full,
                    'DEEPLOSS': DEEPLOSS,
                    'channel_loss': torch.zeros_like(Q),
                    'gate_loss': torch.zeros_like(Q),
                    'q_raw_process': Q,
                    'q_after_channel_loss': Q,
                    'q_after_gate': Q,
                    'total_discharge': Q,
                    'INSC': INSC,
                    'COEF_t': COEF,
                    'SQ_t': SQ_t,
                    'SMSC': SMSC,
                    'ETGAM_t': ETGAM_t,
                    'SUB_t': f_surface,
                    'INTER_t': f_inter,
                    'RECH_t': f_recharge,
                    'CRAK_t': f_recharge,
                    'K_t': K,
                    'LG_t': LG_t,
                    'K_DEEP': K_DEEP,
                    'TT': TT,
                    'SG_CRIT': SG_CRIT,
                    'CFMAX_t': CFMAX_t,
                    'CFR': CFR,
                    'CWH': CWH,
                    'partition_sum_error': torch.abs(f_surface + f_inter + f_recharge - 1.0),
                    'available_before_baseflow': available_before_baseflow,
                    'available_after_baseflow': available_after_baseflow,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SMS, GW, SNOWPACK, MELTWATER, DEEPGW], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)

        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': reg_part
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out

        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelSix_Physical_CDeepGWSlowReturn(MultiInv_DynamicSimHydModelSix_Physical):
    """
    Copy of the retained soft-gate physical-fix model where old groundwater
    loss is converted into internal deep-groundwater recharge and slow return.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 component_routing=True, dry_channel_loss=True, zero_flow_gate=True,
                 channel_loss_max=0.60, zero_gate_hidden=None,
                 gate_variant='soft', gate_strength_max=0.30, deepgw_kwargs=None):
        super(MultiInv_DynamicSimHydModelSix_Physical_CDeepGWSlowReturn, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing,
            dry_channel_loss=dry_channel_loss, zero_flow_gate=zero_flow_gate,
            channel_loss_max=channel_loss_max, zero_gate_hidden=zero_gate_hidden,
            gate_variant=gate_variant, gate_strength_max=gate_strength_max)
        self.nfea = 14
        self.nstaticpm = self.nfea * nmul
        if deepgw_kwargs is None:
            deepgw_kwargs = dict()
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.simhyd = DynamicSimHydModelFiveDifferentiablePhysicalFixDeepGWSlowReturn(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            **deepgw_kwargs)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiablePhysicalFixDeepGWSlowReturn(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            **deepgw_kwargs)


class DynamicSimHydModelFiveDifferentiablePhysicalFixGWNNDeepGW(DynamicSimHydModelFiveDifferentiablePhysicalFix):
    """
    Physical-fix variant that replaces the old groundwater-loss sink with an
    end-to-end differentiable, mass-conserving groundwater redistribution NN.
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True, dynamic_cfmax_snow=True,
                 dynamic_routing_scale=False, dynamic_all=False, dyn_hidden=32, deep_scale=300.0):
        super(DynamicSimHydModelFiveDifferentiablePhysicalFixGWNNDeepGW, self).__init__(
            mode=mode, theta_is_raw=theta_is_raw, smooth=smooth, eps=eps, rain_snow_gain=rain_snow_gain,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, dyn_hidden=dyn_hidden)
        self.deep_scale = deep_scale
        self.gw_redist_net = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def _expand_gwnn(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)
        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = 0.0 + theta[:, 2:3] * (6.0 - 0.0)
        SMSC = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = 0.0 + theta[:, 4:5] * (1.0 - 0.0)
        CRAK = 0.0 + theta[:, 5:6] * (1.0 - 0.0)
        K = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        LG = 0.0 + theta[:, 7:8] * (0.2 - 0.0)
        TT = -2.5 + theta[:, 8:9] * (2.5 - (-2.5))
        CFMAX = 0.5 + theta[:, 9:10] * (10.0 - 0.5)
        CFR = 0.0 + theta[:, 10:11] * (0.1 - 0.0)
        CWH = 0.0 + theta[:, 11:12] * (0.2 - 0.0)
        SG_CRIT = 0.0 + theta[:, 12:13] * (300.0 - 0.0)
        K_DEEP = 0.0001 + theta[:, 13:14] * (0.01 - 0.0001)
        return INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH, SG_CRIT, K_DEEP

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH, SG_CRIT, K_DEEP = self._expand_gwnn(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SMS = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
            DEEPGW = torch.zeros(B, 1, device=device, dtype=dtype)
        else:
            SMS = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]
            SNOWPACK = initial_state[:, 2:3]
            MELTWATER = initial_state[:, 3:4]
            DEEPGW = initial_state[:, 4:5]

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = dict()
        if return_diagnostics is True:
            for name in [
                    'SMS', 'GW', 'DEEPGW', 'SNOWPACK', 'MELTWATER',
                    'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release_to_soil',
                    'interception_storage', 'interception_evaporation', 'actual_ET', 'infiltration',
                    'surface_runoff', 'interflow', 'recharge_to_groundwater', 'soil_overflow',
                    'baseflow_raw', 'baseflow_capped', 'baseflow', 'BAS_NN',
                    'groundwater_loss_raw', 'groundwater_loss_capped', 'groundwater_loss', 'GWLOSS_external',
                    'DEEP_RECH', 'deep_recharge', 'GW_HOLD', 'CAP_RISE', 'Q_DEEP', 'DEEPLOSS',
                    'channel_loss', 'gate_loss', 'q_raw_process', 'q_after_channel_loss', 'q_after_gate',
                    'total_discharge', 'INSC', 'COEF_t', 'SQ_t', 'SMSC', 'ETGAM_t', 'SUB_t', 'INTER_t', 'RECH_t',
                    'CRAK_t', 'K_t', 'LG_t', 'K_DEEP', 'TT', 'SG_CRIT', 'CFMAX_t', 'CFR', 'CWH',
                    'f_base', 'f_deep', 'f_hold', 'f_cap',
                    'partition_sum_error', 'available_before_baseflow', 'available_after_baseflow']:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SMS0 = self._min(self._pos(SMS), SMSC)
            GW0 = self._pos(GW)
            DEEPGW0 = self._pos(DEEPGW)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            wetness = torch.clamp(SMS0 / (SMSC + 1e-8), min=0.0, max=1.5)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0,
                Tt / 20.0,
                E0t / 10.0,
                SMS0 / 300.0,
                wetness,
                GW0 / 300.0,
                SNOWPACK0 / 300.0,
                sin_t,
                cos_t], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)

            if self.dynamic_etgam is True:
                ETGAM_t = 0.25 + 3.75 * torch.sigmoid(dyn_raw[:, self.dyn_slices['etgam']])
            else:
                ETGAM_t = torch.ones_like(SQ)

            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX * m_cf_eff

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()

            RAIN = Pt * frac_rain
            SNOW = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + SNOW
            melt_pot = CFMAX_t * self._pos(Tt - TT)
            melt = self._min(melt_pot, SNOWPACK1)
            MELTWATER1 = MELTWATER0 + melt
            SNOWPACK2 = self._pos(SNOWPACK1 - melt)

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            SNOWPACK3 = SNOWPACK2 + refreezing
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)

            water_holding = CWH * SNOWPACK3
            tosoil_raw = self._pos(MELTWATER2 - water_holding)
            tosoil = self._min(tosoil_raw, MELTWATER2)
            MELTWATER_next = self._pos(MELTWATER2 - tosoil)
            SNOWPACK_next = self._pos(SNOWPACK3)
            Peff = RAIN + tosoil

            IMAX = self._min(INSC, E0t)
            INT = self._min(IMAX, Peff)
            INR = self._pos(Peff - INT)

            infil_cap = COEF * torch.exp(-SQ_t * wetness)
            RMO = self._min(infil_cap, INR)
            IRUN_excess = self._pos(INR - RMO)

            available = wetness * RMO
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_inter = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([
                torch.log(base_surface),
                torch.log(base_inter),
                torch.log(base_recharge)], dim=1)
            if self.dynamic_partition is True:
                part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']]
            else:
                part_logits = base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]

            SRUN = IRUN_excess + f_surface * available
            IFLOW = f_inter * available
            REC = f_recharge * available
            SMF = self._pos(RMO - available)

            POT = self._pos(E0t - INT)
            et_scale = torch.pow(torch.clamp(wetness, min=1e-6, max=1.0), ETGAM_t)
            et_scale = torch.clamp(et_scale, max=1.0)
            ETS_raw = POT * et_scale
            ETS = self._min(ETS_raw, SMS0 + SMF)
            ETS = self._min(ETS, POT)

            REC_total = REC + self._pos(SMS0 + SMF - ETS - SMSC)
            GW_avail = self._pos(GW0 + REC_total)
            soil_deficit = self._pos(SMSC - SMS0)

            gw_inputs = torch.cat([
                SMS0 / torch.clamp(SMSC, min=1e-6),
                GW0 / torch.clamp(SG_CRIT + 1e-6, min=1e-6),
                DEEPGW0 / (self.deep_scale + 1e-6),
                Pt,
                E0t,
                Tt,
                sin_t,
                cos_t], dim=1)
            gw_logits = self.gw_redist_net(gw_inputs)
            gw_frac = torch.softmax(gw_logits, dim=1)
            f_base = gw_frac[:, 0:1]
            f_deep = gw_frac[:, 1:2]
            f_hold = gw_frac[:, 2:3]
            f_cap = gw_frac[:, 3:4]

            BAS_NN = f_base * GW_avail
            DEEP_RECH = f_deep * GW_avail
            GW_HOLD = f_hold * GW_avail
            CAP_RAW = f_cap * GW_avail
            CAP_RISE = self._min(CAP_RAW, soil_deficit)
            CAP_UNUSED = self._pos(CAP_RAW - CAP_RISE)
            GW_HOLD = GW_HOLD + CAP_UNUSED

            SMS_pre = SMS0 + SMF + CAP_RISE - ETS
            SMS_next = torch.clamp(SMS_pre, min=0.0, max=float('inf'))
            SOIL_EXCESS = self._pos(SMS_next - SMSC)
            SMS_next = self._min(SMS_next, SMSC)
            DEEPGW_pre = self._pos(DEEPGW0 + DEEP_RECH)
            Q_DEEP = self._min(K_DEEP * DEEPGW_pre, DEEPGW_pre)
            GW_next = self._pos(GW_HOLD)
            DEEPGW_next = self._pos(DEEPGW_pre - Q_DEEP)
            GWLOSS_external = torch.zeros_like(BAS_NN)
            DEEPLOSS = torch.zeros_like(BAS_NN)
            Q = self._pos(SRUN + IFLOW + BAS_NN + Q_DEEP)

            SMS = SMS_next
            GW = GW_next
            DEEPGW = DEEPGW_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)

            if return_diagnostics is True:
                diag_vals = {
                    'SMS': SMS_next,
                    'GW': GW_next,
                    'DEEPGW': DEEPGW_next,
                    'SNOWPACK': SNOWPACK_next,
                    'MELTWATER': MELTWATER_next,
                    'precipitation': Pt,
                    'rainfall': RAIN,
                    'snowfall': SNOW,
                    'snowmelt': melt,
                    'refreezing': refreezing,
                    'snow_release_to_soil': tosoil,
                    'interception_storage': torch.zeros_like(INT),
                    'interception_evaporation': INT,
                    'actual_ET': ETS,
                    'infiltration': RMO,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'recharge_to_groundwater': REC_total,
                    'soil_overflow': SOIL_EXCESS,
                    'baseflow_raw': BAS_NN,
                    'baseflow_capped': BAS_NN,
                    'baseflow': BAS_NN,
                    'BAS_NN': BAS_NN,
                    'groundwater_loss_raw': GWLOSS_external,
                    'groundwater_loss_capped': GWLOSS_external,
                    'groundwater_loss': GWLOSS_external,
                    'GWLOSS_external': GWLOSS_external,
                    'DEEP_RECH': DEEP_RECH,
                    'deep_recharge': DEEP_RECH,
                    'GW_HOLD': GW_HOLD,
                    'CAP_RISE': CAP_RISE,
                    'Q_DEEP': Q_DEEP,
                    'DEEPLOSS': DEEPLOSS,
                    'channel_loss': torch.zeros_like(Q),
                    'gate_loss': torch.zeros_like(Q),
                    'q_raw_process': Q,
                    'q_after_channel_loss': Q,
                    'q_after_gate': Q,
                    'total_discharge': Q,
                    'INSC': INSC,
                    'COEF_t': COEF,
                    'SQ_t': SQ_t,
                    'SMSC': SMSC,
                    'ETGAM_t': ETGAM_t,
                    'SUB_t': f_surface,
                    'INTER_t': f_inter,
                    'RECH_t': f_recharge,
                    'CRAK_t': f_recharge,
                    'K_t': K,
                    'LG_t': LG,
                    'K_DEEP': K_DEEP,
                    'TT': TT,
                    'SG_CRIT': SG_CRIT,
                    'CFMAX_t': CFMAX_t,
                    'CFR': CFR,
                    'CWH': CWH,
                    'f_base': f_base,
                    'f_deep': f_deep,
                    'f_hold': f_hold,
                    'f_cap': f_cap,
                    'partition_sum_error': torch.abs(f_surface + f_inter + f_recharge - 1.0),
                    'available_before_baseflow': GW_avail,
                    'available_after_baseflow': GW_HOLD,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SMS, GW, SNOWPACK, MELTWATER, DEEPGW], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)

        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': reg_part
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out

        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelSix_Physical_CGWNNDeepGW(MultiInv_DynamicSimHydModelSix_Physical):
    """
    Copy of the retained soft-gate physical-fix model with NN-based
    mass-conserving groundwater redistribution and deep groundwater storage.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 component_routing=True, dry_channel_loss=True, zero_flow_gate=True,
                 channel_loss_max=0.60, zero_gate_hidden=None,
                 gate_variant='soft', gate_strength_max=0.30):
        super(MultiInv_DynamicSimHydModelSix_Physical_CGWNNDeepGW, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing,
            dry_channel_loss=dry_channel_loss, zero_flow_gate=zero_flow_gate,
            channel_loss_max=channel_loss_max, zero_gate_hidden=zero_gate_hidden,
            gate_variant=gate_variant, gate_strength_max=gate_strength_max)
        self.nfea = 14
        self.nstaticpm = self.nfea * nmul
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.simhyd = DynamicSimHydModelFiveDifferentiablePhysicalFixGWNNDeepGW(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiablePhysicalFixGWNNDeepGW(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all)

class DynamicSimHydModelFiveDifferentiableHBVDeepGWHybrid(DynamicSimHydModelFiveDifferentiable):
    """
    Model 6 / HBV deep-groundwater hybrid process simulator.
    It converts discarded losses into internal stores and slow-return pathways.
    """

    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True, dynamic_cfmax_snow=True,
                 dynamic_routing_scale=False, dynamic_all=False, dyn_hidden=32,
                 channel_loss_max=0.60, channel_to_deep_fraction=0.70, channel_true_loss_fraction=0.30,
                 gate_strength_max=0.30, high_flow_bypass_fraction=0.0,
                 high_flow_bypass_surface_fraction=0.70, high_flow_bypass_rain_threshold=10.0,
                 high_flow_bypass_rain_scale=5.0, dynamic_channel_alpha=0.0,
                 dynamic_channel_beta=0.0, dynamic_deep_return_alpha=0.0,
                 deep_return_min=0.001, deep_return_max=0.05,
                 deep_leak_min=0.0, deep_leak_max=0.01,
                 cap_rise_max=2.0, k_channel_min=0.05, k_channel_max=1.0):
        super(DynamicSimHydModelFiveDifferentiableHBVDeepGWHybrid, self).__init__(
            mode=mode, theta_is_raw=theta_is_raw, smooth=smooth, eps=eps, rain_snow_gain=rain_snow_gain,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, dyn_hidden=dyn_hidden)
        self.channel_loss_max = channel_loss_max
        self.channel_to_deep_fraction = channel_to_deep_fraction
        self.channel_true_loss_fraction = channel_true_loss_fraction
        self.gate_strength_max = gate_strength_max
        self.high_flow_bypass_fraction = high_flow_bypass_fraction
        self.high_flow_bypass_surface_fraction = high_flow_bypass_surface_fraction
        self.high_flow_bypass_rain_threshold = high_flow_bypass_rain_threshold
        self.high_flow_bypass_rain_scale = high_flow_bypass_rain_scale
        self.dynamic_channel_alpha = dynamic_channel_alpha
        self.dynamic_channel_beta = dynamic_channel_beta
        self.dynamic_deep_return_alpha = dynamic_deep_return_alpha
        self.deep_return_min = deep_return_min
        self.deep_return_max = deep_return_max
        self.deep_leak_min = deep_leak_min
        self.deep_leak_max = deep_leak_max
        self.cap_rise_max = cap_rise_max
        self.k_channel_min = k_channel_min
        self.k_channel_max = k_channel_max

    def _expand_hybrid(self, theta):
        base = self._expand(theta[:, :13])
        theta_extra = theta[:, 13:19]
        K_SLOW = 0.0005 + theta_extra[:, 0:1] * (0.15 - 0.0005)
        PERC_CAP = 0.0 + theta_extra[:, 1:2] * 10.0
        K_DEEP_RETURN = self.deep_return_min + theta_extra[:, 2:3] * (self.deep_return_max - self.deep_return_min)
        K_DEEP_LEAK = self.deep_leak_min + theta_extra[:, 3:4] * (self.deep_leak_max - self.deep_leak_min)
        CAP_T = 0.0 + theta_extra[:, 4:5] * self.cap_rise_max
        K_CHANNEL = self.k_channel_min + theta_extra[:, 5:6] * (self.k_channel_max - self.k_channel_min)
        return base + (K_SLOW, PERC_CAP, K_DEEP_RETURN, K_DEEP_LEAK, CAP_T, K_CHANNEL)

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, channel_gamma_flat=None, zero_flow_gate=None, component_index=None,
                return_diagnostics=False, return_final_state=False, return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        (INSC, COEF, SQ, SMSC, SUB, CRAK, K_FAST, LG, TT, CFMAX, CFR, CWH, SG_CRIT,
         K_SLOW, PERC_CAP, K_DEEP_RETURN, K_DEEP_LEAK, CAP_T, K_CHANNEL) = self._expand_hybrid(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SMS = torch.zeros(B, 1, device=device, dtype=dtype)
            UZ = torch.zeros(B, 1, device=device, dtype=dtype)
            LZ = torch.zeros(B, 1, device=device, dtype=dtype)
            DEEP_GW = torch.zeros(B, 1, device=device, dtype=dtype)
            CHANNEL_STORE = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
        else:
            SMS = initial_state[:, 0:1]
            UZ = initial_state[:, 1:2]
            LZ = initial_state[:, 2:3]
            DEEP_GW = initial_state[:, 3:4]
            CHANNEL_STORE = initial_state[:, 4:5]
            SNOWPACK = initial_state[:, 5:6]
            MELTWATER = initial_state[:, 6:7]

        if lg_dyn_seq is not None and (lg_dyn_seq.shape[0] != B or lg_dyn_seq.shape[1] != Tlen):
            raise ValueError('lg_dyn_seq must have shape [B, T, 1] matching inputs')
        if channel_gamma_flat is None:
            channel_gamma_flat = torch.ones(B, 1, device=device, dtype=dtype) * (0.5 * self.channel_loss_max)
        else:
            channel_gamma_flat = channel_gamma_flat.to(device=device, dtype=dtype)
        if component_index is None:
            component_index = torch.zeros(B, dtype=torch.long, device=device)
        else:
            component_index = component_index.to(device=device)

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = {}
        deep_leak_hist = []
        precip_hist = []
        if return_diagnostics is True:
            for name in [
                    'SMS', 'UZ', 'LZ', 'DEEP_GW', 'CHANNEL_STORE', 'SNOWPACK', 'MELTWATER',
                    'precipitation', 'rainfall', 'snowfall', 'snowmelt', 'refreezing', 'snow_release_to_soil',
                    'interception_storage', 'interception_evaporation', 'actual_ET', 'infiltration',
                    'surface_runoff', 'interflow', 'recharge_to_groundwater', 'soil_overflow',
                    'high_flow_bypass_total', 'high_flow_bypass_surface', 'high_flow_bypass_uz',
                    'Q_fast_raw', 'Q_fast', 'percolation', 'Q_slow_raw', 'Q_slow',
                    'groundwater_loss_raw', 'GW_to_deep', 'deep_return', 'true_deep_leak',
                    'cap_from_lz', 'cap_from_deep', 'capillary_rise',
                    'channel_loss_raw', 'channel_loss', 'channel_to_deep', 'channel_to_store', 'channel_true_loss',
                    'q_raw_process', 'q_after_channel_loss', 'gate_loss', 'q_after_gate',
                    'channel_release', 'Q_process', 'total_discharge', 'water_balance_residual',
                    'INSC', 'COEF_t', 'SQ_t', 'SMSC', 'ETGAM_t', 'SUB_t', 'INTER_t', 'RECH_t',
                    'CRAK_t', 'K_fast_t', 'K_slow_t', 'PERC_cap_t', 'LG_t', 'K_deep_return_t',
                    'K_deep_leak_t', 'CAP_t', 'K_channel_t', 'K_channel_eff_t', 'K_deep_return_eff_t',
                    'TT', 'SG_CRIT', 'CFMAX_t', 'CFR', 'CWH',
                    'partition_sum_error', 'channel_loss_fraction', 'original_zero_flow_probability',
                    'connectivity_modified_flow_probability']:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SMS0 = self._min(self._pos(SMS), SMSC)
            UZ0 = self._pos(UZ)
            LZ0 = self._pos(LZ)
            DEEP0 = self._pos(DEEP_GW)
            CH0 = self._pos(CHANNEL_STORE)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            wetness = torch.clamp(SMS0 / (SMSC + 1e-8), min=0.0, max=1.5)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0,
                Tt / 20.0,
                E0t / 10.0,
                SMS0 / 300.0,
                wetness,
                (UZ0 + LZ0) / 300.0,
                DEEP0 / 300.0,
                sin_t,
                cos_t], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)

            if self.dynamic_etgam is True:
                ETGAM_t = 0.25 + 3.75 * torch.sigmoid(dyn_raw[:, self.dyn_slices['etgam']])
            else:
                ETGAM_t = torch.ones_like(SQ)

            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX * m_cf_eff

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()

            RAIN = Pt * frac_rain
            SNOW = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + SNOW
            melt_pot = CFMAX_t * self._pos(Tt - TT)
            melt = self._min(melt_pot, SNOWPACK1)
            MELTWATER1 = MELTWATER0 + melt
            SNOWPACK2 = self._pos(SNOWPACK1 - melt)

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            SNOWPACK3 = SNOWPACK2 + refreezing
            MELTWATER2 = self._pos(MELTWATER1 - refreezing)

            water_holding = CWH * SNOWPACK3
            tosoil_raw = self._pos(MELTWATER2 - water_holding)
            tosoil = self._min(tosoil_raw, MELTWATER2)
            MELTWATER_next = self._pos(MELTWATER2 - tosoil)
            SNOWPACK_next = self._pos(SNOWPACK3)
            Peff = RAIN + tosoil

            IMAX = self._min(INSC, E0t)
            INT = self._min(IMAX, Peff)
            INR = self._pos(Peff - INT)

            infil_cap = COEF * torch.exp(-SQ_t * wetness)
            RMO = self._min(infil_cap, INR)
            IRUN_excess = self._pos(INR - RMO)

            available = wetness * RMO
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_inter = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([
                torch.log(base_surface),
                torch.log(base_inter),
                torch.log(base_recharge)], dim=1)
            if self.dynamic_partition is True:
                part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']]
            else:
                part_logits = base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]

            SRUN = IRUN_excess + f_surface * available
            IFLOW = f_inter * available
            REC = f_recharge * available
            SMF = self._pos(RMO - available)

            rain_trigger = torch.sigmoid((Pt - self.high_flow_bypass_rain_threshold) / self.high_flow_bypass_rain_scale)
            wet_trigger = torch.clamp(wetness, 0.0, 1.0)
            bypass_trigger = torch.clamp(rain_trigger * wet_trigger, 0.0, 1.0)
            bypass_pot = self.high_flow_bypass_fraction * bypass_trigger * SMF
            bypass_total = self._min(bypass_pot, SMF)
            bypass_surface = self.high_flow_bypass_surface_fraction * bypass_total
            bypass_uz = self._pos(bypass_total - bypass_surface)
            SMF = self._pos(SMF - bypass_total)
            SRUN = SRUN + bypass_surface

            SMS_pre = SMS0 + SMF
            SOIL_EXCESS = self._pos(SMS_pre - SMSC)
            SMS1 = self._min(self._pos(SMS_pre), SMSC)
            REC_total = REC + SOIL_EXCESS

            UZ1 = UZ0 + REC_total + bypass_uz
            Q_fast_raw = K_FAST * UZ1
            Q_fast = self._min(Q_fast_raw, UZ1)
            UZ2 = self._pos(UZ1 - Q_fast)

            PERC = self._min(PERC_CAP, UZ2)
            UZ3 = self._pos(UZ2 - PERC)
            LZ1 = LZ0 + PERC

            Q_slow_raw = K_SLOW * LZ1
            Q_slow = self._min(Q_slow_raw, LZ1)
            LZ2 = self._pos(LZ1 - Q_slow)

            if lg_dyn_seq is None:
                LG_t = LG
            else:
                LG_dyn_t = torch.clamp(lg_dyn_seq[:, t, :], 0.0, 1.0)
                LG_eff = (1.0 - lg_dyn_weight) * LG + lg_dyn_weight * (LG * LG_dyn_t * 1.5)
                LG_t = torch.clamp(LG_eff, 0.0, 0.2)

            GWLOSS_raw = LG_t * F.softplus(SG_CRIT - LZ2)
            GW_to_deep = self._min(GWLOSS_raw, LZ2)
            LZ3 = self._pos(LZ2 - GW_to_deep)
            DEEP1 = DEEP0 + GW_to_deep

            recession_factor = torch.clamp((SG_CRIT - LZ3) / (SG_CRIT + 1e-6), 0.0, 1.0)
            K_DEEP_RETURN_EFF = torch.clamp(K_DEEP_RETURN * (1.0 + self.dynamic_deep_return_alpha * recession_factor), 0.0, 0.20)
            DEEP_RETURN = self._min(K_DEEP_RETURN_EFF * DEEP1, DEEP1)
            DEEP2 = self._pos(DEEP1 - DEEP_RETURN)
            LZ4 = LZ3 + DEEP_RETURN

            TRUE_DEEP_LEAK = self._min(K_DEEP_LEAK * DEEP2, DEEP2)
            DEEP3 = self._pos(DEEP2 - TRUE_DEEP_LEAK)

            soil_deficit = self._pos(SMSC - SMS1)
            gw_wetness = torch.clamp((LZ4 + DEEP3) / (SMSC + 1e-8), 0.0, 1.5)
            CAP_raw = CAP_T * soil_deficit / (SMSC + 1e-8) * gw_wetness
            CAP_from_LZ = self._min(self._min(CAP_raw, LZ4), soil_deficit)
            LZ5 = self._pos(LZ4 - CAP_from_LZ)
            SMS2 = SMS1 + CAP_from_LZ
            remaining_deficit = self._pos(SMSC - SMS2)
            CAP_from_DEEP = self._min(self._min(self._pos(CAP_raw - CAP_from_LZ), DEEP3), remaining_deficit)
            DEEP4 = self._pos(DEEP3 - CAP_from_DEEP)
            SMS3 = self._min(SMS2 + CAP_from_DEEP, SMSC)
            CAP_total = CAP_from_LZ + CAP_from_DEEP

            POT = self._pos(E0t - INT)
            wetness_et = torch.clamp(SMS3 / (SMSC + 1e-8), min=1e-6, max=1.0)
            et_scale = torch.pow(wetness_et, ETGAM_t)
            et_scale = torch.clamp(et_scale, max=1.0)
            ETS_raw = POT * et_scale
            ETS = self._min(ETS_raw, SMS3)
            ETS = self._min(ETS, POT)
            SMS_next = self._pos(SMS3 - ETS)

            Q_raw = self._pos(SRUN + IFLOW + Q_fast + Q_slow)
            dryness = 1.0 - torch.clamp(SMS_next / (SMSC + 1e-8), 0.0, 1.0)
            loss_frac = 1.0 - torch.exp(-channel_gamma_flat * dryness)
            loss_frac = torch.clamp(loss_frac, 0.0, 0.95)
            channel_loss_raw = Q_raw * loss_frac
            channel_loss = self._min(channel_loss_raw, Q_raw)
            channel_true_loss = self.channel_true_loss_fraction * channel_loss
            channel_to_deep = torch.zeros_like(channel_loss)
            channel_to_store = self._pos(channel_loss - channel_true_loss)
            DEEP5 = DEEP4
            Q_after_channel = self._pos(Q_raw - channel_loss)

            if zero_flow_gate is None:
                p_flow = torch.ones_like(Q_after_channel) * 0.9
            else:
                if inputs.shape[-1] >= 5:
                    sin_gate = inputs[:, t, 3:4]
                    cos_gate = inputs[:, t, 4:5]
                else:
                    sin_gate = torch.zeros_like(Q_after_channel)
                    cos_gate = torch.ones_like(Q_after_channel)
                gate_in = torch.cat([Pt / 20.0, wetness_et, Q_after_channel / 20.0, sin_gate, cos_gate], dim=1)
                logits_all = zero_flow_gate(gate_in)
                logits = logits_all.gather(1, component_index.view(-1, 1))
                p_flow = torch.sigmoid(logits)
            gate_loss_frac = self.gate_strength_max * (1.0 - p_flow)
            gate_loss = self._min(Q_after_channel * gate_loss_frac, Q_after_channel)
            Q_after_gate = self._pos(Q_after_channel - gate_loss)
            CHANNEL1 = CH0 + gate_loss + channel_to_store
            K_CHANNEL_EFF = torch.clamp(
                K_CHANNEL * (1.0 + self.dynamic_channel_alpha * wetness_et + self.dynamic_channel_beta * (Pt / 20.0)),
                self.k_channel_min, self.k_channel_max)
            channel_release = self._min(K_CHANNEL_EFF * CHANNEL1, CHANNEL1)
            CHANNEL_next = self._pos(CHANNEL1 - channel_release)
            Q_process = self._pos(Q_after_gate + channel_release)

            UZ = UZ3
            LZ = LZ5
            DEEP_GW = DEEP5
            CHANNEL_STORE = CHANNEL_next
            SMS = SMS_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            delta_storage = (
                (SMS_next - SMS0) + (UZ3 - UZ0) + (LZ5 - LZ0) + (DEEP5 - DEEP0) +
                (CHANNEL_next - CH0) + (SNOWPACK_next - SNOWPACK0) + (MELTWATER_next - MELTWATER0)
            )
            residual = Pt - Q_process - INT - ETS - TRUE_DEEP_LEAK - channel_true_loss - delta_storage

            q_hist.append(Q_process)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)
            deep_leak_hist.append(TRUE_DEEP_LEAK)
            precip_hist.append(Pt)

            if return_diagnostics is True:
                diag_vals = {
                    'SMS': SMS_next,
                    'UZ': UZ3,
                    'LZ': LZ5,
                    'DEEP_GW': DEEP5,
                    'CHANNEL_STORE': CHANNEL_next,
                    'SNOWPACK': SNOWPACK_next,
                    'MELTWATER': MELTWATER_next,
                    'precipitation': Pt,
                    'rainfall': RAIN,
                    'snowfall': SNOW,
                    'snowmelt': melt,
                    'refreezing': refreezing,
                    'snow_release_to_soil': tosoil,
                    'interception_storage': torch.zeros_like(INT),
                    'interception_evaporation': INT,
                    'actual_ET': ETS,
                    'infiltration': RMO,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'recharge_to_groundwater': REC_total,
                    'soil_overflow': SOIL_EXCESS,
                    'high_flow_bypass_total': bypass_total,
                    'high_flow_bypass_surface': bypass_surface,
                    'high_flow_bypass_uz': bypass_uz,
                    'Q_fast_raw': Q_fast_raw,
                    'Q_fast': Q_fast,
                    'percolation': PERC,
                    'Q_slow_raw': Q_slow_raw,
                    'Q_slow': Q_slow,
                    'groundwater_loss_raw': GWLOSS_raw,
                    'GW_to_deep': GW_to_deep,
                    'deep_return': DEEP_RETURN,
                    'true_deep_leak': TRUE_DEEP_LEAK,
                    'cap_from_lz': CAP_from_LZ,
                    'cap_from_deep': CAP_from_DEEP,
                    'capillary_rise': CAP_total,
                    'channel_loss_raw': channel_loss_raw,
                    'channel_loss': channel_loss,
                    'channel_to_deep': channel_to_deep,
                    'channel_to_store': channel_to_store,
                    'channel_true_loss': channel_true_loss,
                    'q_raw_process': Q_raw,
                    'q_after_channel_loss': Q_after_channel,
                    'gate_loss': gate_loss,
                    'q_after_gate': Q_after_gate,
                    'channel_release': channel_release,
                    'Q_process': Q_process,
                    'total_discharge': Q_process,
                    'water_balance_residual': residual,
                    'INSC': INSC,
                    'COEF_t': COEF,
                    'SQ_t': SQ_t,
                    'SMSC': SMSC,
                    'ETGAM_t': ETGAM_t,
                    'SUB_t': f_surface,
                    'INTER_t': f_inter,
                    'RECH_t': f_recharge,
                    'CRAK_t': f_recharge,
                    'K_fast_t': K_FAST,
                    'K_slow_t': K_SLOW,
                    'PERC_cap_t': PERC_CAP,
                    'LG_t': LG_t,
                    'K_deep_return_t': K_DEEP_RETURN,
                    'K_deep_return_eff_t': K_DEEP_RETURN_EFF,
                    'K_deep_leak_t': K_DEEP_LEAK,
                    'CAP_t': CAP_T,
                    'K_channel_t': K_CHANNEL,
                    'K_channel_eff_t': K_CHANNEL_EFF,
                    'TT': TT,
                    'SG_CRIT': SG_CRIT,
                    'CFMAX_t': CFMAX_t,
                    'CFR': CFR,
                    'CWH': CWH,
                    'partition_sum_error': torch.abs(f_surface + f_inter + f_recharge - 1.0),
                    'channel_loss_fraction': loss_frac,
                    'original_zero_flow_probability': p_flow,
                    'connectivity_modified_flow_probability': 1.0 - gate_loss_frac,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SMS, UZ, LZ, DEEP_GW, CHANNEL_STORE, SNOWPACK, MELTWATER], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)
        if len(deep_leak_hist) > 0:
            deep_leak_seq = torch.stack(deep_leak_hist, dim=1)
            precip_seq = torch.stack(precip_hist, dim=1)
            deep_leak_ratio = torch.sum(deep_leak_seq) / (torch.sum(precip_seq) + 1e-8)
        else:
            deep_leak_ratio = torch.tensor(0.0, device=device, dtype=dtype)

        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': reg_part,
            'deep_leak_ratio': deep_leak_ratio
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out

        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelSix_HBVDeepGWHybrid(MultiInv_DynamicSimHydModelSix_Physical):
    """
    Hybrid Model 6 variant that converts discarded losses to HBV-style
    deep groundwater and channel storage transfers.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 reg_deep_leak_w=0.0, deep_leak_target=0.02,
                 component_routing=True, dry_channel_loss=True, zero_flow_gate=True,
                 channel_loss_max=0.60, zero_gate_hidden=None,
                 gate_variant='soft', gate_strength_max=0.30,
                 channel_to_deep_fraction=0.70, channel_true_loss_fraction=0.30,
                 high_flow_bypass_fraction=0.0, high_flow_bypass_surface_fraction=0.70,
                 high_flow_bypass_rain_threshold=10.0, high_flow_bypass_rain_scale=5.0,
                 dynamic_channel_alpha=0.0, dynamic_channel_beta=0.0,
                 dynamic_deep_return_alpha=0.0,
                 deep_return_min=0.001, deep_return_max=0.05,
                 deep_leak_min=0.0, deep_leak_max=0.01,
                 cap_rise_max=2.0, k_channel_min=0.05, k_channel_max=1.0):
        super(MultiInv_DynamicSimHydModelSix_HBVDeepGWHybrid, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing,
            dry_channel_loss=dry_channel_loss, zero_flow_gate=zero_flow_gate,
            channel_loss_max=channel_loss_max, zero_gate_hidden=zero_gate_hidden,
            gate_variant=gate_variant, gate_strength_max=gate_strength_max)
        self.nfea = 19
        self.reg_deep_leak_w = reg_deep_leak_w
        self.deep_leak_target = deep_leak_target
        self.nstaticpm = self.nfea * nmul
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.simhyd = DynamicSimHydModelFiveDifferentiableHBVDeepGWHybrid(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            channel_loss_max=channel_loss_max,
            channel_to_deep_fraction=channel_to_deep_fraction,
            channel_true_loss_fraction=channel_true_loss_fraction,
            gate_strength_max=gate_strength_max,
            high_flow_bypass_fraction=high_flow_bypass_fraction,
            high_flow_bypass_surface_fraction=high_flow_bypass_surface_fraction,
            high_flow_bypass_rain_threshold=high_flow_bypass_rain_threshold,
            high_flow_bypass_rain_scale=high_flow_bypass_rain_scale,
            dynamic_channel_alpha=dynamic_channel_alpha,
            dynamic_channel_beta=dynamic_channel_beta,
            dynamic_deep_return_alpha=dynamic_deep_return_alpha,
            deep_return_min=deep_return_min,
            deep_return_max=deep_return_max,
            deep_leak_min=deep_leak_min,
            deep_leak_max=deep_leak_max,
            cap_rise_max=cap_rise_max,
            k_channel_min=k_channel_min,
            k_channel_max=k_channel_max)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiableHBVDeepGWHybrid(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            channel_loss_max=channel_loss_max,
            channel_to_deep_fraction=channel_to_deep_fraction,
            channel_true_loss_fraction=channel_true_loss_fraction,
            gate_strength_max=gate_strength_max,
            high_flow_bypass_fraction=high_flow_bypass_fraction,
            high_flow_bypass_surface_fraction=high_flow_bypass_surface_fraction,
            high_flow_bypass_rain_threshold=high_flow_bypass_rain_threshold,
            high_flow_bypass_rain_scale=high_flow_bypass_rain_scale,
            dynamic_channel_alpha=dynamic_channel_alpha,
            dynamic_channel_beta=dynamic_channel_beta,
            dynamic_deep_return_alpha=dynamic_deep_return_alpha,
            deep_return_min=deep_return_min,
            deep_return_max=deep_return_max,
            deep_leak_min=deep_leak_min,
            deep_leak_max=deep_leak_max,
            cap_rise_max=cap_rise_max,
            k_channel_min=k_channel_min,
            k_channel_max=k_channel_max)

    def forward(self, x, z, doDropMC=False, return_diagnostics=False, return_component_diagnostics=False):
        nt_x = x.shape[0]
        ngage = z.shape[1]
        basin_attr = z[-1, :, -self.nattr:]
        if z.shape[2] > self.nattr:
            snow_frac_raw = torch.clamp(z[-1, :, -self.nattr - 1:-self.nattr], min=0.0, max=1.0)
        else:
            snow_frac_raw = None

        staticFeat = self.staticFeat(basin_attr)
        staticParams0 = self.staticOut(staticFeat)
        cursor = 0
        static0 = staticParams0[:, cursor:cursor + self.nstaticpm].view(ngage, self.nfea, self.nmul) + self.compStaticBias
        snowpara = torch.sigmoid(static0)
        cursor += self.nstaticpm
        routpara0 = staticParams0[:, cursor:cursor + self.nroutpm]
        if self.component_routing is True:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        elif self.comprout is False:
            routpara = torch.sigmoid(routpara0)
        else:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        cursor += self.nroutpm
        if self.nwtspm == 0:
            wts = None
        else:
            wtspara = staticParams0[:, cursor:cursor + self.nwtspm] + self.compWeightBias
            wts = F.softmax(wtspara, dim=-1)
        self._last_wts = wts

        if self.lgdyn is False:
            lg_dyn = None
        else:
            lg_dyn_seq = self.lstmdyn(z)
            lg_attr_bias = self.lgAttr(basin_attr).unsqueeze(0).repeat(lg_dyn_seq.shape[0], 1, 1)
            lg_dyn = torch.sigmoid(lg_dyn_seq + lg_attr_bias)

        x_rep = x.unsqueeze(2).repeat(1, 1, self.nmul, 1).view(nt_x, ngage * self.nmul, x.shape[2])
        x_bt = x_rep.permute(1, 0, 2)
        theta = snowpara.permute(0, 2, 1).contiguous().view(ngage * self.nmul, self.nfea)
        lg_bt = None if lg_dyn is None else lg_dyn.permute(1, 2, 0).contiguous().view(ngage * self.nmul, lg_dyn.shape[0], 1)
        if lg_bt is not None and self.inittime > 0:
            lg_bt_main = lg_bt[:, self.inittime:, :]
        else:
            lg_bt_main = lg_bt
        snow_frac_rep = None if snow_frac_raw is None else snow_frac_raw.unsqueeze(1).repeat(1, self.nmul, 1).view(ngage * self.nmul, 1)

        channel_gamma = self.channel_loss_max * torch.sigmoid(self.channelLossHead(basin_attr))
        channel_gamma_rep = channel_gamma.view(ngage, self.nmul, 1).view(ngage * self.nmul, 1)
        component_index = torch.arange(self.nmul, device=x.device).view(1, self.nmul).repeat(ngage, 1).reshape(-1)

        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            _, warm_state = self.simhyd_analysis(
                warm_inputs, theta, lg_dyn_seq=None, lg_dyn_weight=self.lgdynweight,
                snow_frac_raw=snow_frac_rep, channel_gamma_flat=channel_gamma_rep,
                zero_flow_gate=self.zeroFlowGate, component_index=component_index,
                return_final_state=True)
            if return_diagnostics or return_component_diagnostics:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    main_inputs, theta, initial_state=warm_state, lg_dyn_seq=lg_bt_main,
                    lg_dyn_weight=self.lgdynweight, snow_frac_raw=snow_frac_rep,
                    channel_gamma_flat=channel_gamma_rep, zero_flow_gate=self.zeroFlowGate,
                    component_index=component_index, return_diagnostics=True, return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    main_inputs, theta, initial_state=warm_state, lg_dyn_seq=lg_bt_main,
                    lg_dyn_weight=self.lgdynweight, snow_frac_raw=snow_frac_rep,
                    channel_gamma_flat=channel_gamma_rep, zero_flow_gate=self.zeroFlowGate,
                    component_index=component_index, return_regularization=True)
                diag_comp = None
        else:
            if return_diagnostics or return_component_diagnostics:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    x_bt, theta, lg_dyn_seq=lg_bt, lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep, channel_gamma_flat=channel_gamma_rep,
                    zero_flow_gate=self.zeroFlowGate, component_index=component_index,
                    return_diagnostics=True, return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    x_bt, theta, lg_dyn_seq=lg_bt, lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep, channel_gamma_flat=channel_gamma_rep,
                    zero_flow_gate=self.zeroFlowGate, component_index=component_index,
                    return_regularization=True)
                diag_comp = None

        deep_leak_ratio = reg_terms.get('deep_leak_ratio')
        deep_leak_pen = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if deep_leak_ratio is not None:
            deep_leak_pen = torch.relu(deep_leak_ratio - self.deep_leak_target) ** 2
        reg_total = self.reg_amp_w * reg_terms['dynamic_amplitude_loss'] \
            + self.reg_smooth_w * reg_terms['dynamic_smoothness_loss'] \
            + self.reg_part_w * reg_terms['partition_entropy_loss'] \
            + self.reg_deep_leak_w * deep_leak_pen
        reg_terms['deep_leak_penalty'] = deep_leak_pen
        self._last_aux_terms = reg_terms
        self._last_aux_loss = reg_total

        q_comp_raw = q_seq.view(ngage, self.nmul, q_seq.shape[1], 1).permute(2, 0, 1, 3)
        q_comp_raw = torch.clamp(q_comp_raw, min=0.0)
        q_mix_before_routing = self._mix_or_mean(q_comp_raw, wts)
        route_mult_seq = None
        if self.routOpt is True and self.component_routing is True:
            q_for_routing = q_comp_raw.permute(0, 1, 3, 2).contiguous().view(q_comp_raw.shape[0], ngage * self.nmul, 1)
            if self.dynamic_routing_scale is True:
                if diag_comp is not None and 'SMS' in diag_comp:
                    sm_comp = self._component_tensor_4d(diag_comp['SMS'], ngage)
                else:
                    sm_comp = torch.ones_like(q_comp_raw) * 50.0
                smsc_comp = (50.0 + theta.view(ngage, self.nmul, self.nfea)[:, :, 3:4] * (500.0 - 50.0)).unsqueeze(0).repeat(q_comp_raw.shape[0], 1, 1, 1)
                p_rep = x[self.inittime:, :, 0:1].unsqueeze(2).repeat(1, 1, self.nmul, 1) if self.inittime > 0 else x[:, :, 0:1].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                if x.shape[-1] >= 5:
                    x_base = x[self.inittime:, :, :] if self.inittime > 0 else x
                    sin_rep = x_base[:, :, 3:4].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                    cos_rep = x_base[:, :, 4:5].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                else:
                    sin_rep = torch.zeros_like(q_comp_raw)
                    cos_rep = torch.ones_like(q_comp_raw)
                x_route = torch.cat([p_rep, sm_comp, smsc_comp, sin_rep, cos_rep], dim=-1).view(q_comp_raw.shape[0], ngage * self.nmul, 5)
                q_routed_flat, route_mult_flat = self._route_q_dynamic_scale(q_for_routing, routpara, x_route)
                route_mult_4d = route_mult_flat.view(q_comp_raw.shape[0], ngage, self.nmul, 1)
                route_mult_seq = self._mix_or_mean(route_mult_4d, wts)
                self._last_aux_loss = self._last_aux_loss + self.reg_amp_w * torch.mean((route_mult_flat - 1.0) ** 2)
                self._last_aux_loss = self._last_aux_loss + self.reg_smooth_w * torch.mean((route_mult_flat[1:, :, :] - route_mult_flat[:-1, :, :]) ** 2)
                q_routed = q_routed_flat.view(q_comp_raw.shape[0], ngage, self.nmul, 1)
            else:
                q_routed = self._route_q(q_for_routing, routpara).view(q_comp_raw.shape[0], ngage, self.nmul, 1)
            out = self._mix_or_mean(q_routed, wts)
        else:
            if self.routOpt is True:
                out = self._route_q(q_mix_before_routing, routpara)
            else:
                out = q_mix_before_routing
            q_routed = q_comp_raw
        out = torch.clamp(out, min=0.0)
        if return_diagnostics is not True and return_component_diagnostics is not True:
            return out

        diag_out = {}
        if diag_comp is not None:
            for name, tensor_comp in diag_comp.items():
                diag_out[name] = self._mix_component_tensor(tensor_comp, ngage)
        diag_out['total_discharge'] = out
        diag_out['q_mix_before_routing'] = q_mix_before_routing
        if route_mult_seq is not None:
            diag_out['route_b_t_multiplier'] = route_mult_seq
        if return_component_diagnostics is True:
            if diag_comp is not None:
                for name, tensor_comp in diag_comp.items():
                    diag_out[name + '_components'] = self._component_tensor_4d(tensor_comp, ngage).squeeze(-1)
            diag_out['q_process_components'] = q_comp_raw.squeeze(-1)
            diag_out['q_routed_components'] = q_routed.squeeze(-1)
            diag_out['component_weights'] = wts
            routpara_view = torch.sigmoid(routpara0).view(ngage, self.nmul, 2)
            diag_out['route_a_components'] = 0.0 + routpara_view[:, :, 0] * 2.9
            diag_out['route_b_components'] = 0.0 + routpara_view[:, :, 1] * 6.5
        return out, diag_out


class MultiInv_DynamicSimHydModelLowNSE_AridPulse(MultiInv_DynamicSimHydModelSix):
    """
    Model Six + one extra process only:
    an arid threshold storm-pulse runoff term added before routing.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 component_routing=True, dry_channel_loss=True, zero_flow_gate=True,
                 channel_loss_max=0.60, zero_gate_hidden=None,
                 arid_pulse=True, pulse_max=0.50, pthr_min=5.0, pthr_max=35.0,
                 pulse_cap_fraction=0.60):
        super(MultiInv_DynamicSimHydModelLowNSE_AridPulse, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing,
            dry_channel_loss=dry_channel_loss, zero_flow_gate=zero_flow_gate,
            channel_loss_max=channel_loss_max, zero_gate_hidden=zero_gate_hidden)
        self.arid_pulse = arid_pulse
        self.pulse_max = pulse_max
        self.pthr_min = pthr_min
        self.pthr_max = pthr_max
        self.pulse_cap_fraction = pulse_cap_fraction
        self.aridPulseHead = nn.Linear(nattr, nmul * 3)

    def _apply_arid_pulse(self, q_comp, x, diag_comp, theta, basin_attr, ngage):
        if self.arid_pulse is not True:
            zeros = torch.zeros_like(q_comp)
            return q_comp, zeros, zeros, zeros, zeros

        T, B, M, _ = q_comp.shape
        pulse_raw = self.aridPulseHead(basin_attr).view(ngage, self.nmul, 3)
        raw_c = pulse_raw[:, :, 0:1]
        raw_thr = pulse_raw[:, :, 1:2]
        raw_dry = pulse_raw[:, :, 2:3]
        c_pulse = self.pulse_max * torch.sigmoid(raw_c)
        p_threshold = self.pthr_min + (self.pthr_max - self.pthr_min) * torch.sigmoid(raw_thr)
        dry_sensitivity = torch.sigmoid(raw_dry)

        if diag_comp is not None and 'soil_moisture' in diag_comp:
            soil_m = self._component_tensor_4d(diag_comp['soil_moisture'], ngage)
            smsc = self._theta_to_smsc(theta, ngage).unsqueeze(0)
            wetness = torch.clamp(soil_m / torch.clamp(smsc, min=1e-6), 0.0, 1.0)
            dryness = 1.0 - wetness
        else:
            dryness = torch.ones_like(q_comp) * 0.5

        p_t = x[:, :, 0:1].unsqueeze(2).repeat(1, 1, M, 1)
        c_rep = c_pulse.unsqueeze(0)
        thr_rep = p_threshold.unsqueeze(0)
        ds_rep = dry_sensitivity.unsqueeze(0)
        p_excess = F.relu(p_t - thr_rep)
        dry_factor = 0.25 + 0.75 * dryness * ds_rep
        q_pulse = c_rep * dry_factor * p_excess
        q_pulse = torch.minimum(q_pulse, self.pulse_cap_fraction * p_t)
        q_after = torch.clamp(q_comp + q_pulse, min=0.0)
        pulse_frac = q_pulse / torch.clamp(q_after, min=1e-6)
        return q_after, q_pulse, pulse_frac, thr_rep.expand(T, -1, -1, -1), c_rep.expand(T, -1, -1, -1), dryness

    def forward(self, x, z, doDropMC=False, return_diagnostics=False):
        nt_x = x.shape[0]
        ngage = z.shape[1]
        basin_attr = z[-1, :, -self.nattr:]

        if z.shape[2] > self.nattr:
            snow_frac_raw = torch.clamp(z[-1, :, -self.nattr - 1:-self.nattr], min=0.0, max=1.0)
        else:
            snow_frac_raw = None

        staticFeat = self.staticFeat(basin_attr)
        staticParams0 = self.staticOut(staticFeat)

        cursor = 0
        static0 = staticParams0[:, cursor:cursor + self.nstaticpm].view(ngage, self.nfea, self.nmul)
        static0 = static0 + self.compStaticBias
        snowpara = torch.sigmoid(static0)
        cursor += self.nstaticpm

        routpara0 = staticParams0[:, cursor:cursor + self.nroutpm]
        if self.component_routing is True:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        elif self.comprout is False:
            routpara = torch.sigmoid(routpara0)
        else:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        cursor += self.nroutpm

        if self.nwtspm == 0:
            wts = None
        else:
            wtspara = staticParams0[:, cursor:cursor + self.nwtspm] + self.compWeightBias
            wts = F.softmax(wtspara, dim=-1)
        self._last_wts = wts

        if self.lgdyn is False:
            lg_dyn = None
        else:
            lg_dyn_seq = self.lstmdyn(z)
            lg_attr_bias = self.lgAttr(basin_attr).unsqueeze(0).repeat(lg_dyn_seq.shape[0], 1, 1)
            lg_dyn = torch.sigmoid(lg_dyn_seq + lg_attr_bias)

        x_rep = x.unsqueeze(2).repeat(1, 1, self.nmul, 1).view(nt_x, ngage * self.nmul, x.shape[2])
        x_bt = x_rep.permute(1, 0, 2)
        theta = snowpara.permute(0, 2, 1).contiguous().view(ngage * self.nmul, self.nfea)

        if lg_dyn is None:
            lg_bt = None
        else:
            lg_bt = lg_dyn.permute(1, 2, 0).contiguous().view(ngage * self.nmul, lg_dyn.shape[0], 1)

        if snow_frac_raw is None:
            snow_frac_rep = None
        else:
            snow_frac_rep = snow_frac_raw.unsqueeze(1).repeat(1, self.nmul, 1).view(ngage * self.nmul, 1)

        need_process_diag = return_diagnostics or self.dry_channel_loss or self.zero_flow_gate_enabled or self.arid_pulse

        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            x_use = x[self.inittime:, :, :]
            _, warm_state = self.simhyd_analysis(
                warm_inputs,
                theta,
                lg_dyn_seq=None,
                lg_dyn_weight=self.lgdynweight,
                snow_frac_raw=snow_frac_rep,
                return_final_state=True)
            if need_process_diag:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    main_inputs, theta, initial_state=warm_state, lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight, snow_frac_raw=snow_frac_rep,
                    return_diagnostics=True, return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    main_inputs, theta, initial_state=warm_state, lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight, snow_frac_raw=snow_frac_rep,
                    return_regularization=True)
                diag_comp = None
        else:
            x_use = x
            if need_process_diag:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    x_bt, theta, lg_dyn_seq=lg_bt, lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep, return_diagnostics=True,
                    return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    x_bt, theta, lg_dyn_seq=lg_bt, lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep, return_regularization=True)
                diag_comp = None

        reg_total = self.reg_amp_w * reg_terms['dynamic_amplitude_loss'] \
            + self.reg_smooth_w * reg_terms['dynamic_smoothness_loss'] \
            + self.reg_part_w * reg_terms['partition_entropy_loss']
        self._last_aux_terms = reg_terms
        self._last_aux_loss = reg_total

        q_comp_raw = q_seq.view(ngage, self.nmul, q_seq.shape[1], 1).permute(2, 0, 1, 3)
        q_comp_raw = torch.clamp(q_comp_raw, min=0.0)

        q_after_loss, channel_loss, channel_loss_fraction = self._apply_channel_loss(q_comp_raw, diag_comp, theta, basin_attr, ngage)
        q_after_gate, zero_flow_probability = self._apply_zero_flow_gate(q_after_loss, x_use, diag_comp, theta, ngage)
        q_after_gate = torch.clamp(q_after_gate, min=0.0)
        q_comp, arid_pulse_runoff, arid_pulse_fraction, arid_pulse_threshold, arid_pulse_coeff, arid_pulse_dryness = \
            self._apply_arid_pulse(q_after_gate, x_use, diag_comp, theta, basin_attr, ngage)

        q_mix_before_routing = self._mix_or_mean(q_comp, wts)
        route_mult_seq = None
        if self.routOpt is True and self.component_routing is True:
            q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
            if self.dynamic_routing_scale is True:
                if diag_comp is not None and 'soil_moisture' in diag_comp:
                    sm_comp = self._component_tensor_4d(diag_comp['soil_moisture'], ngage)
                else:
                    sm_comp = torch.ones_like(q_comp) * 50.0
                smsc_comp = self._theta_to_smsc(theta, ngage).unsqueeze(0).repeat(q_comp.shape[0], 1, 1, 1)
                p_rep = x_use[:, :, 0:1].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                if x_use.shape[-1] >= 5:
                    sin_rep = x_use[:, :, 3:4].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                    cos_rep = x_use[:, :, 4:5].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                else:
                    sin_rep = torch.zeros_like(q_comp)
                    cos_rep = torch.ones_like(q_comp)
                x_route = torch.cat([p_rep, sm_comp, smsc_comp, sin_rep, cos_rep], dim=-1).view(q_comp.shape[0], ngage * self.nmul, 5)
                q_routed_flat, route_mult_flat = self._route_q_dynamic_scale(q_for_routing, routpara, x_route)
                route_mult_4d = route_mult_flat.view(q_comp.shape[0], ngage, self.nmul, 1)
                route_mult_seq = self._mix_or_mean(route_mult_4d, wts)
                self._last_aux_loss = self._last_aux_loss + self.reg_amp_w * torch.mean((route_mult_flat - 1.0) ** 2)
                self._last_aux_loss = self._last_aux_loss + self.reg_smooth_w * torch.mean(
                    (route_mult_flat[1:, :, :] - route_mult_flat[:-1, :, :]) ** 2)
                q_routed = q_routed_flat.view(q_comp.shape[0], ngage, self.nmul, 1)
            else:
                q_routed = self._route_q(q_for_routing, routpara).view(q_comp.shape[0], ngage, self.nmul, 1)
            out = self._mix_or_mean(q_routed, wts)
        else:
            if self.routOpt is True and self.dynamic_routing_scale is True and self.comprout is False:
                if diag_comp is not None and 'soil_moisture' in diag_comp:
                    sm_mix = self._mix_component_tensor(diag_comp['soil_moisture'], ngage)
                else:
                    sm_mix = torch.ones_like(q_mix_before_routing) * 50.0
                smsc_mix = torch.ones_like(q_mix_before_routing) * 200.0
                x_route = torch.cat([x_use[:, :, 0:1], sm_mix, smsc_mix, x_use[:, :, 3:4], x_use[:, :, 4:5]], dim=2)
                out, route_mult_seq = self._route_q_dynamic_scale(q_mix_before_routing, routpara, x_route)
                self._last_aux_loss = self._last_aux_loss + self.reg_amp_w * torch.mean((route_mult_seq - 1.0) ** 2)
                self._last_aux_loss = self._last_aux_loss + self.reg_smooth_w * torch.mean(
                    (route_mult_seq[1:, :, :] - route_mult_seq[:-1, :, :]) ** 2)
            elif self.routOpt is True:
                if self.comprout is True:
                    q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
                    q_routed = self._route_q(q_for_routing, routpara).view(q_comp.shape[0], ngage, self.nmul, 1)
                    out = self._mix_or_mean(q_routed, wts)
                else:
                    out = self._route_q(q_mix_before_routing, routpara)
            else:
                out = q_mix_before_routing

        out = torch.clamp(out, min=0.0)
        if return_diagnostics is not True:
            return out

        diag_out = dict()
        if diag_comp is not None:
            for name, tensor_comp in diag_comp.items():
                diag_out[name] = self._mix_component_tensor(tensor_comp, ngage)
        diag_out['total_discharge'] = out
        diag_out['q_mix_before_routing'] = q_mix_before_routing
        diag_out['component_discharge_raw'] = self._mix_or_mean(q_comp_raw, wts)
        if self.dry_channel_loss is True:
            diag_out['channel_loss'] = self._mix_or_mean(channel_loss, wts)
            diag_out['channel_loss_fraction'] = self._mix_or_mean(channel_loss_fraction, wts)
        if self.zero_flow_gate_enabled is True:
            diag_out['zero_flow_probability'] = self._mix_or_mean(zero_flow_probability, wts)
        diag_out['arid_pulse_runoff'] = self._mix_or_mean(arid_pulse_runoff, wts)
        diag_out['arid_pulse_fraction_of_total'] = self._mix_or_mean(arid_pulse_fraction, wts)
        diag_out['arid_pulse_threshold'] = self._mix_or_mean(arid_pulse_threshold, wts)
        diag_out['arid_pulse_coefficient'] = self._mix_or_mean(arid_pulse_coeff, wts)
        diag_out['arid_pulse_dryness'] = self._mix_or_mean(arid_pulse_dryness, wts)
        if route_mult_seq is not None:
            diag_out['route_b_t_multiplier'] = route_mult_seq
        return out, diag_out


class DynamicSimHydModelFiveDifferentiableSlowGW(DynamicSimHydModelFiveDifferentiable):
    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True, dynamic_cfmax_snow=True,
                 dynamic_routing_scale=False, dynamic_all=False, dyn_hidden=32,
                 use_slow_gw=True, kslow_min=0.001, kslow_max=0.050, slow_init_max=300.0):
        super(DynamicSimHydModelFiveDifferentiableSlowGW, self).__init__(
            mode=mode, theta_is_raw=theta_is_raw, smooth=smooth, eps=eps, rain_snow_gain=rain_snow_gain,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, dyn_hidden=dyn_hidden)
        self.use_slow_gw = use_slow_gw
        self.kslow_min = kslow_min
        self.kslow_max = kslow_max
        self.slow_init_max = slow_init_max

    def _expand(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)
        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = 0.0 + theta[:, 2:3] * (6.0 - 0.0)
        SMSC = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = 0.0 + theta[:, 4:5] * (1.0 - 0.0)
        CRAK = 0.0 + theta[:, 5:6] * (1.0 - 0.0)
        K = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        LG = 0.0 + theta[:, 7:8] * (0.2 - 0.0)
        TT = -2.5 + theta[:, 8:9] * (2.5 - (-2.5))
        CFMAX = 0.5 + theta[:, 9:10] * (10.0 - 0.5)
        CFR = 0.0 + theta[:, 10:11] * (0.1 - 0.0)
        CWH = 0.0 + theta[:, 11:12] * (0.2 - 0.0)
        SG_CRIT = 0.0 + theta[:, 12:13] * (300.0 - 0.0)
        RHO_SLOW = torch.sigmoid(theta[:, 13:14] * 6.0 - 3.0)
        K_SLOW = self.kslow_min + (self.kslow_max - self.kslow_min) * torch.sigmoid(theta[:, 14:15] * 6.0 - 3.0)
        S_SLOW_INIT = self.slow_init_max * torch.sigmoid(theta[:, 15:16] * 6.0 - 3.0)
        return INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH, SG_CRIT, RHO_SLOW, K_SLOW, S_SLOW_INIT

    @torch.no_grad()
    def denorm_params(self, theta):
        return torch.cat(self._expand(theta), dim=1)

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype
        INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH, SG_CRIT, RHO_SLOW, K_SLOW, S_SLOW_INIT = self._expand(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SMS = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
            SLOWGW = self._pos(S_SLOW_INIT)
        else:
            SMS = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]
            SNOWPACK = initial_state[:, 2:3]
            MELTWATER = initial_state[:, 3:4]
            SLOWGW = initial_state[:, 4:5]

        if lg_dyn_seq is not None and (lg_dyn_seq.shape[0] != B or lg_dyn_seq.shape[1] != Tlen):
            raise ValueError('lg_dyn_seq must have shape [B, T, 1] matching inputs')

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = dict()
        if return_diagnostics is True:
            for name in [
                'snowpack', 'interception_storage', 'soil_moisture', 'groundwater', 'slow_groundwater_storage',
                'rainfall', 'snowfall', 'snowmelt', 'interception_evaporation', 'actual_ET',
                'infiltration', 'recharge_to_groundwater', 'surface_runoff', 'interflow',
                'baseflow', 'groundwater_loss', 'channel_loss', 'total_discharge',
                'COEF_t', 'SQ_t', 'ETGAM_t', 'SUB_t', 'CRAK_t', 'K_t', 'LG_t', 'SG_CRIT', 'CFMAX_t',
                'slow_recharge', 'slow_baseflow', 'fast_recharge', 'fast_baseflow',
                'quick_runoff', 'event_flow_before_loss', 'event_flow_after_loss'
            ]:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SMS0 = self._min(self._pos(SMS), SMSC)
            GW0 = self._pos(GW)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            SLOWGW0 = self._pos(SLOWGW)
            wetness = torch.clamp(SMS0 / (SMSC + 1e-8), min=0.0, max=1.5)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([Pt / 20.0, Tt / 20.0, E0t / 10.0, SMS0 / 300.0, wetness, GW0 / 300.0, SNOWPACK0 / 300.0, sin_t, cos_t], dim=1)
            dyn_raw = None if self.dynHead is None else self.dynHead(dyn_in)

            m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']]) if self.dynamic_sq is True else torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)
            ETGAM_t = 0.25 + 3.75 * torch.sigmoid(dyn_raw[:, self.dyn_slices['etgam']]) if self.dynamic_etgam is True else torch.ones_like(SQ)
            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf_eff = torch.ones_like(SQ)
            CFMAX_t = CFMAX * m_cf_eff

            frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT)) if self.smooth else (Tt >= TT).float()
            RAIN = Pt * frac_rain
            SNOW = Pt * (1.0 - frac_rain)
            SNOWPACK1 = SNOWPACK0 + SNOW
            melt_pot = CFMAX_t * self._pos(Tt - TT)
            melt = self._min(melt_pot, SNOWPACK1)
            MELTWATER1 = MELTWATER0 + melt
            SNOWPACK2 = SNOWPACK1 - melt
            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            SNOWPACK3 = SNOWPACK2 + refreezing
            MELTWATER2 = MELTWATER1 - refreezing
            water_holding = CWH * SNOWPACK3
            tosoil = self._pos(MELTWATER2 - water_holding)
            MELTWATER_next = self._pos(MELTWATER2 - tosoil)
            SNOWPACK_next = self._pos(SNOWPACK3)
            Peff = RAIN + tosoil

            IMAX = self._min(INSC, E0t)
            INT = self._min(IMAX, Peff)
            INR = self._pos(Peff - INT)
            infil_cap = COEF * torch.exp(-SQ_t * wetness)
            RMO = self._min(infil_cap, INR)
            IRUN_excess = self._pos(INR - RMO)

            available = wetness * RMO
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_inter = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([torch.log(base_surface), torch.log(base_inter), torch.log(base_recharge)], dim=1)
            part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']] if self.dynamic_partition is True else base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]

            SRUN = IRUN_excess + f_surface * available
            IFLOW = f_inter * available
            REC = f_recharge * available
            SMF = self._pos(RMO - available)

            POT = self._pos(E0t - INT)
            et_scale = torch.pow(torch.clamp(wetness, min=1e-6, max=1.0), ETGAM_t)
            et_scale = torch.clamp(et_scale, max=1.0)
            ETS = self._min(POT * et_scale, SMS0 + SMF)

            SMS_pre = SMS0 + SMF - ETS
            SOIL_EXCESS = self._pos(SMS_pre - SMSC)
            SMS_next = self._pos(SMS_pre - SOIL_EXCESS)
            REC_total = REC + SOIL_EXCESS

            if lg_dyn_seq is None:
                LG_t = LG
            else:
                LG_dyn_t = torch.clamp(lg_dyn_seq[:, t, :], 0.0, 1.0)
                LG_eff = (1.0 - lg_dyn_weight) * LG + lg_dyn_weight * (LG * LG_dyn_t * 1.5)
                LG_t = torch.clamp(LG_eff, 0.0, 0.2)

            R_slow = RHO_SLOW * REC_total if self.use_slow_gw else torch.zeros_like(REC_total)
            R_fast = REC_total - R_slow
            BAS_FAST = K * F.softplus(GW0 - SG_CRIT)
            GWLOSS = LG_t * F.softplus(SG_CRIT - GW0)
            GW_next = self._pos(GW0 + R_fast - BAS_FAST - GWLOSS)
            Q_slow = K_SLOW * SLOWGW0
            SLOWGW_next = torch.clamp(SLOWGW0 + R_slow - Q_slow, min=0.0)

            quick_runoff = SRUN + IFLOW
            event_flow_before_loss = quick_runoff + BAS_FAST
            Q = self._pos(event_flow_before_loss + Q_slow)

            SMS = SMS_next
            GW = GW_next
            SLOWGW = SLOWGW_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next

            q_hist.append(Q)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)

            if return_diagnostics is True:
                diag_vals = {
                    'snowpack': SNOWPACK_next,
                    'interception_storage': torch.zeros_like(INT),
                    'soil_moisture': SMS_next,
                    'groundwater': GW_next,
                    'slow_groundwater_storage': SLOWGW_next,
                    'rainfall': RAIN,
                    'snowfall': SNOW,
                    'snowmelt': melt,
                    'interception_evaporation': INT,
                    'actual_ET': ETS,
                    'infiltration': RMO,
                    'recharge_to_groundwater': REC_total,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'baseflow': BAS_FAST + Q_slow,
                    'groundwater_loss': GWLOSS,
                    'channel_loss': torch.zeros_like(Q),
                    'total_discharge': Q,
                    'COEF_t': COEF,
                    'SQ_t': SQ_t,
                    'ETGAM_t': ETGAM_t,
                    'SUB_t': f_surface,
                    'CRAK_t': f_recharge,
                    'K_t': K,
                    'LG_t': LG_t,
                    'SG_CRIT': SG_CRIT,
                    'CFMAX_t': CFMAX_t,
                    'slow_recharge': R_slow,
                    'slow_baseflow': Q_slow,
                    'fast_recharge': R_fast,
                    'fast_baseflow': BAS_FAST,
                    'quick_runoff': quick_runoff,
                    'event_flow_before_loss': event_flow_before_loss,
                    'event_flow_after_loss': event_flow_before_loss,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SMS, GW, SNOWPACK, MELTWATER, SLOWGW], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)
        reg_terms = {'dynamic_amplitude_loss': reg_amp, 'dynamic_smoothness_loss': reg_smooth, 'partition_entropy_loss': reg_part}

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out
        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelLowNSE_AridSlowGW(MultiInv_DynamicSimHydModelSix):
    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 component_routing=True, dry_channel_loss=True, zero_flow_gate=True,
                 channel_loss_max=0.60, zero_gate_hidden=None,
                 use_slow_gw=True, kslow_min=0.001, kslow_max=0.050, slow_init_max=300.0):
        super(MultiInv_DynamicSimHydModelLowNSE_AridSlowGW, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing,
            dry_channel_loss=dry_channel_loss, zero_flow_gate=zero_flow_gate,
            channel_loss_max=channel_loss_max, zero_gate_hidden=zero_gate_hidden)
        self.nfea = 16
        self.nstaticpm = self.nfea * nmul
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.simhyd = DynamicSimHydModelFiveDifferentiableSlowGW(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            use_slow_gw=use_slow_gw, kslow_min=kslow_min, kslow_max=kslow_max, slow_init_max=slow_init_max)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiableSlowGW(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all,
            use_slow_gw=use_slow_gw, kslow_min=kslow_min, kslow_max=kslow_max, slow_init_max=slow_init_max)

    def forward(self, x, z, doDropMC=False, return_diagnostics=False):
        nt_x = x.shape[0]
        ngage = z.shape[1]
        basin_attr = z[-1, :, -self.nattr:]
        snow_frac_raw = torch.clamp(z[-1, :, -self.nattr - 1:-self.nattr], min=0.0, max=1.0) if z.shape[2] > self.nattr else None

        staticFeat = self.staticFeat(basin_attr)
        staticParams0 = self.staticOut(staticFeat)
        cursor = 0
        static0 = staticParams0[:, cursor:cursor + self.nstaticpm].view(ngage, self.nfea, self.nmul) + self.compStaticBias
        snowpara = torch.sigmoid(static0)
        cursor += self.nstaticpm
        routpara0 = staticParams0[:, cursor:cursor + self.nroutpm]
        routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2) if self.component_routing is True else torch.sigmoid(routpara0)
        cursor += self.nroutpm
        if self.nwtspm == 0:
            wts = None
        else:
            wtspara = staticParams0[:, cursor:cursor + self.nwtspm] + self.compWeightBias
            wts = F.softmax(wtspara, dim=-1)
        self._last_wts = wts

        if self.lgdyn is False:
            lg_dyn = None
        else:
            lg_dyn_seq = self.lstmdyn(z)
            lg_attr_bias = self.lgAttr(basin_attr).unsqueeze(0).repeat(lg_dyn_seq.shape[0], 1, 1)
            lg_dyn = torch.sigmoid(lg_dyn_seq + lg_attr_bias)

        x_rep = x.unsqueeze(2).repeat(1, 1, self.nmul, 1).view(nt_x, ngage * self.nmul, x.shape[2])
        x_bt = x_rep.permute(1, 0, 2)
        theta = snowpara.permute(0, 2, 1).contiguous().view(ngage * self.nmul, self.nfea)
        lg_bt = None if lg_dyn is None else lg_dyn.permute(1, 2, 0).contiguous().view(ngage * self.nmul, lg_dyn.shape[0], 1)
        snow_frac_rep = None if snow_frac_raw is None else snow_frac_raw.unsqueeze(1).repeat(1, self.nmul, 1).view(ngage * self.nmul, 1)

        need_process_diag = return_diagnostics or self.dry_channel_loss or self.zero_flow_gate_enabled
        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            x_use = x[self.inittime:, :, :]
            _, warm_state = self.simhyd_analysis(
                warm_inputs, theta, lg_dyn_seq=None, lg_dyn_weight=self.lgdynweight,
                snow_frac_raw=snow_frac_rep, return_final_state=True)
            if need_process_diag:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    main_inputs, theta, initial_state=warm_state, lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight, snow_frac_raw=snow_frac_rep,
                    return_diagnostics=True, return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    main_inputs, theta, initial_state=warm_state, lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight, snow_frac_raw=snow_frac_rep,
                    return_regularization=True)
                diag_comp = None
        else:
            x_use = x
            if need_process_diag:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    x_bt, theta, lg_dyn_seq=lg_bt, lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep, return_diagnostics=True, return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    x_bt, theta, lg_dyn_seq=lg_bt, lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep, return_regularization=True)
                diag_comp = None

        reg_total = self.reg_amp_w * reg_terms['dynamic_amplitude_loss'] + self.reg_smooth_w * reg_terms['dynamic_smoothness_loss'] + self.reg_part_w * reg_terms['partition_entropy_loss']
        self._last_aux_terms = reg_terms
        self._last_aux_loss = reg_total

        q_comp_raw = q_seq.view(ngage, self.nmul, q_seq.shape[1], 1).permute(2, 0, 1, 3)
        q_comp_raw = torch.clamp(q_comp_raw, min=0.0)
        if diag_comp is not None and 'slow_baseflow' in diag_comp:
            q_slow = self._component_tensor_4d(diag_comp['slow_baseflow'], ngage)
        else:
            q_slow = torch.zeros_like(q_comp_raw)
        if diag_comp is not None and 'event_flow_before_loss' in diag_comp:
            q_event = self._component_tensor_4d(diag_comp['event_flow_before_loss'], ngage)
        else:
            q_event = torch.clamp(q_comp_raw - q_slow, min=0.0)

        q_event_after_loss, channel_loss, channel_loss_fraction = self._apply_channel_loss(q_event, diag_comp, theta, basin_attr, ngage)
        q_event_after_gate, zero_flow_probability = self._apply_zero_flow_gate(q_event_after_loss, x_use, diag_comp, theta, ngage)
        q_event_after_gate = torch.clamp(q_event_after_gate, min=0.0)
        q_comp = torch.clamp(q_event_after_gate + q_slow, min=0.0)
        q_mix_before_routing = self._mix_or_mean(q_comp, wts)
        route_mult_seq = None
        if self.routOpt is True and self.component_routing is True:
            q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
            if self.dynamic_routing_scale is True:
                if diag_comp is not None and 'soil_moisture' in diag_comp:
                    sm_comp = self._component_tensor_4d(diag_comp['soil_moisture'], ngage)
                else:
                    sm_comp = torch.ones_like(q_comp) * 50.0
                smsc_comp = self._theta_to_smsc(theta, ngage).unsqueeze(0).repeat(q_comp.shape[0], 1, 1, 1)
                p_rep = x_use[:, :, 0:1].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                if x_use.shape[-1] >= 5:
                    sin_rep = x_use[:, :, 3:4].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                    cos_rep = x_use[:, :, 4:5].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                else:
                    sin_rep = torch.zeros_like(q_comp)
                    cos_rep = torch.ones_like(q_comp)
                x_route = torch.cat([p_rep, sm_comp, smsc_comp, sin_rep, cos_rep], dim=-1).view(q_comp.shape[0], ngage * self.nmul, 5)
                q_routed_flat, route_mult_flat = self._route_q_dynamic_scale(q_for_routing, routpara, x_route)
                route_mult_4d = route_mult_flat.view(q_comp.shape[0], ngage, self.nmul, 1)
                route_mult_seq = self._mix_or_mean(route_mult_4d, wts)
                self._last_aux_loss = self._last_aux_loss + self.reg_amp_w * torch.mean((route_mult_flat - 1.0) ** 2)
                self._last_aux_loss = self._last_aux_loss + self.reg_smooth_w * torch.mean((route_mult_flat[1:, :, :] - route_mult_flat[:-1, :, :]) ** 2)
                q_routed = q_routed_flat.view(q_comp.shape[0], ngage, self.nmul, 1)
            else:
                q_routed = self._route_q(q_for_routing, routpara).view(q_comp.shape[0], ngage, self.nmul, 1)
            out = self._mix_or_mean(q_routed, wts)
        else:
            if self.routOpt is True:
                out = self._route_q(q_mix_before_routing, routpara)
            else:
                out = q_mix_before_routing
        out = torch.clamp(out, min=0.0)
        if return_diagnostics is not True:
            return out
        diag_out = dict()
        if diag_comp is not None:
            for name, tensor_comp in diag_comp.items():
                diag_out[name] = self._mix_component_tensor(tensor_comp, ngage)
        diag_out['event_flow'] = self._mix_or_mean(q_event_after_gate, wts)
        diag_out['slow_baseflow'] = self._mix_or_mean(q_slow, wts)
        if diag_comp is not None and 'slow_groundwater_storage' in diag_comp:
            diag_out['slow_groundwater_storage'] = self._mix_component_tensor(diag_comp['slow_groundwater_storage'], ngage)
        if diag_comp is not None and 'slow_recharge' in diag_comp:
            diag_out['slow_recharge'] = self._mix_component_tensor(diag_comp['slow_recharge'], ngage)
        diag_out['total_discharge'] = out
        diag_out['q_mix_before_routing'] = q_mix_before_routing
        diag_out['component_discharge_raw'] = self._mix_or_mean(q_comp_raw, wts)
        if self.dry_channel_loss is True:
            diag_out['channel_loss'] = self._mix_or_mean(channel_loss, wts)
            diag_out['channel_loss_fraction'] = self._mix_or_mean(channel_loss_fraction, wts)
        if self.zero_flow_gate_enabled is True:
            diag_out['zero_flow_probability'] = self._mix_or_mean(zero_flow_probability, wts)
        if route_mult_seq is not None:
            diag_out['route_b_t_multiplier'] = route_mult_seq
        return out, diag_out


class MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleLAIRegimeMoE(
        MultiInv_DynamicSimHydModelSix):
    """
    Regime-adaptive closed Model 6 wrapper that mixes three LAI-aware experts
    before a shared routing step.

    Experts:
    - humid_base: closed simple LAI model
    - dry_pulse: threshold-pulse dry-basin specialist
    - slow_gw: slow-groundwater dry-basin specialist
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 gate_hidden=None, gate_entropy_w=1e-3, gate_smooth_w=1e-3,
                 component_routing=False):
        super(MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleLAIRegimeMoE, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=False, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing,
            dry_channel_loss=False, zero_flow_gate=False, channel_loss_max=0.0, zero_gate_hidden=gate_hidden)
        self.expert_names = ('humid_base', 'dry_pulse', 'slow_gw')
        self.attr_idx_frac_snow = 3
        self.attr_idx_aridity = 4
        self.routeStaticOut = nn.Linear(hiddeninv, self.nroutpm)
        gate_hidden = hiddeninv if gate_hidden is None else gate_hidden
        gate_in_dim = nattr + 8
        self.regimeGate = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, len(self.expert_names)),
        )
        self.regimeStaticBias = nn.Linear(nattr, len(self.expert_names))
        self.gate_entropy_w = gate_entropy_w
        self.gate_smooth_w = gate_smooth_w

        expert_kwargs = dict(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=False, comprout=False, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=False, dynamic_all=dynamic_all,
            reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w, reg_part_w=reg_part_w,
        )
        self.humid_expert = MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimpleLAIEco(
            component_routing=False, **expert_kwargs)
        dry_kwargs = dict(expert_kwargs)
        dry_kwargs.update({"lgdyn": False})
        self.dry_pulse_expert = MultiInv_DynamicSimHydModelLowNSE_AridPulse(
            component_routing=False, dry_channel_loss=False, zero_flow_gate=False,
            channel_loss_max=0.0, **dry_kwargs)
        self.slow_gw_expert = MultiInv_DynamicSimHydModelLowNSE_AridSlowGW(
            component_routing=False, dry_channel_loss=False, zero_flow_gate=False,
            channel_loss_max=0.0, **dry_kwargs)
        nn.init.zeros_(self.regimeGate[-1].weight)
        nn.init.zeros_(self.regimeGate[-1].bias)
        nn.init.zeros_(self.regimeStaticBias.weight)
        with torch.no_grad():
            self.regimeStaticBias.bias.copy_(torch.tensor([1.25, -0.35, -0.55], dtype=self.regimeStaticBias.bias.dtype))

    def _zeros_like_diag(self, like):
        return torch.zeros_like(like)

    def _route_mixed_q(self, q_mix, x, diag_mix, basin_attr):
        if self.routOpt is not True:
            return torch.clamp(q_mix, min=0.0), None
        static_feat = self.staticFeat(basin_attr)
        routpara = torch.sigmoid(self.routeStaticOut(static_feat))
        route_mult_seq = None
        if self.dynamic_routing_scale is True:
            sm_mix = diag_mix.get('Sa', self._zeros_like_diag(q_mix))
            smsc_mix = torch.clamp(diag_mix.get('theta_cap', torch.ones_like(q_mix) * 300.0), min=10.0)
            x_route = torch.cat([x[:, :, 0:1], sm_mix, smsc_mix, x[:, :, 3:4], x[:, :, 4:5]], dim=2)
            out, route_mult_seq = self._route_q_dynamic_scale(q_mix, routpara, x_route)
            return torch.clamp(out, min=0.0), route_mult_seq
        return torch.clamp(self._route_q(q_mix, routpara), min=0.0), route_mult_seq

    def _standardize_expert_diag(self, diag, expert_name):
        like = diag['total_discharge']
        zeros = torch.zeros_like(like)
        standard = {}
        if expert_name == 'humid_base':
            standard.update(diag)
            standard.setdefault('SMS', standard.get('Sa', zeros))
            standard.setdefault('Q_process', standard.get('total_discharge', zeros))
            return standard

        soil = diag.get('soil_moisture', zeros)
        gw_fast = diag.get('groundwater', zeros)
        slow_gw = diag.get('slow_groundwater_storage', zeros)
        gw = gw_fast + slow_gw
        theta_cap = torch.ones_like(soil) * 300.0
        s_moist = torch.clamp(soil / torch.clamp(theta_cap, min=1e-6), 0.0, 1.0)
        q_process = diag.get('q_mix_before_routing', diag.get('event_flow', diag.get('total_discharge', zeros)))
        srun = diag.get('surface_runoff', zeros)
        if expert_name == 'dry_pulse':
            srun = srun + diag.get('arid_pulse_runoff', zeros)
        baseflow = diag.get('baseflow', zeros)
        if expert_name == 'slow_gw':
            baseflow = baseflow + diag.get('slow_baseflow', zeros)

        standard.update({
            'precipitation': diag.get('precipitation', zeros),
            'rainfall': diag.get('rainfall', zeros),
            'snowfall': diag.get('snowfall', zeros),
            'snowmelt': diag.get('snowmelt', zeros),
            'refreezing': zeros,
            'snow_release': zeros,
            'PL': diag.get('precipitation', zeros),
            'interception_evaporation': diag.get('interception_evaporation', zeros),
            'actual_ET': diag.get('actual_ET', zeros),
            'LAI_t': diag.get('LAI_t', zeros),
            'LAI_rel': zeros,
            'LAI_interception_scalar': torch.ones_like(soil),
            'LAI_et_scalar': torch.ones_like(soil),
            'alpha_base': 1.0 - s_moist,
            'alpha_veg_scalar': torch.ones_like(soil),
            'Smoist': s_moist,
            'theta_cap': theta_cap,
            'alpha': torch.clamp(1.0 - s_moist, 0.0, 1.0),
            'P_accessible': zeros,
            'P_inaccessible': zeros,
            'Sa_overflow': zeros,
            'surface_runoff': srun,
            'interflow': diag.get('interflow', zeros),
            'recharge_to_groundwater': diag.get('recharge_to_groundwater', zeros),
            'baseflow': baseflow,
            'Q_process': q_process,
            'groundwater_loss': diag.get('groundwater_loss', zeros),
            'channel_loss': diag.get('channel_loss', zeros),
            'gate_loss': diag.get('gate_loss', zeros),
            'partition_sum_error': diag.get('partition_sum_error', zeros),
            'SNOWPACK': diag.get('snowpack', zeros),
            'MELTWATER': zeros,
            'Sa': soil,
            'SMS': soil,
            'GW': gw,
            'INSC': diag.get('INSC', zeros),
            'INSC_t': diag.get('INSC_t', diag.get('INSC', zeros)),
            'COEF_t': diag.get('COEF_t', zeros),
            'SQ_t': diag.get('SQ_t', zeros),
            'K_t': diag.get('K_t', zeros),
            'TT': diag.get('TT', zeros),
            'CFMAX_t': diag.get('CFMAX_t', zeros),
            'CFR': diag.get('CFR', zeros),
            'CWH': diag.get('CWH', zeros),
        })
        return standard

    def _mix_expert_diags(self, standard_diags, weights):
        mixed = {}
        all_keys = set()
        for diag in standard_diags:
            all_keys.update(diag.keys())
        for key in all_keys:
            vals = []
            for i, diag in enumerate(standard_diags):
                if key in diag:
                    vals.append(diag[key] * weights[:, :, i:i + 1])
            if vals:
                mixed[key] = torch.sum(torch.stack(vals, dim=0), dim=0)
        return mixed

    def _compute_process_residual(self, diag_mix):
        precip = diag_mix.get('precipitation')
        q_proc = diag_mix.get('Q_process')
        if precip is None or q_proc is None:
            return None
        snow = diag_mix.get('SNOWPACK', self._zeros_like_diag(precip))
        melt = diag_mix.get('MELTWATER', self._zeros_like_diag(precip))
        soil = diag_mix.get('Sa', self._zeros_like_diag(precip))
        gw = diag_mix.get('GW', self._zeros_like_diag(precip))
        total_store = snow + melt + soil + gw
        prev_store = torch.cat([total_store[0:1], total_store[:-1]], dim=0)
        return precip - diag_mix.get('interception_evaporation', self._zeros_like_diag(precip)) \
            - diag_mix.get('actual_ET', self._zeros_like_diag(precip)) \
            - q_proc - (total_store - prev_store)

    def _repeat_components(self, tensor_mix):
        if tensor_mix.ndim == 2:
            tensor_mix = tensor_mix.unsqueeze(-1)
        return tensor_mix.repeat(1, 1, self.nmul)

    def get_auxiliary_loss(self):
        if self._last_aux_loss is None:
            return None
        return self._last_aux_loss

    def forward(self, x, z, doDropMC=False, return_diagnostics=False, return_component_diagnostics=False):
        self.humid_expert.inittime = self.inittime
        self.dry_pulse_expert.inittime = self.inittime
        self.slow_gw_expert.inittime = self.inittime

        humid_q, humid_diag = self.humid_expert(x, z, return_diagnostics=True)
        dry_q, dry_diag = self.dry_pulse_expert(x, z, return_diagnostics=True)
        slow_q, slow_diag = self.slow_gw_expert(x, z, return_diagnostics=True)

        expert_diags = [
            self._standardize_expert_diag(humid_diag, 'humid_base'),
            self._standardize_expert_diag(dry_diag, 'dry_pulse'),
            self._standardize_expert_diag(slow_diag, 'slow_gw'),
        ]
        q_h = expert_diags[0].get('total_discharge', humid_q)
        q_d = expert_diags[1].get('total_discharge', dry_q)
        q_s = expert_diags[2].get('total_discharge', slow_q)
        q_stack = torch.stack([q_h, q_d, q_s], dim=2)

        x_use = x[self.inittime:, :, :] if self.inittime > 0 and x.shape[0] > q_h.shape[0] else x
        basin_attr = z[-1, :, -self.nattr:]
        attr_rep = basin_attr.unsqueeze(0).repeat(q_h.shape[0], 1, 1)
        wetness = expert_diags[0].get('Smoist', torch.zeros_like(q_h))
        snow_flag = (expert_diags[0].get('SNOWPACK', torch.zeros_like(q_h)) > 1.0).float()
        lai_t = x_use[:, :, 5:6] if x_use.shape[2] >= 6 else torch.zeros_like(q_h)
        gate_in = torch.cat([
            attr_rep,
            x_use[:, :, 0:1] / 20.0,
            x_use[:, :, 2:3] / 10.0,
            x_use[:, :, 1:2] / 20.0,
            wetness,
            snow_flag,
            torch.clamp(lai_t / 6.0, 0.0, 1.0),
            x_use[:, :, 3:4],
            x_use[:, :, 4:5],
        ], dim=2)
        gate_logits = self.regimeGate(gate_in.reshape(-1, gate_in.shape[-1])).view(q_h.shape[0], q_h.shape[1], len(self.expert_names))
        gate_logits = gate_logits + self.regimeStaticBias(basin_attr).unsqueeze(0)

        aridity = basin_attr[:, self.attr_idx_aridity:self.attr_idx_aridity + 1].unsqueeze(0)
        snow_frac = basin_attr[:, self.attr_idx_frac_snow:self.attr_idx_frac_snow + 1].unsqueeze(0)
        p_norm = torch.clamp(x_use[:, :, 0:1] / 20.0, 0.0, 2.0)
        wetness = torch.clamp(wetness, 0.0, 1.0)
        prior_logits = torch.cat([
            1.75 + 0.65 * snow_frac - 1.15 * aridity - 0.20 * p_norm,
            -0.20 - 0.15 * snow_frac + 0.75 * aridity + 0.35 * p_norm - 0.30 * wetness,
            -0.45 - 0.10 * snow_frac + 0.95 * aridity + 0.30 * wetness,
        ], dim=2)
        learned_weights = F.softmax(gate_logits, dim=2)
        prior_weights = F.softmax(prior_logits, dim=2)
        gate_weights = 0.75 * prior_weights + 0.25 * learned_weights

        q_mix_before_routing = torch.sum(q_stack * gate_weights.unsqueeze(-1), dim=2)
        diag_mix = self._mix_expert_diags(expert_diags, gate_weights)
        diag_mix['Q_process'] = q_mix_before_routing
        diag_mix['q_mix_before_routing'] = q_mix_before_routing
        diag_mix['q_after_gate'] = q_mix_before_routing
        residual = self._compute_process_residual(diag_mix)
        if residual is not None:
            diag_mix['process_local_residual'] = residual

        out, route_mult_seq = self._route_mixed_q(q_mix_before_routing, x_use, diag_mix, basin_attr)
        diag_mix['total_discharge'] = out
        if route_mult_seq is not None:
            diag_mix['route_b_t_multiplier'] = route_mult_seq
        diag_mix['regime_weight_humid'] = gate_weights[:, :, 0:1]
        diag_mix['regime_weight_dry_pulse'] = gate_weights[:, :, 1:2]
        diag_mix['regime_weight_slow_gw'] = gate_weights[:, :, 2:3]

        aux = torch.tensor(0.0, device=out.device, dtype=out.dtype)
        for expert in (self.humid_expert, self.dry_pulse_expert, self.slow_gw_expert):
            expert_aux = expert.get_auxiliary_loss()
            if expert_aux is not None:
                aux = aux + expert_aux
        entropy = -torch.sum(gate_weights * torch.log(torch.clamp(gate_weights, min=1e-6)), dim=2).mean()
        entropy_pen = torch.relu(0.85 - entropy) ** 2
        smooth_pen = torch.tensor(0.0, device=out.device, dtype=out.dtype)
        if gate_weights.shape[0] > 1:
            smooth_pen = torch.mean((gate_weights[1:, :, :] - gate_weights[:-1, :, :]) ** 2)
        self._last_aux_loss = aux + self.gate_entropy_w * entropy_pen + self.gate_smooth_w * smooth_pen
        self._last_aux_terms = {
            'gate_entropy_penalty': entropy_pen,
            'gate_smoothness_penalty': smooth_pen,
        }

        if return_diagnostics is not True and return_component_diagnostics is not True:
            return out

        diag_out = dict(diag_mix)
        if return_component_diagnostics is True:
            for key, val in list(diag_mix.items()):
                if torch.is_tensor(val) and val.ndim == 3 and val.shape[-1] == 1:
                    diag_out[key + '_components'] = self._repeat_components(val.squeeze(-1))
            diag_out['q_raw_process_components'] = self._repeat_components(q_mix_before_routing.squeeze(-1))
            diag_out['q_after_channel_loss_components'] = self._repeat_components(q_mix_before_routing.squeeze(-1))
            diag_out['q_after_gate_components'] = self._repeat_components(q_mix_before_routing.squeeze(-1))
            diag_out['q_routed_components'] = self._repeat_components(out.squeeze(-1))
            diag_out['component_weights'] = torch.ones((out.shape[1], self.nmul), device=out.device, dtype=out.dtype) / float(self.nmul)
        return out, diag_out


class DynamicSimHydModelFiveDifferentiableConnectivity(DynamicSimHydModelFiveDifferentiable):
    def __init__(self, mode='normal', theta_is_raw=False, smooth=True, eps=1e-4, rain_snow_gain=5.0,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True, dynamic_cfmax_snow=True,
                 dynamic_routing_scale=False, dynamic_all=False, dyn_hidden=32):
        super(DynamicSimHydModelFiveDifferentiableConnectivity, self).__init__(
            mode=mode, theta_is_raw=theta_is_raw, smooth=smooth, eps=eps, rain_snow_gain=rain_snow_gain,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, dyn_hidden=dyn_hidden)

    def _expand(self, theta):
        if self.theta_is_raw:
            theta = torch.sigmoid(theta)

        INSC = 0.5 + theta[:, 0:1] * (5.0 - 0.5)
        COEF = 50.0 + theta[:, 1:2] * (400.0 - 50.0)
        SQ = 0.0 + theta[:, 2:3] * (6.0 - 0.0)
        SMSC = 50.0 + theta[:, 3:4] * (500.0 - 50.0)
        SUB = 0.0 + theta[:, 4:5] * (1.0 - 0.0)
        CRAK = 0.0 + theta[:, 5:6] * (1.0 - 0.0)
        K = 0.003 + theta[:, 6:7] * (0.3 - 0.003)
        LG = 0.0 + theta[:, 7:8] * (0.2 - 0.0)
        TT = -2.5 + theta[:, 8:9] * (2.5 - (-2.5))
        CFMAX = 0.5 + theta[:, 9:10] * (10.0 - 0.5)
        CFR = 0.0 + theta[:, 10:11] * (0.1 - 0.0)
        CWH = 0.0 + theta[:, 11:12] * (0.2 - 0.0)
        SG_CRIT = 0.0 + theta[:, 12:13] * (300.0 - 0.0)
        C_MAX = 5.0 + theta[:, 13:14] * (150.0 - 5.0)
        C_INIT_FRAC = theta[:, 14:15]
        K_CONN = 0.005 + theta[:, 15:16] * (0.30 - 0.005)
        A_CONN = 0.01 + theta[:, 16:17] * (1.00 - 0.01)
        C_CRIT_FRAC = 0.05 + theta[:, 17:18] * (0.90 - 0.05)
        return INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH, SG_CRIT, C_MAX, C_INIT_FRAC, K_CONN, A_CONN, C_CRIT_FRAC

    @torch.no_grad()
    def denorm_params(self, theta):
        return torch.cat(self._expand(theta), dim=1)

    def forward(self, inputs, theta, initial_state=None, lg_dyn_seq=None, lg_dyn_weight=0.6,
                snow_frac_raw=None, return_diagnostics=False, return_final_state=False,
                return_regularization=False):
        B, Tlen, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        INSC, COEF, SQ, SMSC, SUB, CRAK, K, LG, TT, CFMAX, CFR, CWH, SG_CRIT, C_MAX, C_INIT_FRAC, K_CONN, A_CONN, C_CRIT_FRAC = self._expand(theta)

        P = self._pos(inputs[:, :, 0:1])
        TEMP = inputs[:, :, 1:2]
        E0 = self._pos(inputs[:, :, 2:3])

        if initial_state is None:
            SMS = torch.zeros(B, 1, device=device, dtype=dtype)
            GW = torch.zeros(B, 1, device=device, dtype=dtype)
            SNOWPACK = torch.zeros(B, 1, device=device, dtype=dtype)
            MELTWATER = torch.zeros(B, 1, device=device, dtype=dtype)
            CONN = torch.clamp(C_INIT_FRAC * C_MAX, min=0.0)
        else:
            SMS = initial_state[:, 0:1]
            GW = initial_state[:, 1:2]
            SNOWPACK = initial_state[:, 2:3]
            MELTWATER = initial_state[:, 3:4]
            CONN = initial_state[:, 4:5]

        if lg_dyn_seq is not None and (lg_dyn_seq.shape[0] != B or lg_dyn_seq.shape[1] != Tlen):
            raise ValueError('lg_dyn_seq must have shape [B, T, 1] matching inputs')

        if snow_frac_raw is None:
            snow_mask = torch.ones(B, 1, device=device, dtype=dtype)
        else:
            snow_mask = (snow_frac_raw > 0.05).float()

        q_hist = []
        diag_hist = dict()
        if return_diagnostics is True:
            for name in [
                    'snowpack', 'interception_storage', 'soil_moisture', 'groundwater',
                    'rainfall', 'snowfall', 'snowmelt', 'interception_evaporation', 'actual_ET',
                    'infiltration', 'recharge_to_groundwater', 'surface_runoff', 'interflow',
                    'baseflow', 'groundwater_loss', 'channel_loss', 'total_discharge',
                    'COEF_t', 'SQ_t', 'ETGAM_t', 'SUB_t', 'CRAK_t', 'K_t', 'LG_t', 'SG_CRIT',
                    'CFMAX_t', 'connectivity_storage', 'connectivity_fraction', 'connectivity_gate']:
                diag_hist[name] = []

        sq_mult_hist = []
        cfmax_mult_hist = []
        recharge_frac_hist = []

        for t in range(Tlen):
            Pt = P[:, t, :]
            Tt = TEMP[:, t, :]
            E0t = E0[:, t, :]

            SMS0 = self._min(self._pos(SMS), SMSC)
            GW0 = self._pos(GW)
            SNOWPACK0 = self._pos(SNOWPACK)
            MELTWATER0 = self._pos(MELTWATER)
            CONN0 = self._pos(CONN)
            wetness = torch.clamp(SMS0 / (SMSC + 1e-8), min=0.0, max=1.5)

            sin_t, cos_t = self._seasonal_feats(inputs, t, device, dtype)
            dyn_in = torch.cat([
                Pt / 20.0,
                Tt / 20.0,
                E0t / 10.0,
                SMS0 / 300.0,
                wetness,
                GW0 / 300.0,
                SNOWPACK0 / 300.0,
                sin_t,
                cos_t], dim=1)
            dyn_raw = self._run_dyn_head(dyn_in)

            if self.dynamic_sq is True:
                m_sq = 0.5 + 1.5 * torch.sigmoid(dyn_raw[:, self.dyn_slices['sq']])
            else:
                m_sq = torch.ones_like(SQ)
            SQ_t = torch.clamp(SQ * m_sq, 0.0, 6.0)

            if self.dynamic_etgam is True:
                ETGAM_t = 0.25 + 3.75 * torch.sigmoid(dyn_raw[:, self.dyn_slices['etgam']])
            else:
                ETGAM_t = torch.ones_like(SQ)

            if self.dynamic_cfmax_snow is True:
                m_cf = 0.7 + 0.8 * torch.sigmoid(dyn_raw[:, self.dyn_slices['cfmax']])
                m_cf_eff = snow_mask * m_cf + (1.0 - snow_mask)
            else:
                m_cf = torch.ones_like(SQ)
                m_cf_eff = m_cf
            CFMAX_t = CFMAX * m_cf_eff

            if self.smooth:
                frac_rain = torch.sigmoid(self.rain_snow_gain * (Tt - TT))
            else:
                frac_rain = (Tt >= TT).float()

            RAIN = Pt * frac_rain
            SNOW = Pt * (1.0 - frac_rain)

            SNOWPACK1 = SNOWPACK0 + SNOW
            melt_pot = CFMAX_t * self._pos(Tt - TT)
            melt = self._min(melt_pot, SNOWPACK1)
            MELTWATER1 = MELTWATER0 + melt
            SNOWPACK2 = SNOWPACK1 - melt

            refreeze_pot = CFR * CFMAX_t * self._pos(TT - Tt)
            refreezing = self._min(refreeze_pot, MELTWATER1)
            SNOWPACK3 = SNOWPACK2 + refreezing
            MELTWATER2 = MELTWATER1 - refreezing

            water_holding = CWH * SNOWPACK3
            tosoil = self._pos(MELTWATER2 - water_holding)
            MELTWATER_next = self._pos(MELTWATER2 - tosoil)
            SNOWPACK_next = self._pos(SNOWPACK3)
            Peff = RAIN + tosoil

            CONN_input = A_CONN * Peff
            CONN_next = torch.clamp(CONN0 + CONN_input - K_CONN * CONN0, min=0.0)
            CONN_next = self._min(CONN_next, C_MAX)
            conn_frac = CONN_next / (C_MAX + self.eps)
            conn_t = torch.sigmoid((conn_frac - C_CRIT_FRAC) / 0.05)

            IMAX = self._min(INSC, E0t)
            INT = self._min(IMAX, Peff)
            INR = self._pos(Peff - INT)

            infil_cap = COEF * torch.exp(-SQ_t * wetness)
            RMO = self._min(infil_cap, INR)
            IRUN_excess = self._pos(INR - RMO)

            available = wetness * RMO
            base_surface = torch.clamp(SUB, 1e-6, 1.0)
            base_recharge = torch.clamp((1.0 - SUB) * CRAK, 1e-6, 1.0)
            base_inter = torch.clamp((1.0 - SUB) * (1.0 - CRAK), 1e-6, 1.0)
            base_logits = torch.cat([
                torch.log(base_surface),
                torch.log(base_inter),
                torch.log(base_recharge)], dim=1)
            if self.dynamic_partition is True:
                part_logits = base_logits + dyn_raw[:, self.dyn_slices['partition']]
            else:
                part_logits = base_logits
            part_frac = F.softmax(part_logits, dim=1)
            f_surface = part_frac[:, 0:1]
            f_inter = part_frac[:, 1:2]
            f_recharge = part_frac[:, 2:3]

            SRUN = IRUN_excess + f_surface * available
            IFLOW = f_inter * available
            REC = f_recharge * available
            SMF = self._pos(RMO - available)

            POT = self._pos(E0t - INT)
            et_scale = torch.pow(torch.clamp(wetness, min=1e-6, max=1.0), ETGAM_t)
            et_scale = torch.clamp(et_scale, max=1.0)
            ETS = POT * et_scale
            ETS = self._min(ETS, SMS0 + SMF)

            SMS_pre = SMS0 + SMF - ETS
            SOIL_EXCESS = self._pos(SMS_pre - SMSC)
            SMS_next = self._pos(SMS_pre - SOIL_EXCESS)
            REC_total = REC + SOIL_EXCESS

            if lg_dyn_seq is None:
                LG_t = LG
            else:
                LG_dyn_t = torch.clamp(lg_dyn_seq[:, t, :], 0.0, 1.0)
                LG_eff = (1.0 - lg_dyn_weight) * LG + lg_dyn_weight * (LG * LG_dyn_t * 1.5)
                LG_t = torch.clamp(LG_eff, 0.0, 0.2)

            BAS = K * F.softplus(GW0 - SG_CRIT)
            GWLOSS = LG_t * F.softplus(SG_CRIT - GW0)
            GW_next = self._pos(GW0 + REC_total - BAS - GWLOSS)
            Q = self._pos(SRUN + IFLOW + BAS)

            SMS = SMS_next
            GW = GW_next
            SNOWPACK = SNOWPACK_next
            MELTWATER = MELTWATER_next
            CONN = CONN_next

            q_hist.append(Q)
            sq_mult_hist.append(m_sq)
            cfmax_mult_hist.append(m_cf_eff)
            recharge_frac_hist.append(f_recharge)

            if return_diagnostics is True:
                diag_vals = {
                    'snowpack': SNOWPACK_next,
                    'interception_storage': torch.zeros_like(INT),
                    'soil_moisture': SMS_next,
                    'groundwater': GW_next,
                    'rainfall': RAIN,
                    'snowfall': SNOW,
                    'snowmelt': melt,
                    'interception_evaporation': INT,
                    'actual_ET': ETS,
                    'infiltration': RMO,
                    'recharge_to_groundwater': REC_total,
                    'surface_runoff': SRUN,
                    'interflow': IFLOW,
                    'baseflow': BAS,
                    'groundwater_loss': GWLOSS,
                    'channel_loss': torch.zeros_like(Q),
                    'total_discharge': Q,
                    'COEF_t': COEF,
                    'SQ_t': SQ_t,
                    'ETGAM_t': ETGAM_t,
                    'SUB_t': f_surface,
                    'CRAK_t': f_recharge,
                    'K_t': K,
                    'LG_t': LG_t,
                    'SG_CRIT': SG_CRIT,
                    'CFMAX_t': CFMAX_t,
                    'connectivity_storage': CONN_next,
                    'connectivity_fraction': conn_frac,
                    'connectivity_gate': conn_t,
                }
                for name, val in diag_vals.items():
                    diag_hist[name].append(val)

        q_seq = torch.stack(q_hist, dim=1)
        final_state = torch.cat([SMS, GW, SNOWPACK, MELTWATER, CONN], dim=1)

        reg_amp = torch.tensor(0.0, device=device, dtype=dtype)
        reg_smooth = torch.tensor(0.0, device=device, dtype=dtype)
        reg_part = torch.tensor(0.0, device=device, dtype=dtype)
        if len(sq_mult_hist) > 0:
            sq_seq = torch.stack(sq_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((sq_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((sq_seq[:, 1:, :] - sq_seq[:, :-1, :]) ** 2)
        if len(cfmax_mult_hist) > 0:
            cf_seq = torch.stack(cfmax_mult_hist, dim=1)
            reg_amp = reg_amp + torch.mean((cf_seq - 1.0) ** 2)
            reg_smooth = reg_smooth + torch.mean((cf_seq[:, 1:, :] - cf_seq[:, :-1, :]) ** 2)
        if len(recharge_frac_hist) > 0:
            r_seq = torch.stack(recharge_frac_hist, dim=1)
            reg_part = torch.mean(torch.relu(r_seq - 0.85) ** 2)

        reg_terms = {
            'dynamic_amplitude_loss': reg_amp,
            'dynamic_smoothness_loss': reg_smooth,
            'partition_entropy_loss': reg_part
        }

        if return_diagnostics is True:
            diag_out = {name: torch.stack(vals, dim=1) for name, vals in diag_hist.items()}
            if return_regularization is True and return_final_state is True:
                return q_seq, diag_out, final_state, reg_terms
            if return_regularization is True:
                return q_seq, diag_out, reg_terms
            if return_final_state is True:
                return q_seq, diag_out, final_state
            return q_seq, diag_out

        if return_regularization is True and return_final_state is True:
            return q_seq, final_state, reg_terms
        if return_regularization is True:
            return q_seq, reg_terms
        if return_final_state is True:
            return q_seq, final_state
        return q_seq


class MultiInv_DynamicSimHydModelAridConnectivity(MultiInv_DynamicSimHydModelSix):
    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 component_routing=True, dry_channel_loss=True, zero_flow_gate=True,
                 channel_loss_max=0.60, zero_gate_hidden=None):
        super(MultiInv_DynamicSimHydModelAridConnectivity, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing,
            dry_channel_loss=dry_channel_loss, zero_flow_gate=zero_flow_gate,
            channel_loss_max=channel_loss_max, zero_gate_hidden=zero_gate_hidden)
        self.nfea = 18
        self.nstaticpm = self.nfea * nmul
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)
        comp_bias = torch.linspace(-0.15, 0.15, nmul).view(1, 1, nmul).repeat(1, self.nfea, 1)
        self.compStaticBias = nn.Parameter(comp_bias)
        self.simhyd = DynamicSimHydModelFiveDifferentiableConnectivity(
            mode='normal', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all)
        self.simhyd_analysis = DynamicSimHydModelFiveDifferentiableConnectivity(
            mode='analysis', theta_is_raw=False, smooth=True,
            dynamic_sq=self.dynamic_sq, dynamic_etgam=self.dynamic_etgam,
            dynamic_partition=self.dynamic_partition, dynamic_cfmax_snow=self.dynamic_cfmax_snow,
            dynamic_routing_scale=self.dynamic_routing_scale, dynamic_all=self.dynamic_all)

    def _apply_channel_loss_connectivity(self, q_comp, diag_comp, theta, basin_attr, ngage, conn_gate):
        if self.dry_channel_loss is not True:
            zeros = torch.zeros_like(q_comp)
            return q_comp, zeros, zeros, zeros

        if 'soil_moisture' in diag_comp:
            soil_m = self._component_tensor_4d(diag_comp['soil_moisture'], ngage)
        else:
            soil_m = torch.ones_like(q_comp) * 100.0

        smsc = self._theta_to_smsc(theta, ngage).unsqueeze(0)
        wetness = torch.clamp(soil_m / torch.clamp(smsc, min=1e-6), 0.0, 1.0)
        dryness = 1.0 - wetness

        gamma = self.channel_loss_max * torch.sigmoid(self.channelLossHead(basin_attr))
        gamma = gamma.unsqueeze(0).unsqueeze(-1)
        loss_frac = 1.0 - torch.exp(-gamma * dryness)
        loss_frac = torch.clamp(loss_frac, 0.0, 0.95)
        loss_frac_conn = torch.clamp(loss_frac * (1.0 - conn_gate), 0.0, 0.95)
        q_after = q_comp * (1.0 - loss_frac_conn)
        channel_loss = q_comp - q_after
        return torch.clamp(q_after, min=0.0), channel_loss, loss_frac, loss_frac_conn

    def _apply_zero_flow_gate_connectivity(self, q_comp, x, diag_comp, theta, ngage, conn_gate):
        if self.zero_flow_gate_enabled is not True:
            ones = torch.ones_like(q_comp)
            return q_comp, ones, ones

        T, B, M, _ = q_comp.shape
        if 'soil_moisture' in diag_comp:
            soil_m = self._component_tensor_4d(diag_comp['soil_moisture'], ngage)
            smsc = self._theta_to_smsc(theta, ngage).unsqueeze(0)
            wetness = torch.clamp(soil_m / torch.clamp(smsc, min=1e-6), 0.0, 1.0)
        else:
            wetness = torch.ones_like(q_comp) * 0.5

        p_t = x[:, :, 0:1].unsqueeze(2).repeat(1, 1, M, 1)
        if x.shape[-1] >= 5:
            sin_t = x[:, :, 3:4].unsqueeze(2).repeat(1, 1, M, 1)
            cos_t = x[:, :, 4:5].unsqueeze(2).repeat(1, 1, M, 1)
        else:
            sin_t = torch.zeros_like(q_comp)
            cos_t = torch.ones_like(q_comp)

        gate_in = torch.cat([p_t / 20.0, wetness, q_comp / 20.0, sin_t, cos_t], dim=-1)
        gate_in_flat = gate_in.view(T * B * M, 5)
        logits_all = self.zeroFlowGate(gate_in_flat)
        comp_idx = torch.arange(M, device=q_comp.device).view(1, 1, M, 1).repeat(T, B, 1, 1).view(T * B * M, 1)
        logits = logits_all.gather(1, comp_idx)
        p_flow = torch.sigmoid(logits).view(T, B, M, 1)
        p_flow_conn = torch.clamp(p_flow + (1.0 - p_flow) * conn_gate, 0.0, 1.0)
        return torch.clamp(p_flow_conn * q_comp, min=0.0), p_flow, p_flow_conn

    def forward(self, x, z, doDropMC=False, return_diagnostics=False):
        nt_x = x.shape[0]
        ngage = z.shape[1]
        basin_attr = z[-1, :, -self.nattr:]

        if z.shape[2] > self.nattr:
            snow_frac_raw = torch.clamp(z[-1, :, -self.nattr - 1:-self.nattr], min=0.0, max=1.0)
        else:
            snow_frac_raw = None

        staticFeat = self.staticFeat(basin_attr)
        staticParams0 = self.staticOut(staticFeat)

        cursor = 0
        static0 = staticParams0[:, cursor:cursor + self.nstaticpm].view(ngage, self.nfea, self.nmul)
        static0 = static0 + self.compStaticBias
        snowpara = torch.sigmoid(static0)
        cursor += self.nstaticpm

        routpara0 = staticParams0[:, cursor:cursor + self.nroutpm]
        if self.component_routing is True:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        elif self.comprout is False:
            routpara = torch.sigmoid(routpara0)
        else:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        cursor += self.nroutpm

        if self.nwtspm == 0:
            wts = None
        else:
            wtspara = staticParams0[:, cursor:cursor + self.nwtspm] + self.compWeightBias
            wts = F.softmax(wtspara, dim=-1)
        self._last_wts = wts

        if self.lgdyn is False:
            lg_dyn = None
        else:
            lg_dyn_seq = self.lstmdyn(z)
            lg_attr_bias = self.lgAttr(basin_attr).unsqueeze(0).repeat(lg_dyn_seq.shape[0], 1, 1)
            lg_dyn = torch.sigmoid(lg_dyn_seq + lg_attr_bias)

        x_rep = x.unsqueeze(2).repeat(1, 1, self.nmul, 1).view(nt_x, ngage * self.nmul, x.shape[2])
        x_bt = x_rep.permute(1, 0, 2)
        theta = snowpara.permute(0, 2, 1).contiguous().view(ngage * self.nmul, self.nfea)

        if lg_dyn is None:
            lg_bt = None
        else:
            lg_bt = lg_dyn.permute(1, 2, 0).contiguous().view(ngage * self.nmul, lg_dyn.shape[0], 1)
        if lg_bt is not None and self.inittime > 0:
            lg_bt_main = lg_bt[:, self.inittime:, :]
        else:
            lg_bt_main = lg_bt

        if snow_frac_raw is None:
            snow_frac_rep = None
        else:
            snow_frac_rep = snow_frac_raw.unsqueeze(1).repeat(1, self.nmul, 1).view(ngage * self.nmul, 1)

        need_process_diag = return_diagnostics or self.dry_channel_loss or self.zero_flow_gate_enabled

        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            x_use = x[self.inittime:, :, :]
            _, warm_state = self.simhyd_analysis(
                warm_inputs,
                theta,
                lg_dyn_seq=None,
                lg_dyn_weight=self.lgdynweight,
                snow_frac_raw=snow_frac_rep,
                return_final_state=True)
            if need_process_diag:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    main_inputs,
                    theta,
                    initial_state=warm_state,
                    lg_dyn_seq=lg_bt_main,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_diagnostics=True,
                    return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    main_inputs,
                    theta,
                    initial_state=warm_state,
                    lg_dyn_seq=lg_bt_main,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_regularization=True)
                diag_comp = None
        else:
            x_use = x
            if need_process_diag:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    x_bt,
                    theta,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_diagnostics=True,
                    return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    x_bt,
                    theta,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_regularization=True)
                diag_comp = None

        reg_total = self.reg_amp_w * reg_terms['dynamic_amplitude_loss'] \
            + self.reg_smooth_w * reg_terms['dynamic_smoothness_loss'] \
            + self.reg_part_w * reg_terms['partition_entropy_loss']
        self._last_aux_terms = reg_terms
        self._last_aux_loss = reg_total

        q_comp_raw = q_seq.view(ngage, self.nmul, q_seq.shape[1], 1).permute(2, 0, 1, 3)
        assert q_comp_raw.shape[1] == ngage and q_comp_raw.shape[2] == self.nmul
        q_comp_raw = torch.clamp(q_comp_raw, min=0.0)

        if diag_comp is not None and 'connectivity_storage' in diag_comp:
            conn_storage = self._component_tensor_4d(diag_comp['connectivity_storage'], ngage)
            conn_frac = self._component_tensor_4d(diag_comp['connectivity_fraction'], ngage)
            conn_gate = self._component_tensor_4d(diag_comp['connectivity_gate'], ngage)
        else:
            conn_storage = torch.zeros_like(q_comp_raw)
            conn_frac = torch.zeros_like(q_comp_raw)
            conn_gate = torch.zeros_like(q_comp_raw)

        q_after_loss, channel_loss, original_loss_frac, modified_loss_frac = self._apply_channel_loss_connectivity(
            q_comp_raw, diag_comp, theta, basin_attr, ngage, conn_gate)
        q_after_gate, original_p_flow, modified_p_flow = self._apply_zero_flow_gate_connectivity(
            q_after_loss, x_use, diag_comp, theta, ngage, conn_gate)
        q_comp = torch.clamp(q_after_gate, min=0.0)

        q_mix_before_routing = self._mix_or_mean(q_comp, wts)

        route_mult_seq = None
        if self.routOpt is True and self.component_routing is True:
            q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
            assert routpara.shape[0] == ngage * self.nmul and routpara.shape[1] == 2
            if self.dynamic_routing_scale is True:
                if diag_comp is not None and 'soil_moisture' in diag_comp:
                    sm_comp = self._component_tensor_4d(diag_comp['soil_moisture'], ngage)
                else:
                    sm_comp = torch.ones_like(q_comp) * 50.0
                smsc_comp = self._theta_to_smsc(theta, ngage).unsqueeze(0).repeat(q_comp.shape[0], 1, 1, 1)
                p_rep = x_use[:, :, 0:1].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                if x_use.shape[-1] >= 5:
                    sin_rep = x_use[:, :, 3:4].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                    cos_rep = x_use[:, :, 4:5].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                else:
                    sin_rep = torch.zeros_like(q_comp)
                    cos_rep = torch.ones_like(q_comp)
                x_route = torch.cat([p_rep, sm_comp, smsc_comp, sin_rep, cos_rep], dim=-1).view(q_comp.shape[0], ngage * self.nmul, 5)
                q_routed_flat, route_mult_flat = self._route_q_dynamic_scale(q_for_routing, routpara, x_route)
                route_mult_4d = route_mult_flat.view(q_comp.shape[0], ngage, self.nmul, 1)
                route_mult_seq = self._mix_or_mean(route_mult_4d, wts)
                self._last_aux_loss = self._last_aux_loss + self.reg_amp_w * torch.mean((route_mult_flat - 1.0) ** 2)
                self._last_aux_loss = self._last_aux_loss + self.reg_smooth_w * torch.mean(
                    (route_mult_flat[1:, :, :] - route_mult_flat[:-1, :, :]) ** 2)
                q_routed = q_routed_flat.view(q_comp.shape[0], ngage, self.nmul, 1)
            else:
                q_routed = self._route_q(q_for_routing, routpara).view(q_comp.shape[0], ngage, self.nmul, 1)
            out = self._mix_or_mean(q_routed, wts)
        else:
            if self.routOpt is True and self.dynamic_routing_scale is True and self.comprout is False:
                if diag_comp is not None and 'soil_moisture' in diag_comp:
                    sm_mix = self._mix_component_tensor(diag_comp['soil_moisture'], ngage)
                else:
                    sm_mix = torch.ones_like(q_mix_before_routing) * 50.0
                smsc_mix = torch.ones_like(q_mix_before_routing) * 200.0
                x_route = torch.cat([x_use[:, :, 0:1], sm_mix, smsc_mix, x_use[:, :, 3:4], x_use[:, :, 4:5]], dim=2)
                out, route_mult_seq = self._route_q_dynamic_scale(q_mix_before_routing, routpara, x_route)
                self._last_aux_loss = self._last_aux_loss + self.reg_amp_w * torch.mean((route_mult_seq - 1.0) ** 2)
                self._last_aux_loss = self._last_aux_loss + self.reg_smooth_w * torch.mean(
                    (route_mult_seq[1:, :, :] - route_mult_seq[:-1, :, :]) ** 2)
            elif self.routOpt is True:
                if self.comprout is True:
                    q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
                    q_routed = self._route_q(q_for_routing, routpara).view(q_comp.shape[0], ngage, self.nmul, 1)
                    out = self._mix_or_mean(q_routed, wts)
                else:
                    assert routpara.shape[0] == ngage and routpara.shape[1] == 2
                    out = self._route_q(q_mix_before_routing, routpara)
            else:
                out = q_mix_before_routing

        out = torch.clamp(out, min=0.0)
        if return_diagnostics is not True:
            return out

        diag_out = dict()
        if diag_comp is not None:
            for name, tensor_comp in diag_comp.items():
                diag_out[name] = self._mix_component_tensor(tensor_comp, ngage)
        diag_out['total_discharge'] = out
        diag_out['q_mix_before_routing'] = q_mix_before_routing
        diag_out['component_discharge_raw'] = self._mix_or_mean(q_comp_raw, wts)
        diag_out['connectivity_storage'] = self._mix_or_mean(conn_storage, wts)
        diag_out['connectivity_fraction'] = self._mix_or_mean(conn_frac, wts)
        diag_out['connectivity_gate'] = self._mix_or_mean(conn_gate, wts)
        diag_out['original_channel_loss_fraction'] = self._mix_or_mean(original_loss_frac, wts)
        diag_out['connectivity_modified_channel_loss_fraction'] = self._mix_or_mean(modified_loss_frac, wts)
        diag_out['original_zero_flow_probability'] = self._mix_or_mean(original_p_flow, wts)
        diag_out['connectivity_modified_flow_probability'] = self._mix_or_mean(modified_p_flow, wts)
        if self.dry_channel_loss is True:
            diag_out['channel_loss'] = self._mix_or_mean(channel_loss, wts)
            diag_out['channel_loss_fraction'] = self._mix_or_mean(modified_loss_frac, wts)
        if self.zero_flow_gate_enabled is True:
            diag_out['zero_flow_probability'] = self._mix_or_mean(modified_p_flow, wts)
        if route_mult_seq is not None:
            diag_out['route_b_t_multiplier'] = route_mult_seq
        return out, diag_out


class MultiInv_DynamicSimHydModelSeven(MultiInv_DynamicSimHydModelSix):
    """
    Model Seven: keep Model Six component-wise routing, but make dry-flow
    corrections safer via a residual low-flow gate and selective, capped
    dry-channel loss.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 component_routing=True, dry_channel_loss=True, zero_flow_gate=True,
                 channel_loss_max=0.60, zero_gate_hidden=None, min_flow_gate=0.35,
                 channel_loss_cap=0.35, dry_selective=True):
        super(MultiInv_DynamicSimHydModelSeven, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w, component_routing=component_routing,
            dry_channel_loss=dry_channel_loss, zero_flow_gate=zero_flow_gate,
            channel_loss_max=channel_loss_max, zero_gate_hidden=zero_gate_hidden)
        self.min_flow_gate = min_flow_gate
        self.channel_loss_cap = channel_loss_cap
        self.dry_selective = dry_selective
        self.drySelectorHead = nn.Linear(nattr, nmul)

    def _apply_channel_loss(self, q_comp, diag_comp, theta, basin_attr, ngage, snow_frac_raw=None):
        # q_comp: [T, B, M, 1]
        if self.dry_channel_loss is not True:
            zeros = torch.zeros_like(q_comp)
            ones = torch.ones_like(q_comp)
            return q_comp, zeros, zeros, ones

        if 'soil_moisture' in diag_comp:
            soil_m = self._component_tensor_4d(diag_comp['soil_moisture'], ngage)
        else:
            soil_m = torch.ones_like(q_comp) * 100.0

        smsc = self._theta_to_smsc(theta, ngage).unsqueeze(0)
        wetness = torch.clamp(soil_m / torch.clamp(smsc, min=1e-6), 0.0, 1.0)
        dryness = 1.0 - wetness

        gamma = self.channel_loss_max * torch.sigmoid(self.channelLossHead(basin_attr))
        gamma = gamma.unsqueeze(0).unsqueeze(-1)
        loss_frac_raw = 1.0 - torch.exp(-gamma * dryness)

        if self.dry_selective is True:
            dry_selector = torch.sigmoid(self.drySelectorHead(basin_attr))
        else:
            dry_selector = torch.ones((ngage, self.nmul), device=q_comp.device, dtype=q_comp.dtype)
        dry_selector = dry_selector.unsqueeze(0).unsqueeze(-1)

        if snow_frac_raw is not None:
            snow_frac_rep = snow_frac_raw.unsqueeze(1).repeat(1, self.nmul, 1).unsqueeze(0)
            dry_selector = dry_selector * (1.0 - snow_frac_rep)

        loss_frac = dry_selector * loss_frac_raw
        loss_frac = torch.clamp(loss_frac, 0.0, self.channel_loss_cap)
        q_after = q_comp * (1.0 - loss_frac)
        channel_loss = q_comp - q_after
        return torch.clamp(q_after, min=0.0), channel_loss, loss_frac, dry_selector

    def _apply_zero_flow_gate(self, q_comp, x, diag_comp, theta, ngage):
        # q_comp: [T, B, M, 1]
        if self.zero_flow_gate_enabled is not True:
            ones = torch.ones_like(q_comp)
            return q_comp, ones

        T, B, M, _ = q_comp.shape
        if 'soil_moisture' in diag_comp:
            soil_m = self._component_tensor_4d(diag_comp['soil_moisture'], ngage)
            smsc = self._theta_to_smsc(theta, ngage).unsqueeze(0)
            wetness = torch.clamp(soil_m / torch.clamp(smsc, min=1e-6), 0.0, 1.0)
        else:
            wetness = torch.ones_like(q_comp) * 0.5

        p_t = x[:, :, 0:1].unsqueeze(2).repeat(1, 1, M, 1)
        if x.shape[-1] >= 5:
            sin_t = x[:, :, 3:4].unsqueeze(2).repeat(1, 1, M, 1)
            cos_t = x[:, :, 4:5].unsqueeze(2).repeat(1, 1, M, 1)
        else:
            sin_t = torch.zeros_like(q_comp)
            cos_t = torch.ones_like(q_comp)

        gate_in = torch.cat([p_t / 20.0, wetness, q_comp / 20.0, sin_t, cos_t], dim=-1)
        gate_in_flat = gate_in.view(T * B * M, 5)
        logits_all = self.zeroFlowGate(gate_in_flat)
        comp_idx = torch.arange(M, device=q_comp.device).view(1, 1, M, 1).repeat(T, B, 1, 1).view(T * B * M, 1)
        logits = logits_all.gather(1, comp_idx)
        flow_keep = self.min_flow_gate + (1.0 - self.min_flow_gate) * torch.sigmoid(logits)
        flow_keep = flow_keep.view(T, B, M, 1)
        q_after = flow_keep * q_comp
        return torch.clamp(q_after, min=0.0), flow_keep

    def forward(self, x, z, doDropMC=False, return_diagnostics=False):
        nt_x = x.shape[0]
        ngage = z.shape[1]
        basin_attr = z[-1, :, -self.nattr:]

        if z.shape[2] > self.nattr:
            snow_frac_raw = torch.clamp(z[-1, :, -self.nattr - 1:-self.nattr], min=0.0, max=1.0)
        else:
            snow_frac_raw = None

        staticFeat = self.staticFeat(basin_attr)
        staticParams0 = self.staticOut(staticFeat)

        cursor = 0
        static0 = staticParams0[:, cursor:cursor + self.nstaticpm].view(ngage, self.nfea, self.nmul)
        static0 = static0 + self.compStaticBias
        snowpara = torch.sigmoid(static0)
        cursor += self.nstaticpm

        routpara0 = staticParams0[:, cursor:cursor + self.nroutpm]
        if self.component_routing is True:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        elif self.comprout is False:
            routpara = torch.sigmoid(routpara0)
        else:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        cursor += self.nroutpm

        if self.nwtspm == 0:
            wts = None
        else:
            wtspara = staticParams0[:, cursor:cursor + self.nwtspm] + self.compWeightBias
            wts = F.softmax(wtspara, dim=-1)
        self._last_wts = wts

        if self.lgdyn is False:
            lg_dyn = None
        else:
            lg_dyn_seq = self.lstmdyn(z)
            lg_attr_bias = self.lgAttr(basin_attr).unsqueeze(0).repeat(lg_dyn_seq.shape[0], 1, 1)
            lg_dyn = torch.sigmoid(lg_dyn_seq + lg_attr_bias)

        x_rep = x.unsqueeze(2).repeat(1, 1, self.nmul, 1).view(nt_x, ngage * self.nmul, x.shape[2])
        x_bt = x_rep.permute(1, 0, 2)
        theta = snowpara.permute(0, 2, 1).contiguous().view(ngage * self.nmul, self.nfea)

        if lg_dyn is None:
            lg_bt = None
        else:
            lg_bt = lg_dyn.permute(1, 2, 0).contiguous().view(ngage * self.nmul, lg_dyn.shape[0], 1)

        if snow_frac_raw is None:
            snow_frac_rep = None
        else:
            snow_frac_rep = snow_frac_raw.unsqueeze(1).repeat(1, self.nmul, 1).view(ngage * self.nmul, 1)

        need_process_diag = return_diagnostics or self.dry_channel_loss or self.zero_flow_gate_enabled

        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            x_use = x[self.inittime:, :, :]
            _, warm_state = self.simhyd_analysis(
                warm_inputs,
                theta,
                lg_dyn_seq=None,
                lg_dyn_weight=self.lgdynweight,
                snow_frac_raw=snow_frac_rep,
                return_final_state=True)
            if need_process_diag:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    main_inputs,
                    theta,
                    initial_state=warm_state,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_diagnostics=True,
                    return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    main_inputs,
                    theta,
                    initial_state=warm_state,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_regularization=True)
                diag_comp = None
        else:
            x_use = x
            if need_process_diag:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    x_bt,
                    theta,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_diagnostics=True,
                    return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    x_bt,
                    theta,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_regularization=True)
                diag_comp = None

        reg_total = self.reg_amp_w * reg_terms['dynamic_amplitude_loss'] \
            + self.reg_smooth_w * reg_terms['dynamic_smoothness_loss'] \
            + self.reg_part_w * reg_terms['partition_entropy_loss']
        self._last_aux_terms = reg_terms
        self._last_aux_loss = reg_total

        q_comp_raw = q_seq.view(ngage, self.nmul, q_seq.shape[1], 1).permute(2, 0, 1, 3)
        assert q_comp_raw.shape[1] == ngage and q_comp_raw.shape[2] == self.nmul
        q_comp_raw = torch.clamp(q_comp_raw, min=0.0)

        q_after_loss, channel_loss, channel_loss_fraction, dry_selector = self._apply_channel_loss(
            q_comp_raw, diag_comp, theta, basin_attr, ngage, snow_frac_raw=snow_frac_raw)
        q_after_gate, zero_flow_keep_fraction = self._apply_zero_flow_gate(q_after_loss, x_use, diag_comp, theta, ngage)
        q_comp = torch.clamp(q_after_gate, min=0.0)

        q_mix_before_routing = self._mix_or_mean(q_comp, wts)

        route_mult_seq = None
        if self.routOpt is True and self.component_routing is True:
            q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
            assert routpara.shape[0] == ngage * self.nmul and routpara.shape[1] == 2
            if self.dynamic_routing_scale is True:
                if diag_comp is not None and 'soil_moisture' in diag_comp:
                    sm_comp = self._component_tensor_4d(diag_comp['soil_moisture'], ngage)
                else:
                    sm_comp = torch.ones_like(q_comp) * 50.0
                smsc_comp = self._theta_to_smsc(theta, ngage).unsqueeze(0).repeat(q_comp.shape[0], 1, 1, 1)
                p_rep = x_use[:, :, 0:1].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                if x_use.shape[-1] >= 5:
                    sin_rep = x_use[:, :, 3:4].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                    cos_rep = x_use[:, :, 4:5].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                else:
                    sin_rep = torch.zeros_like(q_comp)
                    cos_rep = torch.ones_like(q_comp)
                x_route = torch.cat([p_rep, sm_comp, smsc_comp, sin_rep, cos_rep], dim=-1).view(q_comp.shape[0], ngage * self.nmul, 5)
                q_routed_flat, route_mult_flat = self._route_q_dynamic_scale(q_for_routing, routpara, x_route)
                route_mult_4d = route_mult_flat.view(q_comp.shape[0], ngage, self.nmul, 1)
                route_mult_seq = self._mix_or_mean(route_mult_4d, wts)
                self._last_aux_loss = self._last_aux_loss + self.reg_amp_w * torch.mean((route_mult_flat - 1.0) ** 2)
                self._last_aux_loss = self._last_aux_loss + self.reg_smooth_w * torch.mean(
                    (route_mult_flat[1:, :, :] - route_mult_flat[:-1, :, :]) ** 2)
                q_routed = q_routed_flat.view(q_comp.shape[0], ngage, self.nmul, 1)
            else:
                q_routed = self._route_q(q_for_routing, routpara).view(q_comp.shape[0], ngage, self.nmul, 1)
            out = self._mix_or_mean(q_routed, wts)
        else:
            if self.routOpt is True and self.dynamic_routing_scale is True and self.comprout is False:
                if diag_comp is not None and 'soil_moisture' in diag_comp:
                    sm_mix = self._mix_component_tensor(diag_comp['soil_moisture'], ngage)
                else:
                    sm_mix = torch.ones_like(q_mix_before_routing) * 50.0
                smsc_mix = torch.ones_like(q_mix_before_routing) * 200.0
                x_route = torch.cat([x_use[:, :, 0:1], sm_mix, smsc_mix, x_use[:, :, 3:4], x_use[:, :, 4:5]], dim=2)
                out, route_mult_seq = self._route_q_dynamic_scale(q_mix_before_routing, routpara, x_route)
                self._last_aux_loss = self._last_aux_loss + self.reg_amp_w * torch.mean((route_mult_seq - 1.0) ** 2)
                self._last_aux_loss = self._last_aux_loss + self.reg_smooth_w * torch.mean(
                    (route_mult_seq[1:, :, :] - route_mult_seq[:-1, :, :]) ** 2)
            elif self.routOpt is True:
                if self.comprout is True:
                    q_for_routing = q_comp.permute(0, 1, 3, 2).contiguous().view(q_comp.shape[0], ngage * self.nmul, 1)
                    q_routed = self._route_q(q_for_routing, routpara).view(q_comp.shape[0], ngage, self.nmul, 1)
                    out = self._mix_or_mean(q_routed, wts)
                else:
                    assert routpara.shape[0] == ngage and routpara.shape[1] == 2
                    out = self._route_q(q_mix_before_routing, routpara)
            else:
                out = q_mix_before_routing

        out = torch.clamp(out, min=0.0)
        if return_diagnostics is not True:
            return out

        diag_out = dict()
        if diag_comp is not None:
            for name, tensor_comp in diag_comp.items():
                diag_out[name] = self._mix_component_tensor(tensor_comp, ngage)
        diag_out['total_discharge'] = out
        diag_out['q_mix_before_routing'] = q_mix_before_routing
        diag_out['component_discharge_raw'] = self._mix_or_mean(q_comp_raw, wts)
        if self.dry_channel_loss is True:
            diag_out['channel_loss'] = self._mix_or_mean(channel_loss, wts)
            diag_out['channel_loss_fraction'] = self._mix_or_mean(channel_loss_fraction, wts)
            diag_out['dry_selector'] = self._mix_or_mean(dry_selector, wts)
        if self.zero_flow_gate_enabled is True:
            diag_out['zero_flow_keep_fraction'] = self._mix_or_mean(zero_flow_keep_fraction, wts)
        if route_mult_seq is not None:
            diag_out['route_b_t_multiplier'] = route_mult_seq
        return out, diag_out


class MultiInv_DynamicSimHydModelSixRoutingOnly(MultiInv_DynamicSimHydModelFive):
    """
    Model 6v2 / routing-only ablation:
    keep only component-wise routing before mixing and remove dry-flow
    corrective mechanisms such as channel loss and zero-flow gate.
    """

    def __init__(self, *, ninv, nmul=4, nattr=35, hiddeninv=256, drinv=0.5, inittime=0,
                 routOpt=True, comprout=False, compwts=True, lgdyn=True, lgdynweight=0.6,
                 dynamic_sq=True, dynamic_etgam=True, dynamic_partition=True,
                 dynamic_cfmax_snow=True, dynamic_routing_scale=False, dynamic_all=False,
                 reg_amp_w=1e-3, reg_smooth_w=1e-3, reg_part_w=1e-3,
                 component_routing=True, dry_channel_loss=False, zero_flow_gate=False):
        super(MultiInv_DynamicSimHydModelSixRoutingOnly, self).__init__(
            ninv=ninv, nmul=nmul, nattr=nattr, hiddeninv=hiddeninv, drinv=drinv, inittime=inittime,
            routOpt=routOpt, comprout=comprout, compwts=compwts, lgdyn=lgdyn, lgdynweight=lgdynweight,
            dynamic_sq=dynamic_sq, dynamic_etgam=dynamic_etgam, dynamic_partition=dynamic_partition,
            dynamic_cfmax_snow=dynamic_cfmax_snow, dynamic_routing_scale=dynamic_routing_scale,
            dynamic_all=dynamic_all, reg_amp_w=reg_amp_w, reg_smooth_w=reg_smooth_w,
            reg_part_w=reg_part_w)
        self.component_routing = component_routing
        self.dry_channel_loss = dry_channel_loss
        self.zero_flow_gate_enabled = zero_flow_gate
        self.nroutpm = nmul * 2
        self.staticOut = nn.Linear(hiddeninv, self.nstaticpm + self.nroutpm + self.nwtspm)

    def _mix_or_mean(self, tensor4, wts):
        if wts is None:
            return torch.mean(tensor4, dim=2)
        return torch.sum(tensor4 * wts.unsqueeze(0).unsqueeze(-1), dim=2)

    def forward(self, x, z, doDropMC=False, return_diagnostics=False):
        nt_x = x.shape[0]
        ngage = z.shape[1]
        basin_attr = z[-1, :, -self.nattr:]

        if z.shape[2] > self.nattr:
            snow_frac_raw = torch.clamp(z[-1, :, -self.nattr - 1:-self.nattr], min=0.0, max=1.0)
        else:
            snow_frac_raw = None

        staticFeat = self.staticFeat(basin_attr)
        staticParams0 = self.staticOut(staticFeat)

        cursor = 0
        static0 = staticParams0[:, cursor:cursor + self.nstaticpm].view(ngage, self.nfea, self.nmul)
        static0 = static0 + self.compStaticBias
        snowpara = torch.sigmoid(static0)
        cursor += self.nstaticpm

        routpara0 = staticParams0[:, cursor:cursor + self.nroutpm]
        routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        cursor += self.nroutpm

        if self.nwtspm == 0:
            wts = None
        else:
            wtspara = staticParams0[:, cursor:cursor + self.nwtspm] + self.compWeightBias
            wts = F.softmax(wtspara, dim=-1)
        self._last_wts = wts

        if self.lgdyn is False:
            lg_dyn = None
        else:
            lg_dyn_seq = self.lstmdyn(z)
            lg_attr_bias = self.lgAttr(basin_attr).unsqueeze(0).repeat(lg_dyn_seq.shape[0], 1, 1)
            lg_dyn = torch.sigmoid(lg_dyn_seq + lg_attr_bias)

        x_rep = x.unsqueeze(2).repeat(1, 1, self.nmul, 1).view(nt_x, ngage * self.nmul, x.shape[2])
        x_bt = x_rep.permute(1, 0, 2)
        theta = snowpara.permute(0, 2, 1).contiguous().view(ngage * self.nmul, self.nfea)

        if lg_dyn is None:
            lg_bt = None
        else:
            lg_bt = lg_dyn.permute(1, 2, 0).contiguous().view(ngage * self.nmul, lg_dyn.shape[0], 1)

        if snow_frac_raw is None:
            snow_frac_rep = None
        else:
            snow_frac_rep = snow_frac_raw.unsqueeze(1).repeat(1, self.nmul, 1).view(ngage * self.nmul, 1)

        if self.inittime > 0:
            warm_inputs = x_bt[:, :self.inittime, :]
            main_inputs = x_bt[:, self.inittime:, :]
            _, warm_state = self.simhyd_analysis(
                warm_inputs,
                theta,
                lg_dyn_seq=None,
                lg_dyn_weight=self.lgdynweight,
                snow_frac_raw=snow_frac_rep,
                return_final_state=True)
            if return_diagnostics is True:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    main_inputs,
                    theta,
                    initial_state=warm_state,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_diagnostics=True,
                    return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    main_inputs,
                    theta,
                    initial_state=warm_state,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_regularization=True)
                diag_comp = None
        else:
            if return_diagnostics is True:
                q_seq, diag_comp, reg_terms = self.simhyd(
                    x_bt,
                    theta,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_diagnostics=True,
                    return_regularization=True)
            else:
                q_seq, reg_terms = self.simhyd(
                    x_bt,
                    theta,
                    lg_dyn_seq=lg_bt,
                    lg_dyn_weight=self.lgdynweight,
                    snow_frac_raw=snow_frac_rep,
                    return_regularization=True)
                diag_comp = None

        reg_total = self.reg_amp_w * reg_terms['dynamic_amplitude_loss'] \
            + self.reg_smooth_w * reg_terms['dynamic_smoothness_loss'] \
            + self.reg_part_w * reg_terms['partition_entropy_loss']
        self._last_aux_terms = reg_terms
        self._last_aux_loss = reg_total

        q_comp_raw = q_seq.view(ngage, self.nmul, q_seq.shape[1], 1).permute(2, 0, 1, 3)
        q_comp_raw = torch.clamp(q_comp_raw, min=0.0)
        assert q_comp_raw.shape[1] == ngage and q_comp_raw.shape[2] == self.nmul

        q_for_routing = q_comp_raw.permute(0, 1, 3, 2).contiguous().view(q_comp_raw.shape[0], ngage * self.nmul, 1)
        assert routpara.shape[0] == ngage * self.nmul and routpara.shape[1] == 2

        route_mult_seq = None
        if self.dynamic_routing_scale is True:
            if return_diagnostics is True and diag_comp is not None and 'soil_moisture' in diag_comp:
                sm_comp = diag_comp['soil_moisture'].view(ngage, self.nmul, diag_comp['soil_moisture'].shape[1], 1).permute(2, 0, 1, 3)
            else:
                sm_comp = torch.ones_like(q_comp_raw) * 50.0
            smsc_comp = (50.0 + theta.view(ngage, self.nmul, self.nfea)[:, :, 3:4] * (500.0 - 50.0)).unsqueeze(0).repeat(q_comp_raw.shape[0], 1, 1, 1)
            x_base = x[self.inittime:, :, :] if self.inittime > 0 else x
            p_rep = x_base[:, :, 0:1].unsqueeze(2).repeat(1, 1, self.nmul, 1)
            if x_base.shape[-1] >= 5:
                sin_rep = x_base[:, :, 3:4].unsqueeze(2).repeat(1, 1, self.nmul, 1)
                cos_rep = x_base[:, :, 4:5].unsqueeze(2).repeat(1, 1, self.nmul, 1)
            else:
                sin_rep = torch.zeros_like(q_comp_raw)
                cos_rep = torch.ones_like(q_comp_raw)
            x_route = torch.cat([p_rep, sm_comp, smsc_comp, sin_rep, cos_rep], dim=-1).view(q_comp_raw.shape[0], ngage * self.nmul, 5)
            q_routed_flat, route_mult_flat = self._route_q_dynamic_scale(q_for_routing, routpara, x_route)
            q_routed = q_routed_flat.view(q_comp_raw.shape[0], ngage, self.nmul, 1)
            route_mult_4d = route_mult_flat.view(q_comp_raw.shape[0], ngage, self.nmul, 1)
            route_mult_seq = self._mix_or_mean(route_mult_4d, wts)
            self._last_aux_loss = self._last_aux_loss + self.reg_amp_w * torch.mean((route_mult_flat - 1.0) ** 2)
            self._last_aux_loss = self._last_aux_loss + self.reg_smooth_w * torch.mean(
                (route_mult_flat[1:, :, :] - route_mult_flat[:-1, :, :]) ** 2)
        else:
            q_routed = self._route_q(q_for_routing, routpara).view(q_comp_raw.shape[0], ngage, self.nmul, 1)

        out = self._mix_or_mean(q_routed, wts)
        out = torch.clamp(out, min=0.0)

        if return_diagnostics is not True:
            return out

        diag_out = dict()
        if diag_comp is not None:
            for name, tensor_comp in diag_comp.items():
                diag_out[name] = self._mix_component_tensor(tensor_comp, ngage)
        diag_out['total_discharge'] = out
        diag_out['q_mix_before_routing'] = self._mix_or_mean(q_comp_raw, wts)
        diag_out['component_discharge_raw'] = q_comp_raw
        diag_out['component_discharge_routed'] = q_routed
        if wts is not None:
            diag_out['component_weights'] = wts.unsqueeze(0).repeat(q_comp_raw.shape[0], 1, 1)
        else:
            diag_out['component_weights'] = torch.ones((q_comp_raw.shape[0], ngage, self.nmul), device=q_comp_raw.device, dtype=q_comp_raw.dtype) / float(self.nmul)
        diag_out['routing_parameters'] = routpara.view(ngage, self.nmul, 2).unsqueeze(0).repeat(q_comp_raw.shape[0], 1, 1, 1)
        if route_mult_seq is not None:
            diag_out['route_b_t_multiplier'] = route_mult_seq
        return out, diag_out


class MultiInv_HBVModel(torch.nn.Module):
    # class for dPL + HBV with multiple components and static parameters
    def __init__(self, *, ninv, nfea, nmul, hiddeninv, drinv=0.5, inittime=0, routOpt=False, comprout=False,
                 compwts=False, pcorr=None):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVModel, self).__init__()
        self.ninv = ninv
        self.nfea = nfea
        self.hiddeninv = hiddeninv
        self.nmul = nmul
        # get the total number of parameters
        nhbvpm = nfea*nmul
        if comprout is False:
            nroutpm = 2
        else:
            nroutpm = nmul*2
        if compwts is False:
            nwtspm = 0
        else:
            nwtspm = nmul
        if pcorr is None:
            ntp = nhbvpm + nroutpm + nwtspm
        else:
            ntp = nhbvpm + nroutpm + nwtspm + 1 # 1 for potential precipitation correction
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv = CudnnLstmModel(
            nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)

        self.HBV = HBVMul()

        self.gpu = 1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm = nhbvpm
        self.nwtspm = nwtspm
        self.nroutpm = nroutpm
        self.pcorr = pcorr



    def forward(self, x, z, doDropMC=False):
        Gen = self.lstminv(z)
        Params0 = Gen[-1, :, :] # the last time step as learned parameters
        ngage = Params0.shape[0]
        # print(Params0)
        hbvpara0 = Params0[:, 0:self.nhbvpm]
        hbvpara = torch.sigmoid(hbvpara0).view(ngage, self.nfea, self.nmul)
        routpara0 = Params0[:, self.nhbvpm:self.nhbvpm+self.nroutpm] # dim: [Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False: # if do routing for each component
            routpara = torch.sigmoid(routpara0)
        else:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        if self.nwtspm == 0: # 0: simple average instead of weighted average for all components
            wts = None
        else:
            wtspara = Params0[:, self.nhbvpm+self.nroutpm:self.nhbvpm+self.nroutpm+self.nwtspm]
            wts = F.softmax(wtspara, dim=-1)
        if self.pcorr is None:
            corrpara = None
        else:
            corrpara0 = Params0[:, self.nhbvpm+self.nroutpm+self.nwtspm:self.nhbvpm+self.nroutpm+self.nwtspm+1]
            corrpara = torch.sigmoid(corrpara0)
        out = self.HBV(x, parameters=hbvpara, mu=self.nmul, muwts=wts, rtwts=routpara, bufftime=self.inittime,
                       routOpt=self.routOpt, comprout=self.comprout, corrwts=corrpara, pcorr=self.pcorr) # HBV forward
        return out


class HBVMulTD(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version implemented by Dapeng Feng"""
    # we suggest you read the class HBVMul() with static parameters first

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTD, self).__init__()

    def forward(self, x, parameters, staind, tdlst, mu, muwts, rtwts, bufftime=0, outstate=False, routOpt=False,
                comprout=False, dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     x: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET


        PRECS = 1e-5

        # Initialization
        if bufftime > 0:
            with torch.no_grad():
                xinit = x[0:bufftime, :, :]
                initmodel = HBVMul()
                buffpara = parameters[bufftime-1, :, :, :]
                Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = initmodel(xinit, buffpara, mu, muwts, rtwts,
                                                                      bufftime=0, outstate=True, routOpt=False, comprout=False)
        else:

            # Without buff time, initialize state variables with zeros
            Ngrid = x.shape[1]
            SNOWPACK = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            MELTWATER = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SM = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SUZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SLZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()

        P = x[bufftime:, :, 0]
        Pm= P.unsqueeze(2).repeat(1,1,mu)
        T = x[bufftime:, :, 1]
        Tm = T.unsqueeze(2).repeat(1,1,mu)
        ETpot = x[bufftime:, :, 2]
        ETpm = ETpot.unsqueeze(2).repeat(1,1,mu)
        parAll = parameters[bufftime:, :, :, :]
        parAllTrans = torch.zeros_like(parAll)

        ## scale the parameters
        hbvscaLst = [[1,6], [50,1000], [0.05,0.9], [0.01,0.5], [0.001,0.2], [0.2,1],
                        [0,10], [0,100], [-2.5,2.5], [0.5,10], [0,0.1], [0,0.2]]
        routscaLst = [[0,2.9], [0,6.5]]

        for ip in range(len(hbvscaLst)): # not include routing. Scaling the parameters using loop
            parAllTrans[:,:,ip,:] = hbvscaLst[ip][0] + parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep, Ngrid = P.size()

        # deal with the dynamic parameters and dropout
        parstaFull = parAllTrans[staind, :, :, :].unsqueeze(0).repeat([Nstep, 1, 1, 1])  # static matrix
        parhbvFull = torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension to apply dropout
        pmat = torch.ones([1, Ngrid, 1])*dydrop
        for ix in tdlst:
            staPar = parstaFull[:, :, ix-1, :]
            dynPar = parAllTrans[:, :, ix-1, :]
            drmask = torch.bernoulli(pmat).detach_().cuda()  # to drop some dynamic parameters as static
            comPar = dynPar*(1-drmask) + staPar*drmask
            parhbvFull[:, :, ix-1, :] = comPar


        # Initialize time series of model variables
        Qsimmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()
        ETmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()

        # Output the box components of Q
        Qsimmu0 = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()
        Qsimmu1 = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()
        Qsimmu2 = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()


        for t in range(Nstep):
            paraLst = []
            for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                paraLst.append(parhbvFull[t, :, ip, :])

            parBETA, parFC, parK0, parK1, parK2, parLP, parPERC, parUZL, parTT, parCFMAX, parCFR, parCWH = paraLst

            # Separate precipitation into liquid and solid components
            PRECIP = Pm[t, :, :]
            RAIN = torch.mul(PRECIP, (Tm[t, :, :] >= parTT).type(torch.float32))
            SNOW = torch.mul(PRECIP, (Tm[t, :, :] < parTT).type(torch.float32))

            # Snow
            SNOWPACK = SNOWPACK + SNOW
            melt = parCFMAX * (Tm[t, :, :] - parTT)
            # melt[melt < 0.0] = 0.0
            melt = torch.clamp(melt, min=0.0)
            # melt[melt > SNOWPACK] = SNOWPACK[melt > SNOWPACK]
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = parCFR * parCFMAX * (parTT - Tm[t, :, :])
            # refreezing[refreezing < 0.0] = 0.0
            # refreezing[refreezing > MELTWATER] = MELTWATER[refreezing > MELTWATER]
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (parCWH * SNOWPACK)
            # tosoil[tosoil < 0.0] = 0.0
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation
            soil_wetness = (SM / parFC) ** parBETA
            # soil_wetness[soil_wetness < 0.0] = 0.0
            # soil_wetness[soil_wetness > 1.0] = 1.0
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness

            SM = SM + RAIN + tosoil - recharge
            excess = SM - parFC
            # excess[excess < 0.0] = 0.0
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            evapfactor = SM / (parLP * parFC)
            evapfactor  = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = ETpm[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=PRECS) # SM can not be zero for gradient tracking

            # Groundwater boxes
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, parPERC)
            SUZ = SUZ - PERC
            Q0 = parK0 * torch.clamp(SUZ - parUZL, min=0.0)
            SUZ = SUZ - Q0
            Q1 = parK1 * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = parK2 * SLZ
            SLZ = SLZ - Q2
            Qsimmu[t, :, :] = Q0 + Q1 + Q2

            # save components
            Qsimmu0[t, :, :] = Q0
            Qsimmu1[t, :, :] = Q1
            Qsimmu2[t, :, :] = Q2
            ETmu[t, :, :] = ETact


        Qsimave0 = Qsimmu0.mean(-1, keepdim=True)
        Qsimave1 = Qsimmu1.mean(-1, keepdim=True)
        Qsimave2 = Qsimmu2.mean(-1, keepdim=True)
        ETave = ETmu.mean(-1, keepdim=True)


        # get the initial average
        if muwts is None:
            Qsimave = Qsimmu.mean(-1)
        else:
            Qsimave = (Qsimmu * muwts).sum(-1)

        if routOpt is True: # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim = Qsimmu.view(Nstep, Ngrid * mu)
            else:
                # average the components, then do routing
                Qsim = Qsimave

            tempa = routscaLst[0][0] + rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb = routscaLst[1][0] + rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            UH = UH_gamma(routa, routb, lenF=15)  # lenF: folter
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])   # dim:gage*var*time
            UH = UH.permute([1, 2, 0])  # dim: gage*var*time
            Qsrout = UH_conv(rf, UH).permute([2, 0, 1])

            if comprout is True: # Qs is [time, [gage*mult], var] now
                Qstemp = Qsrout.view(Nstep, Ngrid, mu)
                if muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp * muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else: # no routing, output the primary average simulations

            Qs = torch.unsqueeze(Qsimave, -1) # add a dimension

        if outstate is True:
            return Qs, SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            # return Qs
            Qall = torch.cat((Qs, Qsimave0, Qsimave1, Qsimave2, ETave), dim=-1)
            return Qall

class HBVMulTDET(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""
    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET, self).__init__()

    def forward(self, x, parameters, staind, tdlst, mu, muwts, rtwts, bufftime=0, outstate=False, routOpt=False,
                comprout=False, dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS = 1e-5  # keep the numerical calculation stable

        # Initialization to warm-up states
        if bufftime > 0:
            with torch.no_grad():
                xinit = x[0:bufftime, :, :]
                initmodel = HBVMulET()
                buffpara = parameters[bufftime-1, :, :, :]
                Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = initmodel(xinit, buffpara, mu, muwts, rtwts,
                                                                      bufftime=0, outstate=True, routOpt=False, comprout=False)
        else:

            # Without warm-up bufftime=0, initialize state variables with zeros
            Ngrid = x.shape[1]
            SNOWPACK = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            MELTWATER = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SM = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SUZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SLZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()

        P = x[bufftime:, :, 0]
        Pm= P.unsqueeze(2).repeat(1,1,mu) # precip
        T = x[bufftime:, :, 1]
        Tm = T.unsqueeze(2).repeat(1,1,mu) # temperature
        ETpot = x[bufftime:, :, 2]
        ETpm = ETpot.unsqueeze(2).repeat(1,1,mu) # potential ET
        parAll = parameters[bufftime:, :, :, :]
        parAllTrans = torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst = [[1,6], [50,1000], [0.05,0.9], [0.01,0.5], [0.001,0.2], [0.2,1],
                        [0,10], [0,100], [-2.5,2.5], [0.5,10], [0,0.1], [0,0.2], [0.3,5]]  # HBV para
        routscaLst = [[0,2.9], [0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)): # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:] = hbvscaLst[ip][0] + parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep, Ngrid = P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull = parAllTrans[staind, :, :, :].unsqueeze(0).repeat([Nstep, 1, 1, 1])  # static para matrix
        parhbvFull = torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat = torch.ones([1, Ngrid, 1])*dydrop
        for ix in tdlst:
            staPar = parstaFull[:, :, ix-1, :]
            dynPar = parAllTrans[:, :, ix-1, :]
            drmask = torch.bernoulli(pmat).detach_().cuda()  # to drop dynamic parameters as static in some basins
            comPar = dynPar*(1-drmask) + staPar*drmask
            parhbvFull[:, :, ix-1, :] = comPar


        # Initialize time series of model variables to save results
        Qsimmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()
        ETmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()

        # Output the box components of Q
        Qsimmu0 = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()
        Qsimmu1 = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()
        Qsimmu2 = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            paraLst = []
            for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                paraLst.append(parhbvFull[t, :, ip, :])

            parBETA, parFC, parK0, parK1, parK2, parLP, parPERC, parUZL, parTT, parCFMAX, parCFR, parCWH, parBETAET = paraLst
            # Separate precipitation into liquid and solid components
            PRECIP = Pm[t, :, :]
            RAIN = torch.mul(PRECIP, (Tm[t, :, :] >= parTT).type(torch.float32))
            SNOW = torch.mul(PRECIP, (Tm[t, :, :] < parTT).type(torch.float32))

            # Snow process
            SNOWPACK = SNOWPACK + SNOW
            melt = parCFMAX * (Tm[t, :, :] - parTT)
            melt = torch.clamp(melt, min=0.0)
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = parCFR * parCFMAX * (parTT - Tm[t, :, :])
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (parCWH * SNOWPACK)
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation
            soil_wetness = (SM / parFC) ** parBETA
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness

            # Not used, logging states for checking
            # logSM[t,:] = SM.detach().cpu().numpy()
            # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
            # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
            # logRE[t, :] = recharge.detach().cpu().numpy()

            SM = SM + RAIN + tosoil - recharge
            excess = SM - parFC
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
            evapfactor = (SM / (parLP * parFC)) ** parBETAET
            evapfactor  = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = ETpm[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=PRECS) # SM can not be zero for gradient tracking

            # Groundwater boxes
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, parPERC)
            SUZ = SUZ - PERC
            Q0 = parK0 * torch.clamp(SUZ - parUZL, min=0.0)
            SUZ = SUZ - Q0
            Q1 = parK1 * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = parK2 * SLZ
            SLZ = SLZ - Q2
            Qsimmu[t, :, :] = Q0 + Q1 + Q2

            # save components for Q
            Qsimmu0[t, :, :] = Q0
            Qsimmu1[t, :, :] = Q1
            Qsimmu2[t, :, :] = Q2
            ETmu[t, :, :] = ETact

            # Not used, for debug state variables
            # SMlog[t,:, :] = SM.detach().cpu().numpy()
            # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
            # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

        Qsimave0 = Qsimmu0.mean(-1, keepdim=True)
        Qsimave1 = Qsimmu1.mean(-1, keepdim=True)
        Qsimave2 = Qsimmu2.mean(-1, keepdim=True)
        ETave = ETmu.mean(-1, keepdim=True)

        # get the initial average
        if muwts is None: # simple average
            Qsimave = Qsimmu.mean(-1)
        else: # weighted average using learned weights
            Qsimave = (Qsimmu * muwts).sum(-1)

        if routOpt is True: # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim = Qsimmu.view(Nstep, Ngrid * mu)
            else:
                # average the components, then do routing
                Qsim = Qsimave

            # scale two routing parameters
            tempa = routscaLst[0][0] + rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb = routscaLst[1][0] + rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            UH = UH_gamma(routa, routb, lenF=15)  # lenF: folter
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])   # dim:gage*var*time
            UH = UH.permute([1, 2, 0])  # dim: gage*var*time
            Qsrout = UH_conv(rf, UH).permute([2, 0, 1])

            if comprout is True: # Qs is [time, [gage*mult], var] now
                Qstemp = Qsrout.view(Nstep, Ngrid, mu)
                if muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp * muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else: # no routing, output the initial average simulations
            Qs = torch.unsqueeze(Qsimave, -1) # add a dimension

        if outstate is True: # output states
            return Qs, SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            # return Qs
            Qall = torch.cat((Qs, Qsimave0, Qsimave1, Qsimave2, ETave), dim=-1)
            return Qall


class MultiInv_HBVTDModel(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self, *, ninv, nfea, nmul, hiddeninv, drinv=0.5, inittime=0, routOpt=False, comprout=False,
                 compwts=False, staind=-1, tdlst=[], dydrop=0.0, ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel, self).__init__()
        self.ninv = ninv
        self.nfea = nfea
        self.hiddeninv = hiddeninv
        self.nmul = nmul
        # get the total number of parameters
        nhbvpm = nfea*nmul
        if comprout is False:
            nroutpm = 2
        else:
            nroutpm = nmul*2
        if compwts is False:
            nwtspm = 0
        else:
            nwtspm = nmul
        ntp = nhbvpm + nroutpm + nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv = CudnnLstmModel(
            nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV = HBVMulTDET()
        else:
            # the original HBV para
            self.HBV = HBVMulTD()

        self.gpu = 1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm = nhbvpm
        self.nwtspm = nwtspm
        self.nroutpm = nroutpm
        self.staind = staind
        self.tdlst = tdlst
        self.dydrop = dydrop


    def forward(self, x, z, doDropMC=False):
        Params0 = self.lstminv(z) # dim: Time, Gage, Para
        ntstep = Params0.shape[0]
        ngage = Params0.shape[1]
        # print(Params0)
        hbvpara0 = Params0[:, :, 0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara = torch.sigmoid(hbvpara0).view(ntstep, ngage, self.nfea, self.nmul) # hbv scaled para, [0,1]
        routpara0 = Params0[-1, :, self.nhbvpm:self.nhbvpm+self.nroutpm] # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara = torch.sigmoid(routpara0) # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2) # [0,1]
        if self.nwtspm == 0: # simple average for multiple components
            wts = None
        else: # weighted average using learned weights
            wtspara = Params0[-1, :, -self.nwtspm:]
            wts = F.softmax(wtspara, dim=-1)  # softmax to make sure sum to 1
        out = self.HBV(x, parameters=hbvpara, staind=self.staind, tdlst=self.tdlst, mu=self.nmul, muwts=wts, rtwts=routpara,
                          bufftime=self.inittime, routOpt=self.routOpt, comprout=self.comprout, dydrop=self.dydrop)
        return out
