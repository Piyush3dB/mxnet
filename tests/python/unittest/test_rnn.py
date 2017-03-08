import find_mxnet
import mxnet as mx
import numpy as np
import pdb as pdb
pds = pdb.set_trace

import sys
sys.path.append('/v/home/epiyusi/private/Downloads/github/mxnetUtils')
sys.path.append('../../../../mxnet-utils')
from mxnetUtils import printStats, _str2tuple, net2dot

shape = {}

def test_rnn():
    cell = mx.rnn.RNNCell(100, prefix='rnn_')
    outputs, _ = cell.unroll(3, input_prefix='rnn_')
    outputs = mx.sym.Group(outputs)
    assert sorted(cell.params._params.keys()) == ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']
    assert outputs.list_outputs() == ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]

def test_lstm():
    cell = mx.rnn.LSTMCell(100, prefix='rnn_')
    outputs, _ = cell.unroll(3, input_prefix='rnn_')
    outputs = mx.sym.Group(outputs)
    assert sorted(cell.params._params.keys()) == ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']
    assert outputs.list_outputs() == ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]


def test_min_scalar():
    inputs = mx.symbol.Variable('data')
    next_c = mx.symbol._internal._minimum_scalar(inputs, scalar=1, name='state')
    net = next_c
    #net = inputs
    executor = net.bind(mx.cpu(), {'data': mx.nd.array([0, 0.9, 1.1, 2, 3]) })
    res = executor.forward()
    print(res[0].asnumpy())


_num_hidden = 100
_num_unroll = 1

def test_rwa0():
    cell = mx.rnn.RWACell(_num_hidden, prefix='rwa_')
    net, _ = cell.unroll(_num_unroll, input_prefix='rwa_')

    net = net[0]

    shape["rwa_t0_data"] = (16,50)
    printStats(net, shape=shape)

    #v = net2dot(net, shape=shape)
    #v.render('RWA')

    #pds()

    
def test_rwa():

    inputs = mx.symbol.Variable('data')

    name =  'testName' #'%st%d_'%(self._prefix, self._counter)
    i2h = mx.symbol.FullyConnected(data=inputs, num_hidden=_num_hidden, name=name)

    next_c = mx.symbol._internal._minimum_scalar(i2h, scalar=1, name='state')

    net = mx.symbol.exp(-next_c)

    #net = next_c
    #net = inputs

    #executor = net.bind(mx.cpu(), {'data': mx.nd.array([0, 0.9, 1.1, 2, 3]) })

    #res = executor.forward()
    #print(res[0].asnumpy())

    #print net.tojson()




    shape["data"] = (10,100)
    printStats(net, shape=shape)

    v = net2dot(net, shape=shape)
    v.render('RWA.pdf')

    #pds()

    return


    cell = mx.rnn.RWACell(100, prefix='rnn_')
    outputs, _ = cell.unroll(3, input_prefix='rnn_')
    outputs = mx.sym.Group(outputs)
    assert sorted(cell.params._params.keys()) == ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']
    assert outputs.list_outputs() == ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]

def test_stack():
    cell = mx.rnn.SequentialRNNCell()
    for i in range(5):
        cell.add(mx.rnn.LSTMCell(100, prefix='rnn_stack%d_'%i))
    outputs, _ = cell.unroll(3, input_prefix='rnn_')
    outputs = mx.sym.Group(outputs)
    keys = sorted(cell.params._params.keys())
    for i in range(5):
        assert 'rnn_stack%d_h2h_weight'%i in keys
        assert 'rnn_stack%d_h2h_bias'%i in keys
        assert 'rnn_stack%d_i2h_weight'%i in keys
        assert 'rnn_stack%d_i2h_bias'%i in keys
    assert outputs.list_outputs() == ['rnn_stack4_t0_out_output', 'rnn_stack4_t1_out_output', 'rnn_stack4_t2_out_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]

if __name__ == '__main__':
    #test_rnn()
    #test_lstm()
    #test_min_scalar()
    test_rwa0()
    #test_stack()
