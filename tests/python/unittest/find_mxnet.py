MX_PATH='../../../python'
#MX_PATH='/v/space/bullfinch/epiyusi/svn/test/mxnet/python'
try:
    import mxnet as mx
except ImportError:
    import os, sys
    curr_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(curr_path, MX_PATH))
    import mxnet as mx
