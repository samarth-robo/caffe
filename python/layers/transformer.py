import caffe
import random
from IPython.core.debugger import Tracer

class Xformer(caffe.io.Transformer):
  def __init__(self, inputs):
    caffe.io.Transformer.__init__(self, inputs)
    self.crop_dim = {}

  def set_crop_dim(self, in_, crop_dim):
    caffe.io.Transformer._Transformer__check_input(self, in_)
    assert crop_dim < self.inputs[in_][2],\
      'crop_dim should be less than inputs[in_] height'
    assert crop_dim < self.inputs[in_][3], \
      'crop_dim should be less than inputs[in_] width'
    self.crop_dim[in_] = crop_dim

  def preprocess(self, in_, data, phase=caffe.TRAIN):
    caffe_in = caffe.io.Transformer.preprocess(self, in_, data)
    _, h, w = caffe_in.shape
    if phase is caffe.TRAIN:  # random crop
      y = random.randint(0, h-self.crop_dim[in_])
      x = random.randint(0, w-self.crop_dim[in_])
    else:  # center crop
      y = int((h - self.crop_dim[in_][2]) / 2.0)
      x = int((w - self.crop_dim[in_][3]) / 2.0)
    caffe_in = caffe_in[:, y:y+self.crop_dim[in_], x:x+self.crop_dim[in_]]
    return caffe_in
