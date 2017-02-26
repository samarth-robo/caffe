import caffe
import random
from IPython.core.debugger import Tracer

class Xformer(caffe.io.Transformer):
  def __init__(self, inputs):
    caffe.io.Transformer.__init__(self, inputs)
    self.crop_size = {}

  def set_crop_size(self, in_, crop_size):
    caffe.io.Transformer._Transformer__check_input(self, in_)
    assert crop_size <= self.inputs[in_][2],\
      'crop_size should be less than inputs[in_] height'
    assert crop_size <= self.inputs[in_][3], \
      'crop_size should be less than inputs[in_] width'
    self.crop_size[in_] = crop_size

  def preprocess(self, in_, data, phase=caffe.TRAIN):
    caffe_in = caffe.io.Transformer.preprocess(self, in_, data)
    _, h, w = caffe_in.shape
    if phase is caffe.TRAIN:  # random crop
      y = random.randint(0, h-self.crop_size[in_])
      x = random.randint(0, w-self.crop_size[in_])
    else:  # center crop
      y = int((h - self.crop_size[in_]) / 2.0)
      x = int((w - self.crop_size[in_]) / 2.0)
    caffe_in = caffe_in[:, y:y+self.crop_size[in_], x:x+self.crop_size[in_]]
    return caffe_in
