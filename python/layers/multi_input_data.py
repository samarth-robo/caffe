import caffe
import glog
import sys
from random import shuffle
import cv2
import numpy as np
import os.path as osp
from transformer import Xformer
from IPython.core.debugger import Tracer

class MultiInputDataLayer(caffe.Layer):
  """
  A data layer that has multiple inputs (.txt, .h5) and shuffles them in sync
  """
  def setup(self, bottom, top):
    self.params = eval(self.param_str)
    self.params['phase'] = self.phase
    self.check_params()
    self.batch_size = self.params['batch_size']
    self.thread_result = {}

    assert len(top) == len(self.params['top_names']),\
      "Number of tops does not match number of inputs"

    # fix the type of each input
    self.params['type'] = {}
    for top_name, ip in self.params['input'].items():
      if ip.find('txt') >= 0:
        self.params['type'][top_name] = 'txt'
      elif ip.find('h5') >= 0:
        self.params['type'][top_name] = 'h5'
      else:
        glog.info("Wrong input type {:s}".format(ip))
        sys.exit(-1)
    self.batch_loader = BatchLoader(self.params, self.thread_result)

    # reshape tops
    # load one batch to get shapes
    self.batch_loader.load_batch()
    for i, tn in enumerate(self.params['top_names']):
      data = self.thread_result[tn]
      assert data.shape[0] == self.batch_size,\
        '{:s}.shape[0] != batch size'.format(tn)
      top[i].reshape(*data.shape)
    self.batch_loader.clear_counter()

  def reshape(self, bottom, top):
    pass

  def backward(self, bottom, top):
    pass

  def forward(self, bottom, top):
    self.batch_loader.load_batch()
    for i, tn in enumerate(self.params['top_names']):
      data = self.thread_result[tn]
      top[i].data[...] = data

  def check_params(self):
    pass

class BatchLoader:
  def __init__(self, params, result):
    self.params = params
    self.result = result
    self.counter = 0
    self.data = {}
    self.N = 0
    xformer_dict = {}
    for tn, ip in self.params['input'].items():
      if self.params['type'][tn] == 'txt':
        xformer_dict[tn] = (self.params['batch_size'], 3,
            self.params['new_height'][tn], self.params['new_width'][tn])
        root_folder = self.params['root_folder'][tn]
        with open(ip, 'r') as f:
          im_names = [l.rstrip().split(' ')[0] for l in f]
          self.data[tn] = [osp.join(root_folder, im_name) for im_name in im_names] 
      elif self.params['type'][tn] == 'h5':
        self.data[top_name] = None

      if self.N == 0:
        self.N = len(self.data[tn])
      else:
        assert self.N == len(self.data[tn]),\
          "File {:s} does not have same # data points as others".format(ip)

    self.order = [i for i in xrange(self.N)]
    if params['shuffle']:
      shuffle(self.order)

    self.xformer = Xformer(xformer_dict)
    for tn in self.params['top_names']:
      if self.params['type'][tn] is not 'txt':
        continue
      self.xformer.set_crop_dim(tn, self.params['crop_dim'][tn])
      self.xformer.set_transpose(tn, (2, 0, 1))
      # TODO: set mean and other things

  def clear_counter(self):
    self.counter = 0

  def load_batch(self):
    for tn in self.params['top_names']:
      if self.params['type'][tn] == 'txt':
        self.result[tn] = self.load_from_txt(tn, self.data[tn])
      elif self.params['type'][tn] == 'h5':
        self.result[tn] = self.load_from_h5(tn, self.data[tn])

  def load_from_txt(self, top_name, data_src):
    data = np.array([])
    for i in xrange(self.params['batch_size']):
      im_fn = data_src[self.order[self.counter]]
      im = cv2.imread(im_fn)
      self.counter += 1
      if im is not None:
        im_t = self.xformer.preprocess(top_name, im, self.params['phase'])
        data = np.concatenate((data, im_t[np.newaxis, :])) if data.size is not 0 else im_t[np.newaxis, :]
        if self.counter == self.N:
          self.counter = 0
          if self.params['shuffle']:
            shuffle(self.order)
            glog.info('Epoch finished, shuffling again')
      else:
        glog.info('Could not read {:s}'.format(im_fn))
    return data

  def load_from_h5(self, top_name, data_src):
    glog.info('Not implemented')
    return None
