import caffe
import glog
import sys
import random
import numpy as np
import h5py
import json
from proto_utils import proto_to_np
import os.path as osp
from transformer import Xformer
from multiprocessing import Pool
from threading import Thread
from IPython.core.debugger import Tracer

class MultiInputDataLayer(caffe.Layer):
  """
  A data layer that has multiple inputs (.txt, .h5) and shuffles them in sync
  """
  def setup(self, bottom, top):
    self.params = json.loads(self.param_str)
    self.params['phase'] = self.phase
    self.batch_size = self.params['batch_size']
    self.thread_result = {}
    self.thread = None
    if self.params['multithreaded_preprocess']:
      self.pool = Pool(processes=8)
    else:
      self.pool = None

    assert len(top) == len(self.params['top_names']),\
      "Number of tops does not match number of inputs"

    # fix the type of each input
    self.params['type'] = {}
    for top_name, ip in self.params['source'].items():
      if ip.find('txt') >= 0:
        self.params['type'][top_name] = 'txt'
      elif ip.find('h5') >= 0:
        self.params['type'][top_name] = 'h5'
      else:
        glog.info("Wrong input type {:s}".format(ip))
        sys.exit(-1)
    
    self.batch_loader = BatchLoader(self.params, self.thread_result, self.pool)

    # reshape tops
    # load one batch to get shapes
    if self.params['multithreaded_prefetch']:
      self.dispatch_worker_multi_threaded()
      self.join_worker()
    else:
      self.dispatch_worker_single_threaded()

    for i, tn in enumerate(self.params['top_names']):
      data = self.thread_result[tn]
      assert data.shape[0] == self.batch_size,\
        '{:s}.shape[0] != batch size'.format(tn)
      top[i].reshape(*data.shape)
    self.batch_loader.clear_counters()

    if self.params['multithreaded_prefetch']:
      glog.info('Multi-threaded prefetch')
    else:
      glog.info('Single-threaded prefetch')

    if self.params['multithreaded_preprocess']:
      glog.info('Multi-threaded preprocess')
    else:
      glog.info('Single-threaded preprocess')

  def reshape(self, bottom, top):
    pass

  def backward(self, bottom, top):
    pass

  def forward(self, bottom, top):
    if self.params['multithreaded_prefetch']:
      if self.thread is not None:
        self.join_worker()

    for i, tn in enumerate(self.params['top_names']):
      data = self.thread_result[tn]
      top[i].data[...] = data

    if self.params['multithreaded_prefetch']:
      self.dispatch_worker_multi_threaded()
    else:
      self.dispatch_worker_single_threaded()

  def dispatch_worker_multi_threaded(self):
    assert self.thread is None
    self.thread = Thread(target=self.batch_loader.load_batch)
    self.thread.start()

  def dispatch_worker_single_threaded(self):
    self.batch_loader.load_batch()

  def join_worker(self):
    assert self.thread is not None
    self.thread.join()
    self.thread = None

  def __del__(self):
    glog.info('Reached python layer destructor')
    self.batch_loader.__del__()

class ImageProcessor:
  def __init__(self, xform_object, phase, top_name=None):
    self.xform_object = xform_object
    self.phase = phase
    self.top_name = top_name

  def __call__(self, im_fn):
    return im_load_preprocess(im_fn, self.xform_object, self.phase, self.top_name)

def im_load_preprocess(im_fn, xform_object, phase, tn):
  im = caffe.io.load_image(im_fn)
  im_t = xform_object.preprocess(tn, im, phase)
  return im_t

class BatchLoader:
  def __init__(self, params, result, pool):
    self.params = params
    self.result = result
    self.pool = pool
    self.data = {}
    self.h5_files = {}
    self.N = 0
    xformer_dict = {}

    # get data sources
    for tn, ip in self.params['source'].items():
      if self.params['type'][tn] == 'txt':
        xformer_dict[tn] = (self.params['batch_size'], 3,
            self.params['new_height'][tn], self.params['new_width'][tn])
        root_folder = osp.expanduser(self.params['root_folder'][tn])
        with open(osp.expanduser(ip), 'r') as f:
          im_names = [l.rstrip().split(' ')[0] for l in f]
          self.data[tn] = [osp.join(root_folder, im_name) for im_name in im_names]
          glog.info('Read {:d} image names from {:s}'.format(len(self.data[tn]),
            ip))
        data_len = len(self.data[tn])
      elif self.params['type'][tn] == 'h5':
        f = h5py.File(osp.expanduser(ip), 'r')
        self.h5_files[tn] = f
        self.data[tn] = f[f.keys()[0]]
        glog.info('Got dataset of shape {:s} from {:s}'.format(str(self.data[tn].shape), ip))
        data_len = self.data[tn].shape[0]

      if self.N == 0:
        self.N = data_len
      else:
        assert self.N == data_len,\
          "File {:s} does not have same # data points as others".format(ip)
    glog.info('{:d} data points in all input files'.format(self.N))

    # RNGs and counters for shuffling
    self.order = {tn: [i for i in xrange(self.N)] for tn in self.params['top_names']}
    self.counter = {tn: caffe.solver_rank() for tn in self.params['top_names']}
    if params['shuffle']:
      self.rngs = {tn: random.Random() for tn in self.params['top_names']}
      state = random.getstate()
      for tn in self.params['top_names']:
        self.rngs[tn].setstate(state)
        self.rngs[tn].shuffle(self.order[tn])
        glog.info('Shuffled order of data for top {:s}'.format(tn))

    # set up image preprocessing objects
    self.xformer = Xformer(xformer_dict)
    for tn in self.params['top_names']:
      if self.params['type'][tn] is not 'txt':
        continue
      im_mean = proto_to_np(osp.expanduser(self.params['mean_file'][tn]))
      self.xformer.set_mean(tn, im_mean[0, :])
      self.xformer.set_crop_size(tn, self.params['crop_size'][tn])
      self.xformer.set_transpose(tn, (2, 0, 1))
      # because we load image from skimage and not opencv
      self.xformer.set_raw_scale(tn, 255)
      self.xformer.set_channel_swap(tn, (2, 1, 0))

    self.image_processor = ImageProcessor(self.xformer, self.params['phase'])

  def __del__(self):
    glog.info('Reached destructor')
    for _, f in self.h5_files.items():
      f.close()

  def clear_counters(self):
    self.counter = {tn: caffe.solver_rank() for tn in self.params['top_names']}

  def load_batch(self):
    for tn in self.params['top_names']:
      if self.params['type'][tn] == 'txt':
        self.result[tn] = self.load_from_txt(tn, self.data[tn])
      elif self.params['type'][tn] == 'h5':
        self.result[tn] = self.load_from_h5(tn, self.data[tn])

  def load_from_txt(self, tn, data_src):
    im_fns = []
    for i in xrange(self.params['batch_size']):
      im_fns.append(data_src[self.order[tn][self.counter[tn]]])
      self.counter[tn] += caffe.solver_count()
      if self.counter[tn] >= self.N:
        self.counter[tn] = caffe.solver_rank()
        if self.params['shuffle']:
          self.rngs[tn].shuffle(self.order[tn])
          glog.info('Epoch finished, shuffled source for {:s} again'.format(tn))
    self.image_processor.top_name = tn
    if self.params['multithreaded_preprocess']:
      data = self.pool.map(self.image_processor, im_fns)
    else:
      data = map(self.image_processor, im_fns)
    data = np.asarray(data)
    return data

  def load_from_h5(self, tn, data_src):
    data = np.zeros((self.params['batch_size'],)+data_src.shape[1:], dtype=np.float32)
    for i in xrange(self.params['batch_size']):
      idx = self.order[tn][self.counter[tn]]
      data[i, :] = data_src[idx]
      self.counter[tn] += caffe.solver_count()
      if self.counter[tn] >= self.N:
        self.counter[tn] = caffe.solver_rank()
        if self.params['shuffle']:
          self.rngs[tn].shuffle(self.order[tn])
    return data
