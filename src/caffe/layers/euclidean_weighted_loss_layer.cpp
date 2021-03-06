#include <vector>

#include "caffe/layers/euclidean_weighted_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanWeightedLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  CHECK_EQ(bottom[0]->count(1), bottom[2]->count(1))
      << "Loss weights must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  tmp_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanWeightedLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  // tmp_i = (a_i - b_i)
  caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(),
      tmp_.mutable_cpu_data());
  // diff_i = w_i * (a_i - b_i) = w_i * tmp_i
  caffe_mul(count, tmp_.cpu_data(), bottom[2]->cpu_data(), diff_.mutable_cpu_data());
  // dot = sum(w_i * (a_i - b_i)^2) = dot(tmp, diff)
  Dtype dot = caffe_cpu_dot(count, tmp_.cpu_data(), diff_->cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanWeightedLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanWeightedLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanWeightedLossLayer);
REGISTER_LAYER_CLASS(EuclideanWeightedLoss);

}  // namespace caffe
