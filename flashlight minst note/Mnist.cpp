/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * A flashlight introduction to Convolutional Neural Networks on MNIST. The
 * model is based on this tutorial from TensorFlow:
 * https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py
 *
 * To run this demo first download and unpack the mnist dataset from:
 * http://yann.lecun.com/exdb/mnist/
 *
 * Once downloaded run the program with:
 * ./Mnist <path_to_data>
 *
 * Final output should be close to:
 *   Test Loss: 0.0373 Test Error (%): 1.1
 */

#include <iomanip>
#include <iostream>
#include <stdexcept>

#include <arrayfire.h>
#include "flashlight/fl/flashlight.h"

using namespace af;
using namespace fl;

namespace {
const int TRAIN_SIZE = 60000;
const int VAL_SIZE = 5000; /* Held-out from train. */
const int TEST_SIZE = 10000;
const int IM_DIM = 28;
const int PIXEL_MAX = 255;
const int INPUT_IDX = 0;
const int TARGET_IDX = 1;

std::pair<double, double> eval_loop(Sequential& model, BatchDataset& dataset) {
  AverageValueMeter loss_meter;
  FrameErrorMeter error_meter;

  // Place the model in eval mode.
  model.eval();
  for (auto& example : dataset) {
    auto inputs = noGrad(example[INPUT_IDX]);
    auto output = model(inputs);

    // Get the predictions in max_ids
    array max_vals, max_ids;
    max(max_vals, max_ids, output.array(), 0);

    auto target = noGrad(example[TARGET_IDX]);

    // Compute and record the prediction error.
    error_meter.add(reorder(max_ids, 1, 0), target.array());

    // Compute and record the loss.
    auto loss = categoricalCrossEntropy(output, target);
    loss_meter.add(loss.array().scalar<float>());
  }
  // Place the model back into train mode.
  model.train();

  double error = error_meter.value();
  double loss = loss_meter.value()[0];
  return std::make_pair(loss, error);
}

std::pair<array, array> load_dataset(
    const std::string& data_dir,
    bool test = false);

} // namespace

// argc: arguments count
// 第一个参数argc表示向main函数传递的参数的个数，但是它实际上要比你在命令行里输入的数据多一个，因为第一个参数它保存了该程序的路径名，也就是说，如果你向命令行输入3个数，则argc实际上等于4；
// argv: arguments value/vectors
// 第二个参数argv保存命令行输入的参数值
int main(int argc, char** argv) {
  fl::init();
  if (argc != 2) {
    throw af::exception("You must pass a data directory.");
  }
  af::setSeed(1); // 设置随机生成数种子
  std::string data_dir = argv[1]; // 读数据的路径

  float learning_rate = 1e-2;
  int epochs = 10;
  int batch_size = 64;

  // array是ArrayFire提供了一个通用的容器对象，在array上执行函数和数学操作。
  // 该数组array可以表示许多不同的基本数据类型
  array train_x;
  array train_y;
  // std::tie会将变量的引用整合成一个tuple，从而实现批量赋值。
  // 读取数据部分
  std::tie(train_x, train_y) = load_dataset(data_dir);

  // Hold out a dev set
  // 取出验证集
  auto val_x = train_x(span, span, 0, seq(0, VAL_SIZE - 1));
  train_x = train_x(span, span, 0, seq(VAL_SIZE, TRAIN_SIZE - 1));
  auto val_y = train_y(seq(0, VAL_SIZE - 1));
  train_y = train_y(seq(VAL_SIZE, TRAIN_SIZE - 1));

  // Make the training batch dataset
  BatchDataset trainset(
      std::make_shared<TensorDataset>(std::vector<af::array>{train_x, train_y}),
      batch_size);

  // Make the validation batch dataset
  BatchDataset valset(
      std::make_shared<TensorDataset>(std::vector<af::array>{val_x, val_y}),
      batch_size);

  Sequential model;
  auto pad = PaddingMode::SAME;
  model.add(View(af::dim4(IM_DIM, IM_DIM, 1, -1)));
  model.add(Conv2D(
      1 /* input channels */,
      32 /* output channels */,
      5 /* kernel width */,
      5 /* kernel height */,
      1 /* stride x */,
      1 /* stride y */,
      pad /* padding mode */,
      pad /* padding mode */));
  model.add(ReLU());
  model.add(Pool2D(
      2 /* kernel width */,
      2 /* kernel height */,
      2 /* stride x */,
      2 /* stride y */));
  model.add(Conv2D(32, 64, 5, 5, 1, 1, pad, pad));
  model.add(ReLU());
  model.add(Pool2D(2, 2, 2, 2));
  model.add(View(af::dim4(7 * 7 * 64, -1)));
  model.add(Linear(7 * 7 * 64, 1024));
  model.add(ReLU());
  model.add(Dropout(0.5));
  model.add(Linear(1024, 10));
  model.add(LogSoftmax());

  // Make the optimizer
  SGDOptimizer opt(model.params(), learning_rate);

  // The main training loop
  for (int e = 0; e < epochs; e++) {
    AverageValueMeter train_loss_meter;

    // Get an iterator over the data
    for (auto& example : trainset) {
      // Make a Variable from the input array.
      auto inputs = noGrad(example[INPUT_IDX]);

      // Get the activations from the model.
      auto output = model(inputs);

      // Make a Variable from the target array.
      auto target = noGrad(example[TARGET_IDX]);

      // Compute and record the loss.
      auto loss = categoricalCrossEntropy(output, target);
      train_loss_meter.add(loss.array().scalar<float>());

      // Backprop, update the weights and then zero the gradients.
      loss.backward();
      opt.step();
      opt.zeroGrad();
    }

    double train_loss = train_loss_meter.value()[0];

    // Evaluate on the dev set.
    double val_loss, val_error;
    std::tie(val_loss, val_error) = eval_loop(model, valset);

    std::cout << "Epoch " << e << std::setprecision(3)
              << ": Avg Train Loss: " << train_loss
              << " Validation Loss: " << val_loss
              << " Validation Error (%): " << val_error << std::endl;
  }

  array test_x;
  array test_y;
  std::tie(test_x, test_y) = load_dataset(data_dir, true);

  BatchDataset testset(
      std::make_shared<TensorDataset>(std::vector<af::array>{test_x, test_y}),
      batch_size);

  double test_loss, test_error;
  std::tie(test_loss, test_error) = eval_loop(model, testset);
  std::cout << "Test Loss: " << test_loss << " Test Error (%): " << test_error
            << std::endl;

  return 0;
}

namespace {

// MNIST Data loading functions below.
// MNIST 数据集读取
int read_int(std::ifstream& f) {
  int d = 0;
  int c;
  // sizeof(int) 计算本机int的长度
  for (int i = 0; i < sizeof(int); i++) {
    c = 0;
    // read (char* s, streamsize n)
    // 从流中提取n个字符并将其存储在s指向的数组中。
    // 此函数仅复制数据块，而无需检查其内容或在末尾添加空字符。
    // s: Pointer to an array where the extracted characters are stored.
    // n: Number of characters to extract. streamsize is a signed integral type.
    // (char*)&c
    // This is called a cast.In C, a cast lets you convert or
    // reinterpret a value from one type to another.When you take the
    // address of the int, you get a int *; casting that to a char * gives
    // you a pointer referring to the same location in memory, but
    // pretending that what lives there is char data rather than int data
    /// \note 为什么要让c=0??????????
    f.read((char*)&c, 1);
    // 按位或运算
    d |= (c << (8 * (sizeof(int) - i - 1)));
  }
  /// \note ????
  return d;
}

// 定义模板
template <typename T>
// 读取数据集函数
// 当函数形参为int& k时，在函数中对k进行操作
// 比如说赋值，那么主函数中相应的主对象的值也会跟着改变。
// 相当于，指针变量，但是不需要加上间接访问符*，直接就是对k本身操作就OK，且与指针变量有区别，相当于一个静态的指针。
// 就是引用。
array load_data(
    const std::string& im_file,
    const std::vector<long long int>& dims) {
  // 以二进制方式打开, 单字符打开（仅有换行）
  std::ifstream file(im_file, std::ios::binary);
  if (!file.is_open()) {
    throw af::exception("[mnist:load_data] Can't find MNIST file.");
  }
  /// \note ????
  read_int(file); // unused magic
  // size_t是一种数据相关的无符号类型，它被设计得足够大以便能够内存中任意对象的大小。
  // 通常我们用sizeof(XXX)操作，这个操作所得到的结果就是size_t类型。
  size_t elems = 1;
  for (auto d : dims) {
    int read_d = read_int(file);
    elems *= read_d;
    /// \note ????
    if (read_d != d) {
      throw af::exception("[mnist:load_data] Unexpected MNIST dimension.");
    }
  }

  // 运用模板
  std::vector<T> data;
  // reserve的作用是更改vector的容量（capacity），使vector至少可以容纳n个元素。
  // 如果n大于vector当前的容量，reserve会对vector进行扩容。其他情况下都不会重新分配vector的存储空间
  data.reserve(elems);
  for (int i = 0; i < elems; i++) {
    // 无符号版本和有符号版本的区别就是无符号类型能保存2倍于有符号类型的正整数数据。
    unsigned char tmp;
    file.read((char*)&tmp, sizeof(tmp));
    data.push_back(tmp);
  }

  // c.rbegin() 返回一个逆序迭代器，它指向容器c的最后一个元素
  // c.rend() 返回一个逆序迭代器，它指向容器c的第一个元素前面的位置
  std::vector<long long int> rdims(dims.rbegin(), dims.rend());
  // af is column-major
  dim4 af_dims(rdims.size(), rdims.data());
  return array(af_dims, data.data());
}

// 判断数据集文件
std::pair<array, array> load_dataset(
    const std::string& data_dir,
    bool test /* = false */) {
  // 判断读取训练集还是测试集
  std::string f = test ? "t10k" : "train";
  int size = test ? TEST_SIZE : TRAIN_SIZE;

  // 确定读取图片文件的路径和文件名称
  std::string image_file = data_dir + "/" + f + "-images-idx3-ubyte";

  array ims = load_data<float>(image_file, {size, IM_DIM, IM_DIM});
  // Modifies the input dimensions without changing the data order.
  // The shape of the output Variable is specified in descriptor dims.
  ims = moddims(ims, {IM_DIM, IM_DIM, 1, size});
  // Rescale to [-0.5,  0.5]
  ims = (ims - PIXEL_MAX / 2) / PIXEL_MAX;

  // 确定读取图片的标签文件
  std::string label_file = data_dir + "/" + f + "-labels-idx1-ubyte";
  array labels = load_data<int>(label_file, {size});

  return std::make_pair(ims, labels);
}
} // namespace
