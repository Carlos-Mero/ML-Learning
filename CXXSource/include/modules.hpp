#ifndef MODULES_HPP
#define MODULES_HPP

#include <cstddef>
#include <memory>
#include <vector>
#include <string>
#include <torch/torch.h>
#include <fmt/core.h>

namespace torch {

class TestNet: public torch::nn::Module {
	public:
		TestNet(int64_t N, int64_t M) {
    W = register_parameter("W", torch::randn({N, M}));
    b = register_parameter("b", torch::randn(M));
		}
		auto forward(torch::Tensor input) {
			return torch::addmm(b, input, W);
		}
		torch::Tensor W, b;
};

struct TestModel: nn::Module {
	TestModel()
		: seq(
				nn::ReLU()
				) {
			register_module("seq", seq);
		}
	auto forward(torch::Tensor x) {
		return seq->forward(x);
	}
	nn::Sequential seq;
};

class LinReLUStack: public nn::Module {
	public:
		LinReLUStack(int64_t N, int64_t M)
			: seq(
					nn::Flatten(),
					nn::Linear(N, 128),
					nn::ReLU(),
					nn::Linear(128, 64),
					nn::ReLU(),
					nn::Linear(64, M),
					nn::Softmax(1)
					) {
				register_module("seq", seq);
			}
		auto forward(torch::Tensor x) {
			return seq->forward(x);
		}
		nn::Sequential seq;
};

struct ConvLinStack: nn::Module {
	ConvLinStack()
		: covseq (
				nn::Conv2d(nn::Conv2dOptions(1, 10, 5)),
				nn::MaxPool2d(2),
				nn::ReLU(),
				nn::Conv2d(nn::Conv2dOptions(10, 20, 5)),
				nn::Dropout2d(),
				nn::MaxPool2d(2),
				nn::ReLU()
				),
			linreluseq (
				nn::Linear(320, 50),
				nn::Dropout(0.5),
				nn::Linear(50, 10),
				nn::LogSoftmax(1)
					) {
			register_module("covseq", covseq);
			register_module("linreluseq", linreluseq);
		}
	auto forward(torch::Tensor x) {
		x = covseq->forward(x);
		x = x.view({-1, 320});
		return linreluseq->forward(x);
	}
	nn::Sequential covseq;
	nn::Sequential linreluseq;
};

struct LinDropoutStack: nn::Module {
	LinDropoutStack(int64_t Numinput, int64_t Numoutput)
		: seq (
				nn::Flatten(),
				nn::Linear(Numinput, 320),
				nn::ReLU(),
				nn::Dropout(0.5),
				nn::Linear(320, 240),
				nn::ReLU(),
				nn::Dropout(0.5),
				nn::Linear(240, Numoutput),
				nn::LogSoftmax(1)
				) {
			register_module("seq", seq);
		}
	auto forward(torch::Tensor x) {
		return seq->forward(x);
	}
	nn::Sequential seq;
};

class LinearRegression: public nn::Module {
	public:
		static const int64_t input_dim;
		static const int64_t output_dim;
		nn::Linear realin;
		nn::Linear lin;
		std::unique_ptr<torch::optim::SGD> sgd_optimizer;
		torch::Tensor input_t;
		torch::Tensor output_t;

		auto forward(torch::Tensor x);
		void train_process();
		void test_process();
		LinearRegression();
		~LinearRegression();
};

class SoftmaxClassification: public ConvLinStack {
	public:
		static const int64_t input_dim;
		static const int64_t output_dim;
		const char* data_path;
		std::unique_ptr<torch::optim::SGD> sgd_optimizer;
		std::vector<double> test_loss_set;
		std::vector<double> test_accuracy;

		void train_process();
		void test_process();
		//SoftmaxClassification(int64_t M, int64_t N);
		SoftmaxClassification();
		~SoftmaxClassification();
};

class MNISTClassification: public nn::Module {
  public:
    static constexpr int64_t input_dim = 28*28;
    static constexpr int64_t output_dim = 10;
    const char* data_path;

    auto forward(torch::Tensor x);
    MNISTClassification()
      : seq (
          nn::Conv2d(nn::Conv2dOptions(1, 2, 3).padding(1).stride(1))
          ) {
        register_module("seq", seq);
      }
    ~MNISTClassification();

    nn::Sequential seq;
};

class ResBlock: public nn::Module {
  public:
    ResBlock(int64_t in_channels, int64_t out_channels, int64_t stride=1)
      : layers_(
          nn::Conv2d(
            nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1)),
          nn::BatchNorm2d(out_channels),
          nn::ReLU(true),
          nn::Conv2d(
            nn::Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(1)),
          nn::BatchNorm2d(out_channels)
          ) {
      if (stride != 1 || in_channels != out_channels) {
        shortcut_ = nn::Sequential(
              nn::Conv2d(
                nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride)),
              nn::BatchNorm2d(out_channels)
            );
      } else {
        shortcut_ = nullptr;
      }
      // register_module
    }

    auto forward(torch::Tensor x) {
      auto res = x;
      x = layers_->forward(x);
      if (shortcut_) {
        res = shortcut_->forward(res);
      }
      x += res;
      x = torch::relu(x);
      return x;
    }

  private:
    nn::Sequential layers_;
    nn::Sequential shortcut_;

};

class ResNetOnMNIST: public nn::Module {
  public:
    ResNetOnMNIST(int num_classes) {
      int64_t in_channels = 1;
      int64_t out_channels = 4;
      std::vector<int64_t> block_repeats{2, 2, 2};

      layers_->push_back(nn::Conv2d(
            nn::Conv2dOptions(in_channels, out_channels, 5).stride(1).padding(2)));
      layers_->push_back(nn::BatchNorm2d(out_channels));
      layers_->push_back(nn::ReLU(true));
      layers_->push_back(nn::MaxPool2d(2));

      in_channels = out_channels;

      for (const auto& repeats: block_repeats) {
        out_channels *= 2;
        layers_->push_back(make_layer(in_channels, out_channels, repeats, 1));
        in_channels = out_channels;
      }

      layers_->push_back(nn::AdaptiveAvgPool2d(1));
      layers_->push_back(nn::Flatten());
      layers_->push_back(nn::Linear(out_channels, num_classes));
      
      int64_t l_count = 0;

      for (auto layer: *layers_) {
        register_module(std::to_string(l_count),
            std::make_shared<nn::AnyModule>(layer));
        l_count++;
      }
    }

    auto forward(torch::Tensor x) {
      for (auto& layer: *layers_) {
        x = layer.forward(x);
      }
      return x;
    }

  private:
    nn::Sequential make_layer(
        int64_t in_channels, int64_t out_channels, int64_t num_blocks, int64_t stride){
      nn::Sequential layers;
      layers->push_back(ResBlock(in_channels, out_channels, stride));
      for (int64_t i = 1; i < num_blocks; ++i) {
        layers->push_back(ResBlock(out_channels, out_channels, 1));
      }
      return layers;
    }
    nn::Sequential layers_;
};

template<typename T>
class TrainerOnMPS: public nn::Module {
	private:
		torch::Device device;
		std::string data_path;
		std::unique_ptr<torch::optim::SGD> sgd_optimizer;
		std::shared_ptr<T> model;
		int64_t kNumsofEpochs;
		double learning_rate;
		double momentum;
		std::vector<double> train_losses;
		std::vector<double> train_accuracys;
		std::vector<double> test_losses;
		std::vector<double> test_accuracys;

	public:
		void train_process();
		void test_process();
		TrainerOnMPS(
        std::shared_ptr<T> mod, int64_t epochs,
        double lr, double mtum, const char* MNIStpath);
		~TrainerOnMPS();
};

}

#endif
