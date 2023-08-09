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

template<typename T>
class TrainerOnMPS {
	private:
		torch::Device device;
		std::string data_path;
		std::unique_ptr<torch::optim::SGD> sgd_optimizer;
		std::unique_ptr<T> model;
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
		TrainerOnMPS(std::unique_ptr<T> mod, int64_t epochs, double lr, double mtum);
		~TrainerOnMPS();
};

}

#endif
