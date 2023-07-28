#ifndef MODULES_HPP
#define MODULES_HPP

#include <cstddef>
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

class ReLUStack: public nn::Module {
	public:
		ReLUStack(int64_t N, int64_t M)
			: seq(
					nn::Linear(N, 128),
					nn::ReLU(),
					nn::Linear(128, 64),
					nn::ReLU(),
					nn::Linear(64, M)
					) {
				register_module("seq", seq);
			}
		auto forward(torch::Tensor x) {
			return seq->forward(x);
		}
		nn::Sequential seq;
};

}

#endif
