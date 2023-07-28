#include <iostream>

#include <cstddef>
#include <fmt/core.h>
#include <torch/torch.h>

#include "modules.hpp"

using namespace torch;

int main(int argc, char** argv) {
	std::ios::sync_with_stdio(false);
	fmt::println("The main function is called here.");
	auto v = torch::randn({3, 4});
	std::cout << v << std::endl;

	auto net = TestNet(4, 3);
	std::cout << net.forward(v) << std::endl;

	auto model = TestModel();
	std::cout << model.forward(net.forward(v)) << std::endl;

	auto v2 = torch::randn(3);
	auto relustack = ReLUStack(3, 4);
	std::cout << relustack.forward(v2) << std::endl;
	std::cout << relustack << std::endl;
	return 0;
}
