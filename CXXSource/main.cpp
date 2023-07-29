#include <iostream>

#include <cstddef>
#include <fmt/core.h>
#include <torch/torch.h>

#include "modules.hpp"

using namespace torch;

int main(int argc, char** argv) {
	std::ios::sync_with_stdio(false);
	fmt::println("The main function is called here.");

	auto sm_reg = SoftmaxClassification();
	std::cout << sm_reg << std::endl;
	sm_reg.train_process();

//	torch::save(
//			sm_reg.covseq, "../models/SoftmaxClassification_covseq.pt");
//	torch::save(
//			sm_reg.linreluseq, "../models/SoftmaxClassification_linreluseq.pt");

	fmt::println("-----------------------------------");
	fmt::println("Done! The final result seems great!");
	return 0;
}
