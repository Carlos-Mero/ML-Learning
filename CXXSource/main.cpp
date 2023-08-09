#include <iostream>

#include <cstddef>
#include <fmt/core.h>
#include <torch/torch.h>

#include "modules.hpp"

using namespace torch;

int main(int argc, char** argv) {
	std::ios::sync_with_stdio(false);
	fmt::println("The main function is called here.");

	auto trainer = TrainerOnMPS<LinDropoutStack>(
			std::make_unique<LinDropoutStack>(28*28, 10), 10, 0.01, 0.5);
	fmt::println("The trainer is initialized!");
	trainer.train_process();

//	torch::save(
//			sm_reg.covseq, "../models/SoftmaxClassification_covseq.pt");
//	torch::save(
//			sm_reg.linreluseq, "../models/SoftmaxClassification_linreluseq.pt");

	fmt::println("-----------------------------------");
	fmt::println("Done! The final result seems great!");
	return 0;
}
