#include <iostream>

#include <cstddef>
#include <fmt/core.h>
#include <torch/torch.h>

#include "modules.hpp"

using namespace torch;

int main(int argc, char** argv) {
	std::ios::sync_with_stdio(false);
	fmt::println("The main function is called here.");

  int64_t num_classes = 10;

	auto trainer = TrainerOnMPS<ResNetOnMNIST>(
			std::make_shared<ResNetOnMNIST>(num_classes), 10, 0.01, 0.5,
      static_cast<const char*>("../data/FashionMNIST/raw"));
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
