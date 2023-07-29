#include "modules.hpp"

#include <cstddef>
#include <memory>
#include <vector>
#include <torch/torch.h>
#include <torchvision/vision.h>
#include <fmt/core.h>

namespace torch {

// Completion of LinearRegression
const int64_t LinearRegression::input_dim = 128;
const int64_t LinearRegression::output_dim = 64;

LinearRegression::LinearRegression()
	: realin(LinearRegression::input_dim, LinearRegression::output_dim),
		lin(LinearRegression::input_dim, LinearRegression::output_dim) {
			register_module("lin", lin);
			sgd_optimizer = std::make_unique<torch::optim::SGD>(
					this->parameters(), torch::optim::SGDOptions(0.01).momentum(0.5)
					);
	}

LinearRegression::~LinearRegression() {
}

auto LinearRegression::forward(torch::Tensor x) {
	return lin->forward(x);
}

void LinearRegression::train_process() {
	this->train();

	for (int i = 0; i < 128; i++) {
		input_t = torch::randn({128, 128});
		sgd_optimizer->zero_grad();
		torch::Tensor loss_v;
		for (int j = 0; j < 128; j++) {
			output_t = this->forward(input_t[j]);
			loss_v = nn::functional::mse_loss(
					output_t, realin->forward(input_t[j])).sum();
			loss_v.backward();
		}
		if (i % 16 == 15 || i == 0) {
			fmt::println("This is the {}th epoch, and the loss is:", i);
			std::cout << loss_v << std::endl;
		}
		sgd_optimizer->step();
	}
}

void LinearRegression::test_process() {

}

// Completion of Classification
//SoftmaxClassification::SoftmaxClassification(int64_t M, int64_t N)
//	: ReLUStack(M, N){
//		data_path = "./data/FashionMNIST/raw";
//		sgd_optimizer = std::make_unique<torch::optim::SGD>(
//				this->parameters(), torch::optim::SGDOptions(0.01).momentum(0.5)
//				);
//}

SoftmaxClassification::SoftmaxClassification()
	: ConvLinStack(){
		data_path = "./data/FashionMNIST/raw";
		sgd_optimizer = std::make_unique<torch::optim::SGD>(
				this->parameters(), torch::optim::SGDOptions(0.01).momentum(0.5)
				);
	}

SoftmaxClassification::~SoftmaxClassification() {

}

void SoftmaxClassification::train_process() {
	int64_t kNumberofEpoches = 20;
	torch::Device device(torch::kMPS);
	this->to(device);

	auto train_dataset = torch::data::datasets::MNIST(data_path)
		.map(torch::data::transforms::Normalize<>(0.1307, 0.3801))
		.map(torch::data::transforms::Stack<>());
	auto test_dataset = torch::data::datasets::MNIST(
		data_path, torch::data::datasets::MNIST::Mode::kTest)
		.map(torch::data::transforms::Normalize<>(0.1307, 0.3801))
		.map(torch::data::transforms::Stack<>());

	const size_t train_data_size = train_dataset.size().value();

	auto train_loader =
		torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
				std::move(train_dataset), 64);
	auto test_loader =
		torch::data::make_data_loader(
				std::move(test_dataset), 1024);
	sgd_optimizer = std::make_unique<torch::optim::SGD>(
			this->parameters(), torch::optim::SGDOptions(0.003).momentum(0.5)
			);

	std::vector<double> accuracy_set(20);
	const double full_size = 60000.0;

	for (int64_t epoch = 0; epoch < kNumberofEpoches; epoch++) {
		this->train();
		size_t batch_idx = 0;

		for (const auto& batch: *train_loader) {
			auto data = batch.data.to(device);
			auto target = batch.target.to(device);
			sgd_optimizer->zero_grad();

			auto output_t = this->forward(data);
			auto loss = torch::nll_loss(output_t, target);
			loss.backward();
			sgd_optimizer->step();
			accuracy_set[epoch] +=
				(output_t.argmax(1) == target).sum().template item<double>();

			if (++batch_idx % 10 == 1) {
				float accuracy =
					(output_t.argmax(1) == target).sum().template item<float>();
				accuracy /= batch.data.size(0);
				accuracy *= 100;
				fmt::println("Train Epoch: {} [{}/{}], Loss: {}, Accuracy: {}%",
						epoch,
						batch_idx * batch.data.size(0),
						train_data_size,
						loss.template item<float>(),
						accuracy);
			}
		}
	}
	for (int64_t epoch = 0; epoch < kNumberofEpoches; epoch++) {
		fmt::println("-------------------------------------");
		fmt::println("The total accuracy of epoch{} is: {}%",
				epoch,
				accuracy_set[epoch] / full_size * 100);
	}	
}

void SoftmaxClassification::test_process() {

}

}
