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
		data_path = "../data/FashionMNIST/raw";
		sgd_optimizer = std::make_unique<torch::optim::SGD>(
				this->parameters(), torch::optim::SGDOptions(0.003).momentum(0.5)
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

	const size_t train_data_size = train_dataset.size().value();

	auto train_loader =
		torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
				std::move(train_dataset), 64);

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
		this->test_process();
	}
	for (int64_t epoch = 0; epoch < kNumberofEpoches; epoch++) {
		fmt::println("-------------------------------------");
		fmt::println("The total train accuracy of epoch{} is: {}%",
				epoch,
				accuracy_set[epoch] / full_size * 100);
		fmt::println("The test accuracy of epoch{} is: {}%",
				epoch,
				test_accuracy[epoch]);
		fmt::println("The test loss of epoch{} is: {}",
				epoch,
				test_loss_set[epoch]);
	}	
}

void SoftmaxClassification::test_process() {
	torch::Device device(torch::kMPS);
	this->to(device);
	this->eval();

	auto test_dataset = torch::data::datasets::MNIST(
		data_path, torch::data::datasets::MNIST::Mode::kTest)
		.map(torch::data::transforms::Normalize<>(0.1307, 0.3801))
		.map(torch::data::transforms::Stack<>());
	auto test_loader =
		torch::data::make_data_loader(
				std::move(test_dataset), 1000);

	const int64_t test_dataset_size = 1000;
	double test_loss = 0.0;
	double accuracy = 0.0;

	for (const auto& batch: *test_loader) {
		auto data = batch.data.to(device);
		auto target = batch.target.to(device);
		auto output_t = this->forward(data);
		test_loss += torch::nll_loss(
				output_t,
				target,
				{},
				torch::Reduction::Sum
				).template item<double>();
		auto pred = output_t.argmax(1);
		accuracy += pred.eq(target).sum().template item<double>();
	}

	fmt::println("--------------------------------------");
	fmt::println("The test loss of the model is: {}", test_loss);
	fmt::println("The test accuracy of the model is: {}%",
			accuracy / test_dataset_size * 10);

	test_loss_set.push_back(test_loss);
	test_accuracy.push_back(accuracy / test_dataset_size * 10);

}

template<typename T>
TrainerOnMPS<T>::TrainerOnMPS(
		std::unique_ptr<T> mod, int64_t epochs, double lr, double mtum)
	: model(std::move(mod)),
	device(torch::kMPS),
	kNumsofEpochs(epochs),
	learning_rate(lr),
	momentum(mtum) {
	this->sgd_optimizer = std::make_unique<torch::optim::SGD>(
				model->parameters(),
				torch::optim::SGDOptions(learning_rate).momentum(mtum)
			);
	model->to(device);
	data_path = "../data/FashionMNIST/raw";

	fmt::println("Now we're training on model with MPS backends:");
	std::cout << *model << std::endl;
}

template<typename T>
TrainerOnMPS<T>::~TrainerOnMPS() {
	
}

template<typename T>
void TrainerOnMPS<T>::train_process() {
	auto train_dataset = torch::data::datasets::MNIST(data_path)
		.map(torch::data::transforms::Stack<>());

	const size_t train_data_size = train_dataset.size().value();

	auto train_loader =
		torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
				std::move(train_dataset), 64);

	std::vector<double> accuracy_set(20);
	const double full_size = 60000.0;

	for (int64_t epoch = 0; epoch < kNumsofEpochs; epoch++) {
		model->train();
		size_t batch_idx = 0;

		for (const auto& batch: *train_loader) {
			auto data = batch.data.to(device);
			auto target = batch.target.to(device);
			sgd_optimizer->zero_grad();

			auto output_t = model->forward(data);
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
		this->test_process();
	}
	for (int64_t epoch = 0; epoch < kNumsofEpochs; epoch++) {
		fmt::println("-------------------------------------------");
		fmt::println("The total train accuracy of epoch{} is: {}%",
				epoch,
				accuracy_set[epoch] / full_size * 100);
		fmt::println("The test accuracy of epoch{} is: {}%",
				epoch,
				test_accuracys[epoch]);
		fmt::println("The test loss of epoch{} is: {}",
				epoch,
				test_losses[epoch]);
	}	
}

template<typename T>
void TrainerOnMPS<T>::test_process() {
	auto test_dataset = torch::data::datasets::MNIST(
		data_path, torch::data::datasets::MNIST::Mode::kTest)
		.map(torch::data::transforms::Stack<>());

	auto test_loader =
		torch::data::make_data_loader(
				std::move(test_dataset), 1000);

	const int64_t test_dataset_size = 1000;
	double test_loss = 0.0;
	double accuracy = 0.0;

	for (const auto& batch: *test_loader) {
		auto data = batch.data.to(device);
		auto target = batch.target.to(device);
		auto output_t = model->forward(data);
		test_loss += torch::nll_loss(
				output_t,
				target,
				{},
				torch::Reduction::Sum
				).template item<double>();
		auto pred = output_t.argmax(1);
		accuracy += pred.eq(target).sum().template item<double>();
	}

	fmt::println("--------------------------------------");
	fmt::println("The test loss of the model is: {}", test_loss);
	fmt::println("The test accuracy of the model is: {}%",
			accuracy / test_dataset_size * 10);

	test_losses.push_back(test_loss);
	test_accuracys.push_back(accuracy / test_dataset_size * 10);
}

template class torch::TrainerOnMPS<LinDropoutStack>;

}
