

#include "Value.h"
#include <cassert>
#include <memory>
#include <random>
#include <vector>

double generateRandomNumber(double minimum, double maximum) {
  // Seed with a real random value, if available
  std::random_device rd;

  // Initialize a random number generator
  std::mt19937 gen(rd());

  // Define the range and distribution
  std::uniform_real_distribution<> dis(minimum, maximum);

  // Generate and return a random number in the range [-1, 1]
  return dis(gen);
}

class Neuron {

public:
  Neuron(size_t w) {
    for (size_t i = 0; i < w; i++) {
      weights.push_back(Value::create(generateRandomNumber(-1.0, 1.0)));
    }
  }

  const std::vector<ValuePtr> &getWeights() { return weights; }

  ValuePtr call(std::vector<double> inputs) {
    assert(inputs.size() == weights.size());
    size_t size = inputs.size();

    auto activation = Value::create(0);

    for (size_t i = 0; i < size; i++) {
      activation = activation + (weights[i] * Value::create(inputs[i]));
    }

    return relu(activation);
  }

private:
  std::vector<ValuePtr> weights;
};