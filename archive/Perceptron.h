#include "Neuron.h"

class Perceptron {

public:
  // n neurons with w weights each
  Perceptron(size_t w, size_t n) {
    for (size_t i = 0; i < n; i++) {
      neurons.push_back(Neuron(w));
    }
  }

  std::vector<Neuron> getNeurons() { return neurons; }

  std::vector<ValuePtr> call(std::vector<double> inputs) {

    std::vector<ValuePtr> outputs;
    for (size_t i = 0; i < neurons.size(); i++) {
      outputs.push_back(neurons[i].call(inputs));
    }
    return outputs;
  }

private:
  std::vector<Neuron> neurons;
};