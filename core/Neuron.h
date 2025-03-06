#include "Value.h"
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

class Neuron {
public:
  Neuron(size_t number_of_inputs) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(-1.0F, 1.0F);

    _bias = create_value(dis(gen));
    _weights.resize(number_of_inputs);
    for (size_t i = 0; i < number_of_inputs; i++) {
      _weights[i] = create_value(dis(gen));
    }
  }

  auto operator()(const std::vector<ValuePtr> &inputs) -> ValuePtr {

    if (inputs.size() != _weights.size()) {
      throw std::runtime_error("Invalid number of inputs");
    }

    ValuePtr activation = _bias;

    for (size_t i = 0; i < inputs.size(); i++) {
      activation = activation + (inputs[i] * _weights[i]);
    }

    ValuePtr output = relu(activation);
    return output;
  }

private:
  std::vector<ValuePtr> _weights;
  ValuePtr _bias;
};

class Layer {
public:
  Layer(size_t number_of_inputs, size_t number_of_outputs) {
    _neurons.resize(number_of_outputs, Neuron(number_of_inputs));
  }

  auto operator()(const std::vector<ValuePtr> &inputs)
      -> std::vector<ValuePtr> {
    try {

      std::vector<ValuePtr> outputs;
      outputs.resize(_neurons.size());

      for (size_t i = 0; i < _neurons.size(); i++) {
        outputs[i] = _neurons[i](inputs);
      }
      return outputs;

    } catch (std::runtime_error &e) {
      throw std::runtime_error(e.what());
    }
  }

private:
  std::vector<Neuron> _neurons;
};

class MultiLayerPerceptron {
public: 



private: 

};