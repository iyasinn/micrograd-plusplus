#include "Value.h"
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

// need to take an umber and get from -1 to 1
// -1
/*

    then map our number from 0 to 2


*/

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

  auto operator()(const std::vector<ValuePtr>& inputs) -> ValuePtr {

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