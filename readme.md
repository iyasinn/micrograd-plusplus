# micrograd++

A lightweight C++ neural network library inspired by Andrej Karpathy's micrograd. This implementation provides a minimalistic yet powerful neural network framework with automatic differentiation capabilities.

## Features

- Automatic differentiation engine
- Neural network primitives:
  - Neurons with configurable inputs
  - Layers with multiple neurons
  - Multi-layer perceptron (MLP) architecture
- Modern C++ implementation (C++17)
- Header-only library
- Efficient memory management using smart pointers

## Getting Started

### Prerequisites

- C++17 compatible compiler
- CMake (for building)

### Building the Project

```bash
mkdir build
cd build
cmake ..
make
```

## Usage Example

```cpp
#include "core/Neuron.h"

// Create a simple neural network
size_t inputs = 3;
std::vector<size_t> layer_sizes = {4, 2, 1};  // 4 neurons in first layer, 2 in second, 1 in output
MultiLayerPerceptron mlp(inputs, layer_sizes);

// Create input values
std::vector<ValuePtr> input_values = {
    create_value(1.0),
    create_value(0.5),
    create_value(-1.0)
};

// Forward pass
auto output = mlp(input_values);
// Backpropagate
output[0]->backpropagate();
// Visualize 
output[0]->visualize("neural_network");
```

## Architecture

### Core Components

1. **Value Class**
   - Represents a value in the computation graph
   - Handles automatic differentiation

2. **Neuron Class**
   - Basic computational unit
   - Contains weights and bias
   - Applies ReLU activation function

3. **Layer Class**
   - Collection of neurons
   - Handles forward propagation through neurons

4. **MultiLayerPerceptron Class**
   - Complete neural network
   - Manages multiple layers
   - Provides forward propagation through the entire network

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd)
- Built with modern C++ features for efficiency and ease of use

## TODO

- [ ] Add backward propagation examples
- [ ] Implement additional activation functions
- [ ] Add serialization support
- [ ] Include more comprehensive testing
- [ ] Add benchmarking suite
- [ ] Improve documentation with more examples
- [ ] Optimize + Add more features to the NN
