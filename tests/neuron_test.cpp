#include "../core/Neuron.h"
#include <gtest/gtest.h>
#include <iostream>

// Example test
TEST(NeuronTest, ExampleTest) {
  Neuron n(2);
  std::vector<ValuePtr> inputs = {create_value(1.0), create_value(1.0)};
  auto output = n(inputs);
  std::cout << output->get_value();
  std::cout << "Test completed" << std::endl;
}

// Test that a Neuron with the correct number of inputs produces a non-negative
// output.
TEST(NeuronTest, ValidComputation) {
  // Create a neuron with 3 inputs.
  Neuron n(3);

  // Create three input values.
  std::vector<ValuePtr> inputs = {create_value(1.0), create_value(-1.0),
                                  create_value(0.5)};

  // Compute the neuron's output.
  ValuePtr output = n(inputs);

  // Check that the output is not null.
  ASSERT_NE(output, nullptr);

  // Retrieve the computed value.
  double out_value = output->get_value();

  // Since the neuron applies ReLU, the output should be non-negative.
  ASSERT_GE(out_value, 0.0)
      << "Output must be non-negative due to ReLU activation.";
}

// Test that calling the Neuron with an incorrect number of inputs throws an
// exception.
TEST(NeuronTest, InvalidInputSizeThrows) {
  // Create a neuron expecting 2 inputs.
  Neuron n(2);

  // Provide a wrong number of inputs (3 instead of 2).
  std::vector<ValuePtr> inputs = {create_value(1.0), create_value(2.0),
                                  create_value(3.0)};

  // Expect a runtime_error due to mismatched input size.
  EXPECT_THROW({ n(inputs); }, std::runtime_error);
}

// Test that a neuron constructed with zero inputs still produces valid output.
TEST(NeuronTest, ZeroInputs) {
  // Create a neuron with zero inputs.
  Neuron n(0);

  // Prepare an empty input vector.
  std::vector<ValuePtr> inputs;

  // Compute the neuron's output.
  ValuePtr output = n(inputs);

  // Check that the output is not null.
  ASSERT_NE(output, nullptr);

  // Retrieve the computed value.
  double out_value = output->get_value();

  // Even if the neuron has no weights, the output is the result of relu(bias),
  // so it should be non-negative.
  ASSERT_GE(out_value, 0.0)
      << "Output must be non-negative even for a neuron with zero inputs.";
}
