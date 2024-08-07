#include "Perceptron.h"
// #include "Value.h"
using namespace std;

ValuePtr Val(double value) { return Value::create(value); }

// We have inputs, their outputs, and what we wanted
double calculate_loss(vector<double> results, vector<double> targets) {
  double loss = 0;
  for (size_t i = 0; i < results.size(); i++) {
    loss += pow(results[i] - targets[i], 2);
  }
  return loss;
}

int main() {

  auto p = Perceptron(3, 2);
  std::cout << p.call({1, 2, 3})[0]->getValue();
  // for (size_t i = 0; i < p.getNeurons().size(); i++) {
  //   cout << i << ": " << p.getNeurons()[i].call({1, 2, 3})[0]->getValue() <<
  //   endl;
}

// const auto a = Val(100);
// const auto b = Val(25);

// auto c = a + b;

// size_t len = 3;

// Neuron n(len);
// vector<vector<double>> inputs = {
//     {2, 3, -1}, {3, -1, 0.5}, {0.5, 1, 1}, {1, 1, -1}};

// vector<double> targets = {1, -1, -1, 1};

// vector<double> results = {};

// for (size_t i = 0; i < inputs.size(); i++) {
//   ValuePtr output = n.call(inputs[i]);
//   output->startBackpropagation();
//   results.push_back(output->getValue());
// }

// cout << calculate_loss(results, targets) << endl;

// for (size_t i = 0; i < n.getWeights().size(); i++) {
//   cout << i << ": " << n.getWeights()[i]->getValue() << " "
//        << n.getWeights()[i]->getGradient() << endl;
// }

// cout << "\n\n\n" << output->getValue() << "\n\n\n";

// output->startBackpropagation();

// cout << output->getValue() << endl;

// for (size_t i = 0; i < n.getWeights().size(); i++) {
//   cout << n.getWeights()[i]->getValue() << " "
//        << n.getWeights()[i]->getGradient() << endl;
// }

// cout << output;
}