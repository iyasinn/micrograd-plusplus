#include "Neuron.h"
// #include "Value.h"
using namespace std;

ValuePtr Val(double value) { return Value::create(value); }

int main() {

  const auto a = Val(100);
  const auto b = Val(25);

  size_t len = 3;

  Neuron n(len);
  vector<double> inputs = {3, 5, 7};
  ValuePtr output = n.call(inputs);

  for (size_t i = 0; i < n.getWeights().size(); i++) {
    cout << i << ": " << n.getWeights()[i]->getValue() << " "
         << n.getWeights()[i]->getGradient() << endl;
  }

  cout << "\n\n\n" << output->getValue() << "\n\n\n";

  output->startBackpropagation();

  for (size_t i = 0; i < n.getWeights().size(); i++) {
    cout << n.getWeights()[i]->getValue() << " "
         << n.getWeights()[i]->getGradient() << endl;
  }

  cout << output;
}