#include "Neuron.h"
// #include "Value.h"
using namespace std;

ValuePtr Val(double value) { return Value::create(value); }

int main() {

  const auto a = Val(100);
  const auto b = Val(25);

  size_t len = 3;

  Neuron n(len);

  vector<double> inputs(len, 1.0);

  ValuePtr output = n.call(inputs);

  cout << output;
}