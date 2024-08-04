#include "Value.h"

using namespace std;

ValuePtr Constant(double value) { return Value::create(value); }

int main() {

  const auto a = Constant(10);
  const auto b = Constant(20);

  auto c = a + b;

  cout << c;

  c->startBackpropagation();

  cout << c;
}