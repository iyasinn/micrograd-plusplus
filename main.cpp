#include "Value.h"

using namespace std;

ValuePtr Val(double value) { return Value::create(value); }

int main() {

  const auto a = Val(100);
  const auto b = Val(25);

  auto c = a / b;

  c->startBackpropagation();

  cout << c;
}