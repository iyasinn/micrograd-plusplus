#include "Value.h"

using namespace std;

int main() {

  auto a = Value::create(5);
  auto b = Value::create(10);

  auto c = a + a;

  cout << c;

  a->setValue(a->getValue() + 10);

  cout << c;

  c->forwardPass();

  cout << c;
}