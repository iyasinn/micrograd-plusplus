#include "Value.h"

using namespace std;

int main() {

  auto a = Value::create(5);
  auto b = Value::create(10);

  auto c = -a;

  cout << c;
  //   cout << c;
}