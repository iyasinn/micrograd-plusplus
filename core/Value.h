#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

class Value {
public:
  Value(double value_in) : _value(value_in) {}

  void zero_all_gradients();

  void set_gradient_to_one();

  void backpropagate();

  auto get_value() const -> double;

  auto get_gradient() const -> double;

private:
  double _value;
  double _gradient = 0;

  void internal_backpropagate();

  std::function<void()> gradient_func = nullptr;
  std::vector<Value *> prev;

  // * Friend Functions
  // friend auto operator*(ValuePtr, ValuePtr) -> ValuePtr;
  // friend auto operator+(ValuePtr, ValuePtr) -> ValuePtr;
  // friend auto inverse(ValuePtr) -> ValuePtr;
  // friend auto relu(ValuePtr) -> ValuePtr;
};

// auto operator*(ValuePtr left, ValuePtr right) -> ValuePtr;

// // * ------------- Friend Operations ---------------
//
// auto operator-(ValuePtr value) -> ValuePtr;
//
// auto operator+(ValuePtr left, ValuePtr right) -> ValuePtr;
//
// auto operator-(ValuePtr left, ValuePtr right) -> ValuePtr;
//
// auto inverse(ValuePtr value) -> ValuePtr;
//
// auto relu(ValuePtr value) -> ValuePtr;
