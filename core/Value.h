#include <functional>
#include <memory>
#include <vector>

class Value;
using ValuePtr = std::shared_ptr<Value>;

inline ValuePtr create_value(double value) {
  return std::make_shared<Value>(value);
}

class Value {
public:
  Value(double value_in) : _value(value_in) {}

  void zero_all_gradients();

  void set_gradient_to_one();

  void backpropagate();

  double get_value() const;

  double get_gradient() const;

private:
  void internal_backpropagate();

  double _value;
  double _gradient = 0;
  std::function<void()> gradient_func = nullptr;
  std::vector<ValuePtr> prev;

  // * Friend Functions
  friend ValuePtr operator*(ValuePtr, ValuePtr);
  friend ValuePtr operator+(ValuePtr, ValuePtr);
  friend ValuePtr inverse(ValuePtr);
  friend ValuePtr relu(ValuePtr);
};

ValuePtr operator*(ValuePtr left, ValuePtr right);

// * ------------- Friend Operations ---------------

ValuePtr operator-(ValuePtr value);

ValuePtr operator+(ValuePtr left, ValuePtr right);

ValuePtr operator-(ValuePtr left, ValuePtr right);

ValuePtr inverse(ValuePtr value);

ValuePtr relu(ValuePtr value);
