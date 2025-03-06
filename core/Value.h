#include <functional>
#include <memory>
#include <vector>
#include <string>
#include <sstream>

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

  auto get_value() const -> double;

  auto get_gradient() const -> double;

  // Visualization methods
  auto to_dot() const -> std::string;
  void visualize(const std::string& filename = "computation_graph") const;

private:
  void internal_backpropagate();
  void build_dot(std::stringstream& ss, std::vector<const Value*>& visited) const;

  double _value;
  double _gradient = 0;
  std::function<void()> gradient_func = nullptr;
  std::vector<ValuePtr> prev;

  // * Friend Functions
  friend auto operator*(ValuePtr, ValuePtr) -> ValuePtr;
  friend auto operator+(ValuePtr, ValuePtr) -> ValuePtr;
  friend auto inverse(ValuePtr) -> ValuePtr;
  friend auto relu(ValuePtr) -> ValuePtr;
};

auto operator*(ValuePtr left, ValuePtr right) -> ValuePtr;

// * ------------- Friend Operations ---------------

auto operator-(ValuePtr value) -> ValuePtr;

auto operator+(ValuePtr left, ValuePtr right) -> ValuePtr;

auto operator-(ValuePtr left, ValuePtr right) -> ValuePtr;

auto inverse(ValuePtr value) -> ValuePtr;

auto relu(ValuePtr value) -> ValuePtr;
