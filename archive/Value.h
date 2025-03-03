#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class Value;
using ValuePtr = const std::shared_ptr<Value>;

// Class that has member that pointers to a object
// Multiple things can be shared pointers to the same object in memory

// WHen all shared pointers die, then that object will aautomatically die

class Value {

public:
  // * Creates a new Value object with the given value
  static ValuePtr create(double valueIn) {
    return std::shared_ptr<Value>(new Value(valueIn));
  }

  // void forwardPass();

  std::string getString(size_t indent = 0) const;

  double getValue() const { return value; }

  void setValue(double valueIn) { value = valueIn; }

  double getGradient() const { return gradient; }

  void zeroGradients();

  void setGradientToOne() { gradient = 1; }

  void setGradient(double gradientIn) { gradient = gradientIn; }

  // * Does not zero the gradients
  void backpropagation();

  // * Does all important steps for backpropagation
  void startBackpropagation() {
    zeroGradients();
    setGradientToOne();
    backpropagation();
  }

  enum OPERATION { ADD, MULT, NEG, POW, RELU, NONE };

  Value(double valueIn) : value(valueIn), gradient(0), operation(NONE) {}
  Value(double valueIn, const std::vector<ValuePtr> &prevIn, OPERATION operationIn)
      : value(valueIn), gradient(0), prev(prevIn), operation(operationIn) {}

private:
  std::string enumToString(OPERATION operationIn) const;

  // * Member Variables
  double value;
  double gradient; //  dL / dV
  // Honeslty this should be a tupel of max size 2 -> vector of capacity 2
  std::vector<ValuePtr> prev = {};
  OPERATION operation = NONE;

  friend ValuePtr operator+(const ValuePtr &lhs, const ValuePtr &rhs);
  friend ValuePtr operator*(const ValuePtr &lhs, const ValuePtr &rhs);
  friend ValuePtr operator-(const ValuePtr &x);
  friend ValuePtr operator^(ValuePtr lhs, ValuePtr rhs);
  friend ValuePtr relu(const ValuePtr &x);
}; // * class Value

// ! ----------------- OPERATOR OVERLOADS -----------------

inline ValuePtr operator+(const ValuePtr &lhs, const ValuePtr &rhs) {
  return std::shared_ptr<Value>(
      new Value(lhs->value + rhs->value, {lhs, rhs}, Value::ADD));
}

inline ValuePtr operator*(const ValuePtr &lhs, const ValuePtr &rhs) {
  return std::shared_ptr<Value>(
      new Value(lhs->value * rhs->value, {lhs, rhs}, Value::MULT));
}

inline ValuePtr operator^(ValuePtr lhs, ValuePtr rhs) {
  return std::shared_ptr<Value>(
      new Value(std::pow(lhs->value, rhs->value), std::vector<ValuePtr>{lhs, rhs}, Value::POW));
}

inline ValuePtr operator/(const ValuePtr &lhs, const ValuePtr &rhs) {
  return lhs * (rhs ^ (Value::create(-1)));
}

inline ValuePtr operator-(const ValuePtr &x) {
  return std::shared_ptr<Value>(new Value(x->value * -1, {x}, Value::NEG));
}

inline ValuePtr operator-(const ValuePtr &lhs, const ValuePtr &rhs) {
  return lhs + (-rhs);
}

inline std::ostream &operator<<(std::ostream &os, ValuePtr other) {
  os << other.get()->getString() << std::endl;
  return os;
}

inline ValuePtr relu(const ValuePtr &x) {
  return std::shared_ptr<Value>(
      new Value(fmax(-0.00001, x->value), {x}, Value::RELU));
}
