#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class Value;
using ValuePtr = const std::shared_ptr<Value>;

class Value {

public:
  // * Creates a new Value object with the given value
  static ValuePtr create(double valueIn) {
    return std::shared_ptr<Value>(new Value(valueIn));
  }

  void forwardPass();

  std::string getString() const;

  double getValue() const { return value; }

  void setValue(double valueIn) { value = valueIn; }

  void zeroGradients();

  void setGradientToOne() { gradient = 1; }

  // * Does not zero the gradients
  void backpropagation();

  // * Does all important steps for backpropagation
  void startBackpropagation() {
    zeroGradients();
    setGradientToOne();
    backpropagation();
  }

private:
  enum OPERATION { ADD, MULT, NEG, EXP, NONE };

  Value(double valueIn);
  Value(double valueIn, const std::vector<ValuePtr> &prevIn,
        OPERATION operationIn);

  std::string enumToString(OPERATION operationIn) const;

  // * Member Variables
  double value;
  double gradient;
  std::vector<ValuePtr> prev = {};
  OPERATION operation = NONE;

  friend ValuePtr operator+(const ValuePtr &lhs, const ValuePtr &rhs);
  friend ValuePtr operator*(const ValuePtr &lhs, const ValuePtr &rhs);
  friend ValuePtr operator-(const ValuePtr &x);
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
