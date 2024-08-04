#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class Value;
using ValuePtr = std::shared_ptr<Value>;

class Value {

public:
  // * Creates a new Value object with the given value
  static ValuePtr create(double valueIn) {
    return std::shared_ptr<Value>(new Value(valueIn));
  }

  void forwardPass() {

    for (auto &p : prev) {
      p->forwardPass();
    }

    switch (operation) {
    case ADD:
      value = prev[0]->value + prev[1]->value;
      break;
    case MULT:
      value = prev[0]->value * prev[1]->value;
      break;
    case SUB:
      value = prev[0]->value - prev[1]->value;
      break;
    case NONE:
      break;
    }
  }

  std::string getString() const;

  double getValue() const { return value; }

  void setValue(double valueIn) { value = valueIn; }

private:
  enum OPERATION { ADD, MULT, SUB, NONE };
  Value(double valueIn);
  Value(double valueIn, const std::vector<ValuePtr> &prevIn,
        OPERATION operationIn);

  std::string enumToString(OPERATION operationIn) const;

  // * Member Variables
  double value;
  std::vector<ValuePtr> prev = {};
  OPERATION operation = NONE;

  friend ValuePtr operator+(const ValuePtr &lhs, const ValuePtr &rhs);
};

inline ValuePtr operator+(const ValuePtr &lhs, const ValuePtr &rhs) {
  return std::shared_ptr<Value>(
      new Value(lhs->value + rhs->value, {lhs, rhs}, Value::ADD));
}

inline std::ostream &operator<<(std::ostream &os, ValuePtr other) {
  os << other.get()->getString() << std::endl;
  return os;
}
