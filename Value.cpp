#include "Value.h"
#include <memory>

Value::Value(double valueIn, const std::vector<std::shared_ptr<Value>> &prevIn,
             OPERATION operationIn)
    : value(valueIn), prev(prevIn), operation(operationIn) {}

Value::Value(double valueIn) : value(valueIn) {}

std::string Value::enumToString(OPERATION operationIn) const {
  switch (operationIn) {
  case ADD:
    return "+";
  case MULT:
    return "*";
  case SUB:
    return "-";
  case NONE:
    return "NONE";
  }
}

std::string Value::getString() const {
  std::string final = "";
  final += "[VAL:" + std::to_string(value) + ":OP:" + enumToString(operation) +
           ":PREV: (";

  for (ValuePtr ptr : prev) {
    final += ptr->getString();
  }
  final += ")]";
  return final;
}
