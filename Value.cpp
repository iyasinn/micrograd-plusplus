#include "Value.h"
#include <cmath>
#include <memory>
#include <string>

Value::Value(double valueIn, const std::vector<ValuePtr> &prevIn,
             OPERATION operationIn)
    : value(valueIn), prev(prevIn), operation(operationIn) {}

Value::Value(double valueIn) : value(valueIn) {}

std::string Value::enumToString(OPERATION operationIn) const {
  switch (operationIn) {
  case ADD:
    return "ADD";
  case MULT:
    return "MULT";
  case NEG:
    return "NEG";
  case EXP:
    return "EXP";
  case NONE:
    return "NONE";
  }
}

std::string Value::getString() const {
  std::string final = "";
  final += "[VAL=" + std::to_string(value) + ":OP=" + enumToString(operation) +
           ":GRAD=" + std::to_string(gradient) + ":PREV= (";

  for (ValuePtr &ptr : prev) {
    final += ptr->getString();
  }
  final += ")]";
  return final;
}
void Value::forwardPass() {

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
  case NEG:
    value = -prev[0]->value;
    break;
  case EXP:
    value = std::pow(prev[0]->value, prev[1]->value);
  case NONE:
    break;
  }
}
void Value::backpropagation() {
  switch (operation) {
  case ADD:
    prev[0]->gradient += gradient;
    prev[1]->gradient += gradient;
    break;
  case MULT:
    prev[0]->gradient += prev[1]->value * gradient;
    prev[1]->gradient += prev[0]->value * gradient;
    break;
  case NEG:
    prev[0]->gradient += (-1 * gradient);
    break;
  case EXP:
    // TODO: Fix it
    prev[0]->gradient = 1;
  case NONE:
    break;
  }

  for (auto p : prev) {
    p->backpropagation();
  }
}
void Value::zeroGradients() {
  gradient = 0;
  for (auto p : prev) {
    p->zeroGradients();
  }
}
