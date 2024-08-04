#include "Value.h"
#include <cmath>
#include <memory>
#include <sstream>
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
  case POW:
    return "POW";
  case NONE:
    return "NONE";
  }
}

// std::string Value::getString(int indent) const {
//   std::string final = "";
//   final += "[VAL=" + std::to_string(value) + ":OP=" + enumToString(operation)
//   +
//            ":GRAD=" + std::to_string(gradient) + ":\n";

//   std::string indentString = std::string(size_t(indent), '\t');

//   final += "PREV= (";

//   for (ValuePtr &ptr : prev) {
//     final += indentString + ptr->getString(indent + 1) + "\n";
//   }

//   final += ")]";
//   return final;
// }

std::string Value::getString(size_t indent) const {
  std::ostringstream final;
  std::string indentString(indent, '\t');

  final << indentString + "- [VAL=" << value
        << ":OP=" << enumToString(operation) << ":GRAD=" << gradient
        << ":PREV=(";

  if (!prev.empty()) {
    for (const ValuePtr &ptr : prev) {
      final << "\n" << ptr->getString(indent + 1);
    }
    final << "\n"
          << indentString; // Closing the indent before closing the parenthesis
  }

  final << ")]";
  return final.str();
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
  case POW:
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
  case POW:
    // TODO: Only works when power is constant
    prev[0]->gradient +=
        (prev[1]->value * std::pow(prev[0]->value, prev[1]->value - 1) *
         gradient);
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
