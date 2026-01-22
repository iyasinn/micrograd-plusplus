#include "Value.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

// * ------------- Member Functions ---------------
void Value::zero_all_gradients() {
  // TODO: Places to optimize -> Redundant zeros
  // TODO: What if they all pointed to some memory that was zeroed out
  _gradient = 0.0;
  for (ValuePtr &ptr : prev) {
    ptr->zero_all_gradients();
  }
}

void Value::set_gradient_to_one() { _gradient = 1.0; }

void Value::backpropagate() {
  zero_all_gradients();
  set_gradient_to_one();
  internal_backpropagate();
}

auto Value::get_value() const -> double { return _value; }

auto Value::get_gradient() const -> double { return _gradient; }

void Value::internal_backpropagate() {

  if (gradient_func) {
    gradient_func();
  }

  for (auto &p : prev) {
    p->internal_backpropagate();
  }
}

// * ------------- Operator Functions ---------------

auto operator*(ValuePtr left, ValuePtr right) -> ValuePtr {

  auto output = create_value(left->get_value() * right->get_value());
  output->prev.push_back(left);
  output->prev.push_back(right);

  output->gradient_func = [output]() {
    auto first = output->prev[0];
    auto second = output->prev[1];
    output->prev[0]->_gradient += (second->_value * output->_gradient);
    output->prev[1]->_gradient += (first->_value * output->_gradient);
  };

  return output;
}

auto operator-(ValuePtr value) -> ValuePtr {
  return value * create_value(-1.0);
}

auto operator+(ValuePtr left, ValuePtr right) -> ValuePtr {

  auto output = create_value(left->_value + right->_value);
  output->prev.push_back(left);
  output->prev.push_back(right);

  output->gradient_func = [output]() {
    output->prev[0]->_gradient += output->_gradient;
    output->prev[1]->_gradient += output->_gradient;
  };

  return output;
}

auto operator-(ValuePtr left, ValuePtr right) -> ValuePtr {
  return left + (-right);
}

auto inverse(ValuePtr value) -> ValuePtr {
  if (std::abs(value->_value) < 0.0001) {
    throw std::invalid_argument("Division by zero in inverse operation");
  }

  auto output = create_value(1.0 / value->_value);
  output->prev.push_back(value);

  output->gradient_func = [output]() {
    // d/dx(1/x) = -1/x^2
    // We can use output->_value = 1/x to simplify computation
    double grad = -output->_value * output->_value * output->_gradient;
    output->prev[0]->_gradient += grad;
  };

  return output;
}

auto relu(ValuePtr value) -> ValuePtr {
  auto output = create_value(value->_value > 0 ? value->_value : 0.0);
  output->prev.push_back(value);

  output->gradient_func = [output]() {
    output->prev[0]->_gradient +=
        (output->_value > 0 ? output->_gradient : 0.0);
  };

  return output;
}

auto Value::to_dot() const -> std::string {
  std::stringstream ss;
  ss << "digraph G {\n";
  ss << "  rankdir=LR;\n";
  ss << "  node [fontname=\"Arial\"];\n";
  ss << "  edge [fontname=\"Arial\"];\n";
  std::vector<const Value *> visited;
  build_dot(ss, visited);
  ss << "}\n";
  return ss.str();
}

void Value::build_dot(std::stringstream &ss,
                      std::vector<const Value *> &visited) const {
  // Skip if already visited
  if (std::find(visited.begin(), visited.end(), this) != visited.end()) {
    return;
  }
  visited.push_back(this);

  // Create a clean node ID
  std::stringstream node_id;
  node_id << "node_" << visited.size();

  // Create node label with value and gradient
  std::stringstream label;
  label << "Value: " << _value << "\\nGrad: " << _gradient;

  // Add node with styling
  ss << "  " << node_id.str() << " [label=\"" << label.str()
     << "\", shape=box, style=filled, fillcolor=lightblue];\n";

  // Add edges to previous nodes
  for (const auto &prev_value : prev) {
    std::stringstream prev_id;
    prev_id << "node_"
            << std::find(visited.begin(), visited.end(), prev_value.get()) -
                   visited.begin() + 1;
    ss << "  " << node_id.str() << " -> " << prev_id.str() << ";\n";
    prev_value->build_dot(ss, visited);
  }
}

void Value::visualize(const std::string &filename) const {
  // Generate DOT file
  std::ofstream dot_file(filename + ".dot");
  dot_file << to_dot();
  dot_file.close();

  // Generate PNG using dot command
  std::string command =
      "dot -Tpng " + filename + ".dot -o " + filename + ".png";
  system(command.c_str());
}
