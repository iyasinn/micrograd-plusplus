#include "Value.h"
#include <stdexcept>

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


double Value::get_value() const { return _value; }


double Value::get_gradient() const { return _gradient; }


void Value::internal_backpropagate() {

  if (gradient_func) {
    gradient_func();
  }

  for (auto &p : prev) {
    p->internal_backpropagate();
  }
}


// * ------------- Operator Functions ---------------

ValuePtr operator*(ValuePtr left, ValuePtr right) {

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


ValuePtr operator-(ValuePtr value) { return value * create_value(-1.0); }


ValuePtr operator+(ValuePtr left, ValuePtr right) {

  auto output = create_value(left->_value + right->_value);
  output->prev.push_back(left);
  output->prev.push_back(right);

  output->gradient_func = [output]() {
    output->prev[0]->_gradient += output->_gradient;
    output->prev[1]->_gradient += output->_gradient;
  };

  return output;
}


ValuePtr operator-(ValuePtr left, ValuePtr right) {
  return left + (-right);
}


ValuePtr inverse(ValuePtr value) {
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


ValuePtr relu(ValuePtr value) {
  auto output = create_value(value->_value > 0 ? value->_value : 0.0);
  output->prev.push_back(value);

  output->gradient_func = [output]() {
    output->prev[0]->_gradient +=
        (output->_value > 0 ? output->_gradient : 0.0);
  };

  return output;
}
