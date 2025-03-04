#include <functional>
#include <memory>
#include <vector>

class Value;
using ValuePtr = std::shared_ptr<Value>;

inline ValuePtr create_value(double value) {
  return std::make_shared<Value>(value, 0.0);
}

class Value {
public:
  Value(double value_in, double gradient_in)
      : _value(value_in), _gradient(gradient_in) {}

  double _value;
  double _gradient = 0;
  // void (*gradient_func)() = nullptr;
  std::function<void()> gradient_func = nullptr; // Change this line
  std::vector<ValuePtr> prev;
};

inline ValuePtr operator-(ValuePtr value) {

  auto output = create_value(-1 * value->_value);
  output->prev.push_back(value);

  output->gradient_func = [output]() {
    output->prev[0]->_gradient += (-1 * output->_gradient);
  };

  return output;
}

inline ValuePtr operator+(ValuePtr left, ValuePtr right) {

  auto output = create_value(left->_value + right->_value);
  output->prev.push_back(left);
  output->prev.push_back(right);

  output->gradient_func = [output]() {
    output->prev[0]->_gradient += output->_gradient;
    output->prev[1]->_gradient += output->_gradient;
  };

  return output;
}

inline ValuePtr operator-(ValuePtr left, ValuePtr right) {
  return left + (-right);
}

inline ValuePtr operator*(ValuePtr left, ValuePtr right) {

  auto output = create_value(left->_value * right->_value);
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


inline ValuePtr inverse(ValuePtr value) {
    if (value->_value == 0.0) {
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

inline ValuePtr relu(ValuePtr value) {
    auto output = create_value(value->_value > 0 ? value->_value : 0.0);
    output->prev.push_back(value);
    
    output->gradient_func = [output]() {
        output->prev[0]->_gradient += (output->_value > 0 ? output->_gradient : 0.0);
    };
    
    return output;
}


