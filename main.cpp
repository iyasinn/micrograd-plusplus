#include "core/Neuron.h"
#include <iostream>

int main() {

  std::vector<ValuePtr> input = {create_value(1.0), create_value(2.0),
                                 create_value(3.0)};


    MultiLayerPerceptron cool(3, {
        3, 2, 1
    });


    auto output = cool(input);
    std::cout << "size is " << output.size() << std::endl;

    std::cout << output[0]->get_value();

}
