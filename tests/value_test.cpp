#include <gtest/gtest.h>
#include "../core/Value.h"
#include <iostream> 

// Example test
TEST(ValueTest, ExampleTest) {
    // EXPECT_TRUE(true);  // A simple test to verify setup
    std::cout << "hello my boi";
}

TEST(ValueTest, Construction) {
    auto v = create_value(5.0);
    EXPECT_DOUBLE_EQ(v->_value, 5.0);
    EXPECT_DOUBLE_EQ(v->_gradient, 0.0);
}

TEST(ValueTest, Addition) {
    auto a = create_value(2.0);
    auto b = create_value(3.0);
    auto c = a + b;
    EXPECT_DOUBLE_EQ(c->_value, 5.0);
}

TEST(ValueTest, Negation) {
    auto a = create_value(2.0);
    auto b = -a;
    EXPECT_DOUBLE_EQ(b->_value, -2.0);
}

TEST(ValueTest, Subtraction) {
    auto a = create_value(5.0);
    auto b = create_value(3.0);
    auto c = a - b;
    EXPECT_DOUBLE_EQ(c->_value, 2.0);
}

TEST(ValueTest, Multiplication) {
    auto a = create_value(4.0);
    auto b = create_value(2.0);
    auto c = a * b;
    EXPECT_DOUBLE_EQ(c->_value, 8.0);
}

TEST(ValueTest, Inverse) {
    auto a = create_value(2.0);
    auto b = inverse(a);
    EXPECT_DOUBLE_EQ(b->_value, 0.5);
}

TEST(ValueTest, InverseDivideByZero) {
    auto a = create_value(0.0);
    EXPECT_THROW(inverse(a), std::invalid_argument);
}

TEST(ValueTest, Relu) {
    auto pos = create_value(2.0);
    auto neg = create_value(-2.0);
    auto zero = create_value(0.0);
    
    EXPECT_DOUBLE_EQ(relu(pos)->_value, 2.0);
    EXPECT_DOUBLE_EQ(relu(neg)->_value, 0.0);
    EXPECT_DOUBLE_EQ(relu(zero)->_value, 0.0);
}

TEST(ValueTest, GradientComputation) {
    // Test for multiplication gradient
    auto a = create_value(2.0);
    auto b = create_value(3.0);
    auto c = a * b;
    c->_gradient = 1.0;
    c->gradient_func();
    
    EXPECT_DOUBLE_EQ(a->_gradient, 3.0);  // db/da = b = 3
    EXPECT_DOUBLE_EQ(b->_gradient, 2.0);  // db/db = a = 2
}

TEST(ValueTest, ReluGradient) {
    auto a = create_value(2.0);
    auto b = relu(a);
    b->_gradient = 1.0;
    b->gradient_func();
    EXPECT_DOUBLE_EQ(a->_gradient, 1.0);  // Positive input

    auto c = create_value(-2.0);
    auto d = relu(c);
    d->_gradient = 1.0;
    d->gradient_func();
    EXPECT_DOUBLE_EQ(c->_gradient, 0.0);  // Negative input
}