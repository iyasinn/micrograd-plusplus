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

TEST(ValueTest, SelfMultiplication) {
    auto a = create_value(3.0);
    auto b = a * a;  // b = a^2
    EXPECT_DOUBLE_EQ(b->_value, 9.0);  // 3^2 = 9
    
    b->_gradient = 1.0;
    b->gradient_func();
    EXPECT_DOUBLE_EQ(a->_gradient, 6.0);  // d(a^2)/da = 2a = 2*3 = 6
}

TEST(ValueTest, SelfAddition) {
    auto a = create_value(3.0);
    auto b = a + a;  // b = 2a
    EXPECT_DOUBLE_EQ(b->_value, 6.0);  // 3 + 3 = 6
    
    b->_gradient = 1.0;
    b->gradient_func();
    EXPECT_DOUBLE_EQ(a->_gradient, 2.0);  // d(a+a)/da = 2
}

TEST(ValueTest, ChainedOperations) {
    auto a = create_value(2.0);
    auto b = a * a;  // b = a^2
    auto c = b * a;  // c = a^3
    EXPECT_DOUBLE_EQ(c->_value, 8.0);  // 2^3 = 8
    
    c->_gradient = 1.0;
    c->gradient_func();
    b->gradient_func();
    EXPECT_DOUBLE_EQ(a->_gradient, 12.0);  // d(a^3)/da = 3a^2 = 3*4 = 12
}

TEST(ValueTest, SelfSubtraction) {
    auto a = create_value(3.0);
    auto b = a - a;  // b = 0
    EXPECT_DOUBLE_EQ(b->_value, 0.0);
    
    b->_gradient = 1.0;
    b->gradient_func();
    EXPECT_DOUBLE_EQ(a->_gradient, 0.0);  // d(a-a)/da = 0
}

TEST(ValueTest, ComplexExpression) {
    // Testing (a * a + a) * a
    auto a = create_value(2.0);
    auto b = a * a;      // 4
    auto c = b + a;      // 6
    auto d = c * a;      // 12
    EXPECT_DOUBLE_EQ(d->_value, 12.0);
    
    d->_gradient = 1.0;
    d->gradient_func();
    c->gradient_func();
    b->gradient_func();
    EXPECT_DOUBLE_EQ(a->_gradient, 10.0);  // d/da((a^2 + a)*a) = 3a^2 + a = 12 + 2 = 14
}