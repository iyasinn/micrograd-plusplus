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
    EXPECT_DOUBLE_EQ(v->get_value(), 5.0);
    EXPECT_DOUBLE_EQ(v->get_gradient(), 0.0);
}

TEST(ValueTest, Addition) {
    auto a = create_value(2.0);
    auto b = create_value(3.0);
    auto c = a + b;
    EXPECT_DOUBLE_EQ(c->get_value(), 5.0);
}

TEST(ValueTest, Negation) {
    auto a = create_value(2.0);
    auto b = -a;
    EXPECT_DOUBLE_EQ(b->get_value(), -2.0);
}

TEST(ValueTest, Subtraction) {
    auto a = create_value(5.0);
    auto b = create_value(3.0);
    auto c = a - b;
    EXPECT_DOUBLE_EQ(c->get_value(), 2.0);
}

TEST(ValueTest, Multiplication) {
    auto a = create_value(4.0);
    auto b = create_value(2.0);
    auto c = a * b;
    EXPECT_DOUBLE_EQ(c->get_value(), 8.0);
}

TEST(ValueTest, Inverse) {
    auto a = create_value(2.0);
    auto b = inverse(a);
    EXPECT_DOUBLE_EQ(b->get_value(), 0.5);
}

TEST(ValueTest, InverseDivideByZero) {
    auto a = create_value(0.0);
    EXPECT_THROW(inverse(a), std::invalid_argument);
}

TEST(ValueTest, Relu) {
    auto pos = create_value(2.0);
    auto neg = create_value(-2.0);
    auto zero = create_value(0.0);
    
    EXPECT_DOUBLE_EQ(relu(pos)->get_value(), 2.0);
    EXPECT_DOUBLE_EQ(relu(neg)->get_value(), 0.0);
    EXPECT_DOUBLE_EQ(relu(zero)->get_value(), 0.0);
}

TEST(ValueTest, GradientComputation) {
    // Test for multiplication gradient
    auto a = create_value(2.0);
    auto b = create_value(3.0);
    auto c = a * b;

    c->backpropagate();
    
    EXPECT_DOUBLE_EQ(a->get_gradient(), 3.0);  // db/da = b = 3
    EXPECT_DOUBLE_EQ(b->get_gradient(), 2.0);  // db/db = a = 2
}

TEST(ValueTest, ReluGradient) {
    auto a = create_value(2.0);
    auto b = relu(a);
    b->backpropagate();
    EXPECT_DOUBLE_EQ(a->get_gradient(), 1.0);  // Positive input

    auto c = create_value(-2.0);
    auto d = relu(c);

    d->backpropagate();
    EXPECT_DOUBLE_EQ(c->get_gradient(), 0.0);  // Negative input
}

TEST(ValueTest, SelfMultiplication) {
    auto a = create_value(3.0);
    auto b = a * a;  // b = a^2
    EXPECT_DOUBLE_EQ(b->get_value(), 9.0);  // 3^2 = 9
    
    b->backpropagate();
    EXPECT_DOUBLE_EQ(a->get_gradient(), 6.0);  // d(a^2)/da = 2a = 2*3 = 6
}

TEST(ValueTest, SelfAddition) {
    auto a = create_value(3.0);
    auto b = a + a;  // b = 2a
    EXPECT_DOUBLE_EQ(b->get_value(), 6.0);  // 3 + 3 = 6
    
    b->backpropagate();
    EXPECT_DOUBLE_EQ(a->get_gradient(), 2.0);  // d(a+a)/da = 2
}

TEST(ValueTest, ChainedOperations) {
    auto a = create_value(2.0);
    auto b = a * a;  // b = a^2
    auto c = b * a;  // c = a^3
    EXPECT_DOUBLE_EQ(c->get_value(), 8.0);  // 2^3 = 8
    
    c->backpropagate();
    EXPECT_DOUBLE_EQ(a->get_gradient(), 12.0);  // d(a^3)/da = 3a^2 = 3*4 = 12
}

TEST(ValueTest, SelfSubtraction) {
    auto a = create_value(3.0);
    auto b = a - a;  // b = 0
    EXPECT_DOUBLE_EQ(b->get_value(), 0.0);
    
    b->backpropagate();
    EXPECT_DOUBLE_EQ(a->get_gradient(), 0.0);  // d(a-a)/da = 0
}

TEST(ValueTest, ComplexExpression) {
    // Testing (a * a + a) * a
    auto a = create_value(2.0);
    auto b = a * a;      // 4
    auto c = b + a;      // 6
    auto d = c * a;      // 12

    EXPECT_DOUBLE_EQ(d->get_value(), 12.0);
    
    d->backpropagate();
    EXPECT_DOUBLE_EQ(a->get_gradient(), 16.0);  // d/da((a^2 + a)*a) = 3a^2 + a = 12 + 2 = 14
}