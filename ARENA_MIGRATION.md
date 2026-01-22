# Arena Migration Guide

Migrate from `shared_ptr<Value>` to arena-allocated raw pointers.

## Why

- No reference counting overhead
- Better cache locality (values are contiguous in memory)
- Fast bulk deallocation (reset arena instead of freeing each value)
- Simpler code

## The Pattern: Two Arenas

```cpp
MemoryArena model_arena(MB(1));   // Weights live here (persist for training)
MemoryArena scratch_arena(MB(4)); // Intermediates live here (reset each iteration)
```

## Step 1: Update Value.h

**Before:**
```cpp
using ValuePtr = std::shared_ptr<Value>;

class Value {
  double _value;
  double _gradient = 0;
  std::function<void()> gradient_func = nullptr;
  std::vector<ValuePtr> prev;
};
```

**After:**
```cpp
class Value {
public:
  double value;
  double gradient = 0;

  Value* prev[2] = {nullptr, nullptr};  // Fixed size, no heap alloc
  u8 prev_count = 0;

  void (*gradient_func)(Value* self) = nullptr;  // Function pointer
};

// Free functions that take arena
Value* create_value(MemoryArena* arena, double val);
Value* add(MemoryArena* arena, Value* a, Value* b);
Value* mul(MemoryArena* arena, Value* a, Value* b);
Value* relu(MemoryArena* arena, Value* v);
```

## Step 2: Update Value.cpp

**Create:**
```cpp
Value* create_value(MemoryArena* arena, double val) {
  Value* v = arena->push_struct<Value>();
  v->value = val;
  v->gradient = 0;
  v->prev_count = 0;
  v->gradient_func = nullptr;
  return v;
}
```

**Add:**
```cpp
Value* add(MemoryArena* arena, Value* a, Value* b) {
  Value* out = create_value(arena, a->value + b->value);
  out->prev[0] = a;
  out->prev[1] = b;
  out->prev_count = 2;
  out->gradient_func = [](Value* self) {
    self->prev[0]->gradient += self->gradient;
    self->prev[1]->gradient += self->gradient;
  };
  return out;
}
```

**Mul:**
```cpp
Value* mul(MemoryArena* arena, Value* a, Value* b) {
  Value* out = create_value(arena, a->value * b->value);
  out->prev[0] = a;
  out->prev[1] = b;
  out->prev_count = 2;
  out->gradient_func = [](Value* self) {
    self->prev[0]->gradient += self->prev[1]->value * self->gradient;
    self->prev[1]->gradient += self->prev[0]->value * self->gradient;
  };
  return out;
}
```

**ReLU:**
```cpp
Value* relu(MemoryArena* arena, Value* v) {
  Value* out = create_value(arena, v->value > 0 ? v->value : 0);
  out->prev[0] = v;
  out->prev_count = 1;
  out->gradient_func = [](Value* self) {
    self->prev[0]->gradient += (self->value > 0) ? self->gradient : 0;
  };
  return out;
}
```

**Backprop:**
```cpp
void zero_gradients(Value* v) {
  v->gradient = 0;
  for (u8 i = 0; i < v->prev_count; i++) {
    zero_gradients(v->prev[i]);
  }
}

void backward_internal(Value* v) {
  if (v->gradient_func) {
    v->gradient_func(v);
  }
  for (u8 i = 0; i < v->prev_count; i++) {
    backward_internal(v->prev[i]);
  }
}

void backward(Value* v) {
  zero_gradients(v);
  v->gradient = 1.0;
  backward_internal(v);
}
```

## Step 3: Update Neuron.h

```cpp
class Neuron {
public:
  Neuron(MemoryArena* arena, size_t num_inputs) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    _num_weights = num_inputs;
    _weights = arena->push_array<Value*>(num_inputs);
    for (size_t i = 0; i < num_inputs; i++) {
      _weights[i] = create_value(arena, dis(gen));
    }
    _bias = create_value(arena, dis(gen));
  }

  Value* forward(MemoryArena* scratch, Value** inputs, size_t num_inputs) {
    Value* sum = _bias;
    for (size_t i = 0; i < num_inputs; i++) {
      Value* wx = mul(scratch, _weights[i], inputs[i]);
      sum = add(scratch, sum, wx);
    }
    return relu(scratch, sum);
  }

private:
  Value** _weights;
  Value* _bias;
  size_t _num_weights;
};
```

## Step 4: Training Loop

```cpp
int main() {
  MemoryArena model_arena(MB(1));
  MemoryArena scratch_arena(MB(4));

  // Create neuron (weights in model_arena)
  Neuron neuron(&model_arena, 3);

  for (int epoch = 0; epoch < 1000; epoch++) {
    // Mark position before forward pass
    Arena checkpoint = scratch_arena.mark();

    // Create inputs in scratch arena
    Value* inputs[3] = {
      create_value(&scratch_arena, 1.0),
      create_value(&scratch_arena, 2.0),
      create_value(&scratch_arena, 3.0)
    };

    // Forward
    Value* output = neuron.forward(&scratch_arena, inputs, 3);

    // Backward
    backward(output);

    // Update weights (simple SGD)
    // weights live in model_arena, so they persist
    double lr = 0.01;
    // neuron._weights[i]->value -= lr * neuron._weights[i]->gradient;

    // Reset scratch arena - frees all intermediates
    checkpoint.end();
  }

  return 0;
}
```

## Summary

| Before | After |
|--------|-------|
| `std::shared_ptr<Value>` | `Value*` |
| `std::vector<ValuePtr> prev` | `Value* prev[2]` |
| `std::function<void()>` | `void (*fn)(Value*)` |
| `create_value(x)` | `create_value(arena, x)` |
| `a + b` | `add(arena, a, b)` |
| automatic cleanup | `arena.clear()` or `checkpoint.end()` |
