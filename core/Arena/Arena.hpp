#pragma once
#include "../Shared/types.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>

constexpr u64 KB(u64 x) { return x * 1024; }
constexpr u64 MB(u64 x) { return x * 1024 * 1024; }
constexpr u64 GB(u64 x) { return x * 1024 * 1024 * 1024; }

struct Arena;

struct MemoryArena {

  uint8_t *buffer;
  u64 capacity;
  u64 pos;

  MemoryArena(u64 size);
  ~MemoryArena();

  // Remove copying
  MemoryArena(const MemoryArena &) = delete;
  MemoryArena &operator=(const MemoryArena &) = delete;

  // Allocation
  void *push(u64 size, u64 align = 16);
  void *push_zero(u64 size, u64 align = 16);

  template <typename T> T *push_array(u64 count) {
    return static_cast<T *>(push(sizeof(T) * count, alignof(T)));
  }

  template <typename T> T *push_array_zero(u64 count) {
    return static_cast<T *>(push_zero(sizeof(T) * count, alignof(T)));
  }

  template <typename T> T *push_struct() { return push_array<T>(1); }

  template <typename T> T *push_struct_zero() { return push_array_zero<T>(1); }

  // Deallocation
  void pop(u64 size);
  void clear();

  // Position
  u64 get_pos() const;
  void set_pos(u64 new_pos);

  // Scoping
  Arena mark();
  void restore(Arena arena);
};

struct Arena {
  MemoryArena *arena;
  u64 pos;

  void end();
};
