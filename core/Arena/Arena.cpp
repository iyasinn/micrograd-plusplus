#include "Arena.hpp"
#include <cstdlib>
#include <cstring>

MemoryArena::MemoryArena(u64 size) {
  buffer = static_cast<u8 *>(std::malloc(size));
  capacity = size;
  pos = 0;
}

MemoryArena::~MemoryArena() { std::free(buffer); }

void *MemoryArena::push(u64 size, u64 align) {

  u64 aligned_pos = (pos + align - 1) & ~(align - 1);

  // Out of memory
  if (aligned_pos + size > capacity) {
    return nullptr;
  }

  void *ptr = buffer + aligned_pos;
  pos = aligned_pos + size;
  return ptr;
}

void *MemoryArena::push_zero(u64 size, u64 align) {
  auto memory = push(size, align);
  if (memory) {
    std::memset(memory, 0, size);
  }
  return memory;
}

void MemoryArena::pop(u64 size) {
  if (size > pos) {
    pos = 0;
  }
  pos -= size;
}
void MemoryArena::clear() { pos = 0; }

// Position
u64 MemoryArena::get_pos() const { return pos; }

void MemoryArena::set_pos(u64 new_pos) {
  if (pos <= capacity) {
    pos = new_pos;
  }
}

Arena MemoryArena::mark() { return Arena{this, pos}; }

void MemoryArena::restore(Arena arena) { pos = arena.pos; }

void Arena::end() {
  arena->restore(*this); // Restore the position we saved
}
