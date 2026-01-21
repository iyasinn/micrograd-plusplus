#include <gtest/gtest.h>
#include "../core/Arena/Arena.hpp"
#include <cstdint>

// Test basic allocation
TEST(ArenaTest, BasicAllocation) {
    MemoryArena arena(KB(1));
    void* ptr = arena.push(64);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(arena.get_pos(), 64);
}

// Test zero-initialized allocation
TEST(ArenaTest, ZeroInitializedAllocation) {
    MemoryArena arena(KB(1));
    u8* ptr = static_cast<u8*>(arena.push_zero(64));
    ASSERT_NE(ptr, nullptr);

    for (int i = 0; i < 64; i++) {
        EXPECT_EQ(ptr[i], 0) << "Byte at index " << i << " is not zero";
    }
}

// Test alignment correctness
TEST(ArenaTest, Alignment) {
    MemoryArena arena(KB(1));

    // Allocate 1 byte to offset the position
    arena.push(1);

    // Allocate with 16-byte alignment
    void* aligned_ptr = arena.push(32, 16);
    ASSERT_NE(aligned_ptr, nullptr);

    uintptr_t addr = reinterpret_cast<uintptr_t>(aligned_ptr);
    EXPECT_EQ(addr % 16, 0) << "Pointer is not 16-byte aligned";
}

TEST(ArenaTest, VariousAlignments) {
    MemoryArena arena(KB(1));

    // Test 8-byte alignment
    arena.push(1);
    void* ptr8 = arena.push(8, 8);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr8) % 8, 0);

    // Test 32-byte alignment
    arena.push(1);
    void* ptr32 = arena.push(8, 32);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr32) % 32, 0);

    // Test 64-byte alignment
    arena.push(1);
    void* ptr64 = arena.push(8, 64);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr64) % 64, 0);
}

// Test push_array template
TEST(ArenaTest, PushArray) {
    MemoryArena arena(KB(1));

    int* arr = arena.push_array<int>(10);
    ASSERT_NE(arr, nullptr);

    // Write and verify values
    for (int i = 0; i < 10; i++) {
        arr[i] = i * 2;
    }
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(arr[i], i * 2);
    }
}

TEST(ArenaTest, PushArrayZero) {
    MemoryArena arena(KB(1));

    int* arr = arena.push_array_zero<int>(10);
    ASSERT_NE(arr, nullptr);

    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(arr[i], 0);
    }
}

// Test push_struct template
TEST(ArenaTest, PushStruct) {
    struct TestStruct {
        int x;
        double y;
        char z;
    };

    MemoryArena arena(KB(1));
    TestStruct* s = arena.push_struct<TestStruct>();
    ASSERT_NE(s, nullptr);

    s->x = 42;
    s->y = 3.14;
    s->z = 'A';

    EXPECT_EQ(s->x, 42);
    EXPECT_DOUBLE_EQ(s->y, 3.14);
    EXPECT_EQ(s->z, 'A');
}

TEST(ArenaTest, PushStructZero) {
    struct TestStruct {
        int x;
        double y;
        char z;
    };

    MemoryArena arena(KB(1));
    TestStruct* s = arena.push_struct_zero<TestStruct>();
    ASSERT_NE(s, nullptr);

    EXPECT_EQ(s->x, 0);
    EXPECT_DOUBLE_EQ(s->y, 0.0);
    EXPECT_EQ(s->z, 0);
}

// Test mark/restore (scoped allocation)
TEST(ArenaTest, MarkRestore) {
    MemoryArena arena(KB(1));

    // Allocate some initial data
    arena.push(64);
    u64 initial_pos = arena.get_pos();

    // Mark current position
    Arena scope = arena.mark();

    // Allocate more data
    arena.push(128);
    EXPECT_EQ(arena.get_pos(), initial_pos + 128);

    // Restore to marked position
    arena.restore(scope);
    EXPECT_EQ(arena.get_pos(), initial_pos);
}

TEST(ArenaTest, MarkRestoreWithEnd) {
    MemoryArena arena(KB(1));

    arena.push(64);
    u64 initial_pos = arena.get_pos();

    Arena scope = arena.mark();
    arena.push(128);

    // Use Arena::end() method
    scope.end();
    EXPECT_EQ(arena.get_pos(), initial_pos);
}

TEST(ArenaTest, NestedScopes) {
    MemoryArena arena(KB(1));

    arena.push(32);
    u64 pos1 = arena.get_pos();
    Arena scope1 = arena.mark();

    arena.push(32);
    u64 pos2 = arena.get_pos();
    Arena scope2 = arena.mark();

    arena.push(32);
    EXPECT_EQ(arena.get_pos(), pos2 + 32);

    scope2.end();
    EXPECT_EQ(arena.get_pos(), pos2);

    scope1.end();
    EXPECT_EQ(arena.get_pos(), pos1);
}

// Test pop behavior
TEST(ArenaTest, Pop) {
    MemoryArena arena(KB(1));

    arena.push(64);
    arena.push(32);
    EXPECT_EQ(arena.get_pos(), 96);

    arena.pop(32);
    EXPECT_EQ(arena.get_pos(), 64);
}

TEST(ArenaTest, PopMoreThanAllocated) {
    MemoryArena arena(KB(1));

    arena.push(64);
    arena.pop(128);  // Pop more than allocated

    EXPECT_EQ(arena.get_pos(), 0);  // Should clamp to 0
}

// Test clear behavior
TEST(ArenaTest, Clear) {
    MemoryArena arena(KB(1));

    arena.push(256);
    arena.push(128);
    EXPECT_GT(arena.get_pos(), 0);

    arena.clear();
    EXPECT_EQ(arena.get_pos(), 0);
}

// Test out-of-memory handling
TEST(ArenaTest, OutOfMemory) {
    MemoryArena arena(64);  // Small arena

    void* ptr = arena.push(128);  // Request more than capacity
    EXPECT_EQ(ptr, nullptr);
}

TEST(ArenaTest, OutOfMemoryWithAlignment) {
    MemoryArena arena(64);

    arena.push(60);  // Fill most of the arena
    void* ptr = arena.push(8, 16);  // Would need alignment padding
    EXPECT_EQ(ptr, nullptr);
}

TEST(ArenaTest, ExactFit) {
    MemoryArena arena(64);

    void* ptr = arena.push(64);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(arena.get_pos(), 64);

    // Now arena is full
    void* ptr2 = arena.push(1);
    EXPECT_EQ(ptr2, nullptr);
}

// Test edge cases
TEST(ArenaTest, ZeroSizeAllocation) {
    MemoryArena arena(KB(1));

    void* ptr = arena.push(0);
    EXPECT_NE(ptr, nullptr);  // Zero-size allocation should succeed
    u64 pos_after_zero = arena.get_pos();

    void* ptr2 = arena.push(16);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_GT(arena.get_pos(), pos_after_zero);
}

TEST(ArenaTest, MultipleSequentialAllocations) {
    MemoryArena arena(KB(1));

    void* ptrs[10];
    for (int i = 0; i < 10; i++) {
        ptrs[i] = arena.push(32);
        ASSERT_NE(ptrs[i], nullptr) << "Allocation " << i << " failed";
    }

    // Verify no overlap (each pointer should be at least 32 bytes apart)
    for (int i = 1; i < 10; i++) {
        uintptr_t prev = reinterpret_cast<uintptr_t>(ptrs[i-1]);
        uintptr_t curr = reinterpret_cast<uintptr_t>(ptrs[i]);
        EXPECT_GE(curr - prev, 32) << "Allocations " << i-1 << " and " << i << " overlap";
    }
}

// Test position getters/setters
TEST(ArenaTest, GetSetPos) {
    MemoryArena arena(KB(1));

    arena.push(100);
    EXPECT_EQ(arena.get_pos(), 100);

    arena.set_pos(50);
    EXPECT_EQ(arena.get_pos(), 50);

    // set_pos should not exceed capacity
    arena.set_pos(KB(2));  // Beyond capacity
    EXPECT_EQ(arena.get_pos(), 50);  // Should remain unchanged
}

// Test helper size macros
TEST(ArenaTest, SizeMacros) {
    EXPECT_EQ(KB(1), 1024ULL);
    EXPECT_EQ(KB(2), 2048ULL);
    EXPECT_EQ(MB(1), 1048576ULL);
    EXPECT_EQ(GB(1), 1073741824ULL);
}

// Test that clear allows reuse
TEST(ArenaTest, ClearAndReuse) {
    MemoryArena arena(128);

    // Fill the arena
    void* ptr1 = arena.push(128);
    ASSERT_NE(ptr1, nullptr);
    EXPECT_EQ(arena.push(1), nullptr);  // Full

    // Clear and reuse
    arena.clear();
    void* ptr2 = arena.push(128);
    ASSERT_NE(ptr2, nullptr);

    // Should get same memory region
    EXPECT_EQ(ptr1, ptr2);
}
