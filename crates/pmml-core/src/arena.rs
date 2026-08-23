//! Arena allocator — per-`run()` bump allocation, reset after each scoring.
//! Mirrors ONNX Runtime `BFCArena` pattern but single-threaded via `thread_local!`.

use bumpalo::Bump;
use std::cell::RefCell;

thread_local! {
    static THREAD_ARENA: RefCell<Bump> = RefCell::new(Bump::new());
}

/// Execute `f` with a thread-local bump arena. Arena is reset after `f` returns.
pub fn with_arena<R>(f: impl FnOnce(&mut Bump) -> R) -> R {
    THREAD_ARENA.with(|cell| {
        let mut arena = cell.borrow_mut();
        arena.reset();
        let r = f(&mut arena);
        // reset again to free for next run (keeps capacity)
        arena.reset();
        r
    })
}

/// Scratch Vec that reuses arena allocation for `Value` buffers.
/// For v1 we use `Vec<Value>` on heap but retain this helper for future `BumpVec`.
pub fn alloc_vec<T>(arena: &Bump, len: usize, val: T) -> Vec<T>
where
    T: Clone,
{
    // Bump doesn't directly give Vec; we allocate slice then convert.
    // For now, simple Vec but documents intent for later `bumpalo::collections::Vec`.
    let _ = arena;
    vec![val; len]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_resets() {
        with_arena(|arena| {
            let _v = arena.alloc_str("hello");
            assert_eq!(_v, "hello");
        });
        // second run should not see previous allocation
        with_arena(|arena| {
            let _w = arena.alloc_str("world");
            assert_eq!(_w, "world");
        });
    }
}
