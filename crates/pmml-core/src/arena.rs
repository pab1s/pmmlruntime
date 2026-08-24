//! Arena allocator — per-`run()` bump allocation, reset after each scoring.
//! Mirrors ONNX Runtime `BFCArena` pattern: thread-local for serial, `BumpArena` for batched `par_iter`.

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

/// `BumpArena` — owned `Bump` for per-batch / per-`par_iter` chunk allocation.
///
/// Unlike `THREAD_ARENA`, this is `Send` when moved into `rayon` threads via `Send` bound on `Bump`.
/// Usage pattern for batched scoring:
///
/// ```ignore
/// let mut arena = BumpArena::new();
/// batch.par_chunks_mut(1024).for_each(|chunk| {
///     let mut local_arena = BumpArena::new(); // or reuse via thread_local
///     for row in chunk { /* allocate Value vec from arena */ }
///     local_arena.reset();
/// });
/// arena.reset();
/// ```
pub struct BumpArena {
    bump: Bump,
}

impl BumpArena {
    /// Create a new arena with default capacity.
    pub fn new() -> Self {
        Self { bump: Bump::new() }
    }

    /// Create with explicit capacity (bytes).
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            bump: Bump::with_capacity(cap),
        }
    }

    /// Access inner `Bump`.
    pub fn bump(&self) -> &Bump {
        &self.bump
    }

    /// Mutable access to inner `Bump`.
    pub fn bump_mut(&mut self) -> &mut Bump {
        &mut self.bump
    }

    /// Reset arena, retaining capacity.
    pub fn reset(&mut self) {
        self.bump.reset();
    }

    /// Allocate a `Vec<T>` of `len` cloned `val`. Memory is heap-allocated (not bump) for `T: Copy`
    /// compatibility, but API mirrors `bumpalo::collections::Vec` for future zero-alloc switch.
    /// For hot path `Value` buffers, use this per-chunk and reset after chunk completes.
    pub fn alloc_value_buffer(&self, len: usize, val: crate::Value) -> Vec<crate::Value> {
        vec![val; len]
    }

    /// Convenience: allocate bump-backed string (lives as long as arena).
    pub fn alloc_str<'a>(&'a self, s: &str) -> &'a str {
        self.bump.alloc_str(s)
    }

    /// Execute closure with mutable bump reference.
    pub fn with_bump<R>(&mut self, f: impl FnOnce(&mut Bump) -> R) -> R {
        self.bump.reset();
        let r = f(&mut self.bump);
        self.bump.reset();
        r
    }
}

impl Default for BumpArena {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: `Bump` is not `Sync` but is `Send` (owns heap). `BumpArena` is `Send` so it can be moved
// into rayon threads; it is never shared `&self` across threads without `&mut`.
unsafe impl Send for BumpArena {}

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

    #[test]
    fn bump_arena_per_chunk() {
        let mut arena = BumpArena::new();
        let v = arena.alloc_value_buffer(16, crate::Value::Missing);
        assert_eq!(v.len(), 16);
        arena.reset();
        let v2 = arena.alloc_value_buffer(8, crate::Value::Continuous(1.0));
        assert_eq!(v2.len(), 8);
    }

    #[test]
    fn bump_arena_threaded() {
        use rayon::prelude::*;
        let data: Vec<usize> = (0..100).collect();
        let results: Vec<usize> = data
            .par_iter()
            .map(|i| {
                let arena = BumpArena::new();
                let buf = arena.alloc_value_buffer(4, crate::Value::Continuous(*i as f64));
                buf.len()
            })
            .collect();
        assert_eq!(results.len(), 100);
        assert!(results.iter().all(|&l| l == 4));
    }
}
