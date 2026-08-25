//! Arena allocator — per-`run()` bump allocation, reset after each scoring.
//!
//! Mirrors ONNX Runtime `BFCArena` pattern: a fast thread-local arena for serial
//! `Session::run` and an owned [`BumpArena`] for batched [`rayon`] `par_iter` shards.
//! The hot path for `<=64` fields avoids the arena entirely via a stack buffer
//! in `pmml-session`; this module is the overflow and string-interning path.
//!
//! # Mental model
//!
//! - `THREAD_ARENA` is `thread_local!` — one `Bump` per thread, reset before and after each `with_arena` call.
//! - [`BumpArena`] is an owned `Bump` that is `Send` (but not `Sync`) so it can be moved into rayon threads.
//! - Hot `Value` buffers are currently `Vec<Value>` on the heap; the API mirrors `bumpalo::collections::Vec`
//!   for a future zero-alloc switch without changing call sites.

use bumpalo::Bump;
use std::cell::RefCell;

thread_local! {
    static THREAD_ARENA: RefCell<Bump> = RefCell::new(Bump::new());
}

/// Execute `f` with a thread-local bump arena.
///
/// The arena is reset before `f` runs and again after it returns, retaining
/// capacity for the next call. Allocations via `arena.alloc_*` live only until
/// the next `with_arena` call on the same thread.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::arena::with_arena;
/// let s = with_arena(|arena| arena.alloc_str("hello").to_string());
/// assert_eq!(s, "hello");
/// ```
#[must_use]
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

/// Scratch `Vec` that conceptually reuses arena allocation for `Value` buffers.
///
/// For v1 this is a plain heap `Vec` — it documents intent for a future
/// `bumpalo::collections::Vec` switch without changing call sites.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::arena::{with_arena, alloc_vec};
/// with_arena(|arena| {
///     let v = alloc_vec(arena, 4, 0u32);
///     assert_eq!(v, vec![0, 0, 0, 0]);
/// });
/// ```
pub fn alloc_vec<T>(arena: &Bump, len: usize, val: T) -> Vec<T>
where
    T: Clone,
{
    // Bump doesn't directly give Vec; we allocate slice then convert.
    // For now, simple Vec but documents intent for later `bumpalo::collections::Vec`.
    let _ = arena;
    vec![val; len]
}

/// Owned bump arena for per-batch / per-`par_iter` chunk allocation.
///
/// Unlike the `thread_local!` `THREAD_ARENA`, this type owns its `Bump` and is
/// `Send` so it can be moved into [`rayon`] threads. It is **not** `Sync`; do not
/// share `&BumpArena` across threads without `&mut`.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::arena::BumpArena;
/// use rayon::prelude::*;
/// let results: Vec<usize> = (0..4).into_par_iter().map(|_| {
///     let arena = BumpArena::new();
///     arena.alloc_value_buffer(8, pmmlruntime::base::Value::Missing).len()
/// }).collect();
/// assert_eq!(results, vec![8, 8, 8, 8]);
/// ```
///
/// Usage pattern for batched scoring with reuse:
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
    /// Create a new arena with default capacity (1 KB initial, grows geometrically).
    #[must_use]
    pub fn new() -> Self {
        Self { bump: Bump::new() }
    }

    /// Create with explicit capacity in bytes. The bump will not reallocate
    /// until that many bytes are consumed.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            bump: Bump::with_capacity(cap),
        }
    }

    /// Access the inner [`Bump`] for direct `alloc_*` calls.
    #[must_use]
    pub fn bump(&self) -> &Bump {
        &self.bump
    }

    /// Mutable access to the inner [`Bump`].
    pub fn bump_mut(&mut self) -> &mut Bump {
        &mut self.bump
    }

    /// Reset the arena, retaining its heap capacity for reuse.
    ///
    /// Does not deallocate; the next allocation reuses the same memory.
    pub fn reset(&mut self) {
        self.bump.reset();
    }

    /// Allocate a `Vec<Value>` of `len` clones of `val`.
    ///
    /// Currently heap-allocated for `T: Copy` compatibility, but the API
    /// mirrors `bumpalo::collections::Vec` for a future zero-alloc switch.
    /// Call [`reset`](Self::reset) after the chunk completes.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::base::{Value, arena::BumpArena};
    /// let arena = BumpArena::new();
    /// let buf = arena.alloc_value_buffer(4, Value::Missing);
    /// assert_eq!(buf.len(), 4);
    /// ```
    pub fn alloc_value_buffer(&self, len: usize, val: crate::Value) -> Vec<crate::Value> {
        vec![val; len]
    }

    /// Allocate a bump-backed `&str` that lives as long as the arena.
    ///
    /// The returned reference is valid until the next [`reset`](Self::reset).
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::base::arena::BumpArena;
    /// let arena = BumpArena::new();
    /// let s = arena.alloc_str("hello");
    /// assert_eq!(s, "hello");
    /// ```
    pub fn alloc_str<'a>(&'a self, s: &str) -> &'a str {
        self.bump.alloc_str(s)
    }

    /// Reset the arena, run `f` with `&mut Bump`, then reset again.
    ///
    /// Convenience for scoped bump usage without manual `reset` calls.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::base::arena::BumpArena;
    /// let mut arena = BumpArena::new();
    /// let len = arena.with_bump(|b| b.alloc_str("hi").len());
    /// assert_eq!(len, 2);
    /// ```
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
#[allow(clippy::pedantic)]
mod tests {
    use super::*;

    #[test]
    fn arena_resets() {
        let _ = with_arena(|arena| {
            let _v = arena.alloc_str("hello");
            assert_eq!(_v, "hello");
        });
        // second run should not see previous allocation
        let _ = with_arena(|arena| {
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
