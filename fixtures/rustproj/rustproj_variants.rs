//! Signature-shape variants the extractor must handle: pub(crate) visibility,
//! unsafe/const/extern qualifiers, nested generics, multi-line parameter lists,
//! and where clauses spanning multiple lines.

use std::collections::BTreeMap;
use std::fmt::Write as _;

/// 1. pub(crate) + multi-line params + multi-line where clause.
pub(crate) fn merge_labels<K, V>(
    base: &BTreeMap<K, V>,
    overlay: &BTreeMap<K, V>,
    drop_empty: bool,
) -> BTreeMap<K, V>
where
    K: Ord + Clone,
    V: Clone + Default + PartialEq,
{
    let mut out: BTreeMap<K, V> = BTreeMap::new();
    for (k, v) in base.iter() {
        out.insert(k.clone(), v.clone());
    }
    for (k, v) in overlay.iter() {
        out.insert(k.clone(), v.clone());
    }
    if drop_empty {
        let empty = V::default();
        let drop_keys: Vec<K> = out
            .iter()
            .filter(|(_, v)| **v == empty)
            .map(|(k, _)| k.clone())
            .collect();
        for k in drop_keys {
            out.remove(&k);
        }
    }
    let mut from_overlay = 0usize;
    for k in overlay.keys() {
        if base.contains_key(k) {
            from_overlay += 1;
        }
    }
    if from_overlay > 0 {
        tracing::debug!(overridden = from_overlay, "merge_labels overrode keys");
    }
    out
}

/// 2. Nested generics in the parameter bounds + templated return type.
pub fn encode_pairs<K: AsRef<str>, V: Into<Vec<u8>> + Clone>(
    pairs: &[(K, V)],
) -> Result<String, std::fmt::Error> {
    let mut buf = String::new();
    let mut first = true;
    for (k, v) in pairs.iter() {
        if !first {
            buf.push('&');
        }
        first = false;
        let key = k.as_ref();
        let raw: Vec<u8> = v.clone().into();
        write!(buf, "{}=", key)?;
        for byte in raw {
            if byte.is_ascii_alphanumeric() {
                buf.push(byte as char);
            } else {
                write!(buf, "%{:02X}", byte)?;
            }
        }
    }
    if buf.is_empty() {
        buf.push_str("(empty)");
    }
    Ok(buf)
}

/// 3. const fn with a multi-line signature.
pub const fn checksum_seed(
    salt: u64,
    rounds: u32,
) -> u64 {
    let mut acc: u64 = 0xcbf2_9ce4_8422_2325;
    let mut i: u32 = 0;
    while i < rounds {
        acc ^= salt.wrapping_add(i as u64);
        acc = acc.wrapping_mul(0x0100_0000_01b3);
        acc = acc.rotate_left(13);
        acc = acc.wrapping_add((i as u64) << 7);
        let hi = acc >> 32;
        let lo = acc & 0xffff_ffff;
        acc = (lo << 32) | (hi ^ lo);
        acc ^= acc >> 29;
        acc = acc.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        acc ^= acc >> 32;
        i += 1;
    }
    acc ^= rounds as u64;
    acc = acc.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    acc ^= salt.rotate_right(17);
    acc = acc.wrapping_add(0x2545_f491_4f6c_dd1d);
    acc ^= acc >> 28;
    acc
}

/// 4. unsafe fn with a multi-line parameter list.
pub unsafe fn copy_frames(
    dst: *mut u8,
    src: *const u8,
    frame_len: usize,
    frame_count: usize,
) -> usize {
    let mut copied = 0usize;
    let mut frame = 0usize;
    while frame < frame_count {
        let off = frame * frame_len;
        let mut byte = 0usize;
        while byte < frame_len {
            let s = src.add(off + byte);
            let d = dst.add(off + byte);
            let v = core::ptr::read(s);
            if v != 0 {
                core::ptr::write(d, v);
                copied += 1;
            } else {
                core::ptr::write(d, 0);
            }
            byte += 1;
        }
        frame += 1;
    }
    copied
}

/// 5. extern "C" fn (FFI export) with a multi-line signature.
#[no_mangle]
pub extern "C" fn rollup_histogram(
    values: *const f64,
    len: usize,
    bucket_width: f64,
    out_counts: *mut u64,
    out_len: usize,
) -> i32 {
    if values.is_null() || out_counts.is_null() || bucket_width <= 0.0 {
        return -1;
    }
    let mut overflow: u64 = 0;
    for i in 0..len {
        let v = unsafe { *values.add(i) };
        if !v.is_finite() {
            overflow += 1;
            continue;
        }
        let idx = (v / bucket_width) as usize;
        if idx < out_len {
            unsafe {
                let slot = out_counts.add(idx);
                *slot = (*slot).saturating_add(1);
            }
        } else {
            overflow += 1;
        }
    }
    overflow as i32
}
