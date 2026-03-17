//! Tier 1: Transfer function round-trip and sanity tests.
//!
//! Every TransferFunction variant must satisfy:
//! - linearize(gamma(x)) ≈ x for x in [0, 1]
//! - gamma(linearize(x)) ≈ x for x in [0, 1]
//! - Monotonicity: if a < b then linearize(a) <= linearize(b)
//! - Range: output in [0, 1] for input in [0, 1]
//! - Identity at endpoints: f(0) ≈ 0, f(1) ≈ 1

use gainforge::TransferFunction;

const ALL_TFS: &[TransferFunction] = &[
    TransferFunction::Srgb,
    TransferFunction::Rec709,
    TransferFunction::Gamma2p2,
    TransferFunction::Gamma2p8,
    TransferFunction::Smpte428,
    TransferFunction::Bt1361,
    TransferFunction::Linear,
    TransferFunction::HybridLogGamma,
    TransferFunction::PerceptualQuantizer,
];

/// TFs where linearize/gamma are true inverses at reference_display=1.0.
/// PQ and HLG have display-dependent scaling, Smpte428 is DCI cinema.
const ROUNDTRIP_TFS: &[TransferFunction] = &[
    TransferFunction::Srgb,
    TransferFunction::Rec709,
    TransferFunction::Gamma2p2,
    TransferFunction::Gamma2p8,
    TransferFunction::Bt1361,
    TransferFunction::Linear,
];

const TEST_VALUES: &[f32] = &[0.0, 0.01, 0.05, 0.1, 0.18, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0];

fn tf_name(tf: &TransferFunction) -> &'static str {
    match tf {
        TransferFunction::Srgb => "Srgb",
        TransferFunction::Rec709 => "Rec709",
        TransferFunction::Gamma2p2 => "Gamma2p2",
        TransferFunction::Gamma2p8 => "Gamma2p8",
        TransferFunction::Smpte428 => "Smpte428",
        TransferFunction::Bt1361 => "Bt1361",
        TransferFunction::Linear => "Linear",
        TransferFunction::HybridLogGamma => "HLG",
        TransferFunction::PerceptualQuantizer => "PQ",
    }
}

#[test]
fn linearize_then_gamma_roundtrip() {
    // linearize → gamma should recover the original encoded value
    for tf in ROUNDTRIP_TFS {
        let name = tf_name(tf);
        for &v in TEST_VALUES {
            let linear = tf.linearize(v, 1.0);
            let back = tf.gamma(linear);
            assert!(
                (back - v).abs() < 0.02,
                "{name}: linearize({v}) = {linear}, gamma({linear}) = {back}, expected ≈{v}"
            );
        }
    }
}

#[test]
fn gamma_then_linearize_roundtrip() {
    // gamma → linearize should recover the original linear value
    for tf in ROUNDTRIP_TFS {
        let name = tf_name(tf);
        for &v in TEST_VALUES {
            let encoded = tf.gamma(v);
            let back = tf.linearize(encoded, 1.0);
            assert!(
                (back - v).abs() < 0.02,
                "{name}: gamma({v}) = {encoded}, linearize({encoded}) = {back}, expected ≈{v}"
            );
        }
    }
}

#[test]
fn linearize_monotonic() {
    for tf in ALL_TFS {
        let name = tf_name(tf);
        let mut prev = tf.linearize(0.0, 1.0);
        for i in 1..=100 {
            let v = i as f32 / 100.0;
            let cur = tf.linearize(v, 1.0);
            assert!(
                cur >= prev - 1e-6,
                "{name}: linearize not monotonic at {v}: {prev} > {cur}"
            );
            prev = cur;
        }
    }
}

#[test]
fn gamma_monotonic() {
    for tf in ALL_TFS {
        let name = tf_name(tf);
        let mut prev = tf.gamma(0.0);
        for i in 1..=100 {
            let v = i as f32 / 100.0;
            let cur = tf.gamma(v);
            assert!(
                cur >= prev - 1e-6,
                "{name}: gamma not monotonic at {v}: {prev} > {cur}"
            );
            prev = cur;
        }
    }
}

#[test]
fn endpoints() {
    for tf in ALL_TFS {
        let name = tf_name(tf);

        let lin_0 = tf.linearize(0.0, 1.0);
        assert!(
            lin_0.abs() < 0.01,
            "{name}: linearize(0) = {lin_0}, expected ≈0"
        );

        let lin_1 = tf.linearize(1.0, 1.0);
        // Smpte428/HLG/PQ may exceed 1.0 at reference_display=1.0
        if !matches!(
            tf,
            TransferFunction::Smpte428
                | TransferFunction::HybridLogGamma
                | TransferFunction::PerceptualQuantizer
        ) {
            assert!(
                (lin_1 - 1.0).abs() < 0.05,
                "{name}: linearize(1) = {lin_1}, expected ≈1"
            );
        }

        let gam_0 = tf.gamma(0.0);
        assert!(gam_0.abs() < 0.01, "{name}: gamma(0) = {gam_0}, expected ≈0");

        // PQ and HLG gamma(1.0) may not be exactly 1.0 due to reference display scaling
        if !matches!(
            tf,
            TransferFunction::PerceptualQuantizer | TransferFunction::HybridLogGamma
        ) {
            let gam_1 = tf.gamma(1.0);
            assert!(
                (gam_1 - 1.0).abs() < 0.05,
                "{name}: gamma(1) = {gam_1}, expected ≈1"
            );
        }
    }
}

#[test]
fn output_in_range() {
    for tf in ALL_TFS {
        let name = tf_name(tf);
        for &v in TEST_VALUES {
            let linear = tf.linearize(v, 1.0);
            // HLG includes OOTF, Smpte428 uses DCI scaling — can exceed 1.0
            let max_linear = if matches!(
                tf,
                TransferFunction::HybridLogGamma | TransferFunction::Smpte428
            ) {
                5.0
            } else {
                1.5
            };
            assert!(
                linear >= -0.01 && linear <= max_linear,
                "{name}: linearize({v}) = {linear}, out of expected range [0, {max_linear}]"
            );

            let encoded = tf.gamma(v);
            assert!(
                encoded >= -0.01 && encoded <= 1.5,
                "{name}: gamma({v}) = {encoded}, out of expected range"
            );
        }
    }
}

#[test]
fn no_nan_or_inf() {
    for tf in ALL_TFS {
        let name = tf_name(tf);
        for &v in TEST_VALUES {
            let linear = tf.linearize(v, 1.0);
            assert!(!linear.is_nan(), "{name}: linearize({v}) = NaN");
            assert!(!linear.is_infinite(), "{name}: linearize({v}) = Inf");

            let encoded = tf.gamma(v);
            assert!(!encoded.is_nan(), "{name}: gamma({v}) = NaN");
            assert!(!encoded.is_infinite(), "{name}: gamma({v}) = Inf");
        }
    }
}
