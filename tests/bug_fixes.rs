//! Regression tests for three bugs found during code review (March 2026).

use gainforge::TransferFunction;

// ============================================================================
// Bug 1: trc_linear clamps to 0 instead of passing through
// File: src/gamma.rs:272-274
// Was: `v.min(1.).min(0.)` — second min(0.) clamps everything to ≤ 0
// Fix: `v.min(1.).max(0.)` — clamp to [0, 1]
// ============================================================================

#[test]
fn trc_linear_preserves_midrange_values() {
    let tf = TransferFunction::Linear;

    let result = tf.linearize(0.5, 1.0);
    assert!(
        (result - 0.5).abs() < 1e-6,
        "Linear TF linearize(0.5) = {result}, expected 0.5"
    );

    let result = tf.gamma(0.5);
    assert!(
        (result - 0.5).abs() < 1e-6,
        "Linear TF gamma(0.5) = {result}, expected 0.5"
    );
}

#[test]
fn trc_linear_roundtrip() {
    let tf = TransferFunction::Linear;
    for &v in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
        let linearized = tf.linearize(v, 1.0);
        let gamma = tf.gamma(v);
        assert!(
            (linearized - v).abs() < 1e-6,
            "Linear linearize({v}) = {linearized}"
        );
        assert!((gamma - v).abs() < 1e-6, "Linear gamma({v}) = {gamma}");
    }
}

#[test]
fn trc_linear_clamps_out_of_range() {
    let tf = TransferFunction::Linear;
    assert!(
        tf.linearize(1.5, 1.0) <= 1.0,
        "Linear TF should clamp >1.0"
    );
    assert!(
        tf.linearize(-0.5, 1.0) >= 0.0,
        "Linear TF should clamp <0.0"
    );
}

// ============================================================================
// Bug 2: ReinhardJodie uses green channel for blue output
// File: src/mappers.rs:512
// Was: `chunk[2] = lerp(chunk[1] * luma_scale, tv_b, tv_b)` — uses chunk[1]
// Fix: `chunk[2] = lerp(chunk[2] * luma_scale, tv_b, tv_b)` — use chunk[2]
// ============================================================================

#[test]
fn reinhard_jodie_blue_channel_independent_of_green() {
    use gainforge::{
        MappingColorSpace, ToneMappingMethod, create_tone_mapper_rgb,
    };
    use moxcms::ColorProfile;

    let input_profile = ColorProfile::new_srgb();
    let output_profile = ColorProfile::new_srgb();
    let mapper = create_tone_mapper_rgb(
        &input_profile,
        &output_profile,
        ToneMappingMethod::ReinhardJodie,
        MappingColorSpace::Rgb(Default::default()),
    )
    .unwrap();

    // Two pixels: same R and B, different G.
    let src_low_green: Vec<u8> = vec![128, 50, 200];
    let src_high_green: Vec<u8> = vec![128, 200, 200];
    let mut dst_low = vec![0u8; 3];
    let mut dst_high = vec![0u8; 3];

    mapper
        .tonemap_lane(&src_low_green, &mut dst_low)
        .unwrap();
    mapper
        .tonemap_lane(&src_high_green, &mut dst_high)
        .unwrap();

    // With the bug, blue output used the already-modified green value,
    // causing dramatic divergence. Without the bug, blue differs modestly
    // due to luminance-dependent blending only.
    let blue_diff = (dst_low[2] as i32 - dst_high[2] as i32).abs();
    assert!(
        blue_diff < 25,
        "Blue channel should not depend heavily on green. \
         low_green: {dst_low:?}, high_green: {dst_high:?}, blue_diff: {blue_diff}",
    );
}

// ============================================================================
// Bug 3: GainLUT index clamping is inverted
// File: src/iso_gain_map.rs:945-947
// Was: `.min(0).max(N-1)` — min(0) clamps to ≤0, max(N-1) always gives N-1
// Fix: `.max(0).min(N-1)` — proper clamping to [0, N-1]
// ============================================================================

#[cfg(feature = "uhdr")]
#[test]
fn gain_lut_varies_with_input() {
    use gainforge::{GainLUT, GainMap, make_gainmap_weight};

    let metadata = GainMap {
        max_content_boost: [4.0; 3],
        min_content_boost: [1.0; 3],
        gamma: [1.0; 3],
        offset_sdr: [0.0; 3],
        offset_hdr: [0.0; 3],
        hdr_capacity_min: 1.0,
        hdr_capacity_max: 16.0,
        use_base_cg: false,
    };

    let weight = make_gainmap_weight(metadata, 4.0);
    let lut = GainLUT::<1024>::new(metadata, weight);

    // With the clamping bug, ALL gain values return the same LUT entry (last one).
    let factor_at_zero = lut.get_gain_r_factor(0.0);
    let factor_at_half = lut.get_gain_r_factor(0.5);
    let factor_at_one = lut.get_gain_r_factor(1.0);

    assert!(
        (factor_at_zero - factor_at_one).abs() > 0.01,
        "Gain factors must vary: gain=0->{factor_at_zero}, gain=0.5->{factor_at_half}, gain=1->{factor_at_one}"
    );
    assert!(
        factor_at_zero < factor_at_one,
        "Higher gain should produce higher factor: gain=0->{factor_at_zero}, gain=1->{factor_at_one}"
    );
}
