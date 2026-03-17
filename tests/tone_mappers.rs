//! Tier 2: Tone mapper sanity tests for all ToneMappingMethod variants.
//!
//! Every tone mapper must satisfy:
//! - Monotonicity: brighter input → brighter or equal output (per luminance)
//! - Channel independence: changing one channel shouldn't wildly affect others
//! - Zero → zero (black stays black)
//! - No NaN or infinity in output
//! - Output in [0, 255] for u8

use gainforge::{
    GainHdrMetadata, GamutClipping, MappingColorSpace, RgbToneMapperParameters,
    ToneMappingMethod, create_tone_mapper_rgb,
};
use moxcms::ColorProfile;

fn rgb_params() -> MappingColorSpace {
    MappingColorSpace::Rgb(RgbToneMapperParameters {
        exposure: 1.0,
        gamut_clipping: GamutClipping::Clip,
    })
}

/// All methods that don't require special parameters.
fn simple_methods() -> Vec<(ToneMappingMethod, &'static str)> {
    vec![
        (ToneMappingMethod::Filmic, "Filmic"),
        (ToneMappingMethod::Aces, "Aces"),
        (ToneMappingMethod::Reinhard, "Reinhard"),
        (ToneMappingMethod::ExtendedReinhard, "ExtendedReinhard"),
        (ToneMappingMethod::ReinhardJodie, "ReinhardJodie"),
        (ToneMappingMethod::Clamp, "Clamp"),
    ]
}

fn hdr_metadata() -> GainHdrMetadata {
    GainHdrMetadata::new(1000.0, 100.0)
}

fn parametric_methods() -> Vec<(ToneMappingMethod, &'static str)> {
    vec![
        (
            ToneMappingMethod::TunedReinhard(hdr_metadata()),
            "TunedReinhard",
        ),
        (ToneMappingMethod::Itu2408(hdr_metadata()), "Itu2408"),
    ]
}

fn all_methods() -> Vec<(ToneMappingMethod, &'static str)> {
    let mut methods = simple_methods();
    methods.extend(parametric_methods());
    methods
}

#[test]
fn black_stays_black() {
    let profile = ColorProfile::new_srgb();
    for (method, name) in all_methods() {
        let mapper = create_tone_mapper_rgb(&profile, &profile, method, rgb_params()).unwrap();
        let src = vec![0u8; 3];
        let mut dst = vec![255u8; 3];
        mapper.tonemap_lane(&src, &mut dst).unwrap();
        assert!(
            dst[0] <= 1 && dst[1] <= 1 && dst[2] <= 1,
            "{name}: black input gave {:?}, expected near-zero",
            dst
        );
    }
}

#[test]
fn no_nan_or_extreme_output() {
    let profile = ColorProfile::new_srgb();
    let test_pixels: Vec<Vec<u8>> = vec![
        vec![0, 0, 0],
        vec![1, 1, 1],
        vec![128, 128, 128],
        vec![255, 255, 255],
        vec![255, 0, 0],
        vec![0, 255, 0],
        vec![0, 0, 255],
        vec![128, 64, 200],
    ];

    for (method, name) in all_methods() {
        let mapper = create_tone_mapper_rgb(&profile, &profile, method, rgb_params()).unwrap();
        for src in &test_pixels {
            let mut dst = vec![0u8; 3];
            mapper.tonemap_lane(src, &mut dst).unwrap();
            // u8 output is inherently [0, 255], but check we don't panic
            let _ = format!("{name}: {src:?} → {dst:?}");
        }
    }
}

#[test]
fn monotonic_gray_ramp() {
    // Brighter gray input should produce brighter or equal gray output.
    let profile = ColorProfile::new_srgb();
    for (method, name) in all_methods() {
        let mapper = create_tone_mapper_rgb(&profile, &profile, method, rgb_params()).unwrap();
        let mut prev_luma = 0u32;
        for i in (0..=255).step_by(5) {
            let src = vec![i as u8; 3];
            let mut dst = vec![0u8; 3];
            mapper.tonemap_lane(&src, &mut dst).unwrap();
            let luma = dst[0] as u32 + dst[1] as u32 + dst[2] as u32;
            assert!(
                luma + 3 >= prev_luma, // allow ±1 per channel rounding
                "{name}: not monotonic at input={i}: prev_luma={prev_luma}, cur_luma={luma}"
            );
            prev_luma = luma;
        }
    }
}

#[test]
fn channel_independence() {
    // Changing one channel should not wildly affect the other two.
    // The ReinhardJodie bug caused blue to depend on green.
    let profile = ColorProfile::new_srgb();
    for (method, name) in all_methods() {
        let mapper = create_tone_mapper_rgb(&profile, &profile, method, rgb_params()).unwrap();

        // Vary green from 50 to 200 while keeping R=128, B=200 fixed.
        let mut dst_low = vec![0u8; 3];
        let mut dst_high = vec![0u8; 3];
        mapper
            .tonemap_lane(&[128, 50, 200], &mut dst_low)
            .unwrap();
        mapper
            .tonemap_lane(&[128, 200, 200], &mut dst_high)
            .unwrap();

        // Luminance-ratio methods legitimately rescale all channels when luminance
        // changes. Allow up to 60 for those. The ReinhardJodie bug caused >80 diff.
        let red_diff = (dst_low[0] as i32 - dst_high[0] as i32).abs();
        let blue_diff = (dst_low[2] as i32 - dst_high[2] as i32).abs();
        assert!(
            red_diff < 60,
            "{name}: red changed by {red_diff} when only green changed: lo={dst_low:?} hi={dst_high:?}"
        );
        assert!(
            blue_diff < 60,
            "{name}: blue changed by {blue_diff} when only green changed: lo={dst_low:?} hi={dst_high:?}"
        );
    }
}

#[test]
fn white_produces_output() {
    // White input should produce non-zero output (not all-black).
    let profile = ColorProfile::new_srgb();
    for (method, name) in all_methods() {
        let mapper = create_tone_mapper_rgb(&profile, &profile, method, rgb_params()).unwrap();
        let mut dst = vec![0u8; 3];
        mapper.tonemap_lane(&[255, 255, 255], &mut dst).unwrap();
        let sum: u32 = dst.iter().map(|&v| v as u32).sum();
        assert!(
            sum > 100,
            "{name}: white input produced near-black output: {dst:?}"
        );
    }
}
