//! Tier 3: GainLUT correctness tests.

#[cfg(feature = "uhdr")]
mod gain_lut {
    use gainforge::{GainLUT, GainMap, make_gainmap_weight};

    fn test_metadata() -> GainMap {
        GainMap {
            max_content_boost: [4.0; 3],
            min_content_boost: [1.0; 3],
            gamma: [1.0; 3],
            offset_sdr: [0.0; 3],
            offset_hdr: [0.0; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 16.0,
            use_base_cg: false,
        }
    }

    #[test]
    fn lut_varies_across_input_range() {
        let meta = test_metadata();
        let weight = make_gainmap_weight(meta, 4.0);
        let lut = GainLUT::<1024>::new(meta, weight);

        let f0 = lut.get_gain_r_factor(0.0);
        let f_half = lut.get_gain_r_factor(0.5);
        let f1 = lut.get_gain_r_factor(1.0);

        assert!(
            (f0 - f1).abs() > 0.01,
            "gain=0 and gain=1 must differ: {f0} vs {f1}"
        );
        assert!(
            (f0 - f_half).abs() > 0.001,
            "gain=0 and gain=0.5 must differ: {f0} vs {f_half}"
        );
    }

    #[test]
    fn lut_monotonic() {
        let meta = test_metadata();
        let weight = make_gainmap_weight(meta, 4.0);
        let lut = GainLUT::<1024>::new(meta, weight);

        let mut prev = lut.get_gain_r_factor(0.0);
        for i in 1..=100 {
            let gain = i as f32 / 100.0;
            let cur = lut.get_gain_r_factor(gain);
            assert!(
                cur >= prev - 1e-6,
                "LUT not monotonic at gain={gain}: {prev} > {cur}"
            );
            prev = cur;
        }
    }

    #[test]
    fn lut_min_max_range() {
        let meta = test_metadata();
        let weight = make_gainmap_weight(meta, 4.0);
        let lut = GainLUT::<1024>::new(meta, weight);

        let f0 = lut.get_gain_r_factor(0.0);
        let f1 = lut.get_gain_r_factor(1.0);

        assert!(f0 >= 0.5, "gain=0 factor too low: {f0}");
        assert!(f1 <= 8.0, "gain=1 factor too high: {f1}");
        assert!(f0 < f1, "min should be less than max: {f0} vs {f1}");
    }

    #[test]
    fn lut_channels_independent() {
        let meta = GainMap {
            max_content_boost: [2.0, 4.0, 8.0],
            min_content_boost: [1.0; 3],
            gamma: [1.0; 3],
            offset_sdr: [0.0; 3],
            offset_hdr: [0.0; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 16.0,
            use_base_cg: false,
        };
        let weight = make_gainmap_weight(meta, 4.0);
        let lut = GainLUT::<1024>::new(meta, weight);

        let r = lut.get_gain_r_factor(1.0);
        let g = lut.get_gain_g_factor(1.0);
        let b = lut.get_gain_b_factor(1.0);

        assert!(
            g > r && b > g,
            "Channels should reflect different boosts: r={r}, g={g}, b={b}"
        );
    }

    #[test]
    fn lut_gamma_affects_mapping() {
        let meta_linear = test_metadata();
        let meta_gamma = GainMap {
            gamma: [2.2; 3],
            ..meta_linear
        };

        let weight = 0.5;
        let lut_lin = GainLUT::<1024>::new(meta_linear, weight);
        let lut_gam = GainLUT::<1024>::new(meta_gamma, weight);

        let f_lin = lut_lin.get_gain_r_factor(0.5);
        let f_gam = lut_gam.get_gain_r_factor(0.5);
        assert!(
            (f_lin - f_gam).abs() > 0.01,
            "Gamma should affect mapping: linear={f_lin}, gamma2.2={f_gam}"
        );
    }

    #[test]
    fn lut_edge_values() {
        let meta = test_metadata();
        let weight = make_gainmap_weight(meta, 4.0);
        let lut = GainLUT::<1024>::new(meta, weight);

        // Should not panic or produce NaN/Inf at boundaries
        let f_neg = lut.get_gain_r_factor(-0.1);
        let f_over = lut.get_gain_r_factor(1.1);
        assert!(!f_neg.is_nan(), "gain=-0.1 produced NaN");
        assert!(!f_over.is_nan(), "gain=1.1 produced NaN");
        assert!(!f_neg.is_infinite(), "gain=-0.1 produced Inf");
        assert!(!f_over.is_infinite(), "gain=1.1 produced Inf");
    }

    #[test]
    fn weight_varies_with_display_boost() {
        let meta = test_metadata();
        let w1 = make_gainmap_weight(meta, 1.0);
        let w2 = make_gainmap_weight(meta, 2.0);
        let w4 = make_gainmap_weight(meta, 4.0);
        let w16 = make_gainmap_weight(meta, 16.0);

        // Weight should increase with display boost
        assert!(w1 <= w2, "weight should increase: w1={w1}, w2={w2}");
        assert!(w2 <= w4, "weight should increase: w2={w2}, w4={w4}");
        assert!(w4 <= w16, "weight should increase: w4={w4}, w16={w16}");

        // At min boost (1.0), weight should be 0 (no HDR)
        assert!(
            w1.abs() < 0.01,
            "weight at display_boost=1.0 should be ≈0: {w1}"
        );
        // At max capacity, weight should be 1.0
        assert!(
            (w16 - 1.0).abs() < 0.01,
            "weight at display_boost=16.0 should be ≈1.0: {w16}"
        );
    }
}
