"""
Small invariant tests for config_v2 -> legacy runtime bridge.

Run:
  python -m spider.core.config_v2.test_bridge
"""

from __future__ import annotations

import unittest

from spider.core.config_v2 import load_config, to_legacy_runtime_params


def _base_config() -> dict:
    return {
        "io": {
            "dtime_file": "dt.csv",
            "station_file": "sta.csv",
            "catalog_infile": "cat_in.csv",
            "catalog_outfile": "cat_out.csv",
            "samples_outfile": "samples.h5",
            "checkpoint_dir": "checkpoints",
            "checkpoint_interval": 100,
            "sample_write_interval": 25,
            "save_every_n": 10,
            "write_samples": True,
        },
        "model": {
            "model_file": "model.pt",
            "domain": {
                "lon_min": -120.0,
                "lat_min": 32.0,
                "z_min": -2.0,
                "z_max": 50.0,
                "scale": 500.0,
            },
            "priors": {},
            "filters": {
                "dtimes": {
                    "remove_duplicates": True,
                    "max_abs_input_dt": 99999,
                    "dtime_thin_frac": 1.0,
                    "flip_dt_sign": False,
                    "cc_min": 0.0,
                },
                "events": {
                    "min_dtimes": 1,
                    "min_unique_phase_per_event": 2,
                    "min_dtimes_per_pair": 1,
                    "min_event_degree": 0,
                    "min_events_per_cluster": 0,
                    "max_pair_station_ratio": 1.0,
                    "ratio_filter_phase": "before",
                },
                "residual": {
                    "enabled": True,
                    "method": "mad",
                    "mad_sigma": 4.0,
                    "abs_max": 0.5,
                },
            },
            "likelihoods": {
                "locate_map": {
                    "type": "laplace",
                    "phase_unc": [0.02, 0.03],
                },
                "sample": {
                    "type": "correlated_gaussian",
                    "phase_unc": [0.02, 0.03],
                    "shared_event_re": {
                        "enabled": True,
                    "model": {
                        "group_by": "station_phase",
                        "tau_s": [0.03, 0.04],
                        "cluster": {"mode": "none", "k": 1},
                    },
                    "limits": {
                        "max_nodes": 12000,
                        "max_rows": 1000000,
                    },
                    "fallback": {
                        "to_diag": True,
                        "abort_on_pcg_fallback": False,
                    },
                    "numerics": {
                        "jitter0": 1e-8,
                        "jitter_max": 1e-3,
                    },
                        "solver": {
                            "kind": "pcg",
                            "max_iters": 40,
                            "min_iters": 2,
                            "tol": 5e-4,
                            "batched": True,
                        "node_bin_edges": [4096, 8192],
                            "warm_start": True,
                            "cache_max_entries": 32,
                            "prefetch_grouping": False,
                            "profile_micro_steps": False,
                        "merge_sparse_node_bins": True,
                        "min_groups_per_node_bin": 24,
                        "max_node_bins_per_node": 3,
                            "precompute": {"enabled": False, "device": "gpu"},
                        },
                        "edge_weights": {
                            "mode": "distance_power",
                            "power": 0.5,
                            "scale_km": 25.0,
                            "global_scale": 1.0,
                            "normalize": True,
                            "eps_km": 0.001,
                        },
                        "autotune": {
                            "enabled": True,
                            "observe_epochs": 1,
                            "latest_epoch": 2,
                            "min_groups": 64,
                        "max_node_bins": 8,
                        "min_groups_per_node_bin": 16,
                        "min_node_bin": 512,
                            "min_gain": 0.08,
                            "raise_nodes_cap": True,
                            "nodes_cap_max": 65536,
                        },
                        "logging": {"quiet": True, "stats_log_every_epochs": 0},
                    },
                },
            },
        },
        "inference": {
            "compute": {"devices": [0]},
            "sampler": {
                "backend": "psgld",
                "epochs_per_phase": [100, 10, 5, 1000],
                "lr": [1e-3, 2e-3, 2e-3, 2e-3],
                "temperature": 1.0,
                "sghmc_alpha": 0.05,
                "eps": 1e-5,
                "beta": 0.99,
                "freeze_preconditioner_sampling": False,
                "preconditioning": {
                    "enabled": True,
                    "type": "rmsprop",
                    "include_gamma": False,
                },
                "overrides": {"core": {"lr_mult": 0.5}},
            },
            "batching": {
                "standard": {"warmup": 500000, "sgld": 500000, "shuffle": False},
                "event_batches": {
                    "enabled": False,
                    "events_per_batch": 1000,
                    "max_edges_per_batch": 1000000,
                    "bucket_reorder_all": False,
                    "bucket_reuse_epochs": 1,
                },
            },
            "runtime": {
                "cuda_empty_cache_every": 0,
                "reset_batch_numbers": True,
                "clear_samples_on_reset": True,
                "min_samples_to_save": 10,
                "verbose": True,
                "cluster_events": False,
                "gauge_projection": {
                    "enabled": True,
                    "mode": "cluster",
                    "dims": [0, 1, 2, 3],
                    "apply_noise": True,
                    "apply_momentum": False,
                },
                "torch": {"allow_tf32": True},
            },
            "safety": {"max_abs_dX": [2.0, 2.0, 2.0, 1.0]},
        },
        "observability": {
            "wandb": {
                "enabled": True,
                "project_name": "proj",
                "run_name": "run",
            },
            "diagnostics": {
                "pair_count_stats_enable": True,
                "sgld_log_gnoise": False,
                "display_precond_every": 50,
                "wandb": {"enabled": True, "groups": {"core": True, "sampler": True}},
            },
        },
    }


class ConfigBridgeTests(unittest.TestCase):
    def test_defaults_and_observability_mapping(self) -> None:
        cfg = _base_config()
        cfg["io"].pop("sample_write_interval")
        # seed omitted -> default applied by resolver
        resolved = load_config(cfg, mode="sample")
        legacy = to_legacy_runtime_params(resolved, profile="all", require_priors=False)

        self.assertIn("io.sample_write_interval <- io.checkpoint_interval", resolved.defaults_applied)
        self.assertIn("inference.runtime.seed <- 0", resolved.defaults_applied)
        self.assertTrue(legacy["use_wandb"])
        self.assertEqual(legacy["wandb_project_name"], "proj")
        self.assertEqual(legacy["wandb_run_name"], "run")
        self.assertEqual(legacy["sample_write_interval"], legacy["checkpoint_interval"])
        self.assertEqual(legacy["runtime_seed"], 0)
        self.assertIn("core", legacy["_wandb_diag_groups"])

    def test_device_parsing(self) -> None:
        cfg = _base_config()
        cfg["inference"]["compute"]["devices"] = ["cpu", "cuda:2", 3, -1, "7"]
        resolved = load_config(cfg, mode="sample")
        legacy = to_legacy_runtime_params(resolved, profile="all", require_priors=False)
        self.assertEqual(legacy["devices"], [-1, 2, 3, -1, 7])

    def test_shared_event_solver_materialization(self) -> None:
        cfg = _base_config()
        se = cfg["model"]["likelihoods"]["sample"]["shared_event_re"]
        se["solver"]["max_iters"] = 77
        se["solver"]["tol"] = 1e-4
        se["edge_weights"]["power"] = 0.25
        se["edge_weights"]["normalize"] = False

        resolved = load_config(cfg, mode="sample")
        legacy = to_legacy_runtime_params(resolved, profile="all", require_priors=False)
        self.assertTrue(legacy["_shared_event_re_enabled"])
        self.assertEqual(legacy["_shared_event_re_solver_kind"], "pcg")
        self.assertEqual(legacy["_shared_event_re_solver_max_iters"], 77)
        self.assertAlmostEqual(legacy["_shared_event_re_solver_tol"], 1e-4)
        self.assertAlmostEqual(legacy["_shared_event_re_edge_weight_power"], 0.25)
        self.assertFalse(legacy["_shared_event_re_edge_weight_normalize"])

    def test_sampler_lrd_materialization(self) -> None:
        cfg = _base_config()
        pre = cfg["inference"]["sampler"]["preconditioning"]
        pre["type"] = "lrd"
        pre["lrd"] = {
            "rank": 12,
            "mode": "oja",
            "update_every": 7,
            "buffer_size": 40,
            "eta": 0.015,
            "diag_floor": 1e-6,
            "target": "dX_src_only",
        }

        resolved = load_config(cfg, mode="sample")
        legacy = to_legacy_runtime_params(resolved, profile="all", require_priors=False)
        self.assertEqual(legacy["sampler_preconditioner"], "lrd")
        self.assertEqual(legacy["sampler_preconditioning_lrd_rank"], 12)
        self.assertEqual(legacy["sampler_preconditioning_lrd_mode"], "oja")
        self.assertEqual(legacy["sampler_preconditioning_lrd_update_every"], 7)
        self.assertEqual(legacy["sampler_preconditioning_lrd_buffer_size"], 40)
        self.assertAlmostEqual(legacy["sampler_preconditioning_lrd_oja_eta"], 0.015)
        self.assertAlmostEqual(legacy["sampler_preconditioning_lrd_diag_floor"], 1e-6)
        self.assertEqual(legacy["sampler_preconditioning_lrd_target"], "dx_src_only")

    def test_sampler_component_lrd_alias_materialization(self) -> None:
        cfg = _base_config()
        pre = cfg["inference"]["sampler"]["preconditioning"]
        pre["type"] = "cc_lrd"
        pre["lrd"] = {"rank": 8, "mode": "svd"}

        resolved = load_config(cfg, mode="sample")
        legacy = to_legacy_runtime_params(resolved, profile="all", require_priors=False)
        self.assertEqual(legacy["sampler_preconditioner"], "component_lrd")
        self.assertEqual(legacy["sampler_preconditioning_lrd_rank"], 8)
        self.assertEqual(legacy["sampler_preconditioning_lrd_mode"], "svd")

    def test_sampler_blocked_reparameterization_materialization(self) -> None:
        cfg = _base_config()
        sampler = cfg["inference"]["sampler"]
        sampler["reparameterization"] = {
            "enabled": True,
            "spatial_scale": 0.5,
            "dt_scale": 2.0,
        }

        resolved = load_config(cfg, mode="sample")
        legacy = to_legacy_runtime_params(resolved, profile="all", require_priors=False)
        self.assertTrue(legacy["sampler_reparam_blocked_enable"])
        self.assertAlmostEqual(legacy["sampler_reparam_blocked_spatial_scale"], 0.5)
        self.assertAlmostEqual(legacy["sampler_reparam_blocked_dt_scale"], 2.0)

    def test_phase1_lr_schedule_materialization(self) -> None:
        cfg = _base_config()
        # Default: no schedule -> constant lr
        resolved = load_config(cfg, mode="sample")
        legacy = to_legacy_runtime_params(resolved, profile="all", require_priors=False)
        self.assertFalse(legacy["phase1_use_cosine"])
        self.assertIsNone(legacy["phase1_cosine_T_max"])  # resolved per MAP pass at runtime
        # Cosine with explicit floor and horizon
        cfg["inference"]["sampler"]["lr_schedule"] = {"phase1": {"type": "cosine", "eta_min": 1e-5, "T_max": 400}}
        resolved = load_config(cfg, mode="sample")
        legacy = to_legacy_runtime_params(resolved, profile="all", require_priors=False)
        self.assertTrue(legacy["phase1_use_cosine"])
        self.assertAlmostEqual(legacy["phase1_cosine_eta_min"], 1e-5)
        self.assertEqual(legacy["phase1_cosine_T_max"], 400)
        # Invalid type / values are rejected
        cfg["inference"]["sampler"]["lr_schedule"] = {"phase1": {"type": "step"}}
        with self.assertRaises(Exception):
            load_config(cfg, mode="sample")
        cfg["inference"]["sampler"]["lr_schedule"] = {"phase1": {"type": "cosine", "eta_min": -1.0}}
        with self.assertRaises(Exception):
            load_config(cfg, mode="sample")
        cfg["inference"]["sampler"]["lr_schedule"] = {"phase2": {"type": "cosine"}}
        with self.assertRaises(Exception):
            load_config(cfg, mode="sample")

    def test_phase1_two_pass_materialization(self) -> None:
        cfg = _base_config()
        resolved = load_config(cfg, mode="sample")
        legacy = to_legacy_runtime_params(resolved, profile="all", require_priors=False)
        self.assertFalse(legacy["phase1_two_pass_enable"])
        self.assertTrue(legacy["phase1_two_pass_warm_start"])
        self.assertEqual(legacy["phase1_two_pass_epochs"], cfg["inference"]["sampler"]["epochs_per_phase"][0])
        cfg["inference"]["sampler"]["phase1_two_pass"] = {"enabled": True, "epochs": 250, "warm_start": False}
        resolved = load_config(cfg, mode="sample")
        legacy = to_legacy_runtime_params(resolved, profile="all", require_priors=False)
        self.assertTrue(legacy["phase1_two_pass_enable"])
        self.assertEqual(legacy["phase1_two_pass_epochs"], 250)
        self.assertFalse(legacy["phase1_two_pass_warm_start"])
        cfg["inference"]["sampler"]["phase1_two_pass"] = {"enabled": True, "epochs": 0}
        with self.assertRaises(Exception):
            load_config(cfg, mode="sample")
        cfg["inference"]["sampler"]["phase1_two_pass"] = {"enabled": "yes"}
        with self.assertRaises(Exception):
            load_config(cfg, mode="sample")

    def test_residual_filter_phase_materialization(self) -> None:
        cfg = _base_config()
        resolved = load_config(cfg, mode="sample")
        legacy = to_legacy_runtime_params(resolved, profile="all", require_priors=False)
        self.assertTrue(legacy["residual_filter_enable"])
        self.assertEqual(legacy["residual_filter_phase"], "after_phase1")
        cfg["model"]["filters"]["residual"]["phase"] = "before"
        resolved = load_config(cfg, mode="sample")
        legacy = to_legacy_runtime_params(resolved, profile="all", require_priors=False)
        self.assertEqual(legacy["residual_filter_phase"], "before")
        cfg["model"]["filters"]["residual"]["phase"] = "phase2"
        resolved = load_config(cfg, mode="sample")
        with self.assertRaises(ValueError):
            to_legacy_runtime_params(resolved, profile="all", require_priors=False)

    def test_grad_clip_norm_phase_mapping(self) -> None:
        cfg = _base_config()
        cfg["inference"]["sampler"].pop("grad_clip_norm", None)
        legacy = to_legacy_runtime_params(load_config(cfg, mode="sample"), profile="all", require_priors=False)
        self.assertAlmostEqual(legacy["sampler_grad_clip_norm"], 0.0)   # sampler phases: disabled when unset
        self.assertAlmostEqual(legacy["grad_clip_norm"], 100.0)         # Phase 1: legacy default when unset
        cfg["inference"]["sampler"]["grad_clip_norm"] = 5.0
        legacy = to_legacy_runtime_params(load_config(cfg, mode="sample"), profile="all", require_priors=False)
        self.assertAlmostEqual(legacy["sampler_grad_clip_norm"], 5.0)
        self.assertAlmostEqual(legacy["grad_clip_norm"], 5.0)
        cfg["inference"]["sampler"]["grad_clip_norm"] = 0.0
        legacy = to_legacy_runtime_params(load_config(cfg, mode="sample"), profile="all", require_priors=False)
        self.assertAlmostEqual(legacy["grad_clip_norm"], 0.0)

    def test_synth_block_passthrough(self) -> None:
        cfg = _base_config()
        legacy = to_legacy_runtime_params(load_config(cfg, mode="synth"), profile="synth", require_priors=False)
        self.assertIsNone(legacy.get("synth"))
        cfg["synth"] = {"seed": 7, "outfile": "dt_synth.csv", "apply_filters": False}
        legacy = to_legacy_runtime_params(load_config(cfg, mode="synth"), profile="synth", require_priors=False)
        self.assertEqual(legacy["synth"]["seed"], 7)
        self.assertEqual(legacy["synth"]["outfile"], "dt_synth.csv")
        cfg["synth"] = "not-a-dict"
        with self.assertRaises(Exception):
            load_config(cfg, mode="synth")

    def test_event_bounds_mapping(self) -> None:
        cfg = _base_config()
        legacy = to_legacy_runtime_params(load_config(cfg, mode="sample"), profile="all", require_priors=False)
        self.assertNotIn("event_lat_bounds", legacy)
        cfg["model"]["filters"]["events"]["lat_bounds"] = [32.0, 34.5]
        cfg["model"]["filters"]["events"]["lon_bounds"] = [-118.0, -115.0]
        legacy = to_legacy_runtime_params(load_config(cfg, mode="sample"), profile="all", require_priors=False)
        self.assertEqual(legacy["event_lat_bounds"], [32.0, 34.5])
        self.assertEqual(legacy["event_lon_bounds"], [-118.0, -115.0])
        cfg["model"]["filters"]["events"]["lat_bounds"] = [34.5, 32.0]
        with self.assertRaises(ValueError):
            to_legacy_runtime_params(load_config(cfg, mode="sample"), profile="all", require_priors=False)

    def test_synth_profile_skips_sampler_materialization(self) -> None:
        cfg = _base_config()
        resolved = load_config(cfg, mode="synth")
        legacy = to_legacy_runtime_params(resolved, profile="synth", require_priors=False)
        self.assertNotIn("phase1_epochs", legacy)
        self.assertIn("batch_size_warmup", legacy)
        self.assertIn("devices", legacy)


if __name__ == "__main__":
    unittest.main()

