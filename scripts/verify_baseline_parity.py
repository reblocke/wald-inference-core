from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wald_inference.compatibility import compatibility_curve, standardized_distance
from wald_inference.detectability import (
    legacy_critical_effect_distance,
    legacy_critical_effect_markers,
)
from wald_inference.effects import EFFECT_SPECS, to_working_scale
from wald_inference.errors import ValidationError
from wald_inference.grid import build_grid, max_safe_grid_span
from wald_inference.likelihood import (
    log_relative_likelihood,
    relative_likelihood,
    support_comparison,
    support_interval,
    wald_point_summary,
)
from wald_inference.precision import (
    approximate_wald_ci_width,
    information_scaled_standard_error,
    precision_target_results,
)
from wald_inference.reconstruction import reconstruct_wald_from_95_ci
from wald_inference.selection import selection_rule_spec
from wald_inference.type_sm import design_metrics_for_true_effects

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_ROOT = PROJECT_ROOT / "tests" / "regression" / "golden"
MANIFEST_PATH = GOLDEN_ROOT / "manifest.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "baseline-parity.json"

EXPECTED_MANIFEST_SHA256 = "f54bb2d8311788c07adcf23fc9f038e35702449e4a77a474abea9411246cabcc"
EXPECTED_FIXTURE_SET_SHA256 = "81c341b39e711caffc85a444f0c1e4bc1e2d00633474c82e720afeb60def3c4d"
EXPECTED_BEHAVIOR_SOURCE = "830756ecb11b4e8161f8dfe1fc75afc346ef4467"
DEFAULT_SELECTION_RULE = "two_sided_p_lt_alpha"
DEFAULT_CLAIM_DIRECTION = "positive"
DEFAULT_NEAR_NULL_DELTA = 1e-12
Z975 = 1.959963984540054
LOG_MAX_FLOAT = float(np.log(np.finfo(float).max))

APP_ONLY_ERROR_CASES = {
    "B07h-display-range-pair": "paired display-window validation",
    "B07i-design-range-pair": "paired design-shading-range validation",
}

CORE_ERROR_CASES = {
    "B07d-alpha-zero",
    "B07e-alpha-one",
    "B07f-alpha-nonnumeric",
    "B07g-alpha-underflow",
    "B07j-ratio-positive",
    "B08e-unrepresentable-design-distance",
}

APP_OWNED_PREFIXES = (
    "$.status",
    "$.response.meta.display_axis_scale",
    "$.response.meta.grid_points",
    "$.response.meta.show_cutoffs",
    "$.response.meta.thresholds_display",
    "$.response.meta.display_range_active",
    "$.response.meta.display_range_display",
    "$.response.meta.display_range_working",
    "$.response.meta.threshold_support_summaries[*].threshold_display",
    "$.response.meta.threshold_support_summaries[*].direction_from_estimate",
    "$.response.meta.threshold_support_summaries[*].direction_from_null",
    "$.response.meta.s_minus_2_interval.range_display",
    "$.response.summary.critical_effect_markers_display",
    "$.response.warnings",
    "$.response.grid.effect_display",
    "$.response.design.config.enabled",
    "$.response.design.config.claim_threshold_display",
    "$.response.design.config.type_m_scale_note",
    "$.response.design.config.plausible_range_display",
    "$.response.design.config.plausible_range_working",
    "$.response.design.grid.true_effect_display",
    "$.response.design.scenarios[*].label",
    "$.response.design.scenarios[*].source",
    "$.response.design.scenarios[*].note",
    "$.response.design.scenarios[*].true_effect_display",
    "$.response.design.precision_targets[*].target_effect_display",
    "$.response.design.warnings",
)

APP_OWNED_EXACT_PATHS = {
    "$.response.meta.threshold_support_summaries",
    "$.response.design",
    "$.response.design.precision_targets",
}

CORE_OWNED_PREFIXES = (
    "$.response.meta.effect_spec",
    "$.response.meta.estimate_source",
    "$.response.meta.default_null_applied",
    "$.response.meta.se_method",
    "$.response.meta.relative_asymmetry",
    "$.response.meta.thresholds_working",
    "$.response.meta.threshold_support_summaries[*].threshold_working",
    "$.response.meta.threshold_support_summaries[*].relative_likelihood",
    "$.response.meta.threshold_support_summaries[*].log_relative_likelihood",
    "$.response.meta.threshold_support_summaries[*].likelihood_ratio_mle_to_threshold",
    "$.response.meta.threshold_support_summaries[*].log_likelihood_ratio_mle_to_threshold",
    "$.response.meta.threshold_support_summaries[*].likelihood_ratio_threshold_to_null",
    "$.response.meta.threshold_support_summaries[*].log_likelihood_ratio_threshold_to_null",
    "$.response.meta.s_minus_2_interval.support_cutoff",
    "$.response.meta.s_minus_2_interval.relative_likelihood_cutoff",
    "$.response.meta.s_minus_2_interval.likelihood_ratio_mle_to_bound",
    "$.response.meta.s_minus_2_interval.range_working",
    "$.response.summary",
    "$.response.grid.effect_working",
    "$.response.grid.z",
    "$.response.grid.compatibility",
    "$.response.grid.relative_likelihood",
    "$.response.grid.log_relative_likelihood",
    "$.response.design.config.alpha",
    "$.response.design.config.selection_rule",
    "$.response.design.config.selection_rule_label",
    "$.response.design.config.claim_direction",
    "$.response.design.config.claim_threshold_working",
    "$.response.design.config.se_working",
    "$.response.design.config.current_se_working",
    "$.response.design.config.design_se_working",
    "$.response.design.config.information_multiplier",
    "$.response.design.config.current_ci_width_working",
    "$.response.design.config.approx_design_ci_width_working",
    "$.response.design.config.null_working",
    "$.response.design.config.estimate_working",
    "$.response.design.config.near_null_delta",
    "$.response.design.grid.true_effect_working",
    "$.response.design.grid.delta",
    "$.response.design.grid.power",
    "$.response.design.grid.type_s",
    "$.response.design.grid.type_m",
    "$.response.design.grid.expected_selected_abs_z",
    "$.response.design.grid.observed_exaggeration",
    "$.response.design.scenarios[*].true_effect_working",
    "$.response.design.scenarios[*].delta",
    "$.response.design.scenarios[*].power",
    "$.response.design.scenarios[*].type_s",
    "$.response.design.scenarios[*].type_m",
    "$.response.design.scenarios[*].observed_exaggeration",
    "$.response.design.precision_targets[*].target",
    "$.response.design.precision_targets[*].requested_value",
    "$.response.design.precision_targets[*].target_effect_working",
    "$.response.design.precision_targets[*].required_se",
    "$.response.design.precision_targets[*].required_information_multiplier",
    "$.response.design.precision_targets[*].approx_95_ci_width_working",
    "$.response.design.precision_targets[*].achieved_power",
    "$.response.design.precision_targets[*].achieved_type_s",
    "$.response.design.precision_targets[*].achieved_type_m",
    "$.response.design.precision_targets[*].note",
)

EDGE_SUMMARY_KEYS = {
    "length",
    "first",
    "last",
    "none_count",
    "all_numeric_values_finite",
}


@dataclass
class ComparisonStats:
    compared_values: int = 0
    max_absolute_difference: float = 0.0
    max_relative_difference: float = 0.0


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_corpus_integrity(manifest: dict[str, Any]) -> None:
    manifest_text = MANIFEST_PATH.read_text(encoding="utf-8")
    if _sha256_text(manifest_text) != EXPECTED_MANIFEST_SHA256:
        raise AssertionError("Frozen manifest SHA-256 mismatch.")
    if manifest["source"]["commit"] != EXPECTED_BEHAVIOR_SOURCE:
        raise AssertionError("Frozen behavior source mismatch.")
    if manifest["case_count"] != 22 or len(manifest["cases"]) != 22:
        raise AssertionError("Frozen case count must be exactly 22.")

    fixture_hash_inputs: list[str] = []
    for case in manifest["cases"]:
        request_text = (GOLDEN_ROOT / case["request_file"]).read_text(encoding="utf-8")
        expected_text = (GOLDEN_ROOT / case["expected_file"]).read_text(encoding="utf-8")
        request = json.loads(request_text)
        expected = json.loads(expected_text)
        if request_text != _canonical_json(request) or expected_text != _canonical_json(expected):
            raise AssertionError(f"{case['id']}: noncanonical stored JSON.")
        if _sha256_text(request_text) != case["request_sha256"]:
            raise AssertionError(f"{case['id']}: request hash mismatch.")
        if _sha256_text(expected_text) != case["expected_sha256"]:
            raise AssertionError(f"{case['id']}: response hash mismatch.")
        fixture_sha = _sha256_text(_canonical_json({"request": request, "expected": expected}))
        if fixture_sha != case["fixture_sha256"]:
            raise AssertionError(f"{case['id']}: combined fixture hash mismatch.")
        fixture_hash_inputs.append(f"{case['id']}:{fixture_sha}")

    for schema in manifest["export_schemas"]:
        text = (GOLDEN_ROOT / schema["path"]).read_text(encoding="utf-8")
        if text != _canonical_json(json.loads(text)):
            raise AssertionError(f"{schema['path']}: noncanonical stored JSON.")
        observed_sha = _sha256_text(text)
        if observed_sha != schema["sha256"]:
            raise AssertionError(f"{schema['path']}: hash mismatch.")
        fixture_hash_inputs.append(f"{schema['path']}:{observed_sha}")

    observed_set_sha = _sha256_text("\n".join(fixture_hash_inputs) + "\n")
    if observed_set_sha != EXPECTED_FIXTURE_SET_SHA256:
        raise AssertionError("Frozen fixture-set SHA-256 mismatch.")


def _normalized_leaf_paths(value: object, path: str = "$") -> set[str]:
    if isinstance(value, dict):
        if not value:
            return {path}
        paths: set[str] = set()
        for key, nested in value.items():
            paths.update(_normalized_leaf_paths(nested, f"{path}.{key}"))
        return paths
    if isinstance(value, list):
        if not value:
            return {path}
        paths = set()
        for nested in value:
            paths.update(_normalized_leaf_paths(nested, f"{path}[*]"))
        return paths
    return {path}


def _path_matches(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(f"{prefix}.") or path.startswith(f"{prefix}[*]")


def _classify_leaf(case_id: str, path: str) -> str | None:
    if case_id == "B03" and _path_matches(path, "$.response.grid.effect_working"):
        return "app-owned"
    if path in APP_OWNED_EXACT_PATHS:
        return "app-owned"
    for prefix in APP_OWNED_PREFIXES:
        if _path_matches(path, prefix):
            return "app-owned"
    for prefix in CORE_OWNED_PREFIXES:
        if _path_matches(path, prefix):
            return "core-owned"
    return None


def _classify_expected_success(case_id: str, expected: dict[str, Any]) -> tuple[int, int]:
    core_count = 0
    app_count = 0
    unclassified: list[str] = []
    for path in sorted(_normalized_leaf_paths(expected)):
        owner = _classify_leaf(case_id, path)
        if owner == "core-owned":
            core_count += 1
        elif owner == "app-owned":
            app_count += 1
        else:
            unclassified.append(path)
    if unclassified:
        joined = "\n".join(unclassified[:20])
        raise AssertionError(f"{case_id}: unclassified fixture leaves:\n{joined}")
    return core_count, app_count


def _edge_summary(values: object) -> dict[str, Any]:
    items = np.asarray(values, dtype=object).reshape(-1).tolist()
    numeric = [float(value) for value in items if value is not None]
    return {
        "length": len(items),
        "first": items[0] if items else None,
        "last": items[-1] if items else None,
        "none_count": sum(value is None for value in items),
        "all_numeric_values_finite": all(math.isfinite(value) for value in numeric),
    }


def _compare(
    expected: object,
    actual: object,
    *,
    path: str,
    rtol: float,
    atol: float,
    exact_float_paths: set[str],
    stats: ComparisonStats,
) -> None:
    if isinstance(expected, dict) and set(expected) == EDGE_SUMMARY_KEYS:
        actual = _edge_summary(actual)

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"{path}: expected object, got {type(actual).__name__}.")
        if set(expected) != set(actual):
            raise AssertionError(
                f"{path}: key mismatch; expected {sorted(expected)}, got {sorted(actual)}."
            )
        for key in expected:
            _compare(
                expected[key],
                actual[key],
                path=f"{path}.{key}",
                rtol=rtol,
                atol=atol,
                exact_float_paths=exact_float_paths,
                stats=stats,
            )
        return

    if isinstance(expected, list):
        if not isinstance(actual, (list, tuple, np.ndarray)):
            raise AssertionError(f"{path}: expected array, got {type(actual).__name__}.")
        actual_items = np.asarray(actual, dtype=object).reshape(-1).tolist()
        if len(expected) != len(actual_items):
            raise AssertionError(
                f"{path}: length mismatch; expected {len(expected)}, got {len(actual_items)}."
            )
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual_items, strict=True)
        ):
            _compare(
                expected_item,
                actual_item,
                path=f"{path}[{index}]",
                rtol=rtol,
                atol=atol,
                exact_float_paths=exact_float_paths,
                stats=stats,
            )
        return

    if expected is None or isinstance(expected, (str, bool)):
        if actual != expected:
            raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}.")
        stats.compared_values += 1
        return

    if isinstance(expected, int) and not isinstance(expected, bool):
        if actual != expected:
            raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}.")
        stats.compared_values += 1
        return

    if isinstance(expected, float):
        try:
            actual_float = float(actual)
        except (TypeError, ValueError) as exc:
            raise AssertionError(f"{path}: expected float, got {actual!r}.") from exc
        if not math.isfinite(expected) or not math.isfinite(actual_float):
            raise AssertionError(f"{path}: nonfinite comparison value.")
        if path in exact_float_paths:
            if actual_float != expected:
                raise AssertionError(
                    f"{path}: exact float mismatch; expected {expected!r}, got {actual_float!r}."
                )
        elif not math.isclose(actual_float, expected, rel_tol=rtol, abs_tol=atol):
            raise AssertionError(
                f"{path}: expected {expected!r}, got {actual_float!r} (rtol={rtol}, atol={atol})."
            )
        absolute = abs(actual_float - expected)
        relative = (
            absolute / abs(expected) if expected != 0 else (0.0 if absolute == 0 else math.inf)
        )
        stats.max_absolute_difference = max(stats.max_absolute_difference, absolute)
        stats.max_relative_difference = max(stats.max_relative_difference, relative)
        stats.compared_values += 1
        return

    raise AssertionError(f"{path}: unsupported expected type {type(expected).__name__}.")


def _coerce_working_values(effect_type: str, values: object) -> list[float]:
    if values is None:
        return []
    transformed = to_working_scale(effect_type, values)
    return np.asarray(transformed, dtype=float).reshape(-1).tolist()


def _build_grid(
    request: dict[str, Any],
    expected_response: dict[str, Any],
    *,
    effect_type: str,
    estimate_working: float,
    standard_error: float,
    null_working: float,
    thresholds_working: list[float],
    critical_markers_working: tuple[float, float],
) -> np.ndarray:
    if request.get("display_range_lower") is not None:
        return np.asarray(expected_response["grid"]["effect_working"], dtype=float)

    points = int(request.get("grid_points", 801))
    if points % 2 == 0:
        points += 1
    natural_axis = bool(
        request.get("display_natural_axis", True) and EFFECT_SPECS[effect_type].family == "ratio"
    )
    max_span = max_safe_grid_span(
        theta_hat=estimate_working,
        se=standard_error,
        natural_axis_upper_bound=LOG_MAX_FLOAT if natural_axis else None,
    )
    return build_grid(
        theta_hat=estimate_working,
        se=standard_error,
        n=points,
        include_values=(null_working, *thresholds_working, *critical_markers_working),
        max_span=max_span,
    )


def _actual_success_projection(
    case_id: str,
    request: dict[str, Any],
    expected_response: dict[str, Any],
) -> dict[str, Any]:
    effect_type = str(request.get("effect_type", "odds_ratio"))
    reconstruction = reconstruct_wald_from_95_ci(
        effect_type=effect_type,
        estimate=request.get("estimate"),
        lower=request["lower"],
        upper=request["upper"],
        null_value=request.get("null_value"),
    )
    spec = reconstruction.effect_spec
    thresholds_working = _coerce_working_values(effect_type, request.get("thresholds"))
    critical_distance = legacy_critical_effect_distance(reconstruction.standard_error)
    critical_markers = legacy_critical_effect_markers(
        reconstruction.null_working,
        reconstruction.standard_error,
    )
    grid_working = _build_grid(
        request,
        expected_response,
        effect_type=effect_type,
        estimate_working=reconstruction.estimate_working,
        standard_error=reconstruction.standard_error,
        null_working=reconstruction.null_working,
        thresholds_working=thresholds_working,
        critical_markers_working=critical_markers,
    )
    point_summary = wald_point_summary(
        theta_hat=reconstruction.estimate_working,
        se=reconstruction.standard_error,
        candidate_working=reconstruction.null_working,
    )
    threshold_rows = []
    for threshold in thresholds_working:
        comparison = support_comparison(
            threshold,
            reconstruction.null_working,
            theta_hat=reconstruction.estimate_working,
            se=reconstruction.standard_error,
        )
        threshold_rows.append(
            {
                "threshold_working": comparison.candidate_working,
                "relative_likelihood": comparison.relative_likelihood,
                "log_relative_likelihood": comparison.log_relative_likelihood,
                "likelihood_ratio_mle_to_threshold": (comparison.likelihood_ratio_mle_to_candidate),
                "log_likelihood_ratio_mle_to_threshold": (
                    comparison.log_likelihood_ratio_mle_to_candidate
                ),
                "likelihood_ratio_threshold_to_null": (
                    comparison.likelihood_ratio_candidate_to_reference
                ),
                "log_likelihood_ratio_threshold_to_null": (
                    comparison.log_likelihood_ratio_candidate_to_reference
                ),
            }
        )
    interval = support_interval(
        reconstruction.estimate_working,
        reconstruction.standard_error,
    )

    grid_projection: dict[str, Any] = {
        "z": standardized_distance(
            grid_working,
            theta_hat=reconstruction.estimate_working,
            se=reconstruction.standard_error,
        ),
        "compatibility": compatibility_curve(
            grid_working,
            theta_hat=reconstruction.estimate_working,
            se=reconstruction.standard_error,
        ),
        "relative_likelihood": relative_likelihood(
            grid_working,
            theta_hat=reconstruction.estimate_working,
            se=reconstruction.standard_error,
        ),
        "log_relative_likelihood": log_relative_likelihood(
            grid_working,
            theta_hat=reconstruction.estimate_working,
            se=reconstruction.standard_error,
        ),
    }
    if case_id != "B03":
        grid_projection["effect_working"] = grid_working

    actual: dict[str, Any] = {
        "response": {
            "meta": {
                "effect_spec": asdict(spec),
                "estimate_source": reconstruction.estimate_source,
                "default_null_applied": reconstruction.default_null_applied,
                "se_method": reconstruction.se_method,
                "relative_asymmetry": reconstruction.relative_asymmetry,
                "thresholds_working": thresholds_working,
                "threshold_support_summaries": threshold_rows,
                "s_minus_2_interval": {
                    "support_cutoff": interval.support_cutoff,
                    "relative_likelihood_cutoff": interval.relative_likelihood_cutoff,
                    "likelihood_ratio_mle_to_bound": interval.likelihood_ratio_mle_to_bound,
                    "range_working": list(interval.range_working),
                },
            },
            "summary": {
                "estimate_display": reconstruction.estimate_display,
                "estimate_working": reconstruction.estimate_working,
                "ci_display": [
                    reconstruction.lower_display,
                    reconstruction.upper_display,
                ],
                "ci_working": [
                    reconstruction.lower_working,
                    reconstruction.upper_working,
                ],
                "null_display": reconstruction.null_display,
                "null_working": reconstruction.null_working,
                "working_scale_se": reconstruction.standard_error,
                "null_relative_likelihood": point_summary.null_relative_likelihood,
                "log_null_relative_likelihood": (point_summary.log_null_relative_likelihood),
                "likelihood_ratio_mle_to_null": (point_summary.likelihood_ratio_mle_to_null),
                "log_likelihood_ratio_mle_to_null": (
                    point_summary.log_likelihood_ratio_mle_to_null
                ),
                "two_sided_wald_p_value": point_summary.two_sided_wald_p_value,
                "null_z_value": point_summary.null_z_value,
                "critical_effect_markers_working": list(critical_markers),
                "critical_effect_distance_working": critical_distance,
            },
            "grid": grid_projection,
        }
    }

    if bool(request.get("design_enabled", False)):
        actual["response"]["design"] = _actual_design_projection(
            request,
            expected_response,
            reconstruction=reconstruction,
            grid_working=grid_working,
            thresholds_working=thresholds_working,
        )
    return actual


def _actual_design_projection(
    request: dict[str, Any],
    expected_response: dict[str, Any],
    *,
    reconstruction: Any,
    grid_working: np.ndarray,
    thresholds_working: list[float],
) -> dict[str, Any]:
    del thresholds_working
    raw_alpha = request.get("design_alpha", 0.05)
    alpha = 0.05 if raw_alpha is None else raw_alpha
    selection_rule = str(request.get("design_selection_rule", DEFAULT_SELECTION_RULE))
    raw_direction = str(request.get("design_claim_direction", DEFAULT_CLAIM_DIRECTION))
    if selection_rule == "one_sided_positive_p_lt_alpha":
        claim_direction = "positive"
    elif selection_rule == "one_sided_negative_p_lt_alpha":
        claim_direction = "negative"
    else:
        claim_direction = raw_direction
    information_multiplier = float(request.get("design_information_multiplier", 1.0))
    design_se = information_scaled_standard_error(
        reconstruction.standard_error,
        information_multiplier,
    )
    threshold_display = request.get("design_claim_threshold")
    threshold_working = (
        None
        if threshold_display is None
        else float(to_working_scale(reconstruction.effect_spec.key, threshold_display))
    )
    selection = selection_rule_spec(
        selection_rule=selection_rule,
        alpha=alpha,
        null_working=reconstruction.null_working,
        se=design_se,
        claim_direction=claim_direction,
        threshold_working=threshold_working,
    )
    grid_metrics = design_metrics_for_true_effects(
        grid_working,
        null_working=reconstruction.null_working,
        se=design_se,
        estimate_working=reconstruction.estimate_working,
        alpha=alpha,
        selection_rule=selection_rule,
        claim_direction=claim_direction,
        threshold_working=threshold_working,
        near_null_delta=DEFAULT_NEAR_NULL_DELTA,
    )
    scenario_probes = [
        float(row["true_effect_working"]) for row in expected_response["design"]["scenarios"]
    ]
    scenario_metrics = design_metrics_for_true_effects(
        scenario_probes,
        null_working=reconstruction.null_working,
        se=design_se,
        estimate_working=reconstruction.estimate_working,
        alpha=alpha,
        selection_rule=selection_rule,
        claim_direction=claim_direction,
        threshold_working=threshold_working,
        near_null_delta=DEFAULT_NEAR_NULL_DELTA,
    )

    target_display = request.get("design_precision_target_effect")
    precision_rows: list[dict[str, Any]] = []
    if target_display is not None:
        target_working = float(to_working_scale(reconstruction.effect_spec.key, target_display))
        target_power = request.get("design_target_power")
        results = precision_target_results(
            target_working,
            null_working=reconstruction.null_working,
            current_se=reconstruction.standard_error,
            alpha=alpha,
            target_power=0.80 if target_power is None else target_power,
            max_type_s=request.get("design_max_type_s"),
            max_type_m=request.get("design_max_type_m"),
            selection_rule=selection_rule,
            claim_direction=claim_direction,
            threshold_working=threshold_working,
            near_null_delta=DEFAULT_NEAR_NULL_DELTA,
            z975=Z975,
        )
        precision_rows = [
            {
                "target": result.target,
                "requested_value": result.requested_value,
                "target_effect_working": target_working,
                "required_se": result.required_se,
                "required_information_multiplier": result.required_information_multiplier,
                "approx_95_ci_width_working": result.approx_95_ci_width_working,
                "achieved_power": result.achieved_power,
                "achieved_type_s": result.achieved_type_s,
                "achieved_type_m": result.achieved_type_m,
                "note": result.note,
            }
            for result in results
        ]

    return {
        "config": {
            "alpha": float(alpha),
            "selection_rule": selection.key,
            "selection_rule_label": selection.label,
            "claim_direction": selection.claim_direction,
            "claim_threshold_working": threshold_working,
            "se_working": design_se,
            "current_se_working": reconstruction.standard_error,
            "design_se_working": design_se,
            "information_multiplier": information_multiplier,
            "current_ci_width_working": approximate_wald_ci_width(
                reconstruction.standard_error,
                z975=Z975,
            ),
            "approx_design_ci_width_working": approximate_wald_ci_width(
                design_se,
                z975=Z975,
            ),
            "null_working": reconstruction.null_working,
            "estimate_working": reconstruction.estimate_working,
            "near_null_delta": DEFAULT_NEAR_NULL_DELTA,
        },
        "grid": {
            "true_effect_working": [metric.true_effect_working for metric in grid_metrics],
            "delta": [metric.delta for metric in grid_metrics],
            "power": [metric.selected_claim_probability for metric in grid_metrics],
            "type_s": [metric.type_s for metric in grid_metrics],
            "type_m": [metric.type_m for metric in grid_metrics],
            "expected_selected_abs_z": [metric.expected_selected_abs_z for metric in grid_metrics],
            "observed_exaggeration": [metric.observed_exaggeration for metric in grid_metrics],
        },
        "scenarios": [
            {
                "true_effect_working": metric.true_effect_working,
                "delta": metric.delta,
                "power": metric.selected_claim_probability,
                "type_s": metric.type_s,
                "type_m": metric.type_m,
                "observed_exaggeration": metric.observed_exaggeration,
            }
            for metric in scenario_metrics
        ],
        "precision_targets": precision_rows,
    }


def _evaluate_core_error(case_id: str, request: dict[str, Any]) -> None:
    if case_id in {
        "B07d-alpha-zero",
        "B07e-alpha-one",
        "B07f-alpha-nonnumeric",
        "B07g-alpha-underflow",
    }:
        selection_rule_spec(alpha=request["design_alpha"])
        return
    expected_stub = {"grid": {}, "design": {"scenarios": []}}
    _actual_success_projection(case_id, request, expected_stub)


def verify_parity(report_path: Path) -> dict[str, Any]:
    manifest = _read_json(MANIFEST_PATH)
    _verify_corpus_integrity(manifest)

    registry_fixture = _read_json(GOLDEN_ROOT / "export_schemas" / "effect_registry.json")
    if list(EFFECT_SPECS) != registry_fixture["key_order"]:
        raise AssertionError("Effect registry key order differs from the frozen source.")
    if [asdict(spec) for spec in EFFECT_SPECS.values()] != registry_fixture["specs"]:
        raise AssertionError("Effect registry values differ from the frozen source.")

    results: list[dict[str, Any]] = []
    total_stats = ComparisonStats()
    success_count = 0
    core_error_count = 0
    exclusion_count = 0

    for case in manifest["cases"]:
        case_id = case["id"]
        request = _read_json(GOLDEN_ROOT / case["request_file"])
        expected = _read_json(GOLDEN_ROOT / case["expected_file"])
        if case_id in APP_ONLY_ERROR_CASES:
            exclusion_count += 1
            results.append(
                {
                    "case_id": case_id,
                    "status": "excluded/app-owned",
                    "reason": APP_ONLY_ERROR_CASES[case_id],
                }
            )
            continue

        if case_id in CORE_ERROR_CASES:
            try:
                _evaluate_core_error(case_id, request)
            except ValidationError as exc:
                if expected["error_type"] != "ValidationError":
                    raise AssertionError(f"{case_id}: unexpected frozen error type.") from exc
                if str(exc) != expected["message"]:
                    raise AssertionError(
                        f"{case_id}: expected error {expected['message']!r}, got {str(exc)!r}."
                    ) from exc
            else:
                raise AssertionError(f"{case_id}: expected ValidationError was not raised.")
            core_error_count += 1
            results.append(
                {
                    "case_id": case_id,
                    "status": "matched/core-error",
                    "message": expected["message"],
                }
            )
            continue

        core_leaves, app_leaves = _classify_expected_success(case_id, expected)
        actual = _actual_success_projection(case_id, request, expected["response"])
        stats = ComparisonStats()
        _compare(
            expected={"response": _core_expected_projection(case_id, expected["response"])},
            actual=actual,
            path="$",
            rtol=float(case["tolerance"]["rtol"]),
            atol=float(case["tolerance"]["atol"]),
            exact_float_paths=set(case["exact_float_paths"]),
            stats=stats,
        )
        total_stats.compared_values += stats.compared_values
        total_stats.max_absolute_difference = max(
            total_stats.max_absolute_difference,
            stats.max_absolute_difference,
        )
        total_stats.max_relative_difference = max(
            total_stats.max_relative_difference,
            stats.max_relative_difference,
        )
        success_count += 1
        results.append(
            {
                "case_id": case_id,
                "status": "matched/core-values",
                "core_leaf_patterns": core_leaves,
                "app_leaf_patterns_excluded": app_leaves,
                **asdict(stats),
            }
        )

    if (success_count, core_error_count, exclusion_count) != (14, 6, 2):
        raise AssertionError(
            "Expected 14 successful cases, 6 core errors, and 2 app-only exclusions."
        )

    report = {
        "schema_version": 1,
        "verdict": "pass",
        "behavior_source": EXPECTED_BEHAVIOR_SOURCE,
        "baseline_tag": "pre-split-baseline-2026-07-29",
        "baseline_tag_target": "5fd501dd947d9b951d736014cfc2b310efa5e7b0",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "fixture_set_sha256": EXPECTED_FIXTURE_SET_SHA256,
        "tolerance": manifest["default_tolerance"],
        "summary": {
            "successful_cases": success_count,
            "matched_core_error_cases": core_error_count,
            "app_only_error_exclusions": exclusion_count,
            **asdict(total_stats),
        },
        "app_only_path_prefixes": list(APP_OWNED_PREFIXES),
        "cases": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_canonical_json(report), encoding="utf-8")
    print(
        "Baseline parity passed: "
        f"{success_count} success cases, {core_error_count} core errors, "
        f"{exclusion_count} explicit app exclusions; "
        f"{total_stats.compared_values} values compared."
    )
    print(
        "Maximum differences: "
        f"absolute={total_stats.max_absolute_difference:.17g}, "
        f"relative={total_stats.max_relative_difference:.17g}."
    )
    print(f"Machine-readable report: {report_path}")
    return report


def _core_expected_projection(case_id: str, expected_response: dict[str, Any]) -> dict[str, Any]:
    def pick(mapping: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
        return {key: mapping[key] for key in keys}

    meta = expected_response["meta"]
    summary = dict(expected_response["summary"])
    summary.pop("critical_effect_markers_display")
    grid_keys = (
        "z",
        "compatibility",
        "relative_likelihood",
        "log_relative_likelihood",
    )
    if case_id != "B03":
        grid_keys = ("effect_working", *grid_keys)
    projection: dict[str, Any] = {
        "meta": {
            "effect_spec": meta["effect_spec"],
            "estimate_source": meta["estimate_source"],
            "default_null_applied": meta["default_null_applied"],
            "se_method": meta["se_method"],
            "relative_asymmetry": meta["relative_asymmetry"],
            "thresholds_working": meta["thresholds_working"],
            "threshold_support_summaries": [
                pick(
                    row,
                    (
                        "threshold_working",
                        "relative_likelihood",
                        "log_relative_likelihood",
                        "likelihood_ratio_mle_to_threshold",
                        "log_likelihood_ratio_mle_to_threshold",
                        "likelihood_ratio_threshold_to_null",
                        "log_likelihood_ratio_threshold_to_null",
                    ),
                )
                for row in meta["threshold_support_summaries"]
            ],
            "s_minus_2_interval": pick(
                meta["s_minus_2_interval"],
                (
                    "support_cutoff",
                    "relative_likelihood_cutoff",
                    "likelihood_ratio_mle_to_bound",
                    "range_working",
                ),
            ),
        },
        "summary": summary,
        "grid": pick(expected_response["grid"], grid_keys),
    }
    if expected_response["design"] is not None:
        design = expected_response["design"]
        projection["design"] = {
            "config": pick(
                design["config"],
                (
                    "alpha",
                    "selection_rule",
                    "selection_rule_label",
                    "claim_direction",
                    "claim_threshold_working",
                    "se_working",
                    "current_se_working",
                    "design_se_working",
                    "information_multiplier",
                    "current_ci_width_working",
                    "approx_design_ci_width_working",
                    "null_working",
                    "estimate_working",
                    "near_null_delta",
                ),
            ),
            "grid": pick(
                design["grid"],
                (
                    "true_effect_working",
                    "delta",
                    "power",
                    "type_s",
                    "type_m",
                    "expected_selected_abs_z",
                    "observed_exaggeration",
                ),
            ),
            "scenarios": [
                pick(
                    row,
                    (
                        "true_effect_working",
                        "delta",
                        "power",
                        "type_s",
                        "type_m",
                        "observed_exaggeration",
                    ),
                )
                for row in design["scenarios"]
            ],
            "precision_targets": [
                pick(
                    row,
                    (
                        "target",
                        "requested_value",
                        "target_effect_working",
                        "required_se",
                        "required_information_multiplier",
                        "approx_95_ci_width_working",
                        "achieved_power",
                        "achieved_type_s",
                        "achieved_type_m",
                        "note",
                    ),
                )
                for row in design["precision_targets"]
            ],
        }
    return projection


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify core-owned values against the frozen integrated baseline."
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"Machine-readable report path (default: {DEFAULT_REPORT_PATH}).",
    )
    args = parser.parse_args()
    verify_parity(args.json_output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
