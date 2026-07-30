#!/usr/bin/env python3
"""Install the release wheel outside the checkout and exercise public APIs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from check_distribution import WHEEL_NAME

SMOKE_CODE = textwrap.dedent(
    """
    from __future__ import annotations

    import math
    import sys
    from dataclasses import asdict
    from importlib.metadata import version
    from pathlib import Path

    import numpy as np
    import wald_inference
    from wald_inference import legacy
    from wald_inference import (
        ValidationError,
        compatibility_curve,
        critical_effect_for_target_probability,
        design_metrics_for_true_effects,
        from_working_scale,
        get_effect_spec,
        joint_precision_result,
        log_support_ratio,
        precision_sensitivity,
        precision_target_results,
        power_curve,
        reconstruct_wald_from_95_ci,
        relative_likelihood,
        selected_claim_probability,
        support_interval_for_ratio,
        support_ratio,
    )


    def assert_finite_or_none(value: object) -> None:
        if value is None or isinstance(value, (str, bool)):
            return
        if isinstance(value, (int, float)):
            assert math.isfinite(float(value)), value
            return
        if isinstance(value, dict):
            for nested in value.values():
                assert_finite_or_none(nested)
            return
        if isinstance(value, (list, tuple)):
            for nested in value:
                assert_finite_or_none(nested)
            return
        raise AssertionError(f"unexpected smoke value type: {type(value)!r}")


    assert version("wald-inference") == "0.4.1"
    assert wald_inference.__version__ == "0.4.1"
    assert wald_inference.__all__
    for exported_name in wald_inference.__all__:
        assert hasattr(wald_inference, exported_name), exported_name
    assert legacy.__all__ == [
        "ASYMMETRY_RELATIVE_TOLERANCE",
        "DEFAULT_GRID_POINTS",
        "DEFAULT_SOLVER_TOLERANCE",
        "DEFAULT_SPAN_MULTIPLIER",
        "ESTIMATE_MATCH_ABSOLUTE_TOLERANCE",
        "ESTIMATE_MATCH_RELATIVE_TOLERANCE",
        "GRID_EXPANSION_PADDING_MULTIPLIER",
        "LOG_MAX_FLOAT",
        "MAX_FINITE_ABS_Z",
        "MAX_FINITE_SPAN",
        "MAX_FLOAT",
        "MAX_INFORMATION_MULTIPLIER",
        "Z80",
        "Z975",
        "asymmetry_warning",
        "build_grid",
        "confidence_curve",
        "estimate_se",
        "from_working_scale",
        "log_relative_likelihood",
        "relative_likelihood",
        "summaries",
        "to_working_scale",
    ]
    for exported_name in legacy.__all__:
        assert hasattr(legacy, exported_name), exported_name

    module_path = Path(wald_inference.__file__).resolve()
    environment_root = Path(sys.prefix).resolve()
    assert module_path.is_relative_to(environment_root), (module_path, environment_root)
    assert get_effect_spec("odds_ratio").default_null == 1.0

    reconstruction = reconstruct_wald_from_95_ci(
        "mean_difference",
        lower=0.11,
        upper=0.73,
    )
    assert math.isclose(reconstruction.estimate_working, 0.42)
    points = np.asarray(
        [
            reconstruction.lower_working,
            reconstruction.estimate_working,
            reconstruction.upper_working,
        ]
    )
    compatibility = compatibility_curve(
        points,
        reconstruction.estimate_working,
        reconstruction.standard_error,
    )
    likelihood = relative_likelihood(
        points,
        reconstruction.estimate_working,
        reconstruction.standard_error,
    )
    assert math.isclose(float(compatibility[0]), 0.05, rel_tol=1e-12, abs_tol=1e-14)
    assert math.isclose(float(compatibility[1]), 1.0)
    assert math.isclose(float(compatibility[2]), 0.05, rel_tol=1e-12, abs_tol=1e-14)
    assert math.isclose(float(likelihood[1]), 1.0)

    support_interval = support_interval_for_ratio(
        reconstruction.estimate_working,
        reconstruction.standard_error,
        mle_to_bound_ratio=4.0,
    )
    assert math.isclose(support_interval.likelihood_ratio_mle_to_bound, 4.0)
    for endpoint in support_interval.range_working:
        endpoint_log_ratio = float(
            log_support_ratio(
                reconstruction.estimate_working,
                endpoint,
                theta_hat=reconstruction.estimate_working,
                se=reconstruction.standard_error,
            )
        )
        assert math.isclose(endpoint_log_ratio, math.log(4.0), rel_tol=1e-12)
    assert math.isclose(
        support_ratio(
            reconstruction.estimate_working,
            support_interval.lower_working,
            theta_hat=reconstruction.estimate_working,
            se=reconstruction.standard_error,
        ),
        4.0,
        rel_tol=1e-12,
    )
    assert (
        support_ratio(
            0.0,
            40.0,
            theta_hat=0.0,
            se=1.0,
        )
        is None
    )
    assert float(log_support_ratio(0.0, 40.0, theta_hat=0.0, se=1.0)) == 800.0
    try:
        from_working_scale("odds_ratio", -746.0)
    except ValidationError as exc:
        assert "representable as strictly positive" in str(exc)
    else:
        raise AssertionError("underflowing ratio back-transform did not raise ValidationError")
    assert legacy.from_working_scale("odds_ratio", -746.0) == 0.0
    try:
        support_interval_for_ratio(
            float.fromhex("0x1.1ccf385ebc8a0p+1023"),
            1.0183045837972807e292,
            mle_to_bound_ratio=4.0,
        )
    except ValidationError as exc:
        assert "cannot represent the requested log-relative-likelihood cutoff" in str(exc)
    else:
        raise AssertionError("unrepresentable support boundary did not raise ValidationError")

    design = design_metrics_for_true_effects(
        [0.0, 0.3],
        null_working=reconstruction.null_working,
        se=reconstruction.standard_error,
        estimate_working=reconstruction.estimate_working,
    )
    assert math.isclose(design[0].selected_claim_probability, 0.05)
    assert design[0].type_s is None
    assert design[0].type_m is None
    assert design[1].type_s is not None
    assert design[1].type_m is not None

    probabilities = power_curve(
        [0.0, 0.3],
        null_working=reconstruction.null_working,
        standard_error=reconstruction.standard_error,
    )
    assert math.isclose(float(probabilities[0]), 0.05)
    assert math.isclose(
        float(probabilities[1]),
        selected_claim_probability(
            0.3,
            null_working=reconstruction.null_working,
            standard_error=reconstruction.standard_error,
        ),
    )
    critical_effect = critical_effect_for_target_probability(
        null_working=reconstruction.null_working,
        standard_error=reconstruction.standard_error,
        alpha=0.05,
        target_probability=0.8,
    )
    assert critical_effect.critical_delta > 0
    assert math.isclose(critical_effect.achieved_probability, 0.8)

    precision = precision_target_results(
        0.2,
        null_working=reconstruction.null_working,
        current_se=reconstruction.standard_error,
        target_power=0.8,
    )
    assert len(precision) == 1
    assert precision[0].required_se is not None
    assert precision[0].required_information_multiplier is not None
    assert precision[0].feasible
    assert (
        precision[0].achieved_selected_claim_probability
        == precision[0].achieved_power
    )

    joint_precision = joint_precision_result(
        0.2,
        null_working=reconstruction.null_working,
        current_se=reconstruction.standard_error,
        target_power=0.8,
        max_type_s=0.01,
        max_type_m=1.25,
    )
    assert joint_precision.feasible
    assert joint_precision.required_se == min(
        row.required_se
        for row in joint_precision.target_results
        if row.required_se is not None
    )
    assert joint_precision.binding_targets

    current_sufficient_joint = joint_precision_result(
        1.0,
        null_working=0.0,
        current_se=0.1,
        target_power=0.8,
        max_type_s=0.01,
        max_type_m=1.25,
    )
    assert current_sufficient_joint.current_precision_sufficient
    assert current_sufficient_joint.required_information_multiplier == 1.0

    infeasible_joint_precision = joint_precision_result(
        0.0,
        null_working=reconstruction.null_working,
        current_se=reconstruction.standard_error,
        max_type_m=1.25,
    )
    assert not infeasible_joint_precision.feasible
    assert infeasible_joint_precision.required_se is None
    assert len(infeasible_joint_precision.target_results) == 1

    sensitivity = precision_sensitivity(
        [0.2, 0.0, 0.4],
        null_working=reconstruction.null_working,
        current_se=reconstruction.standard_error,
        target_power=0.8,
    )
    assert [item.true_effect_working for item in sensitivity] == [0.2, 0.0, 0.4]
    assert [item.feasible for item in sensitivity] == [True, False, True]

    assert_finite_or_none(asdict(reconstruction))
    for item in design:
        assert_finite_or_none(asdict(item))
    assert_finite_or_none(asdict(critical_effect))
    for item in precision:
        assert_finite_or_none(asdict(item))
    assert_finite_or_none(asdict(joint_precision))
    assert_finite_or_none(asdict(current_sufficient_joint))
    assert_finite_or_none(asdict(infeasible_joint_precision))
    for item in sensitivity:
        assert_finite_or_none(asdict(item))

    try:
        get_effect_spec("not-an-effect")
    except ValidationError:
        pass
    else:
        raise AssertionError("invalid effect type did not raise ValidationError")

    print("Cold-wheel public API smoke passed.")
    """
)


def _venv_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def smoke_wheel(dist_dir: Path) -> None:
    wheel = dist_dir / WHEEL_NAME
    if not wheel.is_file():
        raise RuntimeError(f"missing expected wheel: {wheel}")

    with tempfile.TemporaryDirectory(prefix="wald-inference-smoke-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        execution_directory = root / "empty-workdir"
        execution_directory.mkdir()
        subprocess.run(
            [
                "uv",
                "venv",
                "--no-project",
                "--python",
                sys.executable,
                str(environment),
            ],
            cwd=execution_directory,
            check=True,
        )
        python = _venv_python(environment)
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--no-cache",
                str(wheel),
            ],
            cwd=execution_directory,
            check=True,
        )
        subprocess.run(
            ["uv", "pip", "check", "--python", str(python)],
            cwd=execution_directory,
            check=True,
        )
        smoke_environment = os.environ.copy()
        smoke_environment.pop("PYTHONPATH", None)
        subprocess.run(
            [str(python), "-c", SMOKE_CODE],
            cwd=execution_directory,
            env=smoke_environment,
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        smoke_wheel(args.dist_dir.resolve())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
