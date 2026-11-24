"""Compare Bell412 and VPAIR relative rotation errors."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


RESULTS_DIR = Path(__file__).resolve().parent
SEQUENCES = ("3", "4", "5")
GROUPS = (("Bell412", ""), ("3x Scaled Bell412", "_scale"))
HYBRID_JOIN_METHOD = (
    "ROVO (Hybrid Join)", "rovo", "rovo_{sequence}{suffix}_hybridjoin.txt",
    "vpair_hybridjoin.txt", "#c43c39", "-",
)
BASELINE_METHODS = (
    ("ORB-SLAM3", "orbslam3", "orbslam_{sequence}{suffix}.txt", "vpair.txt",
     "#2a9d8f", "-"),
    ("DPVO", "dpvo", "dpvo_{sequence}{suffix}.txt", "vpair.txt",
     "#2878b5", "--"),
)
ROVO_ABLATION_METHODS = (
    ("ROVO (5-point)", "rovo", "rovo_{sequence}{suffix}_fivepoint.txt",
     "vpair_fivepoint.txt", "#e07a28", "-."),
    ("ROVO (Eigensolver)", "rovo", "rovo_{sequence}{suffix}_eigensolver.txt",
     "vpair_eigensolver.txt", "#8e5ea2", ":"),
    HYBRID_JOIN_METHOD,
)
METHODS = BASELINE_METHODS + (HYBRID_JOIN_METHOD,)
OVERWRITE_METHOD = (
    "ROVO (Hybrid Join Overwrite)", "rovo", "rovo_{sequence}{suffix}_hybridjoin_overwrite.txt",
    "vpair_hybridjoin_overwrite.txt", "#4b6334", "--",
)
MAX_TIME_DIFF = 0.05
OUTPUT_DIR = RESULTS_DIR / "rotation_evaluation" / "curves"


def _load_tum(file_path):
    path = Path(file_path)
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.size == 0 or data.shape[1] != 8 or not np.all(np.isfinite(data)):
        raise ValueError(f"{path}: expected finite timestamp tx ty tz qx qy qz qw rows")
    quaternion_norms = np.linalg.norm(data[:, 4:8], axis=1)
    if np.any(quaternion_norms < 1e-12):
        raise ValueError(f"{path}: zero quaternion")
    data[:, 4:8] /= quaternion_norms[:, None]
    return {"t": data[:, 0], "q": data[:, 4:8]}


def associate_timestamps(reference_times, estimate_times, max_time_diff=MAX_TIME_DIFF):
    """Match the shorter trajectory to the longer one as in evo.core.sync."""
    reference = np.asarray(reference_times, dtype=float)
    estimate = np.asarray(estimate_times, dtype=float)
    if not np.isfinite(max_time_diff) or max_time_diff < 0:
        raise ValueError("max_time_diff must be finite and nonnegative")
    reference_is_shorter = len(reference) < len(estimate)
    shorter, longer = (reference, estimate) if reference_is_shorter else (estimate, reference)
    short_ids, long_ids = [], []
    for i, stamp in enumerate(shorter):
        differences = np.abs(longer - stamp)
        j = int(np.argmin(differences))
        if differences[j] <= max_time_diff:
            short_ids.append(i)
            long_ids.append(j)
    matched = np.asarray(short_ids, dtype=int), np.asarray(long_ids, dtype=int)
    return matched if reference_is_shorter else matched[::-1]


def _pair_errors(reference, estimate, ref_start, ref_end, est_start, est_end):
    ref_delta = R.from_quat(reference["q"][ref_start]).inv() * R.from_quat(reference["q"][ref_end])
    est_delta = R.from_quat(estimate["q"][est_start]).inv() * R.from_quat(estimate["q"][est_end])
    angles = np.degrees((ref_delta.inv() * est_delta).magnitude())
    return {"rre_deg": angles, "rmse_deg": float(np.sqrt(np.mean(angles ** 2)))}


def compute_rre(reference, estimate, max_time_diff=MAX_TIME_DIFF):
    """Rotation-angle RPE in degrees for consecutive synchronized poses."""
    ref_ids, est_ids = associate_timestamps(reference["t"], estimate["t"], max_time_diff)
    if len(ref_ids) < 2:
        raise ValueError(f"RRE requires two associated poses, found {len(ref_ids)}")
    result = _pair_errors(reference, estimate, ref_ids[:-1], ref_ids[1:], est_ids[:-1], est_ids[1:])
    result.update(timestamps=estimate["t"][est_ids[1:]], reference_ids=ref_ids)
    return result


def compute_adjacent_rre(reference, estimate, max_time_diff=MAX_TIME_DIFF):
    """Score adjacent reference frames within the same tracking segment."""
    ref_ids, est_ids = associate_timestamps(reference["t"], estimate["t"], max_time_diff)
    valid = (np.diff(ref_ids) == 1) & (np.diff(est_ids) > 0)
    if "segment_ids" in estimate:
        segments = estimate["segment_ids"][est_ids]
        valid &= (segments[:-1] == segments[1:]) & (segments[1:] > 0)
    ends = ref_ids[1:][valid]
    if not len(ends):
        raise ValueError("No valid adjacent reference-frame pairs")
    result = _pair_errors(reference, estimate, ends - 1, ends,
                          est_ids[:-1][valid], est_ids[1:][valid])
    result.update(timestamps=reference["t"][ends], pair_end_ids=ends, reference_ids=ref_ids)
    return result


def shared_pair_errors(methods):
    """Restrict each method to the intersection of its valid reference pairs."""
    shared = set.intersection(*(set(method["pair_end_ids"]) for method in methods.values()))
    if not shared:
        raise ValueError("No adjacent reference-frame pairs shared by all methods")
    ends = sorted(shared)
    results = {}
    for name, method in methods.items():
        keep = np.isin(method["pair_end_ids"], ends)
        errors = method["rre_deg"][keep]
        results[name] = dict(method, rre_deg=errors, rmse_deg=float(np.sqrt(np.mean(errors ** 2))),
                             timestamps=method["timestamps"][keep], pair_end_ids=method["pair_end_ids"][keep])
    return results


def compute_bell_overlap(reference, estimates, max_time_diff=MAX_TIME_DIFF):
    """Evaluate the same adjacent reference-frame pairs for every Bell method."""
    return shared_pair_errors({name: compute_adjacent_rre(reference, estimate, max_time_diff)
                               for name, estimate in estimates.items()})


def _load_segment_ids(path, timestamps, is_vpair):
    time_key = "nominal_time_s" if is_vpair else "timestamp_s"
    metadata_path = path.with_stem(f"{path.stem}_frames").with_suffix(".csv")
    with metadata_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["estimated"])]
    times = [float(row[time_key]) for row in rows]
    # TUM exports round timestamps to six decimal places.
    if len(times) != len(timestamps) or not np.allclose(times, timestamps, rtol=0, atol=1e-6):
        raise ValueError(f"{metadata_path}: estimated timestamps do not match {path.name}")
    return np.asarray([int(row["segment_id"]) for row in rows])


def failure_intervals(timestamps, valid):
    """Return VPAIR time spans containing reference samples without ORB poses."""
    times = np.asarray(timestamps, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if times.shape != valid.shape or not len(times):
        raise ValueError("timestamps and valid must be nonempty arrays of equal length")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("Failure-interval timestamps must be strictly increasing")
    missing = np.flatnonzero(~valid)
    if not len(missing):
        return np.empty((0, 2), dtype=float)
    runs = np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1)
    intervals = []
    for run in runs:
        first, last = int(run[0]), int(run[-1])
        start = times[first] if first == 0 else (times[first - 1] + times[first]) / 2
        end = times[last] if last == len(times) - 1 else (times[last] + times[last + 1]) / 2
        intervals.append((start, end))
    return np.asarray(intervals, dtype=float)


def mark_failure_intervals(axis, intervals, label="ORB-SLAM3 failed"):
    for index, (start, end) in enumerate(intervals):
        axis.axvspan(start, end, color="#e63946", alpha=.16,
                     label=label if index == 0 else "_nolegend_", zorder=0)


def _load_run(results_dir, sequence, title, suffix, max_time_diff, include_overwrite=False):
    is_vpair = sequence == "vpair"
    reference_path = (results_dir / "vpair_gt" / "vpair_gt.txt" if is_vpair else
                      results_dir / "bell412_gt" / f"bell412_{sequence}_gt.txt")
    reference = _load_tum(reference_path)
    run = {"title": title, "times": reference["t"] - reference["t"][0], "algorithms": {}}
    estimates = {}
    styles = {}
    methods = ROVO_ABLATION_METHODS + (OVERWRITE_METHOD,) if include_overwrite else METHODS
    for method in methods:
        name, folder, bell_filename, vpair_filename, color, linestyle = method
        filename = vpair_filename if is_vpair else bell_filename.format(
            sequence=sequence, suffix=suffix)
        path = results_dir / folder / filename
        if method == HYBRID_JOIN_METHOD and sequence == "5" and suffix == "_scale":
            rerun = path.with_stem(f"{path.stem}_rerun")
            if (not path.is_file() or path.stat().st_size == 0) and rerun.is_file():
                path = rerun
        estimate = _load_tum(path)
        if folder == "rovo":
            estimate["segment_ids"] = _load_segment_ids(path, estimate["t"], is_vpair)
        estimates[name] = estimate
        styles[name] = {"color": color, "linestyle": linestyle}
    if is_vpair and not include_overwrite:
        run["algorithms"] = {name: compute_adjacent_rre(reference, estimate, max_time_diff)
                             for name, estimate in estimates.items()}
        tracked = np.zeros(len(reference["t"]), dtype=bool)
        tracked[run["algorithms"]["ORB-SLAM3"]["reference_ids"]] = True
        run["failure_intervals"] = failure_intervals(run["times"], tracked)
    else:
        run["algorithms"] = compute_bell_overlap(reference, estimates, max_time_diff)
    for name, style in styles.items():
        run["algorithms"][name].update(style)
    run["time_origin"] = reference["t"][0]
    return run


def _plot_rre(axis, run):
    for name, method in run["algorithms"].items():
        times = method["timestamps"] - run["time_origin"]
        errors = method["rre_deg"]
        if "pair_end_ids" in method:
            gaps = np.flatnonzero(np.diff(method["pair_end_ids"]) != 1) + 1
            times = np.insert(times, gaps, np.nan)
            errors = np.insert(errors, gaps, np.nan)
        axis.plot(times, errors,
                  color=method["color"], linestyle=method["linestyle"], linewidth=.9,
                  label=f"{name}: RMSE {method['rmse_deg']:.2f} deg")
    axis.set_ylabel("RRE (deg)")


@plt.rc_context({"font.family": "Times New Roman", "font.size": 14,
                 "axes.titlesize": 14, "figure.titlesize": 14, "legend.fontsize": 10,
                 "axes.titleweight": "bold", "figure.titleweight": "bold"})
def plot_comparison_figures(bell_runs, vpair_run, output_dir):
    """Save exactly one figure for each requested rotation curve."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(16, 10))
    grid = figure.add_gridspec(3, 3, height_ratios=(1, 1, 1.1), hspace=.35, wspace=.25)
    for index, run in enumerate(bell_runs):
        axis = figure.add_subplot(grid[index // 3, index % 3])
        axis.set_title(run["title"])
        _plot_rre(axis, run)
        axis.set_xlabel("Time from reference start (s)")
        axis.grid(True, alpha=.25, linestyle="--")
        axis.legend(loc="best")
    axis = figure.add_subplot(grid[2, :])
    axis.set_title("VPAIR")
    if "failure_intervals" in vpair_run:
        mark_failure_intervals(axis, vpair_run["failure_intervals"])
    _plot_rre(axis, vpair_run)
    axis.set_xlabel("Time from first image (s, 1 Hz)")
    axis.set_xlim(vpair_run["times"][0], vpair_run["times"][-1])
    axis.grid(True, alpha=.25, linestyle="--")
    axis.legend(loc="best", ncol=2)
    figure.suptitle("Relative Rotation Error")
    figure.subplots_adjust(left=.07, right=.99, bottom=.07, top=.92,
                           hspace=.38, wspace=.28)
    path = output_dir / "comparison_rre.png"
    figure.savefig(path, dpi=500, bbox_inches="tight")
    return [figure], [path]


def run_comparison(results_dir=RESULTS_DIR, output_dir=None,
                   sequences=SEQUENCES, groups=GROUPS,
                   max_time_diff=MAX_TIME_DIFF, include_overwrite=False):
    results_dir = Path(results_dir)
    bell_runs = []
    for group_name, suffix in groups:
        for sequence in sequences:
            bell_runs.append(_load_run(results_dir, sequence, f"{group_name}_{sequence}",
                                       suffix, max_time_diff, include_overwrite))
    vpair_run = _load_run(results_dir, "vpair", "VPAIR", "", max_time_diff,
                          include_overwrite)
    if output_dir is None:
        output_dir = OUTPUT_DIR.parent / "ablation" if include_overwrite else OUTPUT_DIR
    figures, paths = plot_comparison_figures(bell_runs, vpair_run, output_dir)
    return bell_runs, vpair_run, figures, paths


def write_metrics(path, bell_runs, vpair_run, include_overwrite):
    """Write the scores and tracking coverage used in the flight tables."""
    comparisons = [(run, "shared", run["algorithms"]) for run in bell_runs]
    methods = vpair_run["algorithms"]
    if include_overwrite:
        comparisons.append((vpair_run, "shared", methods))
    else:
        comparisons.extend([
            (vpair_run, "own", methods),
            (vpair_run, "shared", shared_pair_errors(methods)),
            (vpair_run, "shared_dpvo_rovo", shared_pair_errors(
                {name: methods[name] for name in ("DPVO", HYBRID_JOIN_METHOD[0])})),
        ])

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["sequence", "pair_set", "method", "reference_frames",
                         "tracked_frames", "coverage_percent", "valid_pairs", "rmse_deg"])
        for run, pair_set, methods in comparisons:
            reference_count = len(run["times"])
            for name, method in methods.items():
                tracked = len(np.unique(method["reference_ids"]))
                pairs = len(method["rre_deg"])
                rmse = method["rmse_deg"]
                writer.writerow([run["title"], pair_set, name, reference_count, tracked,
                                 f"{100 * tracked / reference_count:.6f}", pairs, f"{rmse:.6f}"])
                print(f"{run['title']} ({pair_set}) {name}: {pairs} pairs, RMSE {rmse:.6f} deg")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening windows")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-time-diff", type=float, default=MAX_TIME_DIFF)
    parser.add_argument("--ablation", "--include-overwrite", dest="ablation", action="store_true",
                        help="Compare the four ROVO variants on shared reference pairs")
    options = parser.parse_args(argv)
    if options.max_time_diff < 0 or not np.isfinite(options.max_time_diff):
        parser.error("--max-time-diff must be finite and nonnegative")
    if options.no_show:
        plt.switch_backend("Agg")

    bell_runs, vpair_run, figures, paths = run_comparison(
        output_dir=options.output_dir, max_time_diff=options.max_time_diff,
        include_overwrite=options.ablation)
    metrics_path = paths[0].with_name("metrics.csv")
    write_metrics(metrics_path, bell_runs, vpair_run, options.ablation)
    for path in [*paths, metrics_path]:
        print(f"Saved {path.resolve()}")
    if not options.no_show:
        plt.show()
    for figure in figures:
        plt.close(figure)
    return bell_runs, vpair_run


if __name__ == "__main__":
    main()
