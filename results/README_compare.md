# Flight result evaluation

Run the following commands from the repository root.

```sh
python -m pip install numpy scipy matplotlib
python results/compare.py --no-show
python results/compare.py --no-show --ablation
```

The default run evaluates ORB-SLAM3, DPVO, and ROVO (Hybrid Join) and writes `metrics.csv` and `comparison_rre.png` to `results/rotation_evaluation/curves/`. The ablation compares the five-point, eigensolver, Hybrid Join, and Hybrid Join Overwrite variants and writes the same filenames to `results/rotation_evaluation/ablation/`. Overwrite directly replaces stored orientations with the aligned window estimates. The other three variants fuse aligned orientations with the stored estimates. `--output-dir` changes the destination. `--max-time-diff` changes the default tolerance of 0.05 seconds.

## Pairing and metrics

The evaluator associates poses by nearest timestamp within the tolerance, following evo's timestamp association convention. Eligible pairs have adjacent reference frames and two distinct associated poses in temporal order. ROVO pairs also require a shared tracking segment. The relative rotation error (RRE) is the angle in degrees of the reference relative rotation inverse composed with the estimated relative rotation. RMSE is the root mean square of those angles. [evo](https://github.com/MichaelGrupp/evo)

- Bell412 comparison and ablation scores use the intersection of eligible reference pairs for the methods in each run.
- VPAIR comparison scores include each method’s own pairs (`own`), the intersection across all three methods (`shared`), and the intersection specific to DPVO and ROVO (`shared_dpvo_rovo`). Coverage counts distinct matched query frames. VPAIR ablation scores use the same 2,667 pairs for all four variants.
- The released Bell412-5 triple-focal-length Hybrid Join trajectory is `rovo/rovo_5_scale_hybridjoin_rerun.txt`. Its 280 shared comparison pairs give 0.779609° RMSE. The manuscript’s ablation reports 0.7796°.

## Inputs and sources

- `rovo/`, `orbslam3/`, and `dpvo/` contain estimated trajectories in TUM format (`timestamp tx ty tz qx qy qz qw`). ROVO uses zero translation placeholders. Its `*_frames.csv` files identify which frames received an estimate and their tracking segment IDs.
- `bell412_gt/` contains the benchmark references for Bell412 flights 3, 4, and 5. The same references apply to the original and triple-focal-length images. [MUN-FRL dataset paper](https://doi.org/10.1177/02783649241238358)
- `vpair_gt/vpair_gt.txt` contains an attitude reference derived from the published VPAIR roll, pitch, and yaw using the ZYX convention in local NED. Times are nominal seconds from the first query (1 Hz). Translation fields are zero placeholders. The evaluation follows the released attitude convention, with no independently verified optical-camera/IMU extrinsic. VPAIR data are for academic use under the source dataset terms. Cite Schleiss, Rouatbi, and Cremers (2022) when using the reference. [VPAIR dataset and license](https://github.com/AerVisLoc/vpair)
