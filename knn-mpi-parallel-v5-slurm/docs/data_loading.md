# Data Loading Modes

The digits dataset is always split before augmentation. The loader first runs a
stratified `train_test_split` on the original sklearn digits samples, then
augments the train and test splits independently. Each augmented sample keeps the
origin id of the original digit it came from, and the loader asserts that train
and test origin ids are disjoint.

Percentage mode is meant for accuracy and correctness experiments:

```bash
python src/knn_mpi_v5.py --n-total 100000 --test-size 0.2 --k 3
```

This produces final train/test sizes from the requested percentage. The legacy
`--n` flag is kept as an alias for `--n-total`.

Fixed-size mode is meant for performance and scaling experiments:

```bash
python src/knn_mpi_v5.py --n-train 200000 --n-test 5000 --orig-test-size 0.2 --k 3
```

In fixed-size mode, `orig-test-size` controls the representative holdout taken
from the original dataset before augmentation. It does not use the final
`n_test / (n_train + n_test)` ratio, because that can be too small for a useful
original test holdout.
