# omicsmad

A clean, minimal package for filtering genomics datasets (UCSC Xena bulk expression, CNV arrays) using median absolute deviation.

## Installation

```bash
pip install omicsmad
```

## Usage

```python
import numpy as np
import pandas as pd
from omicsmad import OmicsMADSelector

# mocking data matrices
np.random.seed(42)
X_train = pd.DataFrame(np.random.randn(100, 5000), columns=[f"Gene_{i}" for i in range(5000)])

selector = OmicsMADSelector(top_n=1000)
selector.fit(X_train)

# saves to 'figures/mad_distribution.png'
selector.plot_distributions()

# saves processed matrix to 'datasets/processed/filtered_features.tsv'
selector.save_filtered_data(X_train)
```