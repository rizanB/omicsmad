import pytest
import pandas as pd
import numpy as np
from omicsmad import OmicsMADSelector

@pytest.fixture
def mock_genomics_data():
    """Make a test gene expression dataframe (10 samples, 50 genes)."""
    np.random.seed(42)
    
    # 45 random genes and 5 'flat' genes with no variance
    data = np.random.randn(10, 45)
    flat_data = np.zeros((10, 5))
    
    combined_data = np.hstack([data, flat_data])
    gene_names = [f"Gene_{i}" for i in range(50)]
    sample_names = [f"Sample_{i}" for i in range(10)]
    
    return pd.DataFrame(combined_data, index=sample_names, columns=gene_names)

def test_selector_feature_reduction(mock_genomics_data):
    """Test if selector cuts down to top_n columns."""
    top_n = 20
    selector = OmicsMADSelector(top_n=top_n)
    
    # run fit_transform
    X_filtered = selector.fit_transform(mock_genomics_data)
    
    # assertions
    assert X_filtered.shape[0] == mock_genomics_data.shape[0]  # samples unchanged
    assert X_filtered.shape[1] == top_n                       # correct feature count
    assert len(selector.top_features_) == top_n
    assert len(selector.mad_scores_) == mock_genomics_data.shape[1]

def test_transform_before_fit_raises_error(mock_genomics_data):
    """transforming without fitting should trigger a ValueError."""
    selector = OmicsMADSelector(top_n=10)
    with pytest.raises(ValueError, match="Selector must be fitted before transforming"):
        selector.transform(mock_genomics_data)

def test_invalid_input_type():
    """passing raw numpy array instead of a DataFrame should trigger a TypeError."""
    selector = OmicsMADSelector(top_n=5)
    invalid_input = np.random.randn(10, 10)
    with pytest.raises(TypeError, match="Input X must be a pandas DataFrame"):
        selector.fit(invalid_input)

def test_nan_policy_handling():
    """test if nan_policy='omit' correctly ignores NaNs without crashing."""
    # data with a missing entry
    df_with_nan = pd.DataFrame({
        "Gene_A": [1.0, 2.0, np.nan, 4.0, 5.0],
        "Gene_B": [5.0, 5.0, 5.0, 5.0, 5.0]
    })
    
    selector = OmicsMADSelector(top_n=1, nan_policy='omit')
    selector.fit(df_with_nan)
    
    # Gene_A has variation, Gene_B has 0 variation. Gene_A should be chosen.
    assert selector.top_features_[0] == "Gene_A"
    assert not np.isnan(selector.mad_scores_[0])