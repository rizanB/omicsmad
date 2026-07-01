import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation

class OmicsMADSelector:
    """
    feature selector based on Median Absolute Deviation (MAD) 
    for high-throughput biological and numeric matrices.
    """
    def __init__(self, top_n=2000, scale='normal', nan_policy='omit'):
        self.top_n = top_n
        self.scale = scale
        self.nan_policy = nan_policy
        self.mad_scores_ = None
        self.top_features_ = None
        self.feature_names_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        calculates MAD scores across samples for each feature.
        assumes X is a pandas DataFrame where columns = features.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame with features as columns.")
            
        self.feature_names_ = X.columns
        self.mad_scores_ = median_abs_deviation(
            X.values, axis=0, scale=self.scale, nan_policy=self.nan_policy
        )
        
        mad_series = pd.Series(self.mad_scores_, index=self.feature_names_)
        actual_top_n = min(self.top_n, len(self.feature_names_))
        self.top_features_ = mad_series.nlargest(actual_top_n).index
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """filters dataframe to selected top features."""
        if self.top_features_ is None:
            raise ValueError("Selector must be fitted before transforming data.")
        return X[self.top_features_]
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """fit to data, then transform it."""
        return self.fit(X).transform(X)
        
    def plot_distributions(
        self, 
        filename="mad_distribution.png", 
        output_dir="figures",
        title_hist="Feature MAD Distribution", 
        title_elbow="Sorted MAD Scores (Elbow Plot)",
        xlabel="MAD Score", 
        ylabel_hist="Number of Features",
        xlabel_elbow="Feature Rank (by MAD)",
        color="steelblue",
        figsize=(12, 4),
        dpi=150
    ):
        """
        makes histogram and elbow plots
        saves inside 'figures/' directory by default.
        """
        if self.mad_scores_ is None:
            raise ValueError("No data fitted yet to plot. Call .fit() first.")
            
        # make output folder if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, filename)
        else:
            save_path = filename
            
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

        # histogram
        axes[0].hist(self.mad_scores_, bins=80, color=color, edgecolor="white", linewidth=0.4)
        thresh_val = np.sort(self.mad_scores_)[::-1][min(self.top_n, len(self.mad_scores_))-1]
        axes[0].axvline(thresh_val, color="red", linestyle="--", label=f"Top {self.top_n} threshold")
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel_hist)
        axes[0].set_title(title_hist)
        axes[0].legend()
        
        # 2. elbow Plot
        sorted_mad = np.sort(self.mad_scores_)[::-1]
        axes[1].plot(sorted_mad, color=color, linewidth=1)
        axes[1].axvline(self.top_n, color="red", linestyle="--", label=f"N = {self.top_n}")
        axes[1].set_xlabel(xlabel_elbow)
        axes[1].set_ylabel(xlabel)
        axes[1].set_title(title_elbow)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)
        plt.close()
        print(f"MAD diagnostic plots saved successfully → {save_path}")

    def save_filtered_data(self, X: pd.DataFrame, filename="filtered_features.tsv", output_dir="datasets/processed", sep="\t"):
        """
        extracts the top features from X, sorts columns by descending MAD score, 
        and saves filtered matrix directly to disk.
        """
        if self.top_features_ is None:
            raise ValueError("Selector must be fitted before saving filtered data.")
            
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, filename)
        else:
            save_path = filename
            
        # extract and transform data matrix
        filtered_df = self.transform(X)
        
        # write to disk
        filtered_df.to_csv(save_path, sep=sep)
        print(f"Filtered matrix ({filtered_df.shape[0]} samples x {filtered_df.shape[1]} features) saved successfully → {save_path}")