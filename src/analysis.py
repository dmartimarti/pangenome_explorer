"""UMAP dimensionality reduction and analysis utilities."""

import numpy as np
import pandas as pd
import streamlit as st
import umap
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any
import logging

from config import UMAP_PARAMS

logger = logging.getLogger(__name__)


@st.cache_data
def compute_umap_embedding(embeddings: np.ndarray, 
                          n_neighbors: int = None,
                          min_dist: float = None,
                          metric: str = None,
                          random_state: int = None) -> Tuple[np.ndarray, umap.UMAP]:
    """
    Compute UMAP dimensionality reduction on protein embeddings.
    
    Args:
        embeddings: High-dimensional embeddings array (n_samples, n_features)
        n_neighbors: Number of neighbors for UMAP (None uses config default)
        min_dist: Minimum distance for UMAP (None uses config default)
        metric: Distance metric (None uses config default)
        random_state: Random state for reproducibility (None uses config default)
    
    Returns:
        Tuple of (2D_embeddings, fitted_umap_model)
    """
    try:
        # Use config defaults if not specified
        umap_kwargs = UMAP_PARAMS.copy()
        if n_neighbors is not None:
            umap_kwargs['n_neighbors'] = n_neighbors
        if min_dist is not None:
            umap_kwargs['min_dist'] = min_dist
        if metric is not None:
            umap_kwargs['metric'] = metric
        if random_state is not None:
            umap_kwargs['random_state'] = random_state
        
        logger.info(f"Computing UMAP with parameters: {umap_kwargs}")
        logger.info(f"Input embeddings shape: {embeddings.shape}")
        
        # Initialize UMAP
        umap_model = umap.UMAP(**umap_kwargs)
        
        # Fit and transform
        embedding_2d = umap_model.fit_transform(embeddings)
        
        logger.info(f"UMAP completed. Output shape: {embedding_2d.shape}")
        
        return embedding_2d, umap_model
        
    except Exception as e:
        logger.error(f"Error computing UMAP: {e}")
        raise


@st.cache_data
def compute_standardized_umap(embeddings: np.ndarray,
                             standardize: bool = True,
                             **umap_kwargs) -> Tuple[np.ndarray, umap.UMAP, StandardScaler]:
    """
    Compute UMAP with optional standardization of input embeddings.
    
    Args:
        embeddings: High-dimensional embeddings
        standardize: Whether to standardize embeddings before UMAP
        **umap_kwargs: Additional UMAP parameters
    
    Returns:
        Tuple of (2D_embeddings, umap_model, scaler_or_None)
    """
    try:
        scaler = None
        processed_embeddings = embeddings
        
        if standardize:
            logger.info("Standardizing embeddings before UMAP")
            scaler = StandardScaler()
            processed_embeddings = scaler.fit_transform(embeddings)
        
        # Merge with default parameters
        params = UMAP_PARAMS.copy()
        params.update(umap_kwargs)
        
        # Compute UMAP
        embedding_2d, umap_model = compute_umap_embedding(processed_embeddings, **params)
        
        return embedding_2d, umap_model, scaler
        
    except Exception as e:
        logger.error(f"Error computing standardized UMAP: {e}")
        raise


def add_umap_coordinates(df: pd.DataFrame, 
                        umap_coords: np.ndarray,
                        coord_prefix: str = "UMAP") -> pd.DataFrame:
    """
    Add UMAP coordinates to the dataframe.
    
    Args:
        df: DataFrame with protein data
        umap_coords: UMAP coordinates array (n_samples, 2)
        coord_prefix: Prefix for coordinate column names
    
    Returns:
        DataFrame with added UMAP coordinates
    """
    try:
        if len(df) != len(umap_coords):
            raise ValueError(f"DataFrame length {len(df)} doesn't match coordinates length {len(umap_coords)}")
        
        df_with_coords = df.copy()
        df_with_coords[f'{coord_prefix}_1'] = umap_coords[:, 0]
        df_with_coords[f'{coord_prefix}_2'] = umap_coords[:, 1]
        
        return df_with_coords
        
    except Exception as e:
        logger.error(f"Error adding UMAP coordinates: {e}")
        raise


def analyze_umap_clusters(umap_coords: np.ndarray, 
                         labels: np.ndarray = None,
                         cluster_method: str = "kmeans") -> Dict[str, Any]:
    """
    Analyze clusters in UMAP space.
    
    Args:
        umap_coords: 2D UMAP coordinates
        labels: Optional labels for supervised analysis
        cluster_method: Clustering method ('kmeans', 'dbscan', 'hdbscan')
    
    Returns:
        Dictionary with cluster analysis results
    """
    try:
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.metrics import silhouette_score
        
        results = {
            'coordinates_range': {
                'x_min': float(umap_coords[:, 0].min()),
                'x_max': float(umap_coords[:, 0].max()),
                'y_min': float(umap_coords[:, 1].min()),
                'y_max': float(umap_coords[:, 1].max())
            }
        }
        
        # Automatic clustering
        if cluster_method == "kmeans":
            # Try different numbers of clusters
            silhouette_scores = []
            cluster_range = range(2, min(11, len(umap_coords) // 10 + 1))
            
            for n_clusters in cluster_range:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(umap_coords)
                
                if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette
                    score = silhouette_score(umap_coords, cluster_labels)
                    silhouette_scores.append((n_clusters, score))
            
            if silhouette_scores:
                best_k, best_score = max(silhouette_scores, key=lambda x: x[1])
                
                kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(umap_coords)
                
                results['clustering'] = {
                    'method': 'kmeans',
                    'best_k': int(best_k),
                    'silhouette_score': float(best_score),
                    'cluster_labels': cluster_labels.tolist(),
                    'cluster_counts': {int(i): int(count) for i, count in 
                                     zip(*np.unique(cluster_labels, return_counts=True))}
                }
        
        elif cluster_method == "dbscan":
            # DBSCAN clustering
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(umap_coords)
            
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            results['clustering'] = {
                'method': 'dbscan',
                'n_clusters': n_clusters,
                'n_noise_points': n_noise,
                'cluster_labels': cluster_labels.tolist(),
                'cluster_counts': {int(i): int(count) for i, count in 
                                 zip(*np.unique(cluster_labels, return_counts=True))}
            }
        
        # If labels provided, analyze separation
        if labels is not None:
            unique_labels = np.unique(labels)
            results['label_analysis'] = {
                'unique_labels': unique_labels.tolist(),
                'label_counts': {str(label): int(count) for label, count in 
                               zip(*np.unique(labels, return_counts=True))}
            }
            
            # Compute within-label vs between-label distances
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(umap_coords)
            
            within_distances = []
            between_distances = []
            
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    if labels[i] == labels[j]:
                        within_distances.append(distances[i, j])
                    else:
                        between_distances.append(distances[i, j])
            
            if within_distances and between_distances:
                results['label_analysis'].update({
                    'mean_within_distance': float(np.mean(within_distances)),
                    'mean_between_distance': float(np.mean(between_distances)),
                    'separation_ratio': float(np.mean(between_distances) / np.mean(within_distances))
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing UMAP clusters: {e}")
        return {'error': str(e)}


def compute_feature_importance(embeddings: np.ndarray,
                              umap_coords: np.ndarray,
                              method: str = "correlation") -> np.ndarray:
    """
    Compute feature importance for UMAP embedding.
    
    Args:
        embeddings: Original high-dimensional embeddings
        umap_coords: 2D UMAP coordinates
        method: Method for importance calculation
    
    Returns:
        Feature importance scores
    """
    try:
        if method == "correlation":
            # Compute correlation between each feature and UMAP coordinates
            correlations_x = np.abs([np.corrcoef(embeddings[:, i], umap_coords[:, 0])[0, 1] 
                                   for i in range(embeddings.shape[1])])
            correlations_y = np.abs([np.corrcoef(embeddings[:, i], umap_coords[:, 1])[0, 1] 
                                   for i in range(embeddings.shape[1])])
            
            # Combine correlations
            importance = np.sqrt(correlations_x**2 + correlations_y**2)
            
            # Handle NaN values
            importance = np.nan_to_num(importance, nan=0.0)
            
            return importance
            
    except Exception as e:
        logger.error(f"Error computing feature importance: {e}")
        return np.zeros(embeddings.shape[1])