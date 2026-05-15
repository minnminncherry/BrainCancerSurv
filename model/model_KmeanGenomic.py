import torch
import torch.nn as nn


class KMeansGenomics(nn.Module):
    """
    K-means-inspired genomic classifier.

    The model first projects genomic features into a compact embedding space,
    then computes soft assignment weights to learnable cluster centers.
    Those cluster assignments and the cluster-aware pooled embedding are used
    to predict the survival class.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_clusters=8,
        projection_dim=64,
        hidden_dim=None,
        dropout=0.1,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(32, projection_dim)

        self.num_clusters = int(num_clusters)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.cluster_centers = nn.Parameter(torch.randn(self.num_clusters, projection_dim))
        self.cluster_norm = nn.LayerNorm(self.num_clusters)
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim + self.num_clusters, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, output_dim),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.cluster_centers)

    def _compute_soft_assignments(self, projection):
        # Squared Euclidean distance to each learnable centroid.
        distances = torch.cdist(projection, self.cluster_centers, p=2) ** 2
        assignments = torch.softmax(-distances, dim=1)
        return assignments, distances

    def forward(self, x, return_projection=False, return_clusters=False):
        projection = self.encoder(x)
        assignments, distances = self._compute_soft_assignments(projection)
        cluster_features = self.cluster_norm(assignments)
        classifier_input = torch.cat([projection, cluster_features], dim=1)
        logits = self.classifier(classifier_input)

        if return_clusters:
            return logits, projection, assignments, distances
        if return_projection:
            return logits, projection
        return logits
