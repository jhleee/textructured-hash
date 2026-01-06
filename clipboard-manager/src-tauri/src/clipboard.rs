use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use parking_lot::RwLock;
use std::sync::Arc;

use crate::encoder::{MultiScaleEncoder, cosine_similarity};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipboardItem {
    pub id: String,
    pub text: String,
    pub embedding: Vec<f32>,
    pub timestamp: DateTime<Utc>,
    pub cluster_id: Option<usize>,
    pub priority: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cluster {
    pub id: usize,
    pub name: Option<String>,
    pub items: Vec<String>, // item IDs
    pub centroid: Vec<f32>,
    pub is_favorite: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipboardConfig {
    pub max_items: usize,
    pub similarity_threshold: f32,
    pub favorite_priority_boost: f32,
}

impl Default for ClipboardConfig {
    fn default() -> Self {
        Self {
            max_items: 1000,
            similarity_threshold: 0.7,
            favorite_priority_boost: 2.0,
        }
    }
}

pub struct ClipboardManager {
    items: Arc<RwLock<HashMap<String, ClipboardItem>>>,
    clusters: Arc<RwLock<Vec<Cluster>>>,
    encoder: Arc<MultiScaleEncoder>,
    config: Arc<RwLock<ClipboardConfig>>,
    next_cluster_id: Arc<RwLock<usize>>,
}

impl ClipboardManager {
    pub fn new() -> Self {
        Self {
            items: Arc::new(RwLock::new(HashMap::new())),
            clusters: Arc::new(RwLock::new(Vec::new())),
            encoder: Arc::new(MultiScaleEncoder::new(128, 42)),
            config: Arc::new(RwLock::new(ClipboardConfig::default())),
            next_cluster_id: Arc::new(RwLock::new(0)),
        }
    }

    /// Add a new clipboard item
    pub fn add_item(&self, text: String) -> ClipboardItem {
        let id = format!("{}", uuid::Uuid::new_v4());
        let embedding = self.encoder.encode(&text);
        let timestamp = Utc::now();

        // Find best matching cluster
        let cluster_id = self.find_best_cluster(&embedding);

        // Calculate priority based on favorite clusters
        let priority = self.calculate_priority(&embedding);

        let item = ClipboardItem {
            id: id.clone(),
            text,
            embedding,
            timestamp,
            cluster_id,
            priority,
        };

        // Add to items
        {
            let mut items = self.items.write();
            items.insert(id.clone(), item.clone());
        }

        // Update cluster
        if let Some(cluster_id) = cluster_id {
            self.add_to_cluster(cluster_id, id.clone(), &item.embedding);
        } else {
            // Create new cluster
            self.create_cluster(vec![id.clone()], item.embedding.clone());
        }

        // Enforce max items limit
        self.enforce_max_items();

        item
    }

    /// Find the best matching cluster for an embedding
    fn find_best_cluster(&self, embedding: &[f32]) -> Option<usize> {
        let clusters = self.clusters.read();
        let config = self.config.read();

        let mut best_similarity = config.similarity_threshold;
        let mut best_cluster = None;

        for cluster in clusters.iter() {
            let similarity = cosine_similarity(embedding, &cluster.centroid);
            if similarity > best_similarity {
                best_similarity = similarity;
                best_cluster = Some(cluster.id);
            }
        }

        best_cluster
    }

    /// Calculate priority based on similarity to favorite clusters
    fn calculate_priority(&self, embedding: &[f32]) -> f32 {
        let clusters = self.clusters.read();
        let config = self.config.read();

        let mut max_similarity = 0.0f32;

        for cluster in clusters.iter() {
            if cluster.is_favorite {
                let similarity = cosine_similarity(embedding, &cluster.centroid);
                max_similarity = max_similarity.max(similarity);
            }
        }

        if max_similarity > 0.0 {
            1.0 + max_similarity * config.favorite_priority_boost
        } else {
            1.0
        }
    }

    /// Add item to cluster
    fn add_to_cluster(&self, cluster_id: usize, item_id: String, embedding: &[f32]) {
        let mut clusters = self.clusters.write();

        if let Some(cluster) = clusters.iter_mut().find(|c| c.id == cluster_id) {
            cluster.items.push(item_id);

            // Update centroid (running average)
            let n = cluster.items.len() as f32;
            for (i, val) in embedding.iter().enumerate() {
                cluster.centroid[i] = (cluster.centroid[i] * (n - 1.0) + val) / n;
            }
        }
    }

    /// Create a new cluster
    fn create_cluster(&self, item_ids: Vec<String>, centroid: Vec<f32>) {
        let mut clusters = self.clusters.write();
        let mut next_id = self.next_cluster_id.write();

        let cluster = Cluster {
            id: *next_id,
            name: None,
            items: item_ids,
            centroid,
            is_favorite: false,
        };

        clusters.push(cluster);
        *next_id += 1;
    }

    /// Enforce max items limit, removing lowest priority items first
    fn enforce_max_items(&self) {
        let config = self.config.read();
        let max_items = config.max_items;
        drop(config);

        let items = self.items.read();
        if items.len() <= max_items {
            return;
        }
        drop(items);

        // Get all items sorted by priority (ascending)
        let mut items_vec: Vec<(String, f32, DateTime<Utc>)> = {
            let items = self.items.read();
            items.iter()
                .map(|(id, item)| (id.clone(), item.priority, item.timestamp))
                .collect()
        };

        // Sort by priority (ascending), then by timestamp (oldest first)
        items_vec.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.2.cmp(&b.2))
        });

        // Remove items until we're under the limit
        let to_remove = items_vec.len() - max_items;
        let remove_ids: Vec<String> = items_vec.iter()
            .take(to_remove)
            .map(|(id, _, _)| id.clone())
            .collect();

        for id in remove_ids {
            self.remove_item(&id);
        }
    }

    /// Remove an item
    fn remove_item(&self, id: &str) {
        let mut items = self.items.write();
        if let Some(item) = items.remove(id) {
            drop(items);

            // Remove from cluster
            if let Some(cluster_id) = item.cluster_id {
                let mut clusters = self.clusters.write();
                if let Some(cluster) = clusters.iter_mut().find(|c| c.id == cluster_id) {
                    cluster.items.retain(|item_id| item_id != id);

                    // Remove empty clusters
                    if cluster.items.is_empty() {
                        clusters.retain(|c| c.id != cluster_id);
                    }
                }
            }
        }
    }

    /// Get all items
    pub fn get_all_items(&self) -> Vec<ClipboardItem> {
        let items = self.items.read();
        let mut items_vec: Vec<ClipboardItem> = items.values().cloned().collect();
        items_vec.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        items_vec
    }

    /// Get all clusters
    pub fn get_all_clusters(&self) -> Vec<Cluster> {
        self.clusters.read().clone()
    }

    /// Toggle favorite status of a cluster
    pub fn toggle_favorite(&self, cluster_id: usize) -> bool {
        let mut clusters = self.clusters.write();
        if let Some(cluster) = clusters.iter_mut().find(|c| c.id == cluster_id) {
            cluster.is_favorite = !cluster.is_favorite;

            // Recalculate priorities for all items in this cluster
            let items = self.items.read();
            let item_ids: Vec<String> = cluster.items.clone();
            drop(items);

            for item_id in item_ids {
                let items = self.items.read();
                if let Some(item) = items.get(&item_id) {
                    let new_priority = self.calculate_priority(&item.embedding);
                    drop(items);

                    let mut items = self.items.write();
                    if let Some(item) = items.get_mut(&item_id) {
                        item.priority = new_priority;
                    }
                }
            }

            return cluster.is_favorite;
        }
        false
    }

    /// Rename a cluster
    pub fn rename_cluster(&self, cluster_id: usize, name: String) {
        let mut clusters = self.clusters.write();
        if let Some(cluster) = clusters.iter_mut().find(|c| c.id == cluster_id) {
            cluster.name = Some(name);
        }
    }

    /// Update configuration
    pub fn update_config(&self, config: ClipboardConfig) {
        let mut current_config = self.config.write();
        *current_config = config;
    }

    /// Get current configuration
    pub fn get_config(&self) -> ClipboardConfig {
        self.config.read().clone()
    }

    /// Search items by text
    pub fn search(&self, query: &str) -> Vec<ClipboardItem> {
        let items = self.items.read();
        let query_lower = query.to_lowercase();

        items.values()
            .filter(|item| item.text.to_lowercase().contains(&query_lower))
            .cloned()
            .collect()
    }
}

// Add uuid dependency
use uuid::Uuid;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_item() {
        let manager = ClipboardManager::new();
        let item = manager.add_item("test text".to_string());

        assert_eq!(item.text, "test text");
        assert_eq!(item.embedding.len(), 128);
    }

    #[test]
    fn test_clustering() {
        let manager = ClipboardManager::new();

        // Add similar items (phone numbers)
        manager.add_item("010-1234-5678".to_string());
        manager.add_item("010-9876-5432".to_string());
        manager.add_item("010-5555-6666".to_string());

        // Add different items (emails)
        manager.add_item("test@example.com".to_string());
        manager.add_item("hello@test.com".to_string());

        let clusters = manager.get_all_clusters();

        // Should have at least 2 clusters (phones and emails)
        assert!(clusters.len() >= 2);
    }
}
