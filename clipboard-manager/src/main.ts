import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

interface ClipboardItem {
  id: string;
  text: string;
  embedding: number[];
  timestamp: string;
  cluster_id: number | null;
  priority: number;
}

interface Cluster {
  id: number;
  name: string | null;
  items: string[];
  centroid: number[];
  is_favorite: boolean;
}

interface ClipboardConfig {
  max_items: number;
  similarity_threshold: number;
  favorite_priority_boost: number;
}

let currentView: 'all' | 'clusters' = 'all';
let selectedClusterId: number | null = null;
let allItems: ClipboardItem[] = [];
let allClusters: Cluster[] = [];

// Initialize app
async function init() {
  await loadItems();
  await loadClusters();
  await loadConfig();
  setupEventListeners();
  setupClipboardListener();
}

async function loadItems() {
  allItems = await invoke<ClipboardItem[]>('get_all_items');
  renderItems();
}

async function loadClusters() {
  allClusters = await invoke<Cluster[]>('get_all_clusters');
  renderClusters();
}

async function loadConfig() {
  const config = await invoke<ClipboardConfig>('get_config');
  (document.getElementById('max-items') as HTMLInputElement).value = config.max_items.toString();
  (document.getElementById('similarity-threshold') as HTMLInputElement).value = config.similarity_threshold.toString();
  (document.getElementById('threshold-value') as HTMLSpanElement).textContent = config.similarity_threshold.toFixed(2);
  (document.getElementById('priority-boost') as HTMLInputElement).value = config.favorite_priority_boost.toString();
  (document.getElementById('boost-value') as HTMLSpanElement).textContent = config.favorite_priority_boost.toFixed(1);
}

function setupEventListeners() {
  // View controls
  document.getElementById('view-all')?.addEventListener('click', () => {
    currentView = 'all';
    selectedClusterId = null;
    updateViewButtons();
    renderItems();
  });

  document.getElementById('view-clusters')?.addEventListener('click', () => {
    currentView = 'clusters';
    updateViewButtons();
    renderItems();
  });

  // Search
  document.getElementById('search')?.addEventListener('input', async (e) => {
    const query = (e.target as HTMLInputElement).value;
    if (query.trim()) {
      allItems = await invoke<ClipboardItem[]>('search_items', { query });
    } else {
      await loadItems();
    }
    renderItems();
  });

  // Settings modal
  const settingsBtn = document.getElementById('settings-btn');
  const settingsModal = document.getElementById('settings-modal');
  const closeBtn = settingsModal?.querySelector('.close');

  settingsBtn?.addEventListener('click', () => {
    settingsModal?.classList.add('show');
  });

  closeBtn?.addEventListener('click', () => {
    settingsModal?.classList.remove('show');
  });

  window.addEventListener('click', (e) => {
    if (e.target === settingsModal) {
      settingsModal?.classList.remove('show');
    }
  });

  // Settings form
  document.getElementById('similarity-threshold')?.addEventListener('input', (e) => {
    const value = (e.target as HTMLInputElement).value;
    document.getElementById('threshold-value')!.textContent = parseFloat(value).toFixed(2);
  });

  document.getElementById('priority-boost')?.addEventListener('input', (e) => {
    const value = (e.target as HTMLInputElement).value;
    document.getElementById('boost-value')!.textContent = parseFloat(value).toFixed(1);
  });

  document.getElementById('save-settings')?.addEventListener('click', async () => {
    const config: ClipboardConfig = {
      max_items: parseInt((document.getElementById('max-items') as HTMLInputElement).value),
      similarity_threshold: parseFloat((document.getElementById('similarity-threshold') as HTMLInputElement).value),
      favorite_priority_boost: parseFloat((document.getElementById('priority-boost') as HTMLInputElement).value),
    };

    await invoke('update_config', { config });
    document.getElementById('settings-modal')?.classList.remove('show');
    await loadItems();
    await loadClusters();
  });
}

async function setupClipboardListener() {
  await listen<ClipboardItem>('clipboard-update', (event) => {
    allItems.unshift(event.payload);
    loadClusters();
    renderItems();
  });
}

function updateViewButtons() {
  const allBtn = document.getElementById('view-all');
  const clustersBtn = document.getElementById('view-clusters');

  if (currentView === 'all') {
    allBtn?.classList.add('active');
    clustersBtn?.classList.remove('active');
  } else {
    allBtn?.classList.remove('active');
    clustersBtn?.classList.add('active');
  }
}

function renderClusters() {
  const container = document.getElementById('clusters-list');
  if (!container) return;

  if (allClusters.length === 0) {
    container.innerHTML = '<div class="empty-state"><p>No clusters yet</p></div>';
    return;
  }

  container.innerHTML = allClusters
    .map((cluster) => {
      const name = cluster.name || `Cluster ${cluster.id}`;
      const favorite = cluster.is_favorite ? '⭐' : '☆';
      const favoriteClass = cluster.is_favorite ? 'favorite' : '';
      const selectedClass = selectedClusterId === cluster.id ? 'selected' : '';

      return `
        <div class="cluster-item ${favoriteClass} ${selectedClass}" data-cluster-id="${cluster.id}">
          <div class="cluster-header">
            <span class="cluster-name">${escapeHtml(name)}</span>
            <span class="cluster-favorite" data-cluster-id="${cluster.id}">${favorite}</span>
          </div>
          <div class="cluster-stats">${cluster.items.length} items</div>
        </div>
      `;
    })
    .join('');

  // Add event listeners
  container.querySelectorAll('.cluster-item').forEach((el) => {
    el.addEventListener('click', (e) => {
      if ((e.target as HTMLElement).classList.contains('cluster-favorite')) return;
      const clusterId = parseInt((el as HTMLElement).dataset.clusterId!);
      selectedClusterId = clusterId;
      currentView = 'all';
      updateViewButtons();
      renderClusters();
      renderItems();
    });
  });

  container.querySelectorAll('.cluster-favorite').forEach((el) => {
    el.addEventListener('click', async (e) => {
      e.stopPropagation();
      const clusterId = parseInt((el as HTMLElement).dataset.clusterId!);
      await invoke('toggle_favorite', { clusterId });
      await loadClusters();
      await loadItems();
    });
  });
}

function renderItems() {
  const container = document.getElementById('items-container');
  if (!container) return;

  let itemsToRender = allItems;

  // Filter by selected cluster
  if (selectedClusterId !== null) {
    itemsToRender = allItems.filter((item) => item.cluster_id === selectedClusterId);
  }

  if (itemsToRender.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <h3>No items yet</h3>
        <p>Copy some text to get started!</p>
      </div>
    `;
    return;
  }

  if (currentView === 'clusters') {
    renderItemsByCluster(container, itemsToRender);
  } else {
    renderItemsList(container, itemsToRender);
  }
}

function renderItemsList(container: HTMLElement, items: ClipboardItem[]) {
  container.innerHTML = items
    .map((item) => renderItemCard(item))
    .join('');
}

function renderItemsByCluster(container: HTMLElement, items: ClipboardItem[]) {
  const clusterGroups = new Map<number, ClipboardItem[]>();
  const unclustered: ClipboardItem[] = [];

  items.forEach((item) => {
    if (item.cluster_id !== null) {
      if (!clusterGroups.has(item.cluster_id)) {
        clusterGroups.set(item.cluster_id, []);
      }
      clusterGroups.get(item.cluster_id)!.push(item);
    } else {
      unclustered.push(item);
    }
  });

  let html = '';

  // Render clustered items
  clusterGroups.forEach((groupItems, clusterId) => {
    const cluster = allClusters.find((c) => c.id === clusterId);
    if (!cluster) return;

    const name = cluster.name || `Cluster ${clusterId}`;
    const favoriteClass = cluster.is_favorite ? 'favorite' : '';

    html += `
      <div class="cluster-group">
        <div class="cluster-group-header ${favoriteClass}">
          <span class="cluster-group-title">${escapeHtml(name)}</span>
          <div class="cluster-group-actions">
            <button onclick="renameCluster(${clusterId})">Rename</button>
            <button onclick="toggleFavorite(${clusterId})">${cluster.is_favorite ? '⭐' : '☆'}</button>
          </div>
        </div>
        ${groupItems.map((item) => renderItemCard(item)).join('')}
      </div>
    `;
  });

  // Render unclustered items
  if (unclustered.length > 0) {
    html += `
      <div class="cluster-group">
        <div class="cluster-group-header">
          <span class="cluster-group-title">Unclustered</span>
        </div>
        ${unclustered.map((item) => renderItemCard(item)).join('')}
      </div>
    `;
  }

  container.innerHTML = html;
}

function renderItemCard(item: ClipboardItem): string {
  const timestamp = new Date(item.timestamp).toLocaleString();
  const priorityClass = item.priority > 1.5 ? 'high' : '';
  const previewText = item.text.length > 200 ? item.text.substring(0, 200) + '...' : item.text;

  return `
    <div class="clipboard-item">
      <div class="item-header">
        <span class="item-timestamp">${timestamp}</span>
        <span class="item-priority ${priorityClass}">Priority: ${item.priority.toFixed(2)}</span>
      </div>
      <div class="item-text">${escapeHtml(previewText)}</div>
      <div class="item-footer">
        <span class="item-cluster">
          ${item.cluster_id !== null ? `Cluster ${item.cluster_id}` : 'Unclustered'}
        </span>
        <div class="item-actions">
          <button onclick="copyToClipboard('${escapeHtml(item.text)}')">Copy</button>
        </div>
      </div>
    </div>
  `;
}

function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Global functions for inline event handlers
(window as any).renameCluster = async (clusterId: number) => {
  const cluster = allClusters.find((c) => c.id === clusterId);
  const currentName = cluster?.name || `Cluster ${clusterId}`;
  const newName = prompt('Enter new cluster name:', currentName);

  if (newName && newName.trim()) {
    await invoke('rename_cluster', { clusterId, name: newName.trim() });
    await loadClusters();
    renderItems();
  }
};

(window as any).toggleFavorite = async (clusterId: number) => {
  await invoke('toggle_favorite', { clusterId });
  await loadClusters();
  await loadItems();
};

(window as any).copyToClipboard = async (text: string) => {
  try {
    await navigator.clipboard.writeText(text);
  } catch (err) {
    console.error('Failed to copy:', err);
  }
};

// Start the app
init();
