# Clipboard Manager - Tauri Application

## Project Overview

This is a **Windows desktop application** that provides intelligent clipboard management with automatic text clustering based on structural similarity. Built with Tauri (Rust + TypeScript).

### Key Features

1. **Automatic Clipboard Monitoring** - Captures all text copies
2. **Smart Clustering** - Groups similar items using multi-scale text structure analysis
3. **Priority System** - Favorite clusters have higher retention priority
4. **Configurable Limits** - Up to 1000 items (adjustable)
5. **Real-time UI** - Instant updates and search

## Architecture

### Backend (Rust)

**Location**: `clipboard-manager/src-tauri/src/`

#### Core Components

1. **`encoder.rs`** - MultiScale text structure encoder
   - Port of the Python research algorithm (AUC-ROC 0.955)
   - 128-dimensional vector encoding
   - Features: byte-level, unicode, token, pattern statistics
   - ~9000 texts/second encoding speed

2. **`clipboard.rs`** - Clipboard management system
   - `ClipboardManager`: Main state manager
   - `ClipboardItem`: Individual clipboard entry
   - `Cluster`: Group of similar items
   - Auto-clustering with configurable threshold
   - Priority-based deletion policy

3. **`main.rs`** - Tauri app entry point
   - Clipboard monitor thread (500ms polling)
   - Tauri commands for frontend interaction
   - Event emission for real-time updates

#### Key Algorithms

**Clustering Logic**:
```rust
// When new text is copied:
1. Encode text → 128D vector
2. Find best cluster (cosine similarity > threshold)
3. If found: add to cluster, update centroid
4. If not: create new cluster
5. Calculate priority (boost if similar to favorites)
6. Enforce max items limit (delete lowest priority first)
```

**Priority Calculation**:
```
priority = 1.0 + (max_similarity_to_favorites * boost_factor)
```

### Frontend (TypeScript + Vite)

**Location**: `clipboard-manager/src/`

#### Files

1. **`main.ts`** - UI logic
   - Tauri API integration
   - Real-time event listeners
   - View rendering (list / clusters)
   - Search functionality

2. **`style.css`** - Dark theme styling
   - Sidebar for clusters
   - Main content area for items
   - Settings modal
   - Responsive design

3. **`index.html`** - App structure
   - Header with search + settings
   - Sidebar (cluster list)
   - Main view (items)
   - Settings modal

#### UI Views

- **All Items View**: Chronological list of all clipboard items
- **Clusters View**: Items grouped by similarity clusters
- **Settings Panel**: Configure max items, threshold, priority boost

## Development Guide

### Project Structure

```
clipboard-manager/
├── src/                     # Frontend
│   ├── main.ts             # UI logic
│   └── style.css           # Styling
├── src-tauri/              # Backend
│   ├── src/
│   │   ├── main.rs         # Entry point
│   │   ├── encoder.rs      # Encoding algorithm
│   │   └── clipboard.rs    # Clipboard logic
│   ├── Cargo.toml          # Rust dependencies
│   ├── tauri.conf.json     # Tauri config
│   └── build.rs            # Build script
├── index.html              # HTML template
├── vite.config.ts          # Vite config
├── package.json            # Node dependencies
└── tsconfig.json           # TypeScript config
```

### Building

**Prerequisites** (Windows):
- Node.js 18+
- Rust (via rustup)
- Visual Studio Build Tools (C++ tools)

**Commands**:
```bash
cd clipboard-manager

# Install dependencies
npm install

# Development (hot-reload)
npm run tauri:dev

# Production build
npm run tauri:build
# Output: src-tauri/target/release/bundle/nsis/*.exe
```

### Testing

**Unit Tests**:
```bash
cd src-tauri
cargo test
```

**Manual Testing**:
1. Run `npm run tauri:dev`
2. Copy various text types:
   - Phone numbers: 010-1234-5678, 010-9999-8888
   - URLs: https://example.com, https://test.com
   - Emails: test@example.com, hello@test.com
3. Verify clustering in UI
4. Test favorite toggling
5. Test settings changes

## Configuration

### Tauri Config (`tauri.conf.json`)

```json
{
  "productName": "Clipboard Manager",
  "identifier": "com.clipboard.manager",
  "bundle": {
    "targets": ["nsis", "msi"]  // Windows installers
  },
  "app": {
    "windows": [{
      "width": 1200,
      "height": 800
    }]
  }
}
```

### Default Settings

```rust
// clipboard.rs - ClipboardConfig::default()
{
    max_items: 1000,
    similarity_threshold: 0.7,
    favorite_priority_boost: 2.0,
}
```

## API Reference

### Tauri Commands

All commands are defined in `main.rs` and callable from frontend:

```typescript
// Get all clipboard items
invoke<ClipboardItem[]>('get_all_items')

// Get all clusters
invoke<Cluster[]>('get_all_clusters')

// Toggle favorite status
invoke<boolean>('toggle_favorite', { clusterId: number })

// Rename cluster
invoke<boolean>('rename_cluster', { clusterId: number, name: string })

// Update configuration
invoke<boolean>('update_config', { config: ClipboardConfig })

// Get current configuration
invoke<ClipboardConfig>('get_config')

// Search items
invoke<ClipboardItem[]>('search_items', { query: string })

// Manually add item (for testing)
invoke<ClipboardItem>('add_clipboard_item', { text: string })
```

### Events

```typescript
// Listen for clipboard updates
listen<ClipboardItem>('clipboard-update', (event) => {
    // Handle new clipboard item
})
```

## Code Patterns

### Adding a New Tauri Command

1. **Backend** (`main.rs`):
```rust
#[tauri::command]
fn my_command(param: String) -> Result<String, String> {
    Ok(format!("Received: {}", param))
}

// Register in main():
.invoke_handler(tauri::generate_handler![
    my_command,
    // ... other commands
])
```

2. **Frontend** (`main.ts`):
```typescript
const result = await invoke<string>('my_command', { param: 'test' });
```

### Modifying the Encoder

The encoder in `encoder.rs` is a direct port of `legacy/src/encoders/proposed/multiscale.py`:

- Keep the 4-scale structure (byte, unicode, token, pattern)
- Maintain 128-dimensional output
- Preserve L2 normalization
- Test with existing test cases

### Adjusting Clustering

Clustering parameters in `clipboard.rs`:

```rust
// Similarity threshold for cluster membership
config.similarity_threshold  // default: 0.7

// How much to boost priority for favorite-similar items
config.favorite_priority_boost  // default: 2.0
```

Lower threshold = more clusters (stricter grouping)
Higher threshold = fewer clusters (looser grouping)

## Common Tasks

### Adding a New Setting

1. **Add to `ClipboardConfig`** (`clipboard.rs`):
```rust
pub struct ClipboardConfig {
    // ...
    pub new_setting: f32,
}
```

2. **Update `Default` impl**:
```rust
impl Default for ClipboardConfig {
    fn default() -> Self {
        Self {
            // ...
            new_setting: 1.0,
        }
    }
}
```

3. **Add UI controls** (`index.html` + `main.ts`)

4. **Use in logic** (`clipboard.rs`)

### Changing the Encoder Dimension

1. Update `MultiScaleEncoder::new()` call in `clipboard.rs`:
```rust
Arc::new(MultiScaleEncoder::new(256, 42))  // 256 dims
```

2. Update frontend type definitions in `main.ts`

3. Re-run tests

### Adding Persistence

Currently, all data is in-memory. To add persistence:

1. Add SQLite dependency to `Cargo.toml`:
```toml
rusqlite = { version = "0.31", features = ["bundled"] }
```

2. Create database schema in `clipboard.rs`
3. Implement save/load methods
4. Call on startup and shutdown

## Troubleshooting

### Common Build Issues

**Windows:**
- Missing MSVC: Install Visual Studio Build Tools
- Rust not found: Install from https://rustup.rs/

**Linux (dev env):**
- Missing GTK libraries: This is expected, app targets Windows
- Use cross-compilation or build on Windows

### Runtime Issues

**Clipboard not updating:**
- Check polling interval in `main.rs` (currently 500ms)
- Verify clipboard permissions

**High memory:**
- Reduce `max_items` setting
- Check for memory leaks with `cargo flamegraph`

**Clusters not forming:**
- Lower `similarity_threshold`
- Check encoder output with unit tests

## Performance

### Benchmarks (Research Results)

- **Encoding Speed**: ~9,000 texts/second (single-threaded)
- **Memory per Item**: ~600 bytes (128 floats + metadata)
- **Clustering Speed**: O(n) where n = number of clusters
- **Max Recommended Items**: 10,000 (at 1000, ~600KB memory)

### Optimization Tips

1. **Encoder**: Already heavily optimized (pure statistics)
2. **Clustering**: Use KD-tree for >1000 clusters
3. **UI**: Virtual scrolling for >500 items
4. **Storage**: Implement lazy loading with SQLite

## Research Context

This app is based on academic research into **lightweight text structure similarity**:

### Original Research Goals

- Classify text by structure without semantic embeddings
- Memory: ≤256 bytes/text
- Speed: ≥10,000 texts/sec
- Quality: AUC-ROC ≥0.92

### Results

- **Best Model**: MultiScale V1
- **AUC-ROC**: 0.955 ✓
- **Speed**: 9,340 texts/sec ✗ (close to target)
- **Memory**: 512 bytes ✗ (double target)

### Why This Algorithm?

- No ML frameworks needed (pure statistics)
- Fast enough for real-time clustering
- Excellent structural pattern detection
- Works offline

See `legacy/` for original Python implementation and research papers.

## Contributing

When contributing:

1. **Rust Code**: Follow Rust idioms, run `cargo clippy`
2. **TypeScript**: Use strict mode, run `npm run build` to check
3. **Tests**: Add tests for new features
4. **Documentation**: Update this file and README.md

## Future Enhancements

- [ ] Persistent storage (SQLite)
- [ ] Global hotkeys
- [ ] System tray icon
- [ ] Export/import
- [ ] Image clipboard support
- [ ] Cloud sync (optional)
- [ ] Manual cluster editing
- [ ] Binary quantization (reduce vector size)

---

**Last Updated**: 2026-01-06
**Status**: Production-ready (pending Windows build)
**Tech Stack**: Rust, TypeScript, Tauri, Vite
