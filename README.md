# Clipboard Manager

**Smart clipboard manager with text structure clustering**

A Windows desktop application built with Tauri that automatically organizes your clipboard history using advanced text structure similarity algorithms.

## Features

### 🎯 Core Features

- **Automatic Clipboard Monitoring**: Captures all text copied to clipboard
- **Smart Clustering**: Groups similar items using multi-scale text structure analysis
- **Configurable Storage**: Store up to 1000 items (configurable)
- **Priority System**: Favorite clusters get higher priority, reducing deletion risk
- **Real-time Updates**: See new clipboard items instantly
- **Search**: Quickly find clipboard items by text content

### 🧠 Algorithm

The app uses the **MultiScale Encoder** algorithm from our research:
- **AUC-ROC**: 0.955 (excellent classification performance)
- **Speed**: ~9,000 texts/second
- **No ML dependencies**: Pure statistical feature extraction
- **Zero external API calls**: All processing happens locally

The algorithm automatically detects structural similarities:
- Phone numbers cluster together
- URLs group with URLs
- Email addresses with emails
- Korean/English/Japanese text by language
- JSON/XML by format
- And many more patterns!

### ⭐ Advanced Features

- **Favorite Clusters**: Mark important clusters to boost item priority
- **Custom Names**: Rename clusters for better organization
- **Cluster View**: Browse items grouped by similarity
- **Dark Theme**: Easy on the eyes

## Project Structure

```
clipboard-manager/
├── src/                    # Frontend (TypeScript + Vite)
│   ├── main.ts            # UI logic
│   └── style.css          # Styling
├── src-tauri/             # Backend (Rust)
│   ├── src/
│   │   ├── main.rs        # Tauri entry point
│   │   ├── encoder.rs     # MultiScale algorithm
│   │   └── clipboard.rs   # Clipboard management
│   └── Cargo.toml         # Rust dependencies
├── legacy/                # Original research code (Python)
└── README.md              # This file
```

## Getting Started

### Prerequisites

**For Windows Development:**
1. [Node.js](https://nodejs.org/) (v18+)
2. [Rust](https://rustup.rs/)
3. [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) with C++ development tools

### Installation

```bash
cd clipboard-manager
npm install
```

### Development

```bash
npm run tauri:dev
```

This will:
1. Start the Vite dev server (frontend)
2. Launch the Tauri app with hot-reload

### Building for Production

```bash
npm run tauri:build
```

The installer will be created in:
```
src-tauri/target/release/bundle/nsis/Clipboard Manager_1.0.0_x64-setup.exe
```

## Configuration

### Default Settings

- **Max Items**: 1000
- **Similarity Threshold**: 0.7 (70% similarity to join cluster)
- **Favorite Priority Boost**: 2.0x

### Changing Settings

1. Click the **Settings** button in the top-right
2. Adjust values:
   - **Max Items**: Total clipboard items to keep
   - **Similarity Threshold**: Lower = more clusters, Higher = fewer clusters
   - **Favorite Priority Boost**: How much to protect favorite clusters from deletion
3. Click **Save Settings**

## How It Works

### Clipboard Monitoring

The app monitors your system clipboard every 500ms. When new text is detected:

1. **Encoding**: Text is converted to a 128-dimensional vector using the MultiScale algorithm
2. **Clustering**: The vector is compared with existing cluster centroids
3. **Assignment**: If similarity > threshold, item joins that cluster; otherwise creates new cluster
4. **Priority**: If similar to a favorite cluster, item gets priority boost

### Deletion Policy

When the max item limit is reached:
1. Items are sorted by priority (ascending) and timestamp (oldest first)
2. Lowest priority, oldest items are deleted first
3. Favorite cluster items are preserved longer due to higher priority

### Clustering Algorithm

The **MultiScale Encoder** extracts features at multiple scales:

1. **Byte-level**: UTF-8 byte distribution
2. **Unicode-level**: Character categories and scripts (Latin, Hangul, CJK, etc.)
3. **Token-level**: Word statistics and character class ratios
4. **Pattern-level**: Entropy, n-gram diversity, structural patterns

Features are projected to 128 dimensions and L2-normalized for cosine similarity comparison.

## Research Background

This app is based on research into **lightweight text structure similarity** without semantic embeddings:

- **Goal**: Fast, memory-efficient text clustering without ML models
- **Performance**: 1000x faster than OpenAI embeddings
- **Quality**: AUC-ROC 0.955 on 24 text categories
- **Memory**: 512 bytes per vector (vs 1536 bytes for text-embedding-3-small)

See `legacy/` folder for the original research code and experiments.

## License

MIT License

## Contributing

Issues and PRs welcome!

## Troubleshooting

### Build Errors on Windows

**Error: "MSVC not found"**
- Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
- Select "Desktop development with C++"

**Error: "Rust not found"**
- Install Rust: https://rustup.rs/
- Restart terminal after installation

### Runtime Issues

**Clipboard not updating**
- Check Windows clipboard permissions
- Try running as Administrator

**High memory usage**
- Reduce Max Items in Settings
- Clear old clusters (not yet implemented)

## Future Improvements

- [ ] Persistent storage (SQLite)
- [ ] Export/import clipboard history
- [ ] Hotkey support
- [ ] Tray icon with quick access
- [ ] Image clipboard support
- [ ] Cross-platform support (macOS, Linux)
- [ ] Binary quantization for smaller vectors
- [ ] Manual cluster merging/splitting
