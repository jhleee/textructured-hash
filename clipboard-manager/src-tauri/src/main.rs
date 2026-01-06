// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod encoder;
mod clipboard;

use clipboard::{ClipboardManager, ClipboardItem, Cluster, ClipboardConfig};
use once_cell::sync::Lazy;
use std::sync::Arc;
use parking_lot::Mutex;
use tauri::{Manager, AppHandle};
use arboard::Clipboard as SystemClipboard;
use std::time::Duration;

// Global clipboard manager
static CLIPBOARD_MANAGER: Lazy<Arc<ClipboardManager>> = Lazy::new(|| {
    Arc::new(ClipboardManager::new())
});

// Clipboard monitor state
static LAST_CLIPBOARD_TEXT: Lazy<Arc<Mutex<String>>> = Lazy::new(|| {
    Arc::new(Mutex::new(String::new()))
});

#[tauri::command]
fn get_all_items() -> Vec<ClipboardItem> {
    CLIPBOARD_MANAGER.get_all_items()
}

#[tauri::command]
fn get_all_clusters() -> Vec<Cluster> {
    CLIPBOARD_MANAGER.get_all_clusters()
}

#[tauri::command]
fn toggle_favorite(cluster_id: usize) -> bool {
    CLIPBOARD_MANAGER.toggle_favorite(cluster_id)
}

#[tauri::command]
fn rename_cluster(cluster_id: usize, name: String) -> bool {
    CLIPBOARD_MANAGER.rename_cluster(cluster_id, name);
    true
}

#[tauri::command]
fn update_config(config: ClipboardConfig) -> bool {
    CLIPBOARD_MANAGER.update_config(config);
    true
}

#[tauri::command]
fn get_config() -> ClipboardConfig {
    CLIPBOARD_MANAGER.get_config()
}

#[tauri::command]
fn search_items(query: String) -> Vec<ClipboardItem> {
    CLIPBOARD_MANAGER.search(&query)
}

#[tauri::command]
fn add_clipboard_item(text: String) -> ClipboardItem {
    CLIPBOARD_MANAGER.add_item(text)
}

// Clipboard monitoring
fn start_clipboard_monitor(app_handle: AppHandle) {
    std::thread::spawn(move || {
        let mut clipboard = match SystemClipboard::new() {
            Ok(cb) => cb,
            Err(e) => {
                eprintln!("Failed to create clipboard: {}", e);
                return;
            }
        };

        loop {
            std::thread::sleep(Duration::from_millis(500));

            // Read clipboard
            let text = match clipboard.get_text() {
                Ok(text) => text,
                Err(_) => continue,
            };

            // Check if text changed
            let mut last_text = LAST_CLIPBOARD_TEXT.lock();
            if text != *last_text && !text.is_empty() {
                *last_text = text.clone();
                drop(last_text);

                // Add to manager
                let item = CLIPBOARD_MANAGER.add_item(text);

                // Emit event to frontend
                let _ = app_handle.emit("clipboard-update", item);
            }
        }
    });
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let app_handle = app.handle().clone();
            start_clipboard_monitor(app_handle);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_all_items,
            get_all_clusters,
            toggle_favorite,
            rename_cluster,
            update_config,
            get_config,
            search_items,
            add_clipboard_item,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
