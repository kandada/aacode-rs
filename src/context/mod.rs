// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Context management: init.md loading, large-output archiving, and a light
//! project-structure summary injected before a task.
//!
//! Ported from Python `utils/context_manager.py` + `class_method_mapper.py`
//! (structure analysis is simplified to a file/extension summary).

use std::path::{Path, PathBuf};

pub struct ContextManager {
    project_path: PathBuf,
}

const DEFAULT_INIT_MD: &str = r#"# Project Guidelines

## Core Rules
1. Annotate path at top of each code file: `# {relative_path}`
2. Prefer modifying existing files over creating new ones
3. All file operations must stay within the project directory
4. Dangerous commands require caution

## Workflow
1. Analyze requirements first, then plan
2. Small steps, frequent testing
3. Write self-contained test functions
4. Check safety before using tools

## Code Quality
- Follow language best practices
- Keep functions reasonably short
- Add necessary docstrings
- Handle errors gracefully
------
"#;

impl ContextManager {
    pub fn new(project_path: &Path) -> Self {
        let cm = ContextManager {
            project_path: project_path.to_path_buf(),
        };
        let _ = std::fs::create_dir_all(cm.context_dir());
        cm
    }

    fn context_dir(&self) -> PathBuf {
        self.project_path.join(".aacode").join("context")
    }

    /// Load init.md, creating a default one if absent. Returns its contents.
    pub fn load_init_instructions(&self) -> String {
        let path = self.project_path.join("init.md");
        if let Ok(s) = std::fs::read_to_string(&path) {
            if !s.trim().is_empty() {
                return s;
            }
        }
        // create default
        let _ = std::fs::write(&path, DEFAULT_INIT_MD);
        DEFAULT_INIT_MD.to_string()
    }

    /// Archive a large output to `.aacode/context/` and return its path.
    pub fn save_large_output(&self, content: &str, name_hint: &str) -> Option<String> {
        let dir = self.context_dir();
        std::fs::create_dir_all(&dir).ok()?;
        let name = format!(
            "{}_{}.txt",
            sanitize(name_hint),
            uuid::Uuid::new_v4().simple()
        );
        let path = dir.join(name);
        std::fs::write(&path, content).ok()?;
        Some(path.to_string_lossy().to_string())
    }

    /// Produce a light project-structure summary: top-level entries and a
    /// per-extension file count. Bounded output for prompt injection.
    pub fn analyze_project_structure(&self) -> String {
        use std::collections::BTreeMap;
        let mut ext_counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut top_entries: Vec<String> = Vec::new();
        let exclude = [".git", ".aacode", "node_modules", "target", "__pycache__", ".venv"];

        if let Ok(rd) = std::fs::read_dir(&self.project_path) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if exclude.contains(&name.as_str()) {
                    continue;
                }
                let is_dir = e.path().is_dir();
                top_entries.push(if is_dir { format!("{name}/") } else { name });
            }
        }
        top_entries.sort();

        // Walk (bounded) for extension stats.
        let mut stack = vec![self.project_path.clone()];
        let mut visited = 0usize;
        while let Some(dir) = stack.pop() {
            if visited > 5000 {
                break;
            }
            let rd = match std::fs::read_dir(&dir) {
                Ok(r) => r,
                Err(_) => continue,
            };
            for e in rd.flatten() {
                let p = e.path();
                let fname = e.file_name().to_string_lossy().to_string();
                if exclude.contains(&fname.as_str()) {
                    continue;
                }
                if p.is_dir() {
                    stack.push(p);
                } else {
                    visited += 1;
                    let ext = p
                        .extension()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| "(none)".to_string());
                    *ext_counts.entry(ext).or_insert(0) += 1;
                }
            }
        }

        let mut out = String::from("Project structure summary:\n\nTop-level entries:\n");
        for e in top_entries.iter().take(50) {
            out.push_str(&format!("  {e}\n"));
        }
        out.push_str("\nFile counts by extension:\n");
        for (ext, n) in ext_counts.iter() {
            out.push_str(&format!("  .{ext}: {n}\n"));
        }
        // bound size
        if out.len() > 1500 {
            out.truncate(1500);
            out.push_str("\n...(truncated)");
        }
        out
    }
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "aacode_ctx_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn init_md_created_default() {
        let d = tmp();
        let cm = ContextManager::new(&d);
        let content = cm.load_init_instructions();
        assert!(content.contains("Project Guidelines"));
        assert!(d.join("init.md").exists());
    }

    #[test]
    fn init_md_existing_used() {
        let d = tmp();
        std::fs::write(d.join("init.md"), "# Custom\nrules").unwrap();
        let cm = ContextManager::new(&d);
        assert!(cm.load_init_instructions().contains("Custom"));
    }

    #[test]
    fn large_output_archived() {
        let d = tmp();
        let cm = ContextManager::new(&d);
        let path = cm.save_large_output("big content", "run_shell_output").unwrap();
        assert!(std::path::Path::new(&path).exists());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "big content");
    }

    #[test]
    fn structure_summary() {
        let d = tmp();
        std::fs::write(d.join("main.rs"), "fn main(){}").unwrap();
        std::fs::write(d.join("lib.py"), "x=1").unwrap();
        std::fs::create_dir_all(d.join("src")).unwrap();
        let cm = ContextManager::new(&d);
        let s = cm.analyze_project_structure();
        assert!(s.contains("main.rs"));
        assert!(s.contains(".rs:") || s.contains("rs:"));
    }
}
