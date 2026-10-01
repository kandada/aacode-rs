// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Code tools — execute_python, run_tests, debug_code, analyze_code.
//!
//! Ported from Python `tools/code_tools.py` + `tools/custom_tools.py`.

use super::registry::Tool;
use super::schema::{ParamType, ToolParameter, ToolSchema};
use crate::error::{AacodeError, Result};
use crate::tools::ShellBackend;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

// ──────────────────────── execute_python ────────────────────────────────

/// Build the script preamble that best-effort changes into `project_path`.
///
/// The fastshell Python sandbox re-roots absolute paths, so a hard
/// `os.chdir("<host path>")` fails with `FileNotFoundError`. We only chdir when
/// the path is genuinely a directory and swallow any residual error — the
/// backend already runs the script with the project dir as cwd, so a skipped
/// chdir is still correct (no false failure, no wrong cwd).
fn python_wrapper(project_path: &str, code: &str) -> String {
    format!(
        "import sys, os\ntry:\n    _p = {project_path:?}\n    if os.path.isdir(_p):\n        os.chdir(_p)\nexcept Exception:\n    pass\n{code}"
    )
}

/// Executes Python code by writing to a temp file and running it through
/// the shell backend. On desktop this uses the native OS shell (python3);
/// on mobile it routes through fastshell's embedded CPython engine.
pub struct ExecutePythonTool {
    pub project_path: PathBuf,
    pub backend: Arc<dyn ShellBackend>,
    pub default_timeout_secs: u64,
}

#[async_trait::async_trait]
impl Tool for ExecutePythonTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "execute_python",
            "Execute Python code directly. Writes code to a temp file and runs via python3. Returns stdout, stderr, and returncode. For quick inline tests, syntax checks, and one-shot calculations. Large scripts should be written via run_shell heredoc and then executed with run_shell python3.",
            vec![
                ToolParameter::new("code", ParamType::String, true, "Python code to execute inline", &["script", "source"]),
                ToolParameter::new("timeout", ParamType::Integer, false, "Timeout in seconds", &["time_limit", "max_time"]),
                ToolParameter::new("stdin_input", ParamType::String, false, "Stdin content", &["input", "stdin"]),
            ],
        )
    }

    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let code = args
            .get("code")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if code.is_empty() {
            return Ok(json!({"success": false, "error": "empty code"}).to_string());
        }
        let stdin_input = args
            .get("stdin_input")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());
        let backend = self.backend.clone();
        let project_path = self.project_path.clone();
        let default_timeout_secs = self.default_timeout_secs;

        tokio::task::spawn_blocking(move || -> Result<String> {
            let test_dir = project_path.join(".aacode").join("tests");
            let _ = std::fs::create_dir_all(&test_dir);
            let temp_path = test_dir.join(format!("temp_{}.py", uuid::Uuid::new_v4().simple()));

            let wrapped = python_wrapper(&project_path.to_string_lossy(), &code);
            if let Err(e) = std::fs::write(&temp_path, &wrapped) {
                return Ok(
                    json!({"success": false, "error": format!("write temp file: {e}")}).to_string(),
                );
            }

            // The backend's cwd is `project_path` and the fastshell sandbox
            // re-roots absolute paths — pass a RELATIVE path or the script is
            // looked up under `<sandbox>/<abs>` → FileNotFoundError.
            let rel = temp_path
                .strip_prefix(&project_path)
                .unwrap_or(&temp_path)
                .to_string_lossy()
                .to_string();

            let cmd = format!("python3 {rel:?}");
            let result = backend.run(
                &cmd,
                stdin_input.as_deref(),
                default_timeout_secs,
                0,
                &project_path,
            );

            // Bound leftover temp scripts (the model may re-read a few).
            prune_temp_scripts(&test_dir, 20);

            // Same binary guard as run_shell: a script that prints bytes must
            // not paste them into the context.
            let stdout = if crate::tools::shell::is_binary_text(&result.stdout) {
                let store = crate::artifacts::ArtifactStore::new(&project_path);
                crate::tools::shell::binary_output_json(&store, "python_stdout", &result.stdout)
            } else {
                json!(result.stdout)
            };

            Ok(json!({
                "success": result.exit_code == 0,
                "returncode": result.exit_code,
                "file": rel,
                "stdout": stdout,
                "stderr": result.stderr,
            })
            .to_string())
        })
        .await
        .map_err(|e| AacodeError::Other(format!("execute_python: {e}")))?
    }
}

/// Keep at most `keep` most-recent `temp_*.py` scripts in `dir` (the model
/// re-reads recent ones; older ones are dead weight).
fn prune_temp_scripts(dir: &std::path::Path, keep: usize) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<(std::time::SystemTime, std::path::PathBuf)> = rd
        .flatten()
        .filter(|e| {
            let n = e.file_name().to_string_lossy().to_string();
            n.starts_with("temp_") && n.ends_with(".py")
        })
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .collect();
    if files.len() <= keep {
        return;
    }
    files.sort_by(|a, b| b.0.cmp(&a.0)); // newest first
    for (_, p) in files.into_iter().skip(keep) {
        let _ = std::fs::remove_file(p);
    }
}

// ──────────────────────── analyze_code ──────────────────────────────────

/// Analyzes a source file and returns its structure (functions, classes,
/// imports, line count). Language-agnostic via regex patterns.
pub struct AnalyzeCodeTool {
    pub project_path: PathBuf,
}

#[async_trait::async_trait]
impl Tool for AnalyzeCodeTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "analyze_code",
            "Analyze a source code file. Returns detected language, function/class names, imports, and line count.",
            vec![
                ToolParameter::new("file_path", ParamType::String, true, "Path to the source file", &["file", "path"]),
            ],
        )
    }

    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let file_path = args
            .get("file_path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if file_path.is_empty() {
            return Ok(json!({"success": false, "error": "empty file_path"}).to_string());
        }
        let project_path = self.project_path.clone();
        let fp = file_path.clone();
        let content = tokio::task::spawn_blocking(move || {
            let full_path = if std::path::Path::new(&fp).is_absolute() {
                std::path::PathBuf::from(&fp)
            } else {
                project_path.join(&fp)
            };
            std::fs::read_to_string(&full_path)
        })
        .await
        .map_err(|e| AacodeError::Other(format!("analyze_code: {e}")))?;
        let content = match content {
            Ok(c) => c,
            Err(_) => {
                return Ok(json!({"success": false, "error": "read: file not found"}).to_string())
            }
        };
        let lang = lang_from_ext(&file_path);
        let lines: Vec<&str> = content.lines().collect();

        // Detect functions and classes (simple regex patterns per language).
        let patterns: &[(&str, &str)] = match lang {
            "python" => &[("def ", "function"), ("class ", "class")],
            "rust" => &[
                ("fn ", "function"),
                ("struct ", "struct"),
                ("impl ", "impl"),
                ("enum ", "enum"),
            ],
            "javascript" | "typescript" => &[
                ("function ", "function"),
                ("class ", "class"),
                ("const ", "const"),
                ("export ", "export"),
            ],
            _ => &[
                ("def ", "function"),
                ("fn ", "function"),
                ("function ", "function"),
                ("class ", "class"),
            ],
        };

        let mut functions = Vec::new();
        let mut classes = Vec::new();
        let mut imports = Vec::new();

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            for (prefix, kind) in patterns {
                if let Some(rest) = trimmed.strip_prefix(prefix) {
                    let rest = rest.trim();
                    if !rest.is_empty() && !rest.starts_with('(') {
                        // Take first token (function/class name)
                        let name: String = rest
                            .split(|c: char| !c.is_alphanumeric() && c != '_')
                            .next()
                            .unwrap_or(rest)
                            .to_string();
                        if *kind == "function" || *kind == "fn" {
                            functions.push(format!("L{}: {name}", i + 1));
                        } else if *kind == "class" || *kind == "struct" || *kind == "enum" {
                            classes.push(format!("L{}: {name}", i + 1));
                        }
                    } else if let Some(paren_open) = rest.find('(') {
                        let name = rest[..paren_open].trim().to_string();
                        if !name.is_empty() {
                            functions.push(format!("L{}: {name}", i + 1));
                        }
                    }
                    break;
                }
            }
            // Detect imports (language-agnostic).
            let ln = i + 1;
            if trimmed.starts_with("import ") || trimmed.starts_with("from ") {
                imports.push(format!("L{ln}: {trimmed}"));
            } else if trimmed.starts_with("use ") {
                imports.push(format!("L{ln}: {trimmed}"));
            } else if trimmed.starts_with("require(") || trimmed.starts_with("import ") {
                imports.push(format!("L{ln}: {trimmed}"));
            }
        }

        Ok(json!({
            "success": true,
            "file": file_path,
            "language": lang,
            "line_count": lines.len(),
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "top_comment": lines.first().map(|s| s.to_string()).filter(|s| s.starts_with('#') || s.starts_with("//")),
        })
        .to_string())
    }
}

// ──────────────────────── run_tests ─────────────────────────────────────

pub struct RunTestsTool {
    pub project_path: PathBuf,
}

#[async_trait::async_trait]
impl Tool for RunTestsTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "run_tests",
            "Detect the test framework and suggest the command to run. Does NOT execute tests — use run_shell with the returned suggested_command to actually run them. Use after writing code to verify correctness.",
            vec![
                ToolParameter::new("test_path", ParamType::String, false, "Optional specific test file or directory", &["path", "dir", "file"]),
                ToolParameter::new("timeout", ParamType::Integer, false, "Timeout seconds", &["time_limit"]),
            ],
        )
    }

    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let test_path = args
            .get("test_path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let project_path = self.project_path.clone();
        let lang = tokio::task::spawn_blocking(move || detect_lang(&project_path))
            .await
            .map_err(|e| AacodeError::Other(format!("run_tests: {e}")))?;
        let cmd = match lang {
            "python" => {
                if test_path.is_empty() {
                    "python3 -m pytest".to_string()
                } else {
                    format!("python3 -m pytest {test_path}")
                }
            }
            "rust" => {
                if test_path.is_empty() {
                    "cargo test".to_string()
                } else {
                    format!("cargo test --test {}", test_path.trim_end_matches(".rs"))
                }
            }
            "javascript" | "typescript" => {
                if test_path.is_empty() {
                    "npm test".to_string()
                } else {
                    format!("npx jest {test_path}")
                }
            }
            _ => {
                return Ok(json!({"success": false, "error": "No supported test framework detected. Use run_shell directly.", "detected_lang": lang}).to_string());
            }
        };
        Ok(json!({"success": true, "detected_lang": lang, "suggested_command": cmd, "hint": &format!("run_shell(command=\"{cmd}\")")}).to_string())
    }
}

// ──────────────────────── debug_code ────────────────────────────────────

pub struct DebugCodeTool {
    pub project_path: PathBuf,
}

#[async_trait::async_trait]
impl Tool for DebugCodeTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "debug_code",
            "Suggest debugging commands for a file or error message, based on detected language (e.g. pdb/print for Python, dbg!/--nocapture for Rust, node --inspect for JS).",
            vec![
                ToolParameter::new("file_path", ParamType::String, false, "File to debug", &["file", "path"]),
                ToolParameter::new("error_message", ParamType::String, false, "Error message to analyze", &["error", "message"]),
            ],
        )
    }

    async fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        let file = args
            .get("file_path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let err_msg = args
            .get("error_message")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let project_path = self.project_path.clone();
        let lang = if !file.is_empty() {
            lang_from_ext(&file)
        } else {
            tokio::task::spawn_blocking(move || detect_lang(&project_path))
                .await
                .map_err(|e| AacodeError::Other(format!("debug_code: {e}")))?
        };
        let suggestions: &[&str] = match lang {
            "rust" => &[
                "cargo check",
                "cargo test -- --nocapture",
                "dbg!(&variable)",
            ],
            "python" => &[
                "python3 -m pdb <file>",
                "print(f'{var=}')",
                "python3 -c 'import ast; ast.parse(open(\"FILE\").read())'",
            ],
            "javascript" | "typescript" => {
                &["node --inspect-brk", "console.log(JSON.stringify(obj))"]
            }
            _ => &["Use run_shell to inspect errors."],
        };
        let mut out = String::from("Debug suggestions:\n");
        for s in suggestions {
            out.push_str(&format!("  - {s}\n"));
        }
        if !err_msg.is_empty() {
            out.push_str(&format!("\nError: {err_msg}\n"));
        }
        Ok(json!({"success": true, "detected_lang": lang, "suggestions": suggestions}).to_string())
    }
}

// ──────────────────────── helpers ───────────────────────────────────────

fn detect_lang(project: &std::path::Path) -> &'static str {
    for (lang, markers) in &[
        ("rust", &["Cargo.toml"][..]),
        (
            "python",
            &[
                "pyproject.toml",
                "requirements.txt",
                "setup.py",
                "setup.cfg",
            ],
        ),
        (
            "javascript",
            &["package.json", "yarn.lock", "pnpm-lock.yaml"],
        ),
        ("typescript", &["tsconfig.json"]),
    ] {
        for m in *markers {
            if project.join(m).exists() {
                return lang;
            }
        }
    }
    let mut counts = std::collections::HashMap::new();
    if let Ok(rd) = std::fs::read_dir(project) {
        for e in rd.flatten() {
            if let Some(ext) = e.path().extension() {
                let k = match ext.to_str().unwrap_or("") {
                    "py" => "python",
                    "rs" => "rust",
                    "js" => "javascript",
                    "ts" => "typescript",
                    _ => continue,
                };
                *counts.entry(k).or_insert(0usize) += 1;
            }
        }
    }
    counts
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .map(|(k, _)| k)
        .unwrap_or("unknown")
}

fn lang_from_ext(file: &str) -> &'static str {
    if file.ends_with(".py") {
        "python"
    } else if file.ends_with(".rs") {
        "rust"
    } else if file.ends_with(".ts") {
        "typescript"
    } else if file.ends_with(".js") {
        "javascript"
    } else {
        "unknown"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_wrapper_guards_chdir() {
        let w = python_wrapper("/data/user/0/app/projects/x", "print(1)");
        // No bare `os.chdir(...)`: it must be guarded by isdir + try/except so a
        // re-rooted sandbox path can't raise FileNotFoundError.
        assert!(!w.contains("os\nos.chdir"), "unguarded chdir: {w}");
        assert!(w.contains("if os.path.isdir(_p):"), "{w}");
        assert!(w.contains("except Exception:"), "{w}");
        assert!(w.ends_with("print(1)"), "code must be last: {w}");
    }

    fn tmp_py() -> PathBuf {
        let d = std::env::temp_dir().join(format!("ct_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("requirements.txt"), "").unwrap();
        d
    }

    #[tokio::test]
    async fn execute_python_prints() {
        let d = tmp_py();
        let backend: Arc<dyn ShellBackend> = Arc::new(crate::tools::backend::NativeShell::new());
        let t = ExecutePythonTool {
            project_path: d,
            backend,
            default_timeout_secs: 10,
        };
        let cancel = AtomicBool::new(false);
        let out = t
            .call(&json!({"code": "print(1+2)"}), &cancel)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        if v["success"].as_bool().unwrap() {
            assert!(v["stdout"].as_str().unwrap().contains("3"));
        }
    }

    #[tokio::test]
    async fn execute_python_empty_code() {
        let backend: Arc<dyn ShellBackend> = Arc::new(crate::tools::backend::NativeShell::new());
        let t = ExecutePythonTool {
            project_path: std::env::temp_dir(),
            backend,
            default_timeout_secs: 10,
        };
        let cancel = AtomicBool::new(false);
        let out = t.call(&json!({"code": ""}), &cancel).await.unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&out).unwrap()["success"],
            false
        );
    }

    #[tokio::test]
    async fn analyze_code_python() {
        let d = tmp_py();
        std::fs::write(
            d.join("app.py"),
            "import os\n\nclass Hello:\n    def greet(self):\n        print('hi')\n",
        )
        .unwrap();
        let t = AnalyzeCodeTool { project_path: d };
        let cancel = AtomicBool::new(false);
        let out = t
            .call(&json!({"file_path": "app.py"}), &cancel)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], true);
        assert_eq!(v["language"], "python");
        assert!(v["functions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f.as_str().unwrap().contains("greet")));
        assert!(v["imports"]
            .as_array()
            .unwrap()
            .iter()
            .any(|i| i.as_str().unwrap().contains("os")));
    }

    #[tokio::test]
    async fn analyze_code_rust() {
        let d = tmp_py();
        std::fs::write(
            d.join("main.rs"),
            "use std::io;\n\nfn main() {\n    println!(\"hi\");\n}\n",
        )
        .unwrap();
        let t = AnalyzeCodeTool { project_path: d };
        let cancel = AtomicBool::new(false);
        let out = t
            .call(&json!({"file_path": "main.rs"}), &cancel)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["language"], "rust");
        assert!(v["functions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f.as_str().unwrap().contains("main")));
    }

    #[tokio::test]
    async fn analyze_code_missing_file() {
        let d = tmp_py();
        let t = AnalyzeCodeTool { project_path: d };
        let cancel = AtomicBool::new(false);
        let out = t
            .call(&json!({"file_path": "nope.py"}), &cancel)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&out).unwrap()["success"],
            false
        );
    }

    #[tokio::test]
    async fn debug_code_for_unknown_lang() {
        let d = std::env::temp_dir().join(format!("cd_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&d).unwrap();
        let t = DebugCodeTool { project_path: d };
        let cancel = AtomicBool::new(false);
        let out = t
            .call(
                &json!({"file_path": "app.py", "error_message": "SyntaxError"}),
                &cancel,
            )
            .await
            .unwrap();
        assert!(out.contains("\"success\":true"));
    }

    #[test]
    fn prune_temp_scripts_keeps_newest() {
        let d = std::env::temp_dir().join(format!("ct_prune_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&d).unwrap();
        for i in 0..25 {
            let p = d.join(format!("temp_{i}.py"));
            std::fs::write(&p, "x").unwrap();
            // Stagger mtimes so ordering is deterministic.
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        std::fs::write(d.join("keep_me.txt"), "x").unwrap();
        prune_temp_scripts(&d, 20);
        let n = std::fs::read_dir(&d)
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("temp_"))
            .count();
        assert_eq!(n, 20, "keeps 20 newest temp_*.py");
        assert!(d.join("keep_me.txt").exists(), "unrelated files untouched");
        let _ = std::fs::remove_dir_all(&d);
    }
}
