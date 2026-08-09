// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Skills — `run_skills` with three modes (__list__, __info__, execute).
//!
//! Ported from Python `tools/skills_tools.py`, adapted for the no-embedded-
//! Python model: skills are document-type. `execute` returns the SKILL.md
//! instruction guide so the model can follow it using `run_shell`.
//!
//! Three skill paths coexist:
//!
//! | Path                            | Condition                    | Scope         |
//! |---------------------------------|------------------------------|---------------|
//! | Builtin (compiled into binary)  | `user_dir` is set            | Global        |
//! | `user_dir`                      | `user_dir` is set            | Global        |
//! | `<project>/skills/` + `.aacode` | `user_dir` is NOT set (CLI)  | Per-project   |
//!
//! When `user_dir` is configured (mobile hosts / desktop with
//! AACODE_SKILLS_DIR env): builtin skills + user-dir skills are loaded,
//! project directories are NOT scanned (skills are app-level; also prevents
//! prompt injection from cloned repos). A user skill with the same name
//! overrides a builtin.
//!
//! When `user_dir` is not configured (legacy/desktop CLI without
//! AACODE_SKILLS_DIR): only per-project `<project>/skills` and
//! `<project>/.aacode/skills` are scanned, no builtins are injected.

use super::registry::Tool;
use super::schema::{ParamType, ToolParameter, ToolSchema};
use crate::error::Result;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

/// Builtin document skills embedded at compile time. `{SKILLS_DIR}` inside
/// the body is replaced with the configured user skills directory.
///
/// skill_creator and book_writer are always injected when user_dir is set.
/// The others are **gated by the host** via `config.skills.extra_builtins` —
/// only hosts that explicitly declare support (e.g. Android app brings cron
/// scheduling) will see them, keeping desktop/CLI deployments untouched.
const BUILTIN_SKILLS: &[(&str, &str)] = &[
    ("skill_creator", include_str!("builtin_skills/skill_creator.md")),
    ("book_writer", include_str!("builtin_skills/book_writer.md")),
    ("agent_cron", include_str!("builtin_skills/agent_cron.md")),
];

/// Info about a discovered skill.
pub struct SkillInfo {
    pub name: String,
    pub description: String,
    pub full_md: String,
}

/// Legacy per-project skills directories.
fn skills_dirs(project_path: &Path) -> Vec<PathBuf> {
    vec![
        project_path.join("skills"),
        project_path.join(".aacode").join("skills"),
    ]
}

/// Discover skills. See module docs for the two modes.
///
/// When `user_dir` is set:
///   - `skill_creator` is ALWAYS injected (all hosts get it).
///   - `extra_builtins` from the caller determines which OTHER builtin
///     skills appear (e.g. `agent_cron` only when the host explicitly
///     declares it). Hosts that don't pass `extra_builtins` never see
///     platform-specific builtins.
///
/// `vfs_skills_dir` is an optional VFS-internal path (e.g. `/skills`)
/// used for `{SKILLS_DIR}` substitution instead of the absolute `user_dir`.
/// When `None`, substitution falls back to `user_dir` (legacy behaviour).
pub fn discover_skills(
    project_path: &Path,
    user_dir: Option<&Path>,
    extra_builtins: &[String],
    vfs_skills_dir: Option<&str>,
) -> Vec<SkillInfo> {
    let mut map: BTreeMap<String, SkillInfo> = BTreeMap::new();
    // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0

    match user_dir {
        Some(dir) => {
            // dir = absolute path on disk, used only for reading skill files.
            // prompt_dir = VFS-internal path injected into skill prompts,
            //   so the agent sees `/skills` instead of `/data/.../skills`.
            //   Falls back to dir (legacy absolute path) when not set.
            let dir_str = dir.to_string_lossy();
            let prompt_dir = vfs_skills_dir.unwrap_or(&dir_str);
            for (name, body) in BUILTIN_SKILLS {
                // skill_creator and book_writer are always injected. All others are gated.
                if *name != "skill_creator" && *name != "book_writer" && !extra_builtins.iter().any(|x| x == *name) {
                    continue;
                }
                let content = body.replace("{SKILLS_DIR}", prompt_dir);
                map.insert(
                    (*name).to_string(),
                    SkillInfo {
                        name: (*name).to_string(),
                        description: first_description(&content),
                        full_md: content,
                    },
                );
            }
            scan_dir_into(dir, &mut map);
        }
        None => {
            for dir in skills_dirs(project_path) {
                scan_dir_into(&dir, &mut map);
            }
        }
    }

    map.into_values().collect()
}

fn scan_dir_into(dir: &Path, map: &mut BTreeMap<String, SkillInfo>) {
    let rd = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return,
    };
    for e in rd.flatten() {
        if !e.path().is_dir() {
            continue;
        }
        if let Some(md_path) = find_skill_md(&e.path()) {
            if let Ok(content) = std::fs::read_to_string(&md_path) {
                let name = e.file_name().to_string_lossy().to_string();
                map.insert(
                    name.clone(),
                    SkillInfo {
                        name,
                        description: first_description(&content),
                        full_md: content,
                    },
                );
            }
        }
    }
}

fn find_skill_md(dir: &Path) -> Option<PathBuf> {
    for cand in ["SKILL.md", "skill.md"] {
        let p = dir.join(cand);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Extract a short description: the first non-heading, non-empty line, or the
/// content under a `## Description` heading.
fn first_description(md: &str) -> String {
    // Prefer text after "## Description".
    let lower = md.to_lowercase();
    if let Some(pos) = lower.find("## description") {
        let after = &md[pos..];
        for line in after.lines().skip(1) {
            let t = line.trim();
            if !t.is_empty() && !t.starts_with('#') {
                return t.chars().take(120).collect();
            }
        }
    }
    for line in md.lines() {
        let t = line.trim();
        if !t.is_empty() && !t.starts_with('#') {
            return t.chars().take(120).collect();
        }
    }
    String::new()
}

/// Build the skills list string injected into the system prompt.
pub fn skills_list_for_prompt(
    project_path: &Path,
    user_dir: Option<&Path>,
    extra_builtins: &[String],
    vfs_skills_dir: Option<&str>,
) -> String {
    // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
    let skills = discover_skills(project_path, user_dir, extra_builtins, vfs_skills_dir);
    if skills.is_empty() {
        return "(no skills installed)".to_string();
    }
    skills
        .iter()
        .map(|s| format!("      - {}: {}", s.name, s.description))
        .collect::<Vec<_>>()
        .join("\n")
}

pub struct RunSkillsTool {
    pub project_path: PathBuf,
    /// Configured user skills directory (None = legacy project scanning).
    pub user_dir: Option<PathBuf>,
    /// VFS-internal path for {SKILLS_DIR} substitution (e.g. `/skills`).
    pub vfs_skills_dir: Option<String>,
    /// Host-declared extra builtins (e.g. ["agent_cron"]).
    pub extra_builtins: Vec<String>,
}

impl Tool for RunSkillsTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "run_skills",
            "Skills entry (three modes): run_skills(\"__list__\") lists skills; run_skills(\"__info__\", {\"skill_name\":\"x\"}) shows details; run_skills(\"x\", {...}) returns the skill's instruction guide to follow with run_shell.",
            vec![
                ToolParameter::new("skill_name", ParamType::String, false, "Skill name or __list__/__info__", &["skill", "name"]),
                ToolParameter::new("params", ParamType::Object, false, "Skill params (may include skill_name for __info__)", &["arguments", "args", "kwargs"]),
                ToolParameter::new(
                    "timeout",
                    ParamType::Integer,
                    false,
                    "Max seconds for skill discovery. Default: 5.",
                    &["time_limit", "max_time"],
                ),
            ],
        )
    }

    fn call(&self, args: &Value, _c: &AtomicBool) -> Result<String> {
        // (c) 2026 xiefujin <490021684@qq.com> — GPL-3.0
        let name = args
            .get("skill_name")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let params = args.get("params").cloned().unwrap_or(json!({}));

        let name = name.or_else(|| {
            params
                .get("skill_name")
                .or_else(|| params.get("name"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        });

        let timeout = args
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(5);

        let (tx, rx) = std::sync::mpsc::channel();
        let pp = self.project_path.clone();
        let ud = self.user_dir.clone();
        let eb = self.extra_builtins.clone();
        let vsd = self.vfs_skills_dir.clone();
        std::thread::spawn(move || {
            let skills = discover_skills(&pp, ud.as_deref(), &eb, vsd.as_deref());
            let _ = tx.send(skills);
        });

        let skills = match rx.recv_timeout(Duration::from_secs(timeout)) {
            Ok(s) => s,
            Err(_) => return Ok("Error: skill discovery timed out".to_string()),
        };

        match name.as_deref() {
            None | Some("__list__") | Some("list") | Some("--list") | Some("ls") => {
                if skills.is_empty() {
                    return Ok("No skills available.".to_string());
                }
                let mut out = String::from("Available skills:\n");
                for s in &skills {
                    out.push_str(&format!("- {}: {}\n", s.name, s.description));
                }
                Ok(out)
            }
            Some("__info__") | Some("info") | Some("--info") => {
                let target = params
                    .get("skill_name")
                    .or_else(|| params.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if target.is_empty() {
                    return Ok("Error: skill_name required for __info__ mode".to_string());
                }
                match skills.iter().find(|s| s.name == target) {
                    Some(s) => Ok(s.full_md.clone()),
                    None => Ok(format!("Error: skill '{target}' not found")),
                }
            }
            Some(sname) => match skills.iter().find(|s| s.name == sname) {
                Some(s) => Ok(format!(
                    "Skill '{sname}' is a document skill. Follow these instructions using run_shell/other tools:\n\n{}",
                    s.full_md
                )),
                None => Ok(format!("Error: skill '{sname}' not found")),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_skill(project: &Path, name: &str, body: &str) {
        let dir = project.join("skills").join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("SKILL.md"), body).unwrap();
    }

    fn setup_skill_at(root: &Path, name: &str, body: &str) {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("SKILL.md"), body).unwrap();
    }

    fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "aacode_skills_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn tool(project: PathBuf) -> RunSkillsTool {
        RunSkillsTool {
            project_path: project,
            user_dir: None,
            vfs_skills_dir: None,
            extra_builtins: vec![],
        }
    }

    #[test]
    fn discovers_and_lists() {
        let d = tmp();
        setup_skill(&d, "pandas", "## Description\nData analysis helper\n");
        let t = tool(d.clone());
        let cancel = AtomicBool::new(false);
        let out = t.call(&json!({"skill_name": "__list__"}), &cancel).unwrap();
        assert!(out.contains("pandas"));
        assert!(out.contains("Data analysis helper"));
        // Legacy mode must NOT inject builtins at all (not even skill_creator).
        assert!(!out.contains("skill_creator"));
        assert!(!out.contains("agent_cron"));
    }

    #[test]
    fn info_returns_full_md() {
        let d = tmp();
        setup_skill(&d, "numpy", "## Description\nNumeric\n## Usage\nrun stuff\n");
        let t = tool(d.clone());
        let cancel = AtomicBool::new(false);
        let out = t
            .call(&json!({"skill_name": "__info__", "params": {"skill_name": "numpy"}}), &cancel)
            .unwrap();
        assert!(out.contains("## Usage"));
    }

    #[test]
    fn execute_returns_guide() {
        let d = tmp();
        setup_skill(&d, "deploy", "## Description\nDeploy\nSteps: do X\n");
        let t = tool(d.clone());
        let cancel = AtomicBool::new(false);
        let out = t.call(&json!({"skill_name": "deploy"}), &cancel).unwrap();
        assert!(out.contains("document skill"));
        assert!(out.contains("Steps: do X"));
    }

    #[test]
    fn missing_skill() {
        let d = tmp();
        let t = tool(d);
        let cancel = AtomicBool::new(false);
        let out = t.call(&json!({"skill_name": "ghost"}), &cancel).unwrap();
        assert!(out.contains("not found"));
    }

    #[test]
    fn list_for_prompt() {
        let d = tmp();
        setup_skill(&d, "s1", "## Description\nfirst\n");
        let list = skills_list_for_prompt(&d, None, &[], None);
        assert!(list.contains("s1"));
    }

    #[test]
    fn legacy_dedup_project_dirs() {
        let d = tmp();
        setup_skill(&d, "dup", "## Description\nfrom skills dir\n");
        let hidden = d.join(".aacode").join("skills").join("dup");
        std::fs::create_dir_all(&hidden).unwrap();
        std::fs::write(hidden.join("SKILL.md"), "## Description\nfrom aacode dir\n").unwrap();
        let skills = discover_skills(&d, None, &[], None);
        assert_eq!(skills.iter().filter(|s| s.name == "dup").count(), 1);
    }

    // ── user_dir mode ──────────────────────────────────────────────

    #[test]
    fn user_dir_mode_includes_always_injected_and_user_skills() {
        let project = tmp();
        let user = tmp();
        setup_skill_at(&user, "api_probe", "## Description\nProbe an API\n");
        let skills = discover_skills(&project, Some(&user), &[], None);
        let names: Vec<_> = skills.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"skill_creator"), "skill_creator must always appear: {names:?}");
        assert!(names.contains(&"book_writer"), "book_writer must always appear: {names:?}");
        assert!(names.contains(&"api_probe"));
        assert!(!names.contains(&"agent_cron"), "agent_cron must NOT appear without extra_builtins");
    }

    #[test]
    fn extra_builtins_gates_agent_cron() {
        let project = tmp();
        let user = tmp();
        let skills = discover_skills(&project, Some(&user), &["agent_cron".into()], None);
        let names: Vec<_> = skills.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"agent_cron"));
    }

    #[test]
    fn user_dir_placeholder_substituted() {
        let project = tmp();
        let user = tmp();
        let skills = discover_skills(&project, Some(&user), &[], None);
        let creator = skills.iter().find(|s| s.name == "skill_creator").unwrap();
        assert!(!creator.full_md.contains("{SKILLS_DIR}"));
        assert!(creator.full_md.contains(&user.to_string_lossy().to_string()));
    }

    #[test]
    fn user_skill_overrides_builtin() {
        let project = tmp();
        let user = tmp();
        setup_skill_at(&user, "skill_creator", "## Description\ncustom override\n");
        let skills = discover_skills(&project, Some(&user), &[], None);
        let creator = skills.iter().find(|s| s.name == "skill_creator").unwrap();
        assert_eq!(creator.description, "custom override");
    }

    #[test]
    fn user_dir_mode_via_tool_and_prompt_with_agent_cron_gated() {
        let project = tmp();
        let user = tmp();
        setup_skill_at(&user, "remote_box", "## Description\nRemote sandbox\n## Remote Endpoint\nhttps://x\n## Secret\nabc\n");
        // Without extra_builtins, agent_cron is absent.
        let t = RunSkillsTool {
            project_path: project.clone(),
            user_dir: Some(user.clone()),
            vfs_skills_dir: None,
            extra_builtins: vec!["agent_cron".into()],
        };
        let cancel = AtomicBool::new(false);
        let out = t.call(&json!({"skill_name": "__list__"}), &cancel).unwrap();
        assert!(out.contains("remote_box"));
        assert!(out.contains("agent_cron"));
        // The prompt/summary list must never leak endpoint/secret details.
        assert!(!out.contains("https://x"));
        assert!(!out.contains("abc"));
        let info = t
            .call(&json!({"skill_name": "__info__", "params": {"skill_name": "remote_box"}}), &cancel)
            .unwrap();
        assert!(info.contains("## Remote Endpoint"));
        assert!(info.contains("abc"));
        let prompt = skills_list_for_prompt(&project, Some(&user), &["agent_cron".into()], None);
        assert!(prompt.contains("remote_box"));
        assert!(prompt.contains("agent_cron"));
        assert!(!prompt.contains("abc"));
    }
}
