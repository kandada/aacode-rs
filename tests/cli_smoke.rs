// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

use std::process::Command;

fn cli() -> Command {
    Command::new(env!("CARGO_BIN_EXE_aacode"))
}

fn temp_cwd() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "aacode_cli_smoke_{}_{}",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn help_and_version_do_not_require_configuration() {
    let cwd = temp_cwd();
    let help = cli().current_dir(&cwd).arg("--help").output().unwrap();
    assert!(help.status.success());
    let text = String::from_utf8_lossy(&help.stdout);
    assert!(text.contains("USAGE"));
    assert!(text.contains("--project"));
    assert!(text.contains("--interactive"));

    let version = cli().current_dir(&cwd).arg("--version").output().unwrap();
    assert!(version.status.success());
    let text = String::from_utf8_lossy(&version.stdout);
    assert!(text.contains(env!("CARGO_PKG_VERSION")));
    let _ = std::fs::remove_dir_all(&cwd);
}

#[test]
fn missing_api_key_is_a_clean_configuration_error() {
    let cwd = temp_cwd();
    let project = cwd.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let out = cli()
        .current_dir(&cwd)
        .env_remove("LLM_API_KEY")
        .env_remove("OPENAI_API_KEY")
        .args(["-p", project.to_str().unwrap(), "say hi"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("Configuration error"), "{stderr}");
    assert!(stderr.contains("LLM_API_KEY"), "{stderr}");
    let _ = std::fs::remove_dir_all(&cwd);
}

#[test]
fn unknown_flag_is_treated_as_task_text_not_a_crash() {
    let cwd = temp_cwd();
    let project = cwd.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let out = cli()
        .current_dir(&cwd)
        .env_remove("LLM_API_KEY")
        .env_remove("OPENAI_API_KEY")
        .args(["-p", project.to_str().unwrap(), "--definitely-unknown"])
        .output()
        .unwrap();
    // It reaches config validation rather than panicking or silently exiting 0.
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("Configuration error"), "{stderr}");
    let _ = std::fs::remove_dir_all(&cwd);
}

/// Opt-in real-provider CLI E2E. The key comes only from the environment.
#[test]
#[ignore = "requires LLM_API_KEY and network"]
fn live_cli_task_completes() {
    let cwd = temp_cwd();
    let project = cwd.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let api_key = std::env::var("LLM_API_KEY").expect("set LLM_API_KEY");
    let out = cli()
        .current_dir(&cwd)
        .env("LLM_API_KEY", api_key)
        .env(
            "LLM_API_URL",
            std::env::var("LLM_API_URL").unwrap_or_else(|_| "https://api.deepseek.com/v1".into()),
        )
        .env(
            "LLM_MODEL_NAME",
            std::env::var("LLM_MODEL_NAME").unwrap_or_else(|_| "deepseek-flash".into()),
        )
        .env("LLM_GATEWAY", "openai")
        .args([
            "-p",
            project.to_str().unwrap(),
            "Reply with exactly: CLI_OK",
        ])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stdout={stdout}\nstderr={stderr}");
    assert!(
        stdout.contains("CLI_OK") || stdout.contains("status:"),
        "stdout={stdout}"
    );
    let _ = std::fs::remove_dir_all(&cwd);
}
