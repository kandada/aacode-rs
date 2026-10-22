// Copyright (c) 2026 xiefujin <490021684@qq.com>
// Licensed under GPL-3.0, see LICENSE file for full license terms.

//! Version consistency: `Cargo.toml`, the workspace `Cargo.lock` and
//! `PUBLISH.md` must all agree with the compiled crate version, so a release
//! can never ship a mismatched version string.

fn read(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

#[test]
fn versions_are_consistent() {
    let expected = env!("CARGO_PKG_VERSION");
    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    // Cargo.toml `[package] version`.
    let manifest = read(&format!("{manifest_dir}/Cargo.toml"));
    let manifest_version = manifest
        .lines()
        .map(str::trim)
        .find_map(|l| l.strip_prefix("version = "))
        .map(|v| v.trim_matches('"').to_string())
        .expect("version in Cargo.toml");
    assert_eq!(manifest_version, expected, "Cargo.toml version");

    // Workspace Cargo.lock `[[package]] name = "aacode-rs"`.
    let lock = read(&format!("{manifest_dir}/../Cargo.lock"));
    let mut lock_version = None;
    let mut lines = lock.lines();
    while let Some(line) = lines.next() {
        if line.trim() == "name = \"aacode-rs\"" {
            if let Some(next) = lines.next() {
                if let Some(v) = next.trim().strip_prefix("version = ") {
                    lock_version = Some(v.trim_matches('"').to_string());
                    break;
                }
            }
        }
    }
    assert_eq!(
        lock_version.as_deref(),
        Some(expected),
        "Cargo.lock version for aacode-rs"
    );

    // PUBLISH.md example line: `version = "x.y.z"  # ...`.
    let publish = read(&format!("{manifest_dir}/PUBLISH.md"));
    let publish_version = publish
        .lines()
        .map(str::trim)
        .find_map(|l| l.strip_prefix("version = "))
        .map(|v| {
            v.split_whitespace()
                .next()
                .unwrap_or("")
                .trim_matches('"')
                .to_string()
        })
        .expect("version in PUBLISH.md");
    assert_eq!(publish_version, expected, "PUBLISH.md version");
}
