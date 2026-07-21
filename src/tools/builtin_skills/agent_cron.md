# agent_cron

## Description
Meta-skill: how to create, update and manage scheduled (cron) tasks. Follow this guide whenever the user asks to set up a recurring task.

## Parameters
- name: task name — lowercase letters, digits and underscores only
- action: "create", "update", "list", "disable" or "delete"

## Example
run_skills("agent_cron")

## Checking if the host supports cron

Your host (the app you are running inside) may or may not support scheduled
tasks. To check, read the capability marker:

```
run_shell("cat /path/to/cron/.capability")
```

If the output contains `register_cron`, the host supports it. If the command
fails or the content is absent, you MUST reply to the user: "该应用版本不支持定时任务功能" /
"This app version does not support scheduled tasks." Do NOT attempt to create
entries without the marker.

**The exact directory path depends on the host.** Use the environment to
determine it. On Android/iOS hosts the skills directory is the same parent
as the cron directory — check `run_shell` context to locate it:

```
run_shell("ls {SKILLS_DIR}/../cron/.capability")
```

Replace `{SKILLS_DIR}` with the actual skills directory path you see in
your system prompt or working-directory context (e.g. `/skills`).

## Cron entry format

All cron entries live in the cron directory as JSON files:
`<cron_dir>/<name>.json`

```json
{
  "name": "daily-sync",
  "project": "/absolute/path/to/project",
  "schedule": {"days": [], "hour": 9, "minute": 0},
  "session_id": "",
  "task_prompt": "git pull origin main && run tests",
  "enabled": true,
  "created_at": "2026-07-19T10:00:00"
}
```

Field guide:
- `name` — unique id (lowercase, digits, underscores, no spaces)
- `project` — ABSOLUTE path to the project directory. Use your working-directory
  context to determine this. NEVER use a relative path.
- `schedule.days` — empty array [] means EVERY day. Otherwise list weekdays as
  numbers: 1=Monday, 2=Tuesday, ..., 7=Sunday. e.g. `[1,3,5]` = Mon/Wed/Fri.
- `schedule.hour` — 0-23
- `schedule.minute` — 0-59
- `session_id` — "" (empty) = each run creates a NEW session. If you want the
  task to continue an EXISTING conversation, fill in the session id (the user
  can get it from the app's Cron page). For new tasks, leave it empty.
- `task_prompt` — the instructions the agent will receive when the cron fires.
  MUST be a non-empty string describing the task clearly. The agent will be
  informed that this is a cron run and which cron entry triggered it.
- `enabled` — true = active, false = paused (the task is preserved but won't
  run until re-enabled). Prefer disabling over deleting.
- `created_at` — ISO-8601 timestamp (set it to current time).

## How to create a cron entry (write JSON with heredoc)

CRITICAL — before you write the file:
1. Get the absolute project path: `run_shell("pwd")` — this is your CURRENT
   working directory. Use that EXACT string for the `project` field below.
2. The `task_prompt` field MUST contain the user's actual task description
   (the reason they asked for a scheduled task). NEVER put placeholder text
   like "git pull" — use precisely what the user described.

Write the file with `{SKILLS_DIR}/../cron/<name>.json`:
```
cat > {SKILLS_DIR}/../cron/<name>.json <<'EOF'
{
  "name": "<name>",
  "project": "<PWD_OUTPUT>",
  "schedule": {"days": [], "hour": 9, "minute": 0},
  "session_id": "",
  "task_prompt": "<USER'S ACTUAL TASK DESCRIPTION>",
  "enabled": true,
  "created_at": "<CURRENT_ISO_TIMESTAMP>"
}
EOF
```
(The `{SKILLS_DIR}` placeholder is substituted at runtime. Use the value you
see in your system prompt / skill context. Typically it is `/skills` or an
absolute path like `/data/.../files/fastshell/skills`.)

## How to list existing tasks

```
run_shell("ls /path/to/cron/*.json")
```

Or to view details of one: run_shell("cat /path/to/cron/name.json").

## How to update a task

Read the existing JSON, modify the fields, then overwrite with the heredoc
pattern above. Keep the `name` unchanged. Common changes:
- Change `task_prompt` to refine what the cron does
- Set `enabled: false` to pause it
- Set `enabled: true` to resume it
- Adjust `schedule` to change the time

## How to delete a task

```
run_shell("rm /path/to/cron/name.json")
```

## Rules

- Task names: ^[a-z][a-z0-9_]*$ — no spaces, no dashes, no uppercase.
- NEVER propose or create a cron entry unless the `.capability` check passed.
- NEVER overwrite an existing cron entry unless the user explicitly asked to
  modify that specific task. Check with `ls` first.
- Always use ABSOLUTE paths for `project` inside the JSON — relative paths
  will be rejected by the host scheduler.
- `task_prompt` must be clear and self-contained. The agent receiving it will
  only see that prompt (plus a "this is a cron run" note from the host).
