# skill_creator

## Description
Meta-skill: how to create, update and optimize skills in this environment. Follow this guide whenever the user asks to add a new skill or improve an existing one.

## Parameters
- name: skill name — lowercase letters, digits and underscores only (e.g. api_tester)
- action: "create" or "update"

## Example
run_skills("skill_creator")

## Skill storage layout

All skills live in the user skills directory (app-level, shared across projects):

```
{SKILLS_DIR}/<skill_name>/SKILL.md
```

A SKILL.md MUST contain these sections, in this order:

```markdown
# <skill_name>

## Description
One sentence describing what this skill does.

## Parameters
- param1: description of param1
- param2: description of param2

## Example
run_skills("<skill_name>", {"param1": "value1"})
```

Optional sections (append AFTER the three required ones when the skill talks
to a remote service, e.g. a remote test sandbox):

```markdown
## Remote Endpoint
https://example.com/api

## Secret
<the secret or API key required by the endpoint>
```

## How to create a skill

1. Check for name conflicts first: run_skills("__list__") — if the name
   already exists, DO NOT overwrite it unless the user explicitly asked to
   modify/optimize that exact skill. Otherwise pick a different name.
2. Write the file with run_shell (heredoc keeps formatting intact):

```
run_shell: mkdir -p {SKILLS_DIR}/my_skill && cat > {SKILLS_DIR}/my_skill/SKILL.md <<'EOF'
# my_skill

## Description
...

## Parameters
- ...

## Example
run_skills("my_skill", {...})
EOF
```

3. Verify: run_skills("__info__", {"skill_name": "my_skill"}) — confirm the
   content is complete and well-formed. Changes take effect immediately, no
   restart needed.

## How to update / optimize a skill

1. Read the current version: run_skills("__info__", {"skill_name": "x"}).
2. Rewrite the full SKILL.md with run_shell (same heredoc pattern). Keep the
   required section structure; preserve the existing "## Remote Endpoint" and
   "## Secret" sections unless the user asked to change them.

## Rules

- Skill names: ^[a-z][a-z0-9_]*$ — no spaces, no dashes, no uppercase.
- Never write skill files anywhere except {SKILLS_DIR}/<name>/SKILL.md.
- Never copy a "## Secret" value into any other file, command output or
  message; it may only exist inside its own SKILL.md.
- Keep Description to one line — it is shown in every system prompt; details
  belong in the body (progressive disclosure keeps context small).
