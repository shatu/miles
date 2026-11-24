"""Prompt text shared by the TMax data prep and agent loop."""

# open-instruct's scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt, which every TMax
# Terminal-RL run passes as --system_prompt_override_file; it replaces the dataset's system turn.
TMAX_SYSTEM_PROMPT = (
    "You are a helpful assistant that can interact with a computer.\n"
    "\n"
    "Your response must include a THOUGHT section before your action where you\n"
    "explain your reasoning. After the THOUGHT, you must call the `bash` tool\n"
    "with EXACTLY ONE bash command (multiple commands chained with `&&` or `||`\n"
    "count as a single action).\n"
    "\n"
    "Failure to follow these rules — calling no tool, calling a tool other than\n"
    "`bash`, or omitting the THOUGHT — will cause your response to be rejected.\n"
)

# The user turn of every TMax task: the tmax repo's rl_data/generator/vanillux_prompts.yaml instance_template
# (commit 1d2f9dcc), which the tmax-15k prompts were rendered with and Vanillux2Agent renders evaluation tasks
# (SWE-bench Verified among them) with. Render with render_instance(task).
TMAX_INSTANCE_TEMPLATE = """Please solve this task:

{{task}}

You can execute bash commands and edit files (with `sed`, `cat > file << 'EOF'`,
etc.) to implement the necessary changes.

## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your
changes and any possible problems.

1. Analyze the codebase / environment by finding and reading relevant files.
2. If applicable, create a script to reproduce the issue or expected behaviour.
3. Implement the change(s) by editing the source code or environment state.
4. Verify your fix works by running your script (or relevant test) again.
5. Test edge cases to ensure your fix is robust.
6. Submit your changes and finish your work by issuing the following command:
   `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
   Do not combine it with any other command. After this command, you cannot
   continue working on this task.

## Important Rules

1. Every response must contain exactly one tool call to `bash`.
2. Directory and environment-variable changes ARE persistent across calls
   in this harness — you can `cd` and `export` and subsequent commands will
   see the change. (This differs from upstream mini-swe-agent's subshell
   model; treat the shell as a long-running login shell.)
3. Long-running commands: wrap with `timeout`, e.g. `timeout 30 <command>`.
4. Interactive commands are not possible. Use `yes`/`no` piping or
   non-interactive flags as appropriate.
5. Output may be truncated. Use `head`, `tail`, `grep`, `sed -n 'A,Bp'`,
   etc. to filter large outputs.

## Useful command examples

### Create a new file:
`cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF`

### Edit files with sed:
`sed -i 's/old_string/new_string/g' filename.py`        # all occurrences
`sed -i '1s/old_string/new_string/' filename.py`         # first on line 1
`sed -i '1,10s/old_string/new_string/g' filename.py`     # lines 1-10

### View file content:
`nl -ba filename.py | sed -n '10,20p'`
"""


def render_instance(task: str) -> str:
    """The user turn for a task description, as vanillux_solver._render_instance renders it."""
    return TMAX_INSTANCE_TEMPLATE.replace("{{task}}", task)
