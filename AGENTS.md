# SiftRank Agent Instructions

## Purpose

This project uses **beads** for lightweight issue tracking. Beads provides local, git-friendly issue management without external dependencies.

## Agent Responsibilities

### When to Create Tickets

- **Bug discovered**: Create immediately with `-t bug -p 0` for security issues, `-p 1-2` for others
- **Feature idea**: Create with `-t feature` and appropriate priority
- **Refactoring needed**: Create with `-t refactor` to track technical debt
- **Research required**: Create with `-t research` for investigation tasks
- **Documentation gaps**: Create with `-t docs` for documentation work

### When to Update Tickets

- **Starting work**: `bd update <id> --status in_progress`
- **Progress notes**: `bd update <id> --notes "description of progress"`
- **Blocked**: `bd update <id> --status blocked --notes "reason for block"`

### When to Close Tickets

- **Work complete**: `bd close <id> --reason "what was done"`
- **Won't fix**: `bd close <id> --reason "won't fix: explanation"`
- **Duplicate**: `bd close <id> --reason "duplicate of <other-id>"`

## Beads Command Reference

```bash
# View work
bd ready              # Show unblocked tickets ready for work
bd list               # Show all open tickets
bd stats              # Show summary statistics
bd show <id>          # Show ticket details

# Create work
bd create "title" -t TYPE -p PRIORITY -d "description"
# Types: bug, feature, refactor, research, docs
# Priority: 0 (highest) to 4 (lowest)

# Update work
bd update <id> --status STATUS    # pending, in_progress, blocked
bd update <id> --notes "text"     # Add progress notes

# Dependencies
bd dep add <blocked> <blocker>    # blocked-id depends on blocker-id
bd dep rm <blocked> <blocker>     # Remove dependency

# Complete work
bd close <id> --reason "description"
```

## Project-Specific Guidance

### Security First (Priority 0)

SiftRank handles user files and API credentials. Security issues take absolute priority:
- Weak RNG for cryptographic operations
- Path traversal vulnerabilities
- File permission issues
- Error handling that could leak information

### Provider Abstraction Foundation

The provider abstraction (Priority 1) enables all future LLM integrations:
- Interface design must be stable before implementations
- Auth strategy pattern supports diverse authentication needs
- Factory pattern enables runtime provider selection

### Dependency Chains

Many tickets have dependencies. Check `bd ready` before starting work to see what's unblocked. Don't start blocked tickets - resolve dependencies first.

### Testing Requirements

All implementations require tests:
- Security fixes need regression tests proving the fix
- New features need unit and integration tests
- Refactors must maintain existing test coverage

## Workflow Example

```bash
# Start your session
bd ready              # See what's available

# Pick a ticket
bd show siftrank-1    # Review details
bd update siftrank-1 --status in_progress

# Do the work...

# Complete
bd close siftrank-1 --reason "Replaced math/rand with crypto/rand in pkg/openai/client.go"
```
