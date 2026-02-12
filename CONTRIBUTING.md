# Contributing to Siftrank

## Git Workflow

### Branch Naming Convention
Use the following branch naming format:
```
<type>/siftrank-<id>-<description>
```

Where `<type>` is one of:
- `feat` for new features.
- `fix` for bug fixes.
- `refactor` for code refactoring.
- `docs` for documentation updates.

Example:
```
feat/siftrank-123-add-new-feature
```

## Commit Message Format

Adhere to the Conventional Commits format:
```
<type>(<scope>): <description>

Refs: siftrank-X
```

Where `<type>` is one of `fix`, `feat`, `docs`, `style`, `refactor`, `perf`, `test`, etc.

Example:
```
feat(api): add user authentication

Refs: siftrank-456
```

## Pull Request Process

1. **Create a Branch**: Start by creating a new branch from the main branch using your naming convention.
   ```bash
   git checkout -b feat/siftrank-<id>-<description>
   ```

2. **Commit Changes**: Make changes and commit them with the specified format.
   ```bash
   git add .
   git commit -m "feat(api): add user authentication

Refs: siftrank-456"
   ```

3. **Push Branch**: Push your branch to the remote repository.
   ```bash
   git push origin feat/siftrank-<id>-<description>
   ```

4. **Create Pull Request (PR)**: Create a PR in GitHub, ensuring it references the relevant issue.

5. **Wait for Review**: Wait for human review; do not merge your own PRs.

## Beads Integration

1. **Update Ticket Status**: Set the ticket status to `in_progress` when you start working on an issue.
2. **Reference Tickets**: Include the reference in each commit message (as shown above).
3. **Close Tickets**: Close the ticket once the PR is merged and changes are live.

## Testing Requirements

Before creating a pull request, ensure all tests pass by running:

1. **Static Analysis**:
   ```bash
   gosec ./...
   ```

2. **Unit Tests**:
   ```bash
   go test -cover ./...
   ```

3. **Build Verification**:
   ```bash
   go build
   ```

By following these guidelines, you help maintain the quality and consistency of the Siftrank project.

---

Thank you for your contributions!
