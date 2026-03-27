# Git Quick Guide

## 1. Load a Previous Commit (Detached HEAD)

```bash
git log --oneline
git checkout <commit-hash>
```

> You are now in a *detached HEAD* state. Changes here are not on any branch unless you create one.

To create a branch from this state:

```bash
git checkout -b new-branch-name
```

---

## 2. Return to Main Branch & Delete Local Changes

### Discard ALL local changes (dangerous)

```bash
git checkout main
git reset --hard
```

### Also remove untracked files:

```bash
git clean -fd
```

---

## 3. Commit on a Separate Branch

### Create and switch to a new branch:

```bash
git checkout -b feature-branch
```

### Stage and commit:

```bash
git add .
git commit -m "Your message"
```

---

## 4. Merge Branch into Main

### Step 1: Go to main

```bash
git checkout main
```

### Step 2: Merge branch

```bash
git merge feature-branch
```

---

## 5. Optional: Delete Branch After Merge

```bash
git branch -d feature-branch
```

---

## Notes (Worth Knowing)

* `git status` → always check before doing anything
* `git log --oneline --graph` → visualize history
* Avoid `--hard` unless you’re sure you want to lose work
* If merge conflicts happen: fix files → `git add .` → `git commit`

---
