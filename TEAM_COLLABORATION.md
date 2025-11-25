# Team Collaboration Guide

## Branch Structure

We use a **branch-based workflow** to avoid conflicts and maintain clean code.

### Current Branches:

```
main          → Production-ready code (protected)
yila-dev      → Yila's development branch
ngo-dev       → Ngo's development branch
```

---

## Workflow for Each Team Member

### 🔵 For Yila (You)

#### 1. Start Working (in Colab or Local)

```python
# In Colab, after running setup (Cell 1)
!git checkout yila-dev
!git pull origin yila-dev  # Get latest changes
```

#### 2. Make Your Changes

- Edit code, run experiments, refine algorithms
- Test your changes

#### 3. Commit and Push

```python
# Check what changed
!git status

# Add files
!git add .

# Commit with descriptive message
!git commit -m "Updated recommendation scoring algorithm"

# Push to YOUR branch
!git push origin yila-dev
```

#### 4. Create Pull Request (When Ready)

When your feature is complete and tested:
1. Go to: https://github.com/yilalalalala/jewelry-recommendation-system
2. Click "Pull requests" → "New pull request"
3. Base: `main` ← Compare: `yila-dev`
4. Add description of changes
5. Request review from teammate
6. Merge after approval

---

### 🟢 For Ngo (Teammate)

#### 1. First Time Setup

```python
# In Colab, after running setup (Cell 1)
!git checkout -b ngo-dev  # Create your branch
!git push -u origin ngo-dev  # Push to GitHub
```

#### 2. Start Working

```python
# Switch to your branch
!git checkout ngo-dev
!git pull origin ngo-dev  # Get latest changes
```

#### 3. Make Your Changes

- Edit code, run experiments, refine algorithms
- Test your changes

#### 4. Commit and Push

```python
# Check what changed
!git status

# Add files
!git add .

# Commit with descriptive message
!git commit -m "Added data preprocessing improvements"

# Push to YOUR branch
!git push origin ngo-dev
```

#### 5. Create Pull Request (When Ready)

When your feature is complete and tested:
1. Go to: https://github.com/yilalalalala/jewelry-recommendation-system
2. Click "Pull requests" → "New pull request"
3. Base: `main` ← Compare: `ngo-dev`
4. Add description of changes
5. Request review from teammate
6. Merge after approval

---

## Staying Synchronized

### Getting Latest Changes from Main

If your teammate merged something to `main` and you want it:

```python
# Update main branch
!git checkout main
!git pull origin main

# Merge main into your branch
!git checkout yila-dev  # or ngo-dev
!git merge main

# Push updated branch
!git push origin yila-dev  # or ngo-dev
```

### Checking All Branches

```python
# See all branches
!git branch -a

# See current branch
!git branch --show-current
```

---

## Communication Tips

### Before Starting Work:
- 💬 Tell your teammate what you're working on
- 📋 Divide tasks clearly (e.g., "I'll work on the ML model, you work on the UI")

### While Working:
- 🔄 Push frequently (at least daily)
- 📝 Write clear commit messages
- ✅ Test your changes before pushing

### Before Merging:
- 🧪 Make sure tests pass
- 👀 Request review from teammate
- 📖 Update documentation if needed

---

## Common Scenarios

### Scenario 1: Both Working on Same File

If you both edited the same file:

**Person B (who pushes second):**
```python
!git pull origin main  # This may show conflicts

# If conflicts, edit the file to resolve
# Look for markers like <<<<<<< HEAD

# After fixing:
!git add .
!git commit -m "Resolved merge conflicts"
!git push origin ngo-dev
```

### Scenario 2: Want to Test Teammate's Work

```python
# Switch to teammate's branch
!git fetch origin
!git checkout ngo-dev  # or yila-dev

# Run their code
# Test it

# Switch back to your branch
!git checkout yila-dev  # or ngo-dev
```

### Scenario 3: Made Changes on Wrong Branch

```python
# Stash your changes
!git stash

# Switch to correct branch
!git checkout yila-dev

# Apply your changes
!git stash pop

# Commit on correct branch
!git add .
!git commit -m "Your changes"
!git push
```

---

## Colab-Specific Commands

### Quick Reference for Colab:

```python
# ===== SETUP (Run once per session) =====
!git checkout yila-dev  # or ngo-dev
!git pull origin yila-dev  # or ngo-dev

# ===== WHILE WORKING =====
# (make your changes)

# ===== SAVING WORK =====
!git status  # See what changed
!git add .
!git commit -m "Describe your changes"
!git push origin yila-dev  # or ngo-dev

# ===== CHECK CURRENT STATE =====
!git branch --show-current  # Which branch am I on?
!git log --oneline -5  # Recent commits
!git status  # What's changed?
```

---

## Branch Protection (Optional but Recommended)

To protect `main` branch from accidental pushes:

1. Go to: https://github.com/yilalalalala/jewelry-recommendation-system/settings/branches
2. Click "Add rule"
3. Branch name pattern: `main`
4. Check:
   - ✅ Require a pull request before merging
   - ✅ Require approvals (1)
5. Save

This ensures all changes to `main` go through Pull Requests with review.

---

## Visual Workflow

```
YOU (Yila)                          TEAMMATE (Ngo)
    |                                      |
    git checkout yila-dev                  git checkout ngo-dev
    |                                      |
    Make changes                           Make changes
    |                                      |
    git commit & push                      git commit & push
    |                                      |
    Create PR → main                       Create PR → main
          ↓                                      ↓
          ←──────── REVIEW & MERGE ──────────→
                         ↓
                    MAIN BRANCH
                   (Updated code)
```

---

## Getting Help

### Check Branch Status:
```python
!git branch -a  # All branches
!git branch --show-current  # Current branch
!git log --oneline --graph --all  # Visual branch history
```

### Undo Last Commit (Not Pushed Yet):
```python
!git reset --soft HEAD~1  # Undo commit, keep changes
!git reset --hard HEAD~1  # Undo commit, discard changes (DANGEROUS!)
```

### Discard Local Changes:
```python
!git checkout -- filename.py  # Discard changes to one file
!git reset --hard  # Discard ALL changes (DANGEROUS!)
```

---

## Summary of Rules

1. ✅ **NEVER push directly to `main`**
2. ✅ **Always work on your own branch** (`yila-dev` or `ngo-dev`)
3. ✅ **Pull before you start working** (`git pull origin your-branch`)
4. ✅ **Commit frequently** with clear messages
5. ✅ **Use Pull Requests** to merge to main
6. ✅ **Review each other's code** before merging
7. ✅ **Communicate** about what you're working on

---

## Questions?

- Check GitHub's branch documentation: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-branches
- Ask your teammate before making major changes
- Test thoroughly before creating Pull Requests

Happy collaborating! 🚀👥
