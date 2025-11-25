# Google Colab Setup Guide

## Quick Start: Open in Colab

**Option 1: Direct Link (Easiest)**

Click this link to open the notebook directly in Colab:
```
https://colab.research.google.com/github/yilalalalala/jewelry-recommendation-system/blob/main/notebooks/Jewelry_Recommendation_Colab.ipynb
```

**Option 2: From GitHub**

1. Go to your repository: https://github.com/yilalalalala/jewelry-recommendation-system
2. Navigate to `notebooks/Jewelry_Recommendation_Colab.ipynb`
3. Click the "Open in Colab" button (if you see one) OR
4. Copy the file URL and paste it in Colab's GitHub tab

**Option 3: Upload Manually**

1. Go to https://colab.research.google.com/
2. Click "Upload" tab
3. Upload `notebooks/Jewelry_Recommendation_Colab.ipynb`

---

## First Time Setup

### 1. Run the Setup Cell (Cell 1)

This will:
- Clone your GitHub repository
- Install all dependencies
- Set up the Python path

### 2. Configure Git (Cell 2)

Replace these with your info:
```python
!git config --global user.name "Cao Yila"  # Your name
!git config --global user.email "your.email@example.com"  # Your email
```

### 3. Create a GitHub Personal Access Token

You need this to push changes from Colab back to GitHub.

**Steps:**
1. Go to https://github.com/settings/tokens
2. Click **"Generate new token (classic)"**
3. Give it a name: `Colab Access`
4. Set expiration: 90 days (or your preference)
5. Select scope: Check **`repo`** (full control of private repositories)
6. Scroll down and click **"Generate token"**
7. **COPY THE TOKEN** (you won't see it again!)
8. Save it somewhere secure (notes app, password manager)

---

## Working in Colab

### Typical Workflow:

1. **Run Setup** (Cell 1) - Clone repo and install dependencies
2. **Import Libraries** (Cell 3) - Load your code
3. **Explore/Experiment** (Cells 4-8) - Test features, refine algorithms
4. **Save Changes** (Cell 9) - Push back to GitHub

### Making Code Changes:

If you want to edit the actual Python files (`src/recommendation_system.py`, etc.):

```python
# Open and edit files
with open('src/recommendation_system.py', 'r') as f:
    content = f.read()

# Make your changes to 'content' here

# Save back
with open('src/recommendation_system.py', 'w') as f:
    f.write(content)
```

Or use Colab's file browser (left sidebar) to edit directly.

---

## Pushing Changes Back to GitHub

### Method 1: With Prompt (Simple)

Run in a cell:
```python
!git add .
!git commit -m "Description of your changes"
!git push origin main
```

When prompted:
- **Username:** `yilalalalala`
- **Password:** Paste your Personal Access Token (not your GitHub password!)

### Method 2: With Token (No Prompt)

Use Cell 9 (Alternative) in the notebook - it will ask for your token securely.

---

## Tips for Colab

### Reconnecting to Runtime

If your Colab session disconnects:
1. Reconnect runtime
2. Re-run Cell 1 (Setup) to clone repo again
3. Continue working

### Checking Your Changes on GitHub

After pushing:
1. Go to https://github.com/yilalalalala/jewelry-recommendation-system
2. Check the "commits" to see your changes
3. Verify files were updated

### Pulling Latest Changes

If you made changes locally and want to sync to Colab:
```python
!git pull origin main
```

### Common Issues

**Problem:** "Permission denied" when pushing
**Solution:** Make sure you're using your Personal Access Token, not your password

**Problem:** "Repository not found"
**Solution:** Check that the repository URL is correct and you have access

**Problem:** "Nothing to commit"
**Solution:** No changes were made. Check `!git status` to see what's tracked

---

## What You Can Do in Colab

- ✅ Test both recommendation systems
- ✅ Experiment with different parameters
- ✅ Analyze the data
- ✅ Refine algorithms
- ✅ Create visualizations
- ✅ Modify code and push changes
- ✅ Add new features
- ✅ Generate reports

---

## Need Help?

- **Colab Documentation:** https://colab.research.google.com/notebooks/intro.ipynb
- **GitHub Tokens:** https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token
- **Your Repository:** https://github.com/yilalalalala/jewelry-recommendation-system

---

## Security Notes

⚠️ **Important:**
- Never commit your GitHub token to the repository
- Don't share notebooks that contain your token
- If you accidentally expose your token, delete it immediately and create a new one
- Colab notebooks in the repository should NOT contain sensitive data

Happy coding! 🚀
