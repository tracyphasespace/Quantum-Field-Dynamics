# Quick Start: Syncing Results to Your Local Machine

This guide shows you how to easily pull all QFD V15 publication figures and results to your local computer with a single command.

## 🚀 One-Command Sync

### Option 1: Bash Script (Linux/Mac)

```bash
cd projects/astrophysics/qfd-supernova-v15
./pull_results.sh
```

**Default destination**: `~/QFD_Results`

**Custom destination**:
```bash
./pull_results.sh --local-dir /path/to/your/directory
```

### Option 2: Python Script (Cross-platform: Linux/Mac/Windows)

```bash
cd projects/astrophysics/qfd-supernova-v15
python pull_results.py
```

**Custom destination**:
```bash
python pull_results.py --local-dir /path/to/your/directory
```

## 📦 What Gets Synced

The scripts automatically:
1. ✓ Pull latest changes from git
2. ✓ Copy all publication figures (Fig 4, 5, 6, 8)
3. ✓ Copy validation plots (Fig 1, 2, 3)
4. ✓ Copy per-survey diagnostic CSVs
5. ✓ Copy Stage 3 results and posterior samples
6. ✓ Copy all documentation (validation reports, templates)
7. ✓ Create an INDEX.md file for easy navigation

## 📂 Result Structure

After running the sync script, you'll have:

```
~/QFD_Results/                           # (or your custom path)
├── INDEX.md                            # Start here!
├── figures/
│   ├── fig4_hubble_diagram.png
│   ├── fig5_corner_plot.png
│   ├── fig6_per_survey_residuals.png
│   ├── fig8_holdout_performance.png
│   └── composite_all_figures.png
├── validation_plots/
│   ├── figure1_alpha_pred_validation.png
│   ├── figure2_wiring_bug_detection.png
│   └── figure3_stage3_guard.png
├── reports/
│   ├── summary_overall.csv
│   ├── summary_by_survey_alpha.csv
│   ├── zbin_alpha_by_survey.csv
│   ├── train_rms_by_survey.csv
│   └── test_rms_by_survey.csv
├── stage3_results.csv                  # 300 SNe residuals
├── posterior_samples.csv               # 2000 MCMC samples
├── PUBLICATION_FIGURES_SUMMARY.md      # Detailed figure descriptions
├── PUBLICATION_TEMPLATE.md             # Paper template
├── CODE_VERIFICATION.md                # Code verification report
└── HOTFIX_VALIDATION.md                # Validation summary
```

## 🔄 Re-sync After Updates

Whenever new improvements are made, just run the script again:

```bash
# It will automatically pull latest changes and update all files
./pull_results.sh
# or
python pull_results.py
```

## 💡 Quick Tips

**View all figures at once**:
- Open `figures/composite_all_figures.png` to see all 4 publication figures in one view

**Read the summary first**:
- Start with `PUBLICATION_FIGURES_SUMMARY.md` for detailed descriptions of each figure

**Load data in Python**:
```python
import pandas as pd
df = pd.read_csv("stage3_results.csv")
posterior = pd.read_csv("posterior_samples.csv")
```

**Load data in R**:
```r
df <- read.csv("stage3_results.csv")
posterior <- read.csv("posterior_samples.csv")
```

## 🎯 Example Workflow

```bash
# 1. Clone or navigate to repository
cd /path/to/Quantum-Field-Dynamics/projects/astrophysics/qfd-supernova-v15

# 2. Sync all results (first time)
./pull_results.sh

# 3. Open results folder
cd ~/QFD_Results

# 4. View the index
cat INDEX.md

# 5. Open figures in your image viewer
open figures/  # Mac
xdg-open figures/  # Linux
explorer figures/  # Windows

# Later: Re-sync after improvements
cd /path/to/Quantum-Field-Dynamics/projects/astrophysics/qfd-supernova-v15
./pull_results.sh  # Updates everything automatically
```

## 🆘 Troubleshooting

**Script not executable**:
```bash
chmod +x pull_results.sh pull_results.py
```

**Git pull conflicts**:
```bash
# Stash local changes first
git stash
./pull_results.sh
```

**Python not found (Windows)**:
- Make sure Python 3.6+ is installed
- Try `python3 pull_results.py` instead

**Permission errors**:
- Choose a directory you have write access to:
  ```bash
  ./pull_results.sh --local-dir ~/Documents/QFD_Results
  ```

## 📧 Questions?

Check the full documentation in `PUBLICATION_FIGURES_SUMMARY.md` or the repository README.

---

**Pro tip**: Bookmark `~/QFD_Results` in your file browser for instant access to the latest figures!
