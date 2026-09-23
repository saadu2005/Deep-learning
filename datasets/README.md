# Datasets

## Purpose

This file explains where to keep local datasets for the repository and how to avoid committing large data files.

## Real-Life Example

For a sales-forecasting exercise, keep a small `sales.csv` with dates and units sold here or document a public download source. A model can learn from past rows to estimate future demand; large raw datasets should stay outside Git.

Place local datasets in this folder when a project requires them.

Do not commit very large datasets to GitHub. Prefer:
- public dataset links
- dataset download scripts
- small sample datasets
- `.gitignore` for downloaded data
