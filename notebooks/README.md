# Notebooks Directory

This directory contains Jupyter notebooks for exploration, analysis, and documentation.

## Purpose

Notebooks are for:
- Exploratory data analysis (EDA)
- Prototyping models and features
- Creating visualizations
- Documenting analysis results
- Communicating findings

## Naming Convention

Use a naming convention that includes a number (for ordering), the creator's initials, and a short description:

```
<number>-<initials>-<description>.ipynb

Examples:
1.0-jp-initial-data-exploration.ipynb
2.0-jp-feature-engineering.ipynb
3.0-jp-model-training.ipynb
```

## Best Practices

1. **One notebook per analysis**: Keep notebooks focused on a single topic
2. **Use version control**: Commit notebooks to Git (consider using nbdime or jupytext)
3. **Clear all outputs before committing**: Reduces Git conflicts
4. **Add markdown cells**: Document your thought process
5. **Promote good code to src/**: Once code is working, move it to the `src/` directory
6. **Keep notebooks runnable**: Ensure notebooks can be run from top to bottom

## From Notebooks to Production

As your analysis matures:
1. Start with exploratory work in notebooks
2. Refactor working code into functions
3. Move stable functions to `src/` modules
4. Import those modules in your notebooks
5. Keep notebooks for visualization and communication

## Running Notebooks

```bash
# Install Jupyter if not already installed
pip install jupyter

# Start Jupyter server
jupyter notebook

# Or use JupyterLab
pip install jupyterlab
jupyter lab
```
