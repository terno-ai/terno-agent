---
name: data-visualization
description: Create charts and plots from query results. Use when the user asks for a chart, graph, plot, dashboard, or visual summary.
---

# Data Visualization

Use Plotly by default. Follow the File Saving Rules section for the
directory (`out_dir`) and the `output_{file_suffix}.<ext>` naming
convention — save charts as interactive HTML:

```python
fig.write_html(os.path.join(out_dir, "output_{file_suffix}.html"), include_plotlyjs="cdn")
```

Use matplotlib only if:
- The user explicitly asks for it
- Plotly cannot generate the required format

Save matplotlib plots as `output_{file_suffix}.png`. Configure
`MPLCONFIGDIR` before importing matplotlib (see the File Saving Rules
section). Use appropriate scaling for axes.
