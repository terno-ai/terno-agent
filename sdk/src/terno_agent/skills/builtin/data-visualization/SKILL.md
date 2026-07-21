---
name: data-visualization
description: Create charts and plots from query results. Use when the user asks for a chart, graph, plot, dashboard, or visual summary.
---

# Data Visualization

Use Plotly by default. Save charts as interactive HTML using the file
naming convention from the File Saving Rules section:

```python
fig.write_html(os.path.join(out_dir, "output_{file_suffix}.html"), include_plotlyjs="cdn")
```

Use matplotlib only if:
- The user explicitly asks for it
- Plotly cannot generate the required format

Save matplotlib plots a:
`output_{file_suffix}.png`. 

In case using matplotlib, we need to configure config directory before importing the module.
Use appropriate scaling for axes.
