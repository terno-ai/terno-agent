---
name: interactive-dashboard-builder
description: Build self-contained interactive HTML dashboards with Chart.js, dropdown filters, and professional styling. Use when creating dashboards, building interactive reports, or generating shareable HTML files with charts and filters that work without a server.
---

# Interactive Dashboard Builder Skill

Patterns and techniques for building self-contained HTML/JS dashboards with Chart.js, filters, interactivity, and professional styling.

## Code Generation Rules

Read these before writing anything. Each one exists because breaking it
produces a dashboard that fails in the browser.

### Use the functions in this skill

`COLORS`, `formatValue`, `renderKPI`, `createLineChart`, `createBarChart`,
`createDoughnutChart`, `updateChart`, `populateFilter`, `getFilterValue`,
`filterByDateRange` and `renderTable` are the whole vocabulary of a
dashboard. Copy them into the `<script>` block **as written** and call them.

- Do not write a bespoke charting, formatting or table routine when one of
  these covers the job.
- Do not inline a modified copy of one of them at a call site.
- If a chart needs a Chart.js option these don't expose, pass it through the
  existing `options` argument -- don't hand-roll a second `new Chart(...)`.
- Copy `COLORS` and `formatValue` first. Every other function depends on
  them, and a missing `COLORS` is the single most common cause of a
  dashboard that loads with empty chart boxes.

Keep the custom JavaScript you write down to the parts that are genuinely
specific to this dashboard: the `Dashboard` class body, the filter
predicates in `applyFilters`, and the calls that wire data into the
functions above. Everything else is a copy.

### Never compress JavaScript onto one line

Write one statement per line, with the indentation shown in this skill. Do
not minify, do not chain a whole render into a single expression, and do not
collapse a function body onto its signature line.

A syntax error in one-lined JavaScript is reported as "line 1" and cannot be
located -- neither by you nor by the browser console. Keeping the code on
separate lines is what makes an error fixable.

### Never build JavaScript inside a Python f-string

This is the most common source of syntax errors in generated dashboards. CSS
uses `{`, and JavaScript template literals use `${...}` -- both collide with
f-string and `.format()` placeholders, producing either a Python
`KeyError`/`SyntaxError` or, worse, HTML that is silently mangled.

Assemble the file with plain (non-f) strings and `str.replace` on explicit
placeholders, and pass data as JSON rather than interpolating values into
code:

```python
import json

# Non-f strings. Nothing in here is interpolated by Python.
STYLES = """
    .kpi-card { background: #fff; border-radius: 8px; }
"""

SCRIPT = """
    const COLORS = ['#4C72B0', '#DD8452', '#55A868'];

    function formatValue(value, format) {
        // ... copied from this skill, unchanged
    }

    const DATA = JSON.parse(document.getElementById('dashboard-data').textContent);
    const dashboard = new Dashboard(DATA);
"""

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.5.1"></script>
<style>__STYLES__</style>
</head>
<body>
__BODY__
<script id="dashboard-data" type="application/json">__DATA__</script>
<script>__SCRIPT__</script>
</body>
</html>
"""

html = TEMPLATE
for token, value in (
    ("__TITLE__", title),
    ("__STYLES__", STYLES),
    ("__BODY__", body_html),
    ("__DATA__", json.dumps(records).replace("</", "<\\/")),
    ("__SCRIPT__", SCRIPT),
):
    html = html.replace(token, value)
```

Two details in that pattern matter:

- **Embed the data as JSON in a `<script type="application/json">` block**
  and read it with `JSON.parse`, instead of writing `const DATA = ` followed
  by interpolated Python. The JSON is inert text, so no value in the data
  can break the surrounding JavaScript.
- **`json.dumps` emits bare `NaN` for a missing float**, which is valid
  Python output but invalid JavaScript -- `JSON.parse` throws and the page
  renders blank. Replace missing values before serialising
  (`df.where(df.notna(), None)`), or pass `allow_nan=False` to find out at
  build time. The `.replace("</", "<\\/")` guards against a `</script>`
  inside a string value closing the data block early.

### Validate the JavaScript before showing the dashboard

After writing the HTML, extract the inline script and check that it parses:

```python
import re, shutil, subprocess, tempfile, os

html = open(out_path, encoding="utf-8").read()
blocks = [
    body for attrs, body in re.findall(
        r"<script\b([^>]*)>(.*?)</script\s*>", html, re.DOTALL)
    if "src=" not in attrs and "application/json" not in attrs
]

if shutil.which("node"):
    with tempfile.TemporaryDirectory() as tmp:
        for i, body in enumerate(blocks):
            # Check each block on its own: concatenating them produces bogus
            # "Identifier has already been declared" errors.
            path = os.path.join(tmp, f"block_{i}.js")
            open(path, "w", encoding="utf-8").write(body)
            done = subprocess.run(["node", "--check", path],
                                  capture_output=True, text=True)
            if done.returncode != 0:
                raise SystemExit(f"script block {i} has a syntax error:\n{done.stderr}")
```

Node is not installed in every sandbox, so treat a missing `node` as "check
skipped", not as a failure. These checks work everywhere and are worth
running regardless:

- The file ends with `</html>` (catches a truncated write).
- `html.count("<script") == html.count("</script")`.
- Every `<canvas id="...">` you reference has a matching `getElementById`.
- No line in a script block is longer than ~200 characters (a proxy for
  accidentally minified code).

Report a failed check to the user instead of handing them a broken file.

## HTML/JS Dashboard Patterns

### Base Template

Every dashboard follows this structure:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard Title</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.5.1"></script>
    <!-- Only load the date adapter if a chart uses a time-scale axis.
         Prefer pre-formatted category labels and omit it. -->
    <style>
        /* Dashboard styles go here */
    </style>
</head>
<body>
    <div class="dashboard-container">
        <header class="dashboard-header">
            <h1>Dashboard Title</h1>
            <div class="filters">
                <!-- Filter controls -->
            </div>
        </header>

        <section class="kpi-row">
            <!-- KPI cards -->
        </section>

        <section class="chart-row">
            <!-- Chart containers -->
        </section>

        <section class="table-section">
            <!-- Data table -->
        </section>

        <footer class="dashboard-footer">
            <span>Data as of: <span id="data-date"></span></span>
        </footer>
    </div>

    <!-- Data goes in its own JSON block, never interpolated into the code
         below. See "Never build JavaScript inside a Python f-string". -->
    <script id="dashboard-data" type="application/json">[]</script>

    <script>
        const DATA = JSON.parse(document.getElementById('dashboard-data').textContent);

        // COLORS, formatValue and the create*Chart / renderKPI / renderTable
        // helpers are copied here unchanged from this skill.

        // Dashboard logic
        class Dashboard {
            constructor(data) {
                this.rawData = data;
                this.filteredData = data;
                this.charts = {};
                this.init();
            }

            init() {
                this.setupFilters();
                this.renderKPIs();
                this.renderCharts();
                this.renderTable();
            }

            applyFilters() {
                // Filter logic
                this.filteredData = this.rawData.filter(row => {
                    // Apply each active filter
                    return true; // placeholder
                });
                this.renderKPIs();
                this.updateCharts();
                this.renderTable();
            }

            // ... methods for each section
        }

        const dashboard = new Dashboard(DATA);
    </script>
</body>
</html>
```

### KPI Card Pattern

```html
<div class="kpi-card">
    <div class="kpi-label">Total Revenue</div>
    <div class="kpi-value" id="kpi-revenue">$0</div>
    <div class="kpi-change positive" id="kpi-revenue-change">+0%</div>
</div>
```

```javascript
function renderKPI(elementId, value, previousValue, format = 'number') {
    const el = document.getElementById(elementId);
    const changeEl = document.getElementById(elementId + '-change');

    // Format the value
    el.textContent = formatValue(value, format);

    // Calculate and display change
    if (previousValue && previousValue !== 0) {
        const pctChange = ((value - previousValue) / previousValue) * 100;
        const sign = pctChange >= 0 ? '+' : '';
        changeEl.textContent = `${sign}${pctChange.toFixed(1)}% vs prior period`;
        changeEl.className = `kpi-change ${pctChange >= 0 ? 'positive' : 'negative'}`;
    }
}

function formatValue(value, format) {
    // Guard first: this runs inside Chart.js tick and tooltip callbacks,
    // where a single throw takes down the entire chart.
    if (value === null || value === undefined || value === '') return '\u2014';
    if (typeof value === 'number' && !isFinite(value)) return '\u2014';

    switch (format) {
        case 'currency':
            if (value >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
            if (value >= 1e3) return `$${(value / 1e3).toFixed(1)}K`;
            return `$${value.toFixed(0)}`;
        case 'percent':
            return `${value.toFixed(1)}%`;
        case 'number':
            if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
            if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
            return value.toLocaleString();
        default:
            return value.toString();
    }
}
```

### Chart Container Pattern

```html
<div class="chart-container">
    <h3 class="chart-title">Monthly Revenue Trend</h3>
    <canvas id="revenue-chart"></canvas>
</div>
```

## Chart.js Integration

### Shared Palette

Every chart function below reads `COLORS`. Copy this into the `<script>`
block before them -- if it is missing, the charts throw
`COLORS is not defined` and render as empty boxes.

```javascript
const COLORS = [
    '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
    '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD'
];
```

These are the same six accents as the `--color-1 ... --color-6` CSS
variables in the Color System section, plus four more for charts with many
series. Keep the two lists consistent if you change either.

### Line Chart

```javascript
function createLineChart(canvasId, labels, datasets) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets.map((ds, i) => ({
                label: ds.label,
                data: ds.data,
                borderColor: COLORS[i % COLORS.length],
                backgroundColor: COLORS[i % COLORS.length] + '20',
                borderWidth: 2,
                fill: ds.fill || false,
                tension: 0.3,
                pointRadius: 3,
                pointHoverRadius: 6,
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: { usePointStyle: true, padding: 20 }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${formatValue(context.parsed.y, 'currency')}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false }
                },
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return formatValue(value, 'currency');
                        }
                    }
                }
            }
        }
    });
}
```

### Bar Chart

```javascript
function createBarChart(canvasId, labels, data, options = {}) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const isHorizontal = options.horizontal || labels.length > 8;

    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: options.label || 'Value',
                data: data,
                backgroundColor: options.colors || COLORS.map(c => c + 'CC'),
                borderColor: options.colors || COLORS,
                borderWidth: 1,
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: isHorizontal ? 'y' : 'x',
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return formatValue(context.parsed[isHorizontal ? 'x' : 'y'], options.format || 'number');
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: { display: isHorizontal },
                    ticks: isHorizontal ? {
                        callback: function(value) {
                            return formatValue(value, options.format || 'number');
                        }
                    } : {}
                },
                y: {
                    beginAtZero: !isHorizontal,
                    grid: { display: !isHorizontal },
                    ticks: !isHorizontal ? {
                        callback: function(value) {
                            return formatValue(value, options.format || 'number');
                        }
                    } : {}
                }
            }
        }
    });
}
```

### Doughnut Chart

```javascript
function createDoughnutChart(canvasId, labels, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: COLORS.map(c => c + 'CC'),
                borderColor: '#ffffff',
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '60%',
            plugins: {
                legend: {
                    position: 'right',
                    labels: { usePointStyle: true, padding: 15 }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const pct = ((context.parsed / total) * 100).toFixed(1);
                            return `${context.label}: ${formatValue(context.parsed, 'number')} (${pct}%)`;
                        }
                    }
                }
            }
        }
    });
}
```

### Updating Charts on Filter Change

```javascript
function updateChart(chart, newLabels, newData) {
    chart.data.labels = newLabels;

    if (Array.isArray(newData[0])) {
        // Multiple datasets
        newData.forEach((data, i) => {
            chart.data.datasets[i].data = data;
        });
    } else {
        chart.data.datasets[0].data = newData;
    }

    chart.update('none'); // 'none' disables animation for instant update
}
```

## Filter and Interactivity Implementation

### Dropdown Filter

```html
<div class="filter-group">
    <label for="filter-region">Region</label>
    <select id="filter-region" onchange="dashboard.applyFilters()">
        <option value="all">All Regions</option>
    </select>
</div>
```

```javascript
function populateFilter(selectId, data, field) {
    const select = document.getElementById(selectId);
    const values = [...new Set(data.map(d => d[field]))].sort();

    // Keep the "All" option, add unique values
    values.forEach(val => {
        const option = document.createElement('option');
        option.value = val;
        option.textContent = val;
        select.appendChild(option);
    });
}

function getFilterValue(selectId) {
    const val = document.getElementById(selectId).value;
    return val === 'all' ? null : val;
}
```

### Date Range Filter

```html
<div class="filter-group">
    <label>Date Range</label>
    <input type="date" id="filter-date-start" onchange="dashboard.applyFilters()">
    <span>to</span>
    <input type="date" id="filter-date-end" onchange="dashboard.applyFilters()">
</div>
```

```javascript
function filterByDateRange(data, dateField, startDate, endDate) {
    return data.filter(row => {
        const rowDate = new Date(row[dateField]);
        if (startDate && rowDate < new Date(startDate)) return false;
        if (endDate && rowDate > new Date(endDate)) return false;
        return true;
    });
}
```

### Combined Filter Logic

```javascript
applyFilters() {
    const region = getFilterValue('filter-region');
    const category = getFilterValue('filter-category');
    const startDate = document.getElementById('filter-date-start').value;
    const endDate = document.getElementById('filter-date-end').value;

    this.filteredData = this.rawData.filter(row => {
        if (region && row.region !== region) return false;
        if (category && row.category !== category) return false;
        if (startDate && row.date < startDate) return false;
        if (endDate && row.date > endDate) return false;
        return true;
    });

    this.renderKPIs();
    this.updateCharts();
    this.renderTable();
}
```

### Sortable Table

```javascript
function renderTable(containerId, data, columns) {
    const container = document.getElementById(containerId);
    let sortCol = null;
    let sortDir = 'desc';

    function render(sortedData) {
        let html = '<table class="data-table">';

        // Header
        html += '<thead><tr>';
        columns.forEach(col => {
            const arrow = sortCol === col.field
                ? (sortDir === 'asc' ? ' ▲' : ' ▼')
                : '';
            html += `<th onclick="sortTable('${col.field}')" style="cursor:pointer">${col.label}${arrow}</th>`;
        });
        html += '</tr></thead>';

        // Body
        html += '<tbody>';
        sortedData.forEach(row => {
            html += '<tr>';
            columns.forEach(col => {
                const value = col.format ? formatValue(row[col.field], col.format) : row[col.field];
                html += `<td>${value}</td>`;
            });
            html += '</tr>';
        });
        html += '</tbody></table>';

        container.innerHTML = html;
    }

    window.sortTable = function(field) {
        if (sortCol === field) {
            sortDir = sortDir === 'asc' ? 'desc' : 'asc';
        } else {
            sortCol = field;
            sortDir = 'desc';
        }
        const sorted = [...data].sort((a, b) => {
            const aVal = a[field], bVal = b[field];
            const cmp = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
            return sortDir === 'asc' ? cmp : -cmp;
        });
        render(sorted);
    };

    render(data);
}
```

## CSS Styling for Dashboards

### Color System

```css
:root {
    /* Background layers */
    --bg-primary: #f8f9fa;
    --bg-card: #ffffff;
    --bg-header: #1a1a2e;

    /* Text */
    --text-primary: #212529;
    --text-secondary: #6c757d;
    --text-on-dark: #ffffff;

    /* Accent colors for data */
    --color-1: #4C72B0;
    --color-2: #DD8452;
    --color-3: #55A868;
    --color-4: #C44E52;
    --color-5: #8172B3;
    --color-6: #937860;

    /* Status colors */
    --positive: #28a745;
    --negative: #dc3545;
    --neutral: #6c757d;

    /* Spacing */
    --gap: 16px;
    --radius: 8px;
}
```

### Layout

```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.5;
}

.dashboard-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: var(--gap);
}

.dashboard-header {
    background: var(--bg-header);
    color: var(--text-on-dark);
    padding: 20px 24px;
    border-radius: var(--radius);
    margin-bottom: var(--gap);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}

.dashboard-header h1 {
    font-size: 20px;
    font-weight: 600;
}
```

### KPI Cards

```css
.kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: var(--gap);
    margin-bottom: var(--gap);
}

.kpi-card {
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}

.kpi-label {
    font-size: 13px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}

.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 4px;
}

.kpi-change {
    font-size: 13px;
    font-weight: 500;
}

.kpi-change.positive { color: var(--positive); }
.kpi-change.negative { color: var(--negative); }
```

### Chart Containers

```css
.chart-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: var(--gap);
    margin-bottom: var(--gap);
}

.chart-container {
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}

.chart-container h3 {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 16px;
}

.chart-container canvas {
    max-height: 300px;
}
```

### Filters

```css
.filters {
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
}

.filter-group {
    display: flex;
    align-items: center;
    gap: 6px;
}

.filter-group label {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.7);
}

.filter-group select,
.filter-group input[type="date"] {
    padding: 6px 10px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.1);
    color: var(--text-on-dark);
    font-size: 13px;
}

.filter-group select option {
    background: var(--bg-header);
    color: var(--text-on-dark);
}
```

### Data Table

```css
.table-section {
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    overflow-x: auto;
}

.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}

.data-table thead th {
    text-align: left;
    padding: 10px 12px;
    border-bottom: 2px solid #dee2e6;
    color: var(--text-secondary);
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    white-space: nowrap;
    user-select: none;
}

.data-table thead th:hover {
    color: var(--text-primary);
    background: #f8f9fa;
}

.data-table tbody td {
    padding: 10px 12px;
    border-bottom: 1px solid #f0f0f0;
}

.data-table tbody tr:hover {
    background: #f8f9fa;
}

.data-table tbody tr:last-child td {
    border-bottom: none;
}
```

### Responsive Design

```css
@media (max-width: 768px) {
    .dashboard-header {
        flex-direction: column;
        align-items: flex-start;
    }

    .kpi-row {
        grid-template-columns: repeat(2, 1fr);
    }

    .chart-row {
        grid-template-columns: 1fr;
    }

    .filters {
        flex-direction: column;
        align-items: flex-start;
    }
}

@media print {
    body { background: white; }
    .dashboard-container { max-width: none; }
    .filters { display: none; }
    .chart-container { break-inside: avoid; }
    .kpi-card { border: 1px solid #dee2e6; box-shadow: none; }
}
```

## Performance Considerations for Large Datasets

### Data Size Guidelines

| Data Size | Approach |
|---|---|
| <1,000 rows | Embed directly in HTML. Full interactivity. |
| 1,000 - 10,000 rows | Embed in HTML. May need to pre-aggregate for charts. |
| 10,000 - 100,000 rows | Pre-aggregate server-side. Embed only aggregated data. |
| >100,000 rows | Not suitable for client-side dashboard. Use a BI tool or paginate. |

### Pre-Aggregation Pattern

Instead of embedding raw data and aggregating in the browser:

```javascript
// DON'T: embed 50,000 raw rows
const RAW_DATA = [/* 50,000 rows */];

// DO: pre-aggregate before embedding
const CHART_DATA = {
    monthly_revenue: [
        { month: '2024-01', revenue: 150000, orders: 1200 },
        { month: '2024-02', revenue: 165000, orders: 1350 },
        // ... 12 rows instead of 50,000
    ],
    top_products: [
        { product: 'Widget A', revenue: 45000 },
        // ... 10 rows
    ],
    kpis: {
        total_revenue: 1980000,
        total_orders: 15600,
        avg_order_value: 127,
    }
};
```

### Chart Performance

- Limit line charts to <500 data points per series (downsample if needed)
- Limit bar charts to <50 categories
- For scatter plots, cap at 1,000 points (use sampling for larger datasets)
- Disable animations for dashboards with many charts: `animation: false` in Chart.js options
- Use `Chart.update('none')` instead of `Chart.update()` for filter-triggered updates

### DOM Performance

- Limit data tables to 100-200 visible rows. Add pagination for more.
- Use `requestAnimationFrame` for coordinated chart updates
- Avoid rebuilding the entire DOM on filter change -- update only changed elements

```javascript
// Efficient table pagination
function renderTablePage(data, page, pageSize = 50) {
    const start = page * pageSize;
    const end = Math.min(start + pageSize, data.length);
    const pageData = data.slice(start, end);
    // Render only pageData
    // Show pagination controls: "Showing 1-50 of 2,340"
}
```
