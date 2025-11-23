import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load data
df = pd.read_csv("/workspaces/maxsold/data/auction/items_all_2025-11-21.csv")
df['current_bid'] = pd.to_numeric(df['current_bid'], errors='coerce')
df['bid_count'] = pd.to_numeric(df['bid_count'], errors='coerce')

# Create static dashboard
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Bid Distribution", "Price Box Plot", "Views vs Bids", "Top Items")
)

# Add plots
fig1 = px.histogram(df, x='bid_count')
fig2 = px.box(df[df['current_bid'] > 0], y='current_bid')
fig3 = px.scatter(df, x='viewed', y='bid_count', size='current_bid')

for trace in fig1.data:
    fig.add_trace(trace, row=1, col=1)
for trace in fig2.data:
    fig.add_trace(trace, row=1, col=2)
for trace in fig3.data:
    fig.add_trace(trace, row=2, col=1)

fig.update_layout(height=800, showlegend=False, title_text="MaxSold Analytics")
fig.write_html("docs/index.html")
print("✅ Static dashboard exported to docs/index.html")