import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Load data
df = pd.read_csv("/workspaces/maxsold/data/auction/items_all_2025-11-21.csv")
df['current_bid'] = pd.to_numeric(df['current_bid'], errors='coerce')
df['bid_count'] = pd.to_numeric(df['bid_count'], errors='coerce')
df['viewed'] = pd.to_numeric(df['viewed'], errors='coerce')

# Initialize app
app = dash.Dash(__name__, title="MaxSold Analytics")

app.layout = html.Div([
    html.H1("MaxSold Auction Analytics", style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            html.Label("Select Auction:"),
            dcc.Dropdown(
                id='auction-dropdown',
                options=[{'label': 'All', 'value': 'All'}] + 
                        [{'label': str(x), 'value': x} for x in sorted(df['auction_id'].unique())],
                value='All'
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),
    ]),
    
    html.Div(id='metrics-row', style={'padding': '20px'}),
    
    html.Div([
        html.Div([
            dcc.Graph(id='bid-distribution')
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='price-box')
        ], style={'width': '50%', 'display': 'inline-block'}),
    ]),
    
    html.Div([
        dcc.Graph(id='scatter-views-bids')
    ])
])

@app.callback(
    [Output('metrics-row', 'children'),
     Output('bid-distribution', 'figure'),
     Output('price-box', 'figure'),
     Output('scatter-views-bids', 'figure')],
    [Input('auction-dropdown', 'value')]
)
def update_dashboard(selected_auction):
    filtered_df = df if selected_auction == 'All' else df[df['auction_id'] == selected_auction]
    
    # Metrics
    metrics = html.Div([
        html.Div([
            html.H3(f"{len(filtered_df):,}"),
            html.P("Total Items")
        ], style={'display': 'inline-block', 'margin': '20px', 'textAlign': 'center'}),
        
        html.Div([
            html.H3(f"{int(filtered_df['bid_count'].sum()):,}"),
            html.P("Total Bids")
        ], style={'display': 'inline-block', 'margin': '20px', 'textAlign': 'center'}),
        
        html.Div([
            html.H3(f"${filtered_df['current_bid'].mean():.2f}"),
            html.P("Avg Current Bid")
        ], style={'display': 'inline-block', 'margin': '20px', 'textAlign': 'center'}),
    ])
    
    # Charts
    fig1 = px.histogram(filtered_df, x='bid_count', nbins=30, title="Bid Count Distribution")
    fig2 = px.box(filtered_df[filtered_df['current_bid'] > 0], y='current_bid', title="Bid Price Distribution")
    fig3 = px.scatter(filtered_df, x='viewed', y='bid_count', size='current_bid', 
                      hover_data=['title'], title="Views vs Bids")
    
    return metrics, fig1, fig2, fig3

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)