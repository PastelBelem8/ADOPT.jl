import json
from textwrap import dedent as d

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# CSS
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Create a dashboard using the CSS
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

color = 'rgb('

# Create the dashboard
app.layout = html.Div([
    # Create a plot
    dcc.Graph(
        id='basic-interactions',
        figure={
            'data': [
                {
                    'x': [1, 2, 3, 4],
                    'y': [4, 1, 3, 5],
                    'text': ['a', 'b', 'c', 'd'],
                    'customdata': [[89, 90], [1, 91], [2, 92], [3, 93]],
                    'name': 'Trace 1',
                    'mode': 'markers',
                    'marker': {'size': 12}
                }
            ],
            'layout': {
                'clickmode': 'event+select'
            }
        }
    ),

    # Create the bottom row
    html.Div(className='row', children=[
        html.Div([
            dcc.Markdown(d("""
                **Click Data**

                Click on points in the graph.
            """)),
            html.Pre(id='click-data', style=styles['pre']),
        ], className='three columns'),
    ])
])

# Add callback for when clicking
@app.callback(
    Output('click-data', 'children'),
    [Input('basic-interactions', 'clickData')])
def display_click_data(clickData):
    if clickData is not None:
        print(clickData)
        print(clickData['points'][0]['customdata'])
        # pi/2, 1, pi/3, 3, pi/10, 5
        j.spiked_truss(1.07, 1, 1, 3, 0.314, 30)
    return json.dumps(clickData, indent=2)


import julia
jl = julia.Julia()
# j.include("Example.jl")

# j.spiked_truss(1.07, 1, 1, 3, 0.314, 30)

# if __name__ == '__main__':
#    app.run_server(debug=True)


