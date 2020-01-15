import pandas as pd
import numpy as np
import warnings
import plotly.graph_objects as go
import dash
import dash_table as dt
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from sklearn.preprocessing import StandardScaler, FunctionTransformer
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

external_stylesheets = ["https://fonts.googleapis.com/css?family=Open+Sans:300,400,700", "https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets = external_stylesheets)

app.title = "Clustering Tool"

app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

application = app.server

app.layout = html.Div(children=[

    # hidden divs; these are used for storing the data shared across callbacks
    html.Div(id="initial_data", style={"display": "none"}),
    html.Div(id="selected_data", style={"display": "none"}),

    # tabs
    html.Div(className="tabs", children=[

        dcc.Tabs(value="tab1", className="custom-tabs", children=[

            # first tab
            dcc.Tab(label="Data Preprocessing", value="tab1", className="custom-tab", selected_className="custom-tab--selected", children=[

                html.Div(children=[

                    # dropdown used for selecting the input file with the dataset
                    html.Div(children=[

                        html.Label("Dataset:", style={"margin": "1vw 0vw 0.3vw 0vw"}),
                        dcc.Dropdown(id="selected_file", options=[{"label": "Adult", "value": "adult"}], value="adult",
                            style={"font-size": "95%"}, placeholder="Select Data", optionHeight=35, multi=False, clearable=False, searchable=True),

                    ], style={"margin": "0vw 0vw 0vw 1vw", "width": "16%", "display": "inline-block", "vertical-align": "top"}),

                    # dropdown used for selecting the columns of the dataset
                    html.Div(children=[

                        html.Label("Columns:", style={"margin": "1vw 0vw 0.3vw 0vw"}),
                        dcc.Dropdown(id="selected_columns", style={"font-size": "95%"}, optionHeight=35, multi=True,
                            searchable=True, clearable=True, placeholder="Select Columns"),

                    ], style={"margin": "0vw 0vw 0vw 1vw", "width": "16%", "display": "inline-block", "vertical-align": "top"}),

                    # dropdown used for selecting the rows of the dataset
                    html.Div(children=[

                        html.Label("Rows:", style={"margin": "1vw 0vw 0.3vw 0vw"}),
                        dcc.Dropdown(id="selected_rows", style={"font-size": "95%"}, optionHeight=35, multi=True,
                            searchable=True, clearable=True, placeholder="Select Rows"),

                    ], style={"margin": "0vw 0vw 0vw 1vw", "width": "16%", "display": "inline-block", "vertical-align": "top"}),

                    # radio buttons used for selecting the approach for numerical missing values
                    html.Div(children=[

                        html.Label("Missing Numerical Values:", style={"margin": "1vw 0vw 1vw 0vw"}),
                        dcc.RadioItems(id="missing_numerical",  value="drop", options=[{"label": "Mean", "value": "mean"},
                            {"label": "Median", 'value': "median"}, {"label": "Mode", "value": "mode"}, {"label": "Drop", 'value': "drop"}],
                                labelStyle={"font-size": "95%", "display": "inline-block", "margin": "0vw 0.5vw 0vw 0vw"}),

                    ], style={"margin": "0vw 0vw 0vw 1vw", "width": "21%", "display": "inline-block", "vertical-align": "top"}),

                    # radio buttons used for selecting the approach for categorical missing values
                    html.Div(children=[

                        html.Label("Missing Categorical Values:", style={"margin": "1vw 0vw 1vw 0vw"}),
                        dcc.RadioItems(id="missing_categorical", value="drop", options=[{"label": "Mode", "value": "mode"}, {"label": "Drop", 'value': "drop"}], 
                            labelStyle={"font-size": "95%", "display": "inline-block", "margin": "0vw 0.5vw 0vw 0vw"}),

                    ], style={"margin": "0vw 0vw 0vw 0vw", "width": "16%", "display": "inline-block", "vertical-align": "top"}),

                    # run button used for updating the table after making a selection
                    html.Div(children=[

                        html.Label("Update Table:", style={"margin": "1vw 0vw 0.3vw 0vw"}),
                        html.Button(id="table_button", n_clicks=0, children=["update"], style={"background-color": "#F46D43", "font-size": "80%", 
                                        "font-weight": "500", "text-align": "center", "width": "99%", "color": "white"}),

                    ], style={"margin": "0vw 0vw 0vw 0.5vw", "width": "9%", "display": "inline-block", "vertical-align": "top"}),

                ], className="row"),

                # spinner
                dcc.Loading(children=[

                    # table with the selected data
                    html.Div(children=[

                        dt.DataTable(id="data_table", style_as_list_view=False, style_data_conditional=[{"if": {"row_index": "odd"}, "background-color": "#ffffd2"}],
                            style_table={"display": "block", "max-height": "calc(20rem + 10vw)", "max-width": "95%", "overflow-y": "scroll", "margin": "2vw 1vw 2vw 1vw"},
                            style_cell={"text-align": "center", "font-family": "Open Sans", "font-size": "90%", "height": "2vw"},
                            style_header={"background-color": "#66C2A5", "text-align": "center", "color": "white", "text-transform": "uppercase", 
                                     "font-family": "Open Sans", "font-size": "85%", "font-weight": "500", "height": "2vw"})

                    ], className="row", style={"height": "calc(20rem + 15vw)", "width": "100%", "background-color": "white"}),

                ], type="circle", color="#F46D43", style={"background-color": "white", "height": "calc(10rem + 8vw)", "width": "100%", "position": "relative", "top": "calc(10rem + 6vw)"}),

            ], style={"background-color": "white"}),

            # second tab
            dcc.Tab(label="Exploratory Data Analysis", value="tab2", className="custom-tab", selected_className="custom-tab--selected", children=[

                html.Div(children=[

                    html.Div(children=[

                        # dropdown used for selecting the features
                        html.Label("Features:", style={"margin": "1vw 0vw 0.5vw 1vw"}),
                        dcc.Dropdown(id="selected_features", style={"font-size": "95%", "margin-left": "0.5vw"}, optionHeight=35, multi=True,
                            searchable=True, clearable=True, placeholder="Select Features"),

                        # radio buttons used for selecting the transformation
                        html.Label("Transformation:", style={"margin": "2vw 0vw 0.5vw 1vw"}),
                        dcc.RadioItems(id="features_transformation", value="none", options=[{"label": "Logarithm", "value": "log"},
                             {"label": "Z-score", 'value': "z-score"}, {"label": "None", 'value': "none"}], labelStyle={"font-size": "95%",
                                "display": "inline-block", "margin": "0vw 0.5vw 0vw 0vw"}, style={"margin-left": "1vw"}),

                        # run button used for updating the results
                        html.Label("Update Results:", style={"margin": "2vw 0vw 0.3vw 1vw"}),
                        html.Button(id="stats_button", n_clicks=0, children=["update"], style={"background-color": "#F46D43", "font-size": "80%",
                              "margin-left": "1vw", "font-weight": "500", "text-align": "center", "width": "60%", "color": "white"}),

                        # explanatory text
                        html.P(children=["Note that the correlation matrix can display at most 10 features."],
                                 style={"margin": "2vw 0vw 1vw 1vw", "font-size": "95%"})

                    ], className="three columns", style={"height": "30vw"}),

                    # correlation matrix
                    html.Div(children=[

                        html.Label("Correlation Matrix:", style={"margin": "1vw 0vw 0.5vw 1vw"}),

                        dcc.Loading(children=[

                            dcc.Graph(id="correlation_matrix", config={"responsive": True, "autosizable": True,
                                "showTips":True, "displayModeBar": False}, style={"height": "28vw", "width": "95%"}),

                        ], type="circle", color="#F46D43", style={"background-color": "white", "height": "28vw", "width": "100%", "position": "relative", "top": "10vw"}),

                     ], className="nine columns", style={"height": "30vw"}),


                ], className="row"),

                # table with the descriptive statistics
                html.Div(children=[

                    html.Label("Descriptive Statistics:", style={"margin": "1vw 0vw 0.5vw 1vw"}),

                    dcc.Loading(children=[

                        dt.DataTable(id="stats_table", style_as_list_view=False, style_data_conditional=[
                            {"if": {"row_index": "odd"}, "background-color": "#ffffd2"}, {"if": {"column_id": "Stat."}, "background-color": "#66C2A5", "color": "white"}],
                            style_table={"display": "block", "max-width": "95%", "overflow-y": "scroll", "margin": "0vw 0vw 2vw 1vw"},
                            style_cell={"text-align": "center", "font-family": "Open Sans", "font-size": "90%", "height": "2vw"},
                            style_header={"background-color": "#66C2A5", "text-align": "center", "color": "white", "text-transform": "uppercase",
                                "font-family": "Open Sans", "font-size": "85%", "font-weight": "500", "height": "2vw"})

                    ], type="circle", color="#F46D43", style={"background-color": "white", "height": "25vw", "width": "100%", "position": "relative", "top": "5vw"}),

                ], className="row", style={"width": "100%", "height": "25vw", "background-color": "white"}),

            ], style={"background-color": "white"}),

            # third tab
            dcc.Tab(label="Cluster Analysis", value="tab3", className="custom-tab",
                    selected_className="custom-tab--selected",
                    children=["To be completed"]),

        ]),
    ]),
])

@app.callback([Output("selected_rows", "options"), Output("selected_columns", "options"),
               Output("initial_data","children")], [Input("selected_file", "value")])
def load_data(selected_file):

    if selected_file=="adult":

        # import the data
        df = pd.read_csv("data/adult.csv")
        columns = list(df.columns)
        rows = list(df.index)

        # process the missing values
        df[df == "-"] = np.nan
        df[df == "?"] = np.nan
        df[df == "."] = np.nan
        df[df == " "] = np.nan

        # include the row ids in the first columns
        df["id"] = rows
        names = ["id"]
        names.extend(columns)
        df = df[names]

        # save the data in the hidden div
        data = df.to_json(orient="split")

        # transform the categorical variables into dummy variables
        df = pd.get_dummies(df, dummy_na=False, drop_first=True)

        # get the new column ids
        columns = list(df.columns)
        columns.remove("id")

        # create the list of column ids to be shown in the dropdown menu
        column_options = [{"value": columns[0], "label": columns[0]}]

        if len(columns) > 1:
           for j in range(1, len(columns)):
               column_options.append({"value": columns[j], "label": columns[j]})

        # create the list of row ids to be shown in the dropdown menu
        row_options = [{"value": rows[0], "label": rows[0]}]

        if len(rows) > 1:
            for j in range(1, len(rows)):
                row_options.append({"value": rows[j], "label": rows[j]})

        # delete the unnecessary objects
        del df, names, rows, columns

    return [row_options, column_options, {"data": data}]

@app.callback([Output("data_table", "data"), Output("data_table", "columns"), Output("selected_data", "children"),
               Output("selected_features", "options")], [Input("table_button", "n_clicks"), Input("initial_data", "children")],
              [State("selected_rows", "value"), State("selected_columns", "value"), State("missing_numerical", "value"),
               State("missing_categorical", "value")])
def update_table(n_clicks, initial_data, selected_rows, selected_columns, missing_numerical, missing_categorical):

    # load the initial data from the hidden div
    df = pd.read_json(initial_data["data"], orient="split")

    # process the missing values
    num = df.loc[:, np.logical_and(df.dtypes != "object", df.dtypes != "category")]

    if missing_numerical == "drop":

        num.dropna(inplace=True)
        df = df.iloc[num.index, :]
        df.reset_index(inplace=True, drop=True)

    elif missing_numerical == "mean":

        keys = list(num.columns)
        values = list(num.mean(axis=0).round().values.flatten())
        df.fillna(value=dict(zip(keys, values)), inplace=True)

    elif missing_numerical == "median":

        keys = list(num.columns)
        values = list(num.median(axis=0).round().values.flatten())
        df.fillna(value=dict(zip(keys, values)), inplace=True)

    elif missing_numerical == "mode":

        keys = list(num.columns)
        values = list(num.mode(axis=0).round().values.flatten())
        df.fillna(value=dict(zip(keys, values)), inplace=True)

    cat = df.loc[:, np.logical_or(df.dtypes == "object", df.dtypes == "category")]

    if missing_categorical == "drop":

        cat.dropna(inplace=True)
        df = df.iloc[cat.index, :]
        df = pd.get_dummies(df, dummy_na=False, drop_first=True)
        df.reset_index(inplace=True, drop=True)

    elif missing_categorical == "mode":

        keys = list(cat.columns)
        values = list(cat.mode(axis=0).values.flatten())
        df.fillna(value=dict(zip(keys, values)), inplace=True)
        df = pd.get_dummies(df, dummy_na=False, drop_first=True)

    # extract the selected rows and columns
    if selected_columns is not None and selected_rows is not None:

        columns = list(set(selected_columns))
        indices = list(set(selected_rows))

        if len(columns) > 0:

            names = ["id"]
            names.extend(columns)

        else:

            names = list(df.columns)

        if len(indices) > 0:

            rows = list(df.index[df["id"].isin(indices)])

        else:

            rows = list(df.index)

    elif selected_columns is None and selected_rows is not None:

        names = list(df.columns)
        indices = list(set(selected_rows))

        if len(indices) > 0:

            rows = list(df.index[df["id"].isin(indices)])

        else:

            rows = list(df.index)

    elif selected_columns is not None and selected_rows is None:

        columns = list(set(selected_columns))
        rows = list(df.index)

        if len(columns) > 0:

            names = ["id"]
            names.extend(columns)

        else:

            names = list(df.columns)

    else:

        names = list(df.columns)
        rows = list(df.index)

    df = df[names].iloc[rows, :]
    df.reset_index(inplace=True, drop=True)

    table_data = df.to_dict("records")
    table_columns = [{"id": x, "name": x} for x in list(df.columns)]

    # save the selected data in the hidden div
    names.remove("id")
    df = df[names]
    data = df.to_json(orient="split")

    # create the list of features to be shown in the dropdown menu
    features_options = [{"value": names[0], "label": names[0]}]

    if len(names) > 1:
        for j in range(1, len(names)):
            features_options.append({"value": names[j], "label": names[j]})

    # delete the unnecessary objects
    del df, names, rows

    return [table_data, table_columns, {"data": data}, features_options]

@app.callback([Output("stats_table", "data"), Output("stats_table", "columns"), Output("correlation_matrix", "figure")],
              [Input("stats_button", "n_clicks"), Input("selected_data", "children")], [State("selected_features", "value"),
               State("features_transformation", "value")])
def update_stats(n_clicks, selected_data, selected_features, features_transformation):

    # load the selected data from the hidden div
    df = pd.read_json(selected_data["data"], orient="split")

    # extract the selected features
    if selected_features is not None:
        if len(selected_features) > 0:
            df = df[list(selected_features)]

    # apply the selected transformation
    if features_transformation == "log":

        df = pd.DataFrame(data=FunctionTransformer(np.log1p, validate=True).fit_transform(df), columns=df.columns, index=df.index)

    elif features_transformation == "z-score":

        df = pd.DataFrame(data=StandardScaler().fit_transform(df), columns=df.columns, index=df.index)

    # calculate the descriptive statistics
    stats = df.describe()
    stats = stats.apply(lambda x: np.round(x,2))
    stats["Stat."] = stats.index
    names = ["Stat."]
    names.extend(list(stats.columns[stats.columns!="Stat."]))
    stats = stats[names]
    table_data = stats.to_dict("records")
    table_columns = [{"id": x, "name": x} for x in list(stats.columns)]

    # calculate the sample correlation matrix
    sigma = df.iloc[:,:10].corr()
    x = list(sigma.columns)
    y = list(sigma.index)
    z = np.nan_to_num(sigma.values)

    # plot the sample correlation matrix
    annotations = []
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            annotations.append(dict(x=x[i], y=y[j], text=str(np.round(z[i,j]*100,2))+"%", showarrow=False))

    layout = dict(annotations=annotations, margin=dict(t=5), xaxis=dict(tickangle=45), yaxis=dict(tickangle=0),
                  font=dict(family="Open Sans", size=10))

    traces = []
    traces.append(go.Heatmap(z=z, x=x, y=y, zmin=-1, zmax=1, colorscale="Spectral"))

    correlation_matrix = go.Figure(data=traces, layout=layout).to_dict()

    return [table_data, table_columns, correlation_matrix]

if __name__ == "__main__":
    app.run_server(port=8080, debug=False) # use this for running on local machine
    #application.run_server(port=8080, debug=False) # use this for deploying on AWS