import pandas as pd
import numpy as np
import hdbscan
import warnings
import urllib.parse
import base64
import io
import plotly.graph_objects as go
import dash
import dash_table as dt
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer, LabelEncoder
from sklearn.utils.random import sample_without_replacement
from sklearn.manifold import TSNE
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

external_stylesheets = ["https://fonts.googleapis.com/css?family=Open+Sans:300,400,700",
                        "https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets = external_stylesheets)

app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
app.config.suppress_callback_exceptions = True

application = app.server

app.layout = html.Div(children=[

    # header
    html.Div(children=[

        html.Label(children=["Data Cleaning & Clustering Tool"], style={"font-size": "150%", "font-weight": "500",
        "letter-spacing": "1px", "margin": "1vw 0vw 1vw 1vw", "color": "white"}),

    ], className="row"),

    html.Div(children=[

        # tabs
        html.Div(children=[

            dcc.Tabs(id="control_tab", value="tab1", parent_className="custom-tabs", className="custom-tabs-container",
                     children=[

                # first tab
                dcc.Tab(label="Data Preparation", value="tab1", className="custom-tab",
                        selected_className="custom-tab--selected", children=[

                    # component used for uploading the file with the data
                    html.Div(children=[

                        html.Label("Upload Dataset:", style={"margin": "1vw 0vw 0.3vw 0vw"}),
                        dcc.Upload(id="uploaded_file", children=html.Div([

                            html.P("Drag and Drop or ", style={"display": "inline"}),
                            html.A("Select File", style={"display": "inline", "text-decoration": "underline"}),

                        ]), style={"border": "1px dashed #D9D9D9", "border-radius": "5px", "text-align": "center",
                                   "height": "35px", "line-height": "35px", "width": "90%"}, multiple=False),

                    ], style={"margin": "0vw 0vw 0vw 1vw"}),

                    # run button used for updating the data after making a selection
                    html.Div(children=[

                        html.Label("Update Dataset:", style={"margin": "1vw 0vw 0.3vw 0vw"}),
                        html.Button(id="data_button", n_clicks=0, children=["update"],
                        style={"background-color": "#3288BD", "font-size": "80%", "font-weight": "500",
                        "text-align": "center", "width": "60%", "color": "white"}),

                    ], style={"margin": "0vw 0vw 0vw 1vw"}),

                ]),

                # second tab
                dcc.Tab(label="Cluster Analysis", value="tab2", className="custom-tab",
                        selected_className="custom-tab--selected", children=[

                    # radio buttons used for choosing whether to perform random sampling
                    html.Label("Random Sampling:", style={"margin": "1vw 0vw 0.5vw 1vw"}),
                    dcc.RadioItems(id="cluster_random_sampling", value="False", options=[{"label": "True", "value": "True"},
                    {"label": "False", "value": "False"}], labelStyle={"font-size": "95%", "display": "inline-block",
                    "margin": "0vw 0.5vw 0vw 0vw"}, style={"margin-left": "1vw"}),

                    # numeric input used for entering the size of the subsample
                    html.Label("Sample Size:", style={"margin": "1vw 0vw 0vw 1vw"}),
                    html.P("If using random sampling, enter the size of the subsample as an integer between 1 "
                    "(corresponding to 1%) and 100 (corresponding to 100%).", style={"font-size": "80%",
                    "margin": "0vw 2vw 0vw 1vw", "text-align": "justify"}),
                    dcc.Input(id="cluster_sample_size", type="number", min=1, max=100, placeholder=80,
                    style={"margin-left": "1vw", "font-size": "95%"}),

                    # radio buttons used for selecting the dimension reduction technique
                    html.Label("Dimension Reduction:", style={"margin": "1vw 0vw 0.5vw 1vw"}),
                    dcc.RadioItems(id="cluster_dimension_reduction", value="pca", options=[{"label": "PCA", "value": "pca"},
                    {"label": "T-SNE", "value": "tsne"}, {"label": "None", "value": "none"}], style={"margin-left": "1vw"},
                    labelStyle={"font-size": "95%", "display": "inline-block", "margin": "0vw 0.5vw 0vw 0vw"}),

                    # numeric input used for entering the number of components
                    html.Label("Number of Components:", style={"margin": "1vw 0vw 0vw 1vw"}),
                    html.P("If using PCA or T-SNE, enter the number of components.", style={"font-size": "80%",
                    "margin": "0vw 2vw 0vw 1vw", "text-align": "justify"}),
                    dcc.Input(id="cluster_components", type="number", placeholder=3, value=3, min=1,
                    style={"margin-left": "1vw", "font-size": "95%"}),

                    # radio buttons used for selecting the clustering algorithm
                    html.Label("Clustering Algorithm:", style={"margin": "1vw 0vw 0.5vw 1vw"}),
                    dcc.RadioItems(id="cluster_algorithm", value="kmeans", options=[{"label": "K-Means", "value": "kmeans"},
                    {"label": "HDBSCAN", "value": "hdbscan"}], labelStyle={"font-size": "95%", "display": "inline-block",
                    "margin": "0vw 0.5vw 0vw 0vw"}, style={"margin-left": "1vw"}),

                    # numeric input used for entering the number of clusters
                    html.Label("Number of Clusters:", style={"margin": "1vw 0vw 0vw 1vw"}),
                    html.P("If using K-Means, enter the number of clusters.", style={"font-size": "80%",
                    "margin": "0vw 2vw 0vw 1vw", "text-align": "justify"}),
                    dcc.Input(id="cluster_number", type="number", placeholder=3, value=3, min=1,
                    style={"margin-left": "1vw", "font-size": "95%"}),

                    # numeric input used for entering the minimum cluster size
                    html.Label("Minimum Cluster Size:", style={"margin": "1vw 0vw 0vw 1vw"}),
                    html.P("If using HDBSCAN, enter the minimum cluster size.", style={"font-size": "80%",
                    "margin": "0vw 2vw 0vw 1vw", "text-align": "justify"}),
                    dcc.Input(id="cluster_size", type="number", placeholder=2, min=2, value=2,
                    style={"margin-left": "1vw", "font-size": "95%"}),

                    # run button used for updating the results
                    html.Label("Update Results:", style={"margin": "1vw 0vw 0.3vw 1vw"}),
                    html.Button(id="cluster_button", n_clicks=0, children=["update"], style={"background-color": "#3288BD",
                    "font-size": "80%", "margin-left": "1vw", "font-weight": "500", "text-align": "center", "width": "60%",
                    "color": "white"}),

                ]),

            ]),

        ]),

    ], style={"display": "inline-block", "vertical-align": "top", "width": "30vw", "height": "70vw",
    "margin": "0vw 0vw 2vw 0vw"}),

    html.Div(children=[

        # initial div used for alerting the user to upload a file
        html.Div(id="alert_output", children=[

            html.Label(children=["Upload a file to start."], style={"display": "block", "font-size": "120%",
            "color": "#BDBDBD", "margin-top": "22.5vw", "text-align": "center"}),

        ], style={"display": "none"}),

        # div used for displaying the data preprocessing output
        html.Div(id="data_output", children=[

            dcc.Tabs(value="data_tab1", parent_className="data-tabs", className="custom-tabs-container", children=[

                # first tab
                dcc.Tab(label="Raw Data", value="data_tab1", className="data-tab",
                        selected_className="data-tab--selected", children=[

                    html.Div(children=[

                        html.Div(id="data_controls", className="row"),

                        html.Div(id="data_container", children=[

                            dt.DataTable(id="raw_data_table", style_as_list_view=False, style_data_conditional=[
                            {"if": {"row_index": "odd"}, "background-color": "#ffffd2"}], style_cell={"height": "2vw",
                            "text-align": "center", "font-family": "Open Sans", "font-size": "90%", "width": "15vw",
                            "min-width": "15vw", "max-width": "15vw"}, style_header={"background-color": "#3288BD",
                            "color": "white", "text-align": "center", "text-transform": "uppercase", "font-size": "85%",
                            "font-family": "Open Sans", "font-weight": "500", "height": "2vw", "width": "15vw",
                            "min-width": "15vw", "max-width": "15vw"}, style_table={"display": "block", "width": "100%",
                            "max-height": "16vw", "overflow-y": "scroll", "overflow-x": "hidden",
                            "scrollbar-width": "1vw", "margin": "0vw 1vw 0vw 1vw"}),

                        ], className="row"),

                    ], className="row", style={"overflow-x": "scroll"}),

                ]),

                # second tab
                dcc.Tab(label="Preprocessed Data", value="data_tab2", className="data-tab",
                        selected_className="data-tab--selected", children=[

                    dt.DataTable(id="preprocessed_data_table", style_as_list_view=False,
                    style_data_conditional=[{"if": {"row_index": "odd"}, "background-color": "#ffffd2"}],
                    style_table={"display": "block", "max-height": "35vw", "max-width": "97%",
                    "overflow-y": "scroll", "overflow-x": "scroll", "margin": "2vw 1vw 2vw 1vw"},
                    style_cell={"text-align": "center", "font-family": "Open Sans", "font-size": "90%",
                    "height": "2vw"}, style_header={"background-color": "#3288BD", "text-align": "center",
                    "color": "white", "text-transform": "uppercase", "font-family": "Open Sans",
                    "font-size": "85%", "font-weight": "500", "height": "2vw"})

                ]),

                # third tab
                dcc.Tab(label="Descriptive Statistics", value="data_tab3", className="data-tab",
                        selected_className="data-tab--selected", children=[

                    dt.DataTable(id="stats_data_table", style_as_list_view=False,
                    style_data_conditional=[{"if": {"row_index": "odd"}, "background-color": "#ffffd2"},
                    {"if": {"column_id": "feature"}, "text-align": "left"}], style_table={"display": "block",
                    "max-height": "60vw", "max-width": "97%", "overflow-y": "scroll", "overflow-x": "scroll",
                    "margin": "2vw 1vw 2vw 1vw"}, style_cell={"text-align": "center", "font-family": "Open Sans",
                    "font-size": "90%", "height": "2vw"}, style_header={"background-color": "#3288BD", "color": "white",
                    "text-align": "center", "text-transform": "uppercase", "font-family": "Open Sans", "font-size": "85%",
                    "font-weight": "500", "height": "2vw"})

                ]),

                # fourth tab
                dcc.Tab(label="Correlation Matrix", value="data_tab4", className="data-tab",
                          selected_className="data-tab--selected", children=[

                    html.Div(children=[

                        html.Label("Select Features:", style={"margin": "1vw 0vw 0.5vw 0vw"}),
                        dcc.Dropdown(id="correlation_features", style={"font-size": "95%"}, optionHeight=25,
                        multi=True, searchable=True, clearable=True, placeholder="Select Features"),

                    ], style={"width": "97%", "margin": "0vw 0vw 0vw 1vw"}),

                    dcc.Loading(children=[

                        html.Div(id="correlation_plot", style={"height": "31vw", "width": "62vw", "margin-top": "1vw"}),

                    ], type="circle", color="#3288BD", style={"height": "31vw", "width": "62vw", "margin-top": "1vw",
                    "position": "relative", "top": "13vw"}),

                ]),

                # fifth tab
                dcc.Tab(label="Histograms", value="data_tab5", className="data-tab",
                        selected_className="data-tab--selected", children=[

                    html.Div(children=[

                        html.Label("Select Feature:", style={"margin": "1vw 0vw 0.5vw 0vw"}),
                        dcc.Dropdown(id="histogram_features", style={"font-size": "95%"}, optionHeight=25,
                        multi=False, searchable=True, clearable=False, placeholder="Select Feature"),

                    ], style={"display": "inline-block", "vertical-align": "top", "width": "20vw",
                    "margin": "0vw 0vw 0vw 1vw"}),

                    html.Div(children=[

                        dcc.Loading(children=[

                            html.Div(id="histogram_plot", style={"height": "30vw", "width": "40vw",
                            "margin": "1vw 0vw 0vw 2vw"}),

                        ], type="circle", color="#3288BD", style={"height": "30vw", "width": "40vw",
                        "margin": "1vw 0vw 0vw 2vw", "position": "relative", "top": "12vw"}),

                    ], style={"display": "inline-block", "vertical-align": "top"}),

                ]),

            ]),

        ], style={"display": "none"}),

        # div used for displaying the clustering output
        html.Div(id="cluster_output", children=[

            dcc.Tabs(value="cluster_tab1", parent_className="cluster-tabs", className="custom-tabs-container",
                    children=[

                # first tab
                dcc.Tab(label="Cluster Visualization", value="cluster_tab1", className="custom-tab",
                        selected_className="custom-tab--selected", children=[

                    html.Div(children=[

                        # radio buttons used for selecting the dimension reduction technique
                        html.Label("Dimension Reduction:", style={"margin": "1vw 0vw 0.5vw 1vw"}),
                        dcc.RadioItems(id="plot_dimension_reduction", value="pca",
                        options=[{"label": "PCA", "value": "pca"}, {"label": "T-SNE", "value": "tsne"},
                        {"label": "None", "value": "none"}], style={"margin-left": "1vw"},
                        labelStyle={"font-size": "95%", "display": "inline-block", "margin": "0vw 0.5vw 0vw 0vw"}),

                        # numeric input used for entering the number of components
                        html.Label("Number of Components:", style={"margin": "1vw 0vw 0vw 1vw"}),
                        html.P("If using PCA or T-SNE, enter the number of components.",
                        style={"font-size": "80%", "margin": "0vw 2vw 0vw 1vw", "text-align": "justify"}),
                        dcc.Input(id="plot_components", type="number", placeholder=3, value=3, min=1,
                        style={"margin-left": "1vw", "font-size": "95%"}),

                        # radio buttons used for selecting the plot dimensions
                        html.Label("Plot Dimensions:", style={"margin": "1vw 0vw 0.5vw 1vw"}),
                        dcc.RadioItems(id="plot_dimensions", value="2d", options=[{"label": "2-D", "value": "2d"},
                        {"label": "3-D", "value": "3d"}], style={"margin-left": "1vw"}, labelStyle={"font-size": "95%",
                        "display": "inline-block", "margin": "0vw 0.5vw 0vw 0vw"}),

                        # run button used for updating the plot
                        html.Div(children=[

                            html.Label("Update Plot:", style={"margin": "1vw 0vw 0.3vw 0vw"}),
                            html.Button(id="plot_button", n_clicks=0, children=["update"],
                            style={"background-color": "#3288BD", "font-size": "80%", "font-weight": "500",
                            "text-align": "center", "width": "60%", "color": "white"}),

                        ], style={"margin": "0vw 0vw 0vw 1vw"}),

                    ], style={"display": "inline-block", "vertical-align": "top", "width": "17.5vw"}),

                    html.Div(children=[

                        html.Div(children=[

                            html.Div(children=[

                                 # dropdown used for selecting the feature to plot on the X axis
                                 html.Label("X-Axis:", style={"margin": "1vw 0vw 0.5vw 0vw"}),
                                 dcc.Dropdown(id="x-axis", style={"font-size": "95%"}, optionHeight=25,
                                 multi=False, searchable=True, clearable=True, placeholder="Select Feature"),

                            ], style={"display": "inline-block", "width": "30%", "margin-left": "1vw"}),

                            html.Div(children=[

                                 # dropdown used for selecting the feature to plot on the Y axis
                                 html.Label("Y-Axis:", style={"margin": "1vw 0vw 0.5vw 0vw"}),
                                 dcc.Dropdown(id="y-axis", style={"font-size": "95%"}, optionHeight=25,
                                 multi=False, searchable=True, clearable=True, placeholder="Select Feature"),

                            ], style={"display": "inline-block", "width": "30%", "margin-left": "1.5vw"}),

                            html.Div(children=[

                                 # dropdown used for selecting the feature to plot on the Z axis
                                 html.Label("Z-Axis:", style={"margin": "1vw 0vw 0.5vw 0vw"}),
                                 dcc.Dropdown(id="z-axis", style={"font-size": "95%"}, optionHeight=25,
                                 multi=False, searchable=True, clearable=True, placeholder="Select Feature"),

                            ], style={"display": "inline-block", "width": "30%", "margin-left": "1.5vw"}),

                        ], className="row", style={"display": "flex"}),

                        html.Div(id="cluster_plot", style={"margin": "1vw 1vw 1vw 1vw"}),

                    ], style={"display": "inline-block", "vertical-align": "top", "width": "40vw",
                              "margin": "0vw 0vw 0vw 1vw"}),

                ]),

                # second tab
                dcc.Tab(label="Clustered Data", value="cluster_tab2", className="data-tab",
                        selected_className="data-tab--selected", children=[

                    html.Div(children=[

                        html.Label(children=["Download CSV"], style={"margin": "1vw 0vw 0vw 50vw",
                           "display": "inline-block", "vertical-align": "middle"}),

                            html.A(id="cluster_data_link", target="_blank", children=[

                                html.Img(id="cluster_data_button", src=app.get_asset_url("icon.png"),
                                style={"height": "1.5vw", "width": "2vw", "margin": "0", "padding": "0"}),

                            ], style={"cursor": "pointer", "display": "inline-block", "margin": "1vw 0vw 0vw 1vw",
                                      "vertical-align": "middle"}),

                    ], className="row", style={"display": "flex"}),

                    dt.DataTable(id="cluster_data_table", style_as_list_view=False,
                    style_data_conditional=[{"if": {"row_index": "odd"}, "background-color": "#ffffd2"}],
                    style_table={"display": "block", "max-height": "60vw", "max-width": "97%",
                    "overflow-y": "scroll", "overflow-x": "scroll", "margin": "0.5vw 1vw 2vw 1vw"},
                    style_cell={"text-align": "center", "font-family": "Open Sans", "font-size": "90%",
                    "height": "2vw"}, style_header={"background-color": "#3288BD", "text-align": "center",
                    "color": "white", "text-transform": "uppercase", "font-family": "Open Sans",
                    "font-size": "85%", "font-weight": "500", "height": "2vw"}),

                ]),

                # third tab
                dcc.Tab(label="Scree Plot", value="cluster_tab3", className="data-tab",
                        selected_className="data-tab--selected", children=[

                        html.Div(id="scree_plot", style={"margin": "1vw 1vw 1vw 1vw"}),

                ]),

            ]),

        ], style={"display": "none"}),

    ], style={"display": "inline-block", "vertical-align": "top", "width": "63vw", "height": "70vw",
              "margin": "0vw 1vw 0vw 3vw", "background-color": "white", "border": "0.1vw solid #D9D9D9"}),

    # hidden divs used for storing the data shared across callbacks
    html.Div(id="uploaded_data", style={"display": "none"}),
    html.Div(id="raw_data", style={"display": "none"}),
    html.Div(id="processed_data", style={"display": "none"}),
    html.Div(id="clustered_data", style={"display": "none"}),
    html.Div(id="plot_data", style={"display": "none"}),

])

####################################################
# maximum number of features to be processed;
# it can be changed if required
N = 20
####################################################

@app.callback([Output("alert_output", "style"), Output("data_output", "style"), Output("cluster_output", "style")],
              [Input("uploaded_data", "children"), Input("control_tab", "value")])
def render_switch(uploaded_data, tab):

    if uploaded_data is None:

        return [{"display": "block"}, {"display": "none"}, {"display": "none"}]

    elif tab == "tab1":

        return [{"display": "none"}, {"display": "block"}, {"display": "none"}]

    elif tab == "tab2":

        return [{"display": "none"}, {"display": "none"}, {"display": "block"}]

def parse_contents(contents, filename):

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:

        if "csv" in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        elif "xls" in filename:
            df = pd.read_excel(io.BytesIO(decoded))

        elif "json" in filename:
            df = pd.read_json(contents)

        elif "txt" or "tsv" in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), delimiter=r"\s+")

    except Exception as e:
        print(e)

    return df.to_json()

@app.callback(Output("uploaded_data", "children"), [Input("uploaded_file", "contents")],
              [State("uploaded_file", "filename")])
def load_file(contents, file_name):

    if contents is not None:

        df_json = parse_contents(contents, file_name)

        return df_json

@app.callback([Output("data_controls", "children"), Output("data_controls", "style"),
               Output("raw_data_table", "data"), Output("raw_data_table", "columns"),
               Output("data_container", "style"), Output("raw_data", "children")],
              [Input("uploaded_data", "children")])
def load_data(data):

    if data is not None:

        # load the data
        df = pd.read_json(data)

        # process the missing values
        df[df == "-"] = np.nan
        df[df == "?"] = np.nan
        df[df == "."] = np.nan
        df[df == " "] = np.nan

        # drop the missing values
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # include the indices in the first columns
        df.rename(columns={"Unnamed: 0": "index"}, inplace=True)

        # make sure that the indices are treated as integers
        df["index"] = df["index"].astype(int)

        # display the data in the table
        data_rows = df.to_dict(orient="records")
        data_columns = [{"id": x, "name": x} for x in list(df.columns)]
        data_container_style = {"white-space": "nowrap", "height": "17vw", "width": str(1 + len(df.columns) * 15) +"vw",
        "min-width": str(1 + len(df.columns) * 15) +"vw", "max-width": str(1 + len(df.columns) * 15) +"vw"}
        data_controls_style = {"white-space": "nowrap", "height": "5rem + 10vw", "margin": "0.5vw 0vw 0vw 16vw",
        "padding-bottom": "2vw", "overflow-x": "hidden", "overflow-y": "hidden", "scrollbar-width": "1vw",
        "width": str(1 + len(df.columns) * 15) +"vw", "min-width": str(1 + len(df.columns) * 15) +"vw",
        "max-width": str(1 + len(df.columns) * 15) +"vw"}

        # define the initial data types, weights and transformations
        selection = pd.DataFrame({"feature": df.columns, "type": df.dtypes, "weight": np.ones(df.shape[1]),
        "transformation": ["none"] * df.shape[1]})
        selection["type"][(selection["type"] == "object") | (selection["type"] == "category")] = "categorical"
        selection["type"][-(selection["type"] == "categorical")] = "numerical"
        selection.reset_index(drop=True, inplace=True)

        # display the data types, weights and transformations in the dropdown menus
        dropdowns = []
        selections = []

        for j in range(N):

            if j == 0:

                dropdowns.append(

                    html.Div(children=[

                        dcc.Dropdown(id="type_" + str(j), multi=False, value="none"),
                        dcc.Input(id="weight_" + str(j), type="number", value=1),
                        dcc.Dropdown(id="transformation_" + str(j), multi=False, value="none"),

                    ], style={"display": "none"})

                )

                selections.append(html.Div(id="selections_" + str(j), children=selection.to_json()))

            elif j != 0 and j < selection.shape[0]:

                dropdowns.append(

                    html.Div(children=[

                        html.Label(children=selection["feature"][j], style={"line-height": "2.01vw", "height": "2.01vw",
                        "width": "15.01vw", "border": "0.5px solid #D9D9D9", "text-transform": "uppercase", "font-size": "85%",
                        "color": "white", "font-weight": "500", "margin": "0.5vw 0vw 0.5vw 0vw", "text-align": "center",
                        "background-color": "#3288BD", "text-overflow": "ellipsis", "box-sizing": "border-box"}),

                        html.Div(children=[

                            html.Label(children=["Transformation"], style={"margin": "0.25vw 0vw 0.1vw 0vw"}),
                            dcc.Dropdown(id="transformation_" + str(j), value=selection["transformation"][j], multi=False,
                            searchable=True, clearable=False, optionHeight=25, options=[{"label": "Logarithm", "value": "log"},
                            {"label": "Z-Score", "value": "z-score"}, {"label": "MinMax", "value": "minmax"},
                            {"label": "None", "value": "none"}], style={"font-size": "95%"}),

                        ], style={"width": "14vw"}),

                        html.Div(children=[

                            html.Label(children=["Type"], style={"margin": "0.25vw 0vw 0.1vw 0vw"}),
                            dcc.Dropdown(id="type_" + str(j), value=selection["type"][j], multi=False, searchable=True,
                            clearable=False, optionHeight=25, options=[{"label": "Numerical", "value": "numerical"},
                            {"label": "Categorical", "value": "categorical"}, {"label": "None", "value": "none"}],
                            style={"font-size": "95%"}),

                        ], style={"width": "14vw"}),

                        html.Label(children=["Weight"], style={"margin": "0.25vw 0vw 0.1vw 0vw"}),
                        dcc.Input(id="weight_" + str(j), type="number", value=selection["weight"][j],
                        style={"font-size": "95%", "height": "35px", "line-height": "35px", "width": "14vw"}),

                    ], style={"display": "inline-block", "vertical-align": "top"}),

                )

                selections.append(html.Div(id="selections_" + str(j)))

            else:

                dropdowns.append(

                    html.Div(children=[

                        dcc.Dropdown(id="transformation_" + str(j), multi=False, value="none"),
                        dcc.Dropdown(id="type_" + str(j), multi=False, value="none"),
                        dcc.Input(id="weight_" + str(j), type="number", value=1),

                    ], style={"display": "none"})

                )

                selections.append(html.Div(id="selections_" + str(j)))

        data_controls = [html.Div(children=dropdowns, className="row"),
                         html.Div(children=selections, style={"display": "none"})]

        return [data_controls, data_controls_style, data_rows, data_columns, data_container_style, df.to_json()]

for j in range(1, N):
    @app.callback(Output("selections_" + str(j), "children"), [Input("type_" + str(j), "value"),
            Input("weight_" + str(j), "value"), Input("transformation_" + str(j), "value"),
            Input("selections_" + str(j-1), "children")])
    def update_selection(dtype, weight, transformation, selection, k=j):

        selection = pd.read_json(selection)

        selection["type"][k] = dtype
        selection["weight"][k] = weight
        selection["transformation"][k] = transformation

        return selection.to_json()

@app.callback([Output("processed_data", "children"), Output("preprocessed_data_table", "data"),
               Output("preprocessed_data_table", "columns"), Output("correlation_features", "options"),
               Output("histogram_features", "options")], [Input("data_button", "n_clicks"),
               Input("raw_data", "children")], [State("selections_" + str(0), "children"),
               State("selections_" + str(N-1), "children")])
def process_data(n_clicks, data, initial_selection, final_selection):

    if data is not None:

        # load the raw data from the hidden div
        df = pd.read_json(data)

        # load the user's selection from the hidden div
        if final_selection is None:

            selection = pd.read_json(initial_selection)

        else:

            selection = pd.read_json(final_selection)

        # discard the features with zero weight and / or "none" data type
        selection = selection[(selection["weight"] != 0) & (selection["type"] != "none")]
        df = df[list(selection["feature"])]

        # discard the index
        selection = selection[selection["feature"] != "index"]
        features = list(selection["feature"])

        # normalize the weights
        selection["weight"] = selection["weight"] / selection["weight"].sum()

        # process the data
        for feature in features:

            x = df[[feature,]]

            dtype = selection["type"][selection["feature"] == feature].values[0]
            weight = selection["weight"][selection["feature"] == feature].values[0]
            transformation = selection["transformation"][selection["feature"] == feature].values[0]

            if (x.dtypes == "object")[0] and dtype == "numerical": # from categorical to numerical

                x = LabelEncoder().fit_transform(x)

                x = weight * x

                x = x.reshape(-1,1)

                if transformation == "log":

                    x = FunctionTransformer(np.log1p, validate=True).fit_transform(x)

                elif transformation == "z-score":

                    x = StandardScaler().fit_transform(x)

                elif transformation == "minmax":

                    x = MinMaxScaler().fit_transform(x)

                df[feature] = x

            elif (x.dtypes != "object")[0] and dtype == "categorical": # from numerical to categorical

                x = pd.get_dummies(pd.cut(x[feature], 3, labels=[feature + "_cat1", feature + "_cat2", feature + "_cat3"]))

                x = weight * x

                if transformation == "log":

                    x = pd.DataFrame(FunctionTransformer(np.log1p, validate=True).fit_transform(x), index=x.index, columns=x.columns)

                elif transformation == "z-score":

                    x = pd.DataFrame(StandardScaler().fit_transform(x), index=x.index, columns=x.columns)

                elif transformation == "minmax":

                    x = pd.DataFrame(MinMaxScaler().fit_transform(x), index=x.index, columns=x.columns)

                df.drop(feature, axis=1, inplace=True)

                df = df.join(x)

            elif (x.dtypes != "object")[0] and dtype == "numerical": # from numerical to numerical

                x = weight * x.values

                x = x.reshape(-1, 1)

                if transformation == "log":

                    x = FunctionTransformer(np.log1p, validate=True).fit_transform(x)

                elif transformation == "z-score":

                    x = StandardScaler().fit_transform(x)

                elif transformation == "minmax":

                    x = MinMaxScaler().fit_transform(x)

                df[feature] = x

            elif (x.dtypes == "object")[0] and dtype == "categorical": # from categorical to categorical

                x = pd.get_dummies(x)

                x = weight * x

                if transformation == "log":

                    x = pd.DataFrame(FunctionTransformer(np.log1p, validate=True).fit_transform(x), index=x.index, columns=x.columns)

                elif transformation == "z-score":

                    x = pd.DataFrame(StandardScaler().fit_transform(x), index=x.index, columns=x.columns)

                elif transformation == "minmax":

                    x = pd.DataFrame(MinMaxScaler().fit_transform(x), index=x.index, columns=x.columns)

                df.drop(feature, axis=1, inplace=True)

                df = df.join(x)

        # round all values to 2 digits
        processed_data = df.astype(float).round(2)

        # display the data in the table
        processed_data_rows = processed_data.to_dict(orient="records")
        processed_data_columns = [{"id": x, "name": x} for x in list(processed_data.columns)]

        # create the lists of features to be shown in the dropdown menus
        new_features = list(df.columns)
        new_features.remove("index")

        new_features_list = [{"value": new_features[0], "label": new_features[0]}]

        if len(new_features) > 1:
            for j in range(1, len(new_features)):
                new_features_list.append({"value": new_features[j], "label": new_features[j]})

        correlation_features = new_features_list
        histogram_features = new_features_list

        # save the data in the hidden div
        processed_data = df.to_json()

        return [processed_data, processed_data_rows, processed_data_columns, correlation_features, histogram_features]

@app.callback([Output("stats_data_table", "data"), Output("stats_data_table", "columns")],
              [Input("processed_data", "children")])
def update_statistics(data):

    if data is not None:

        # load the processed data from the hidden div
        df = pd.read_json(data)

        # drop the index
        df.drop("index", axis=1, inplace=True)

        # calculate the descriptive statistics
        stats = df.describe().transpose()

        # round all values to 2 digits
        stats = stats.astype(float).round(2)

        # display the names of the features in the first column
        stats["feature"] = stats.index
        names = ["feature"]
        names.extend(list(stats.columns[stats.columns != "feature"]))
        stats = stats[names]
        stats.reset_index(drop=True, inplace=True)

        # display the results in the table
        stats_data_rows = stats.to_dict(orient="records")
        stats_data_columns = [{"id": x, "name": x} for x in list(stats.columns)]

        return [stats_data_rows, stats_data_columns]

@app.callback(Output("correlation_plot", "children"), [Input("processed_data", "children"),
              Input("correlation_features", "value")])
def update_correlation_matrix(data, features):

    if data is not None:

        # load the processed data from the hidden div
        df = pd.read_json(data)

        # drop the index
        df.drop("index", axis=1, inplace=True)

        # calculate the sample correlation matrix
        if features is not None:

            if len(features) > 0:

                sigma = df[features].corr()

            else:

                sigma = df.iloc[:,:10].corr()

        else:

            sigma = df.iloc[:, :10].corr()

        # plot the sample correlation matrix
        y = list(sigma.index)
        x = list(sigma.columns)
        z = np.nan_to_num(sigma.values)

        annotations = []
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                annotations.append(dict(x=x[i], y=y[j], text=str(np.round(z[i, j] * 100, 2)) + "%", showarrow=False))

        layout = dict(annotations=annotations, xaxis=dict(tickangle=45), yaxis=dict(tickangle=0),
                      font=dict(family="Open Sans", size=9), margin=dict(t=5, l=5, r=5, b=5, pad=0))

        traces = []
        traces.append(go.Heatmap(z=z, x=x, y=y, zmin=-1, zmax=1, colorscale="Spectral"))

        figure = go.Figure(data=traces, layout=layout).to_dict()

        correlation_plot = dcc.Graph(figure=figure, config={"responsive": True, "autosizable": True,
                            "showTips": True, "displaylogo": False})

        return correlation_plot

@app.callback(Output("histogram_plot", "children"), [Input("processed_data", "children"),
              Input("histogram_features", "value")])
def update_histogram(data, features):

    if data is not None:

        # load the processed data from the hidden div
        df = pd.read_json(data)

        # drop the index
        df.drop("index", axis=1, inplace=True)

        # plot the histogram
        if features is not None:

            if len(features) > 0:

                data = df[features].values
                name = features

            else:

                data = list(df.iloc[:, 0].values)
                name = df.columns[0]

        else:

            data = list(df.iloc[:,0].values)
            name = df.columns[0]

        layout = dict(plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                      font=dict(family="Open Sans", size=9), margin=dict(t=5, l=5, r=5, b=5, pad=0),
                      xaxis=dict(zeroline=False, showgrid=True, mirror=True, linecolor="#d9d9d9", tickangle=0),
                      yaxis=dict(zeroline=False, showgrid=True, mirror=True, linecolor="#d9d9d9", tickangle=0))

        traces = []
        traces.append(go.Histogram(x=data, name=name, text=name, hoverinfo="text+x+y",
                       marker=dict(color="#98c3de", line=dict(color="#3288BD", width=1))))

        figure = go.Figure(data=traces, layout=layout).to_dict()

        histogram_plot = [

            html.Label(children=["Histogram of " + name], style={"margin": "0vw 0vw 0.5vw 0vw", "text-align": "center"}),
            dcc.Graph(figure=figure, config={"responsive": True, "autosizable": True, "showTips": True, "displaylogo": False},
                      style={"height": "28vw", "width": "38vw"}),

        ]

        return histogram_plot

@app.callback(Output("scree_plot", "children"), [Input("processed_data", "children")])
def update_scree_plot(data):

    if data is not None:

        # load the processed data from the hidden div
        df = pd.read_json(data)

        # drop the index
        df.drop("index", axis=1, inplace=True)

        # run the PCA
        y = list(PCA(n_components=np.min([10, df.shape[1]]), random_state=0).fit(df).explained_variance_ratio_)
        x = [z + 1 for z in range(np.min([10, df.shape[1]]))]

        # generate the scree plot
        layout = dict(plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                    font=dict(family="Open Sans", size=9), margin=dict(t=20, l=20, r=20, b=20),
                    xaxis=dict(zeroline=False, showgrid=False, mirror=True, linecolor="#d9d9d9", tickangle=0,
                    tickmode="array", tickvals=x, title_text="Component"),
                    yaxis=dict(range=[-0.05, 1.05], zeroline=False, showgrid=False, mirror=True, linecolor="#d9d9d9",
                    tickangle=0, tickmode="array", tickvals=list(np.linspace(0,1,11)), title_text="Explained Variance Ratio"))

        traces = []
        traces.append(go.Scatter(x=x, y=y, mode="lines+markers", hoverinfo="x+y", marker=dict(size=10, color="#cbe1ee",
                line=dict(color="#3288BD", width=1)), line=dict(color="#3288BD", width=2)))

        figure = go.Figure(data=traces, layout=layout).to_dict()

        scree_plot = dcc.Graph(figure=figure, config={"responsive": True, "autosizable": True,
        "showTips": True, "displaylogo": False}, style={"height": "30vw", "width": "60vw"})

        return scree_plot

@app.callback([Output("cluster_data_table", "data"), Output("cluster_data_table", "columns"),
               Output("clustered_data", "children")], [Input("cluster_button", "n_clicks"),
               Input("processed_data", "children")], [State("cluster_random_sampling", "value"),
               State("cluster_sample_size", "value"), State("cluster_dimension_reduction", "value"),
               State("cluster_components", "value"), State("cluster_algorithm", "value"),
               State("cluster_number", "value"), State("cluster_size", "value")])
def cluster_analysis(clicks, data, random_sampling, sample_size, dimension_reduction, num_components,
                     cluster_algorithm, num_clusters, cluster_size):

    if data is not None:

        # load the processed data from the hidden div
        df = pd.read_json(data)

        # create a copy of the data frame
        df_copy = df.copy()

        # perform random sampling
        if random_sampling == "True":

            if sample_size is not None:

                # calculate the number of samples
                n_samples = np.int(sample_size * df.shape[0] / 100)

                # generate the random sample
                sample = sample_without_replacement(n_population=df.shape[0], n_samples=n_samples, random_state=0)
                sample = np.sort(sample)

                # extract the random sample
                df = df.iloc[sample, :]
                df.reset_index(inplace=True, drop=True)

        # drop the indices
        indices = df["index"]
        df.drop("index", axis=1, inplace=True)

        # run the dimension reduction algorithm
        if dimension_reduction == "pca":

            if num_components > 0 and num_components <= df.shape[1]:

                df = PCA(n_components=np.int(num_components), random_state=0).fit_transform(df)
                df = pd.DataFrame(data=df, columns=["Component " + str(x) for x in range(1, num_components + 1)])

            else:

                df = PCA(n_components=3, random_state=0).fit_transform(df)
                df = pd.DataFrame(data=df, columns=["Component " + str(x) for x in range(1, 4)])

        elif dimension_reduction == "tsne":

            if num_components > 0 and num_components <= 3:

                df = TSNE(n_components=np.int(num_components), random_state=0).fit_transform(df)
                df = pd.DataFrame(data=df, columns=["Component" + str(x) for x in range(1, num_components + 1)])

            else:

                df = TSNE(n_components=3, random_state=0).fit_transform(df)
                df = pd.DataFrame(data=df, columns=["Component" + str(x) for x in range(1, 4)])

        # run the clustering algorithm
        if cluster_algorithm == "kmeans":

            if num_clusters > 0 and num_clusters <= df.shape[0]:

                algo = KMeans(n_clusters=num_clusters, random_state=0).fit(df)

            else:

                algo = KMeans(n_clusters=3).fit(df)

        elif cluster_algorithm == "hdbscan":

            if cluster_size > 1 and cluster_size <= df.shape[0]:

                algo = hdbscan.HDBSCAN(min_cluster_size=cluster_size).fit(df)

            else:

                algo = hdbscan.HDBSCAN(min_cluster_size=2).fit(df)

        # add the cluster labels
        df["cluster labels"] = algo.labels_
        df["cluster labels"] = 1 + df["cluster labels"]

        # add back the indices
        df["index"] = indices
        df = df[["index", "cluster labels"]]
        df = pd.merge(left=df, right=df_copy, on="index", how="left")

        # save the results in the hidden div
        cluster_data = df.to_json()

        # round all values to 2 digits
        df.iloc[:,2:] = df.iloc[:,2:].astype(float).round(2)

        # display the results in the table
        cluster_data_rows = df.to_dict("records")
        cluster_data_columns = [{"id": x, "name": x} for x in list(df.columns)]

        return [cluster_data_rows, cluster_data_columns, cluster_data]

@app.callback([Output("x-axis", "options"), Output("y-axis", "options"), Output("z-axis", "options"),
               Output("plot_data", "children")], [Input("clustered_data", "children"), Input("plot_button", "n_clicks")],
              [State("plot_dimension_reduction", "value"), State("plot_components", "value")])
def update_plot_data(data, plot_button, plot_dimension_reduction, plot_components):

    if data is not None:

        # load the clustering results from the hidden div
        df = pd.read_json(data)

        # drop the indices and the cluster labels
        indices = df["index"]
        labels = df["cluster labels"]
        df.drop(["index", "cluster labels"], axis=1, inplace=True)

        # run the dimension reduction algorithm
        if plot_dimension_reduction == "pca":

            if plot_components > 0 and plot_components <= df.shape[1]:

                df = PCA(n_components=np.int(plot_components), random_state=0).fit_transform(df)
                df = pd.DataFrame(data=df, columns=["Component " + str(x) for x in range(1, plot_components + 1)])

            else:

                df = PCA(n_components=3, random_state=0).fit_transform(df)
                df = pd.DataFrame(data=df, columns=["Component " + str(x) for x in range(1, 4)])

        elif plot_dimension_reduction == "tsne":

            if plot_components > 0 and plot_components <= 3:

                df = TSNE(n_components=np.int(plot_components), random_state=0).fit_transform(df)
                df = pd.DataFrame(data=df, columns=["Component" + str(x) for x in range(1, plot_components + 1)])

            else:

                df = TSNE(n_components=3, random_state=0).fit_transform(df)
                df = pd.DataFrame(data=df, columns=["Component" + str(x) for x in range(1, 4)])

        # create the lists of features to be shown in the dropdown menus
        columns = list(df.columns)
        features_list = [{"value": columns[0], "label": columns[0]}]

        if len(columns) > 1:
            for j in range(1, len(columns)):
                features_list.append({"value": columns[j], "label": columns[j]})

        x_axis_options = features_list
        y_axis_options = features_list
        z_axis_options = features_list

        # add back the indices and the cluster labels
        df["index"] = indices
        df["cluster labels"] = labels
        plot_data = df.to_json()

        return [x_axis_options, y_axis_options, z_axis_options, plot_data]

@app.callback(Output("cluster_plot", "children"), [Input("plot_data", "children"), Input("x-axis", "value"),
              Input("y-axis", "value"), Input("z-axis", "value")], [State("plot_dimensions", "value")])
def update_scatter_plot(data, x_axis, y_axis, z_axis, plot_dimensions):

    if data is not None:

        df = pd.read_json(data)

        if plot_dimensions == "2d":

            if x_axis is None or y_axis is None:

                layout = dict(paper_bgcolor="white", plot_bgcolor="white", showlegend=False,
                margin=dict(t=20, b=20, r=20, l=20), font=dict(family="Open Sans", size=9),
                xaxis=dict(zeroline=False, showgrid=False, mirror=True, linecolor="#d9d9d9", tickangle=0, title_text=df.columns[0]),
                yaxis=dict(zeroline=False, showgrid=False, mirror=True, linecolor="#d9d9d9", tickangle=0, title_text=df.columns[1]))

                traces = []
                traces.append(go.Scatter(x=list(df.iloc[:,0]), y=list(df.iloc[:,1]), mode="markers", hoverinfo="text",
                text=["Cluster " + str(x) + " (Index " + str(y) + ")" for x, y in zip(list(df["cluster labels"]), list(df["index"]))],
                marker=dict(color=list(df["cluster labels"]), colorscale="Spectral", size=9, line=dict(width=1))))

                figure = go.Figure(data=traces, layout=layout).to_dict()

            else:

                layout = dict(paper_bgcolor="white", plot_bgcolor="white", showlegend=False,
                margin=dict(t=20, b=20, r=20, l=20), font=dict(family="Open Sans", size=9),
                xaxis=dict(zeroline=False, showgrid=False, mirror=True, linecolor="#d9d9d9", tickangle=0, title_text=x_axis),
                yaxis=dict(zeroline=False, showgrid=False, mirror=True, linecolor="#d9d9d9", tickangle=0, title_text=y_axis))

                traces = []
                traces.append(go.Scatter(x=list(df[x_axis]), y=list(df[y_axis]), mode="markers", hoverinfo="text",
                text=["Cluster " + str(x) + " (Index " + str(y) + ")" for x, y in zip(list(df["cluster labels"]), list(df["index"]))],
                marker=dict(color=list(df["cluster labels"]), colorscale="Spectral", size=9, line=dict(width=1))))

                figure = go.Figure(data=traces, layout=layout).to_dict()

            scatter_plot = dcc.Graph(figure=figure, config={"responsive": True, "autosizable": True, "showTips": True,
            "displaylogo": False}, style={"height": "25vw", "width": "40vw"})

        elif plot_dimensions == "3d":

            if x_axis is None or y_axis is None or z_axis is None:

                layout = dict(paper_bgcolor="white", plot_bgcolor="white", margin=dict(t=0, l=0, r=0, b=0, pad=0),
                font=dict(family="Open Sans", size=9), scene=dict(xaxis=dict(backgroundcolor="white", zeroline=False,
                showgrid=True, mirror=True, linecolor="#d9d9d9", gridcolor="#d9d9d9", tickangle=0, title_text=df.columns[0]),
                yaxis=dict(backgroundcolor="white", zeroline=False, showgrid=True, mirror=True, linecolor="#d9d9d9",
                gridcolor="#d9d9d9", tickangle=0, title_text=df.columns[1]), zaxis=dict(backgroundcolor="white",
                zeroline=False, showgrid=True, mirror=True, linecolor="#d9d9d9", gridcolor="#d9d9d9", tickangle=0,
                title_text=df.columns[2])))

                traces = []
                traces.append(go.Scatter3d(x=list(df.iloc[:,0]), y=list(df.iloc[:,1]), z=list(df.iloc[:,2]), mode="markers", hoverinfo="text",
                text=["Cluster " + str(x) + " (Index " + str(y) + ")" for x, y in zip(list(df["cluster labels"]), list(df["index"]))],
                marker=dict(color=list(df["cluster labels"]), colorscale="Spectral", size=7, line=dict(width=2))))

                figure = go.Figure(data=traces, layout=layout).to_dict()

            else:

                layout = dict(paper_bgcolor="white", plot_bgcolor="white", margin=dict(t=0, l=0, r=0, b=0, pad=0),
                font=dict(family="Open Sans", size=9), scene=dict(xaxis=dict(backgroundcolor="white", zeroline=False,
                showgrid=True, mirror=True, linecolor="#d9d9d9", gridcolor="#d9d9d9", tickangle=0, title_text=df.columns[0]),
                yaxis=dict(backgroundcolor="white", zeroline=False, showgrid=True, mirror=True, linecolor="#d9d9d9",
                gridcolor="#d9d9d9", tickangle=0, title_text=df.columns[1]), zaxis=dict(backgroundcolor="white",
                zeroline=False, showgrid=True, mirror=True, linecolor="#d9d9d9", gridcolor="#d9d9d9", tickangle=0,
                title_text=df.columns[2])))

                traces = []
                traces.append(go.Scatter3d(x=list(df[x_axis]), y=list(df[y_axis]), z=list(df[z_axis]), mode="markers", hoverinfo="text",
                text=["Cluster " + str(x) + " (Index " + str(y) + ")" for x, y in zip(list(df["cluster labels"]), list(df["index"]))],
                marker=dict(color=list(df["cluster labels"]), colorscale="Spectral", size=7, line=dict(width=2))))

                figure = go.Figure(data=traces, layout=layout).to_dict()

            scatter_plot = dcc.Graph(figure=figure, config={"responsive": True, "autosizable": True, "showTips": True,
            "displaylogo": False}, style={"height": "25vw", "width": "40vw"})

        return scatter_plot

@app.callback([Output("cluster_data_link", "href"), Output("cluster_data_link", "download")],
              [Input("cluster_data_button", "n_clicks"), Input("clustered_data", "children")])
def download_file(n_clicks, data):

    if data is not None:

        # load the clustering results from the hidden div
        df = pd.read_json(data)

        # convert the table to csv
        csv = df.to_csv(index=False, encoding="utf-8")

        # create the file for download
        file_for_download = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv)

        # use the current time as the file name
        file_name = "clustering_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"

        return file_for_download, file_name

if __name__ == "__main__":
    app.run_server(port=8080, debug=False) # use this for running on local machine
    #application.run_server(port=8080, debug=False) # use this for deploying on AWS
