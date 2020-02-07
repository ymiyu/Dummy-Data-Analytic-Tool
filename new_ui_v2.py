# -*- coding: utf-8 -*-

# Author: Yiming Yu

import base64
from datetime import datetime
import io
import urllib.parse
import warnings

import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table as dt
import dash_daq as daq
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler
from sklearn.utils.random import sample_without_replacement
from sklearn.manifold import TSNE
import hdbscan

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

external_stylesheets = ["https://fonts.googleapis.com/css?family=Open+Sans:300,400,700",
                        "https://github.com/facultyai/dash-bootstrap-components"]

app = dash.Dash(__name__, external_stylesheets = external_stylesheets)

app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
app.config.suppress_callback_exceptions = True

app.title = "Clustering Tool"

app.layout = html.Div(children=[

    # header
    # TODO revert back to header and place the image on the right of div
    html.Nav(
        className="navbar navbar-expand-lg navbar-dark bg-dark",
        children=[
            html.A(
                html.H3('Data Analysis & Clustering Tool',
                        style={'color': 'white'}),
            ),

            # html.A(
            #     html.Img(
            #         src="assets/AP_Logo-H-White-new.png",
            #         alt=""
            #     ),
            #     className='float-right',
            #     href='https://analyticpartners.com/'
            # ),

        ]
    ),

    html.Div(children=[

        # tabs
        html.Div(children=[

            dcc.Tabs(id="control_tab",
                     value="tab1",
                     parent_className="custom-tabs",
                     className="custom-tabs-container",
                     children=[
                         # first tab
                         dcc.Tab(label="Data Preprocessing",
                                 value="tab1",
                                 className="custom-tab",
                                 selected_className="custom-tab--selected",
                                 children=[
                                     html.Div(children=[
                                     # component used for uploading the file with the data
                                     html.Div(children=[
                                         html.Label("Upload Dataset:",
                                                    style={"margin": "1vw 0vw 0.3vw 0vw"}),
                                         dcc.Upload(id="uploaded_file",
                                                    children=html.Div([
                                                        html.P("Drag and Drop or ",
                                                               style={"display": "inline"}),
                                                        html.A("Select File",
                                                               style={"display": "inline",
                                                                      "text-decoration": "underline",
                                                                      "cursor": "pointer"}),
                                                    ]),
                                                    style={"border": "1px dashed #D9D9D9",
                                                           "border-radius": "5px",
                                                           "text-align": "center",
                                                           "height": "35px",
                                                           "line-height": "35px",
                                                           "width": "90%"},
                                                    multiple=False),
                                         dbc.Alert("Must have a column named 'index' before uploading.",
                                                   style={'width': "90%"})

                                     ], style={"margin": "0vw 0vw 0vw 1vw"}),

                                     # dropdown menu used for selecting the features
                                     html.Div(children=[
                                         html.Label("Select Features:",
                                                    style={"margin": "1vw 0vw 0.3vw 0vw"}),
                                         dcc.Dropdown(id="data_features",
                                                      style={"font-size": "95%"},
                                                      optionHeight=35,
                                                      multi=True,
                                                      searchable=True,
                                                      clearable=True,
                                                      placeholder="Select Features"),
                                         dbc.Alert("maximum 10 number of features allowed for Correlation Matrix display."),
                                     ],
                                         style={"margin": "0vw 0vw 0vw 1vw",
                                                "width": "90%"}),

                                     # dropdown menu used for selecting the weights
                                     html.Div(children=[
                                         dbc.Label("Select Weights:",
                                                   style={"margin": "1vw 0vw 0.3vw 0vw"}
                                                   ),
                                         dbc.Input(id="data_weights",
                                                   placeholder="Enter a list of weights separated by '.' for each feature.",
                                                   type="text"),
                                         # dcc.Dropdown(id="data_weights",
                                         #              style={"font-size": "95%"},
                                         #              optionHeight=35,
                                         #              multi=True,
                                         #              searchable=True,
                                         #              clearable=True,
                                         #              placeholder="Select Weights"
                                         #              ),

                                     ], style={"margin": "0vw 0vw 0vw 1vw",
                                               "width": "90%"}),

                                     # radio buttons used for selecting the transformation
                                     html.Div(children=[
                                         html.Label("Select Transformation:",
                                                    style={"margin": "1vw 0vw 0.5vw 0vw"}),
                                         dbc.RadioItems(id="data_transformation",
                                                        value="none",
                                                        options=[{"label": "Logarithm", "value": "log"},
                                                                 {"label": "Z-score", "value": "z-score"},
                                                                 {"label": "MinMax", "value": "minmax"},
                                                                 {"label": "None", "value": "none"}
                                                                 ],
                                                        labelStyle={"font-size": "95%",
                                                                    "display": "inline-block",
                                                                    "margin": "0vw 0.5vw 0vw 0vw"}
                                                        ),
                                     ],
                                         style={"margin": "0vw 0vw 0vw 1vw"}),

                    # radio buttons used for selecting the approach for numerical missing values
                    # html.Div(children=[
                    #
                    #     html.Label("Process Missing Numerical Values:", style={"margin": "1vw 0vw 0.5vw 0vw"}),
                    #     dcc.RadioItems(id="missing_numerical", value="drop", options=[{"label": "Mean", "value": "mean"},
                    #     {"label": "Median", 'value': "median"}, {"label": "Mode", "value": "mode"},
                    #     {"label": "Drop", 'value': "drop"}], labelStyle={"font-size": "95%", "display": "inline-block",
                    #     "margin": "0vw 0.5vw 0vw 0vw"}),
                    #
                    # ], style={"margin": "0vw 0vw 0vw 1vw"}),
                    #
                    # # radio buttons used for selecting the approach for categorical missing values
                    # html.Div(children=[
                    #
                    #     html.Label("Process Missing Categorical Values:", style={"margin": "1vw 0vw 0.5vw 0vw"}),
                    #     dcc.RadioItems(id="missing_categorical", value="drop", options=[{"label": "Mode", "value": "mode"},
                    #     {"label": "Drop", 'value': "drop"}], labelStyle={"font-size": "95%", "display": "inline-block",
                    #     "margin": "0vw 0.5vw 0vw 0vw"}),
                    #
                    # ], style={"margin": "0vw 0vw 0vw 1vw"}),

                    # run button used for updating the data after making a selection
                                     html.Div(children=[
                                         html.Label("Update Dataset:",
                                                    style={"margin": "1vw 0vw 0.3vw 0vw"}),
                                         dbc.Button(id="data_button",
                                                     n_clicks=0,
                                                     children=["update"],
                                                     style={"background-color": "#3288BD",
                                                            "font-size": "80%",
                                                            "margin-left": "1vw",
                                                            "font-weight": "500",
                                                            "text-align": "center",
                                                            "width": "30%",
                                                            "color": "white"}
                                                     ),
                                     ],
                                         style={"margin": "0vw 0vw 0vw 1vw"}),
                                         ],
                                         style={"overflow": "auto"}
                                     ),
                                 ],
                                 ),

                         # second tab
                         dcc.Tab(label="Cluster Analysis",
                                 value="tab2",
                                 className="custom-tab",
                                 selected_className="custom-tab--selected",
                                 children=[
                                     html.Div(
                                         children=[
                                     # radio buttons used for choosing whether to perform random sampling
                                     dbc.FormGroup(children=[
                                         html.Label("Random Sampling:",
                                                    style={"margin": "1vw 0vw 0.5vw 1vw"}),
                                         dbc.RadioItems(id="cluster_random_sampling",
                                                        value="False",
                                                        options=[{"label": "True", "value": "True"},
                                                                 {"label": "False", "value": "False"}],
                                                        labelStyle={"font-size": "95%",
                                                                    "display": "inline-block",
                                                                    "margin": "0vw 0.5vw 0vw 0vw"},
                                                        style={"margin-left": "1vw"}),
                                     ]),
                                     # numeric input used for entering the size of the subsample
                                     dbc.FormGroup(children=[
                                         # TODO add sample size trigger in input
                                         html.Label("Sample Size:",
                                                    style={"margin": "1vw 0vw 0vw 1vw"}),
                                         dbc.Alert("If using random sampling, enter the size of the subsample as an integer between 1 "
                                                   "(corresponding to 1%) and 100 (corresponding to 100%).",
                                                   style={"font-size": "80%",
                                                          "margin": "1vw 0vw 0vw 1vw",
                                                          "text-align": "justify",
                                                          "width": "90%"}),
                                         dbc.Input(id="cluster_sample_size",
                                                   type="number",
                                                   min=1,
                                                   max=100,
                                                   placeholder=80,
                                                   style={"margin-left": "1vw",
                                                          "width": "90%"}),
                                     ]),
                                     # radio buttons used for selecting the dimension reduction technique
                                     dbc.FormGroup(children=[
                                         html.Label("Dimension Reduction:",
                                                    style={"margin": "1vw 0vw 0.5vw 1vw"}),
                                         dbc.RadioItems(id="cluster_dimension_reduction",
                                                        value="pca",
                                                        options=[{"label": "PCA", "value": "pca"},
                                                                 {"label": "T-SNE", "value": "tsne"},
                                                                 {"label": "None", "value": "none"}],
                                                        style={"margin-left": "1vw"},
                                                        labelStyle={"font-size": "95%",
                                                                    "display": "inline-block",
                                                                    "margin": "0vw 0.5vw 0vw 0vw"}),
                                     ]),

                                     # numeric input used for entering the number of components
                                     dbc.FormGroup([
                                         html.Label("Number of Components:",
                                                    style={"margin": "1vw 0vw 0vw 1vw"}),
                                         html.P("If using PCA or T-SNE, enter the number of components.",
                                                style={"font-size": "80%",
                                                       "margin": "0vw 2vw 0vw 1vw",
                                                       "text-align": "justify"}),
                                         dbc.Input(id="cluster_components",
                                                   type="number",
                                                   placeholder=3,
                                                   value=3,
                                                   min=1,
                                                   style={"margin-left": "1vw",
                                                          "width": "90%"}),
                                     ]),

                                     # radio buttons used for selecting the clustering algorithm
                                     dbc.FormGroup([
                                         html.Label("Clustering Algorithm:",
                                                    style={"margin": "1vw 0vw 0.5vw 1vw"}),
                                         dbc.RadioItems(id="cluster_algorithm",
                                                        value="kmeans",
                                                        options=[{"label": "K-Means", "value": "kmeans"},
                                                                 {"label": "HDBSCAN", "value": "hdbscan"}],
                                                        labelStyle={"font-size": "95%",
                                                                    "display": "inline-block",
                                                                    "margin": "0vw 0.5vw 0vw 0vw"},
                                                        style={"margin-left": "1vw"}),
                                     ]),

                                     # numeric input used for entering the number of clusters
                                     dbc.FormGroup([
                                         html.Label("Number of Clusters:",
                                                    style={"margin": "1vw 0vw 0vw 1vw"}),
                                         html.P("If using K-Means, enter the number of clusters.",
                                                style={"font-size": "80%",
                                                       "margin": "0vw 2vw 0vw 1vw",
                                                       "text-align": "justify"}),
                                         dbc.Input(id="cluster_number",
                                                   type="number",
                                                   placeholder=3,
                                                   value=3,
                                                   min=1,
                                                   style={"margin-left": "1vw",
                                                          "width": "90%"}),
                                     ]),

                                     # numeric input used for entering the minimum cluster size
                                     dbc.FormGroup([
                                         html.Label("Minimum Cluster Size:",
                                                    style={"margin": "1vw 0vw 0vw 1vw"}),
                                         html.P("If using HDBSCAN, enter the minimum cluster size.",
                                                style={"font-size": "80%",
                                                       "margin": "0vw 2vw 0vw 1vw",
                                                       "text-align": "justify"}),
                                         dbc.Input(id="cluster_size",
                                                   type="number",
                                                   placeholder=2,
                                                   min=2,
                                                   value=2,
                                                   style={"margin-left": "1vw",
                                                          "width": "90%"}
                                                   ),
                                     ]),

                                     # run button used for updating the results
                                     dbc.FormGroup([
                                         html.Label("Update Clustering Results:",
                                                    style={"margin": "1vw 0vw 0.3vw 1vw"}),
                                         dbc.Button(id="cluster_button",
                                                     n_clicks=0,
                                                     children=["update"],
                                                     style={"background-color": "#3288BD",
                                                            "font-size": "80%",
                                                            "margin-left": "1vw",
                                                            "font-weight": "500",
                                                            "text-align": "center",
                                                            "width": "30%",
                                                            "color": "white"})
                                     ]),
                                 ],
                                         style={"overflow": "auto"}
                                     )
                                 ]),

                     ]
                     ),

        ],
            style={"display": "inline-block",
                   "vertical-align": "top",
                   "width": "30vw",
                   "height": "70vw",
                   "margin": "0vw 0vw 2vw 0vw"}
        ),

        html.Div(children=[
            # initial div used for alerting the user to upload a file
            html.Div(id="alert_output", children=[

                html.Label(children=["Upload a file to start."],
                           style={"display": "block",
                                  "font-size": "120%",
                                  "color": "#BDBDBD",
                                  "margin-top": "22.5vw",
                                  "text-align": "center"}),
            ],
                     style={"display": "none"}),

            # div used for displaying the data preprocessing output
            html.Div(id="data_output", children=[

                dcc.Tabs(value="data_tab1",
                         parent_className="data-tabs",
                         className="custom-tabs-container",
                         children=[
                             # tab 1 subtab 1
                             dcc.Tab(label="Data Table",
                                     value="data_tab1",
                                     className="data-tab",
                                     selected_className="data-tab--selected",
                                     children=[
                                         html.Br(),
                                         html.Div(id="display_table"),
                                         # html.Label("Showing the first 10 rows of the data set:"),
                                     ]
                                     ),

                             # tab 1 subtab 2
                             dcc.Tab(label="Descriptive Statistics",
                                     value="data_tab2",
                                     className="data-tab",
                                     selected_className="data-tab--selected",
                                     children=[html.Br(),
                                               dt.DataTable(id="stats_data_table",
                                                            style_as_list_view=False,
                                                            style_data_conditional=[{"if": {"row_index": "odd"},
                                                                                    "background-color": "#ffffd2"},
                                                                                   {"if": {"column_id": "feature"},
                                                                                    "text-align": "left"}],
                                                            style_table={"display": "block",
                                                                         "max-height": "60vw",
                                                                         "max-width": "97%",
                                                                         "overflow-y": "scroll",
                                                                         "overflow-x": "scroll",
                                                                         "margin": "2vw 1vw 2vw 1vw"},
                                                            style_cell={"text-align": "center",
                                                                        "font-family": "Open Sans",
                                                                        "font-size": "90%", "height": "2vw"},
                                                            style_header={"background-color": "#3288BD",
                                                                          "text-align": "center",
                                                                          "color": "white",
                                                                          "text-transform": "uppercase",
                                                                          "font-family": "Open Sans",
                                                                          "font-size": "85%",
                                                                          "font-weight": "500",
                                                                          "height": "2vw"}
                                                            )
                                               ]
                                     ),

                             # tab1 subtab 3
                             dcc.Tab(label="Correlation Matrix",
                                     value="data_tab3",
                                     className="data-tab",
                                     selected_className="data-tab--selected",
                                     children=[
                                         html.Div(children=[
                                             dbc.Label("Select Features:",
                                                        style={"margin": "1vw 0vw 0.5vw 0vw"}
                                                        ),
                                             dcc.Dropdown(id="correlation_features",
                                                          style={"font-size": "95%"},
                                                          optionHeight=35,
                                                          multi=True,
                                                          searchable=True,
                                                          clearable=True,
                                                          placeholder="Select Features"),
                                         ],
                                             style={"width": "97%",
                                                    "margin": "0vw 0vw 0vw 1vw"}
                                         ),
                                         dcc.Loading(children=[
                                             html.Div(id="correlation_plot",
                                                      style={"height": "31vw",
                                                             "width": "62vw",
                                                             "margin-top": "1vw"}
                                                      ),
                                         ],
                                             type="circle",
                                             color="#3288BD",
                                             style={"height": "31vw",
                                                    "width": "62vw",
                                                    "margin-top": "1vw",
                                                    "position": "relative",
                                                    "top": "13vw"}
                                         ),
                                     ]),

                             # fourth tab
                             dcc.Tab(label="Histograms",
                                     value="data_tab4",
                                     className="data-tab",
                                     selected_className="data-tab--selected",
                                     children=[
                                         html.Div(children=[
                                             dbc.Label("Select Feature:",
                                                        style={"margin": "1vw 0vw 0.5vw 0vw"}),
                                             dcc.Dropdown(id="histogram_features",
                                                          style={"font-size": "95%"},
                                                          optionHeight=35,
                                                          multi=False,
                                                          searchable=True,
                                                          clearable=False,
                                                          placeholder="Select Feature"),
                                         ],
                                             style={"display": "inline-block",
                                                    "vertical-align": "top",
                                                    "width": "20vw",
                                                    "margin": "0vw 0vw 0vw 1vw"}
                                         ),
                                         html.Div(children=[
                                             dcc.Loading(children=[
                                                 html.Div(id="histogram_plot",
                                                          style={"height": "30vw",
                                                                 "width": "40vw",
                                                                 "margin": "1vw 0vw 0vw 2vw"}
                                                          ),
                                             ],
                                                 type="circle",
                                                 color="#3288BD",
                                                 style={"height": "30vw",
                                                        "width": "40vw",
                                                        "margin": "1vw 0vw 0vw 2vw",
                                                        "position": "relative",
                                                        "top": "12vw"}
                                             ),
                                         ],
                                             style={"display": "inline-block",
                                                    "vertical-align": "top"}
                                         ),
                                     ]
                                     ),

                             # fifth tab
                             dcc.Tab(label="Scatter Plot",
                                     value="data_tab5",
                                     className="data-tab",
                                     selected_className="data-tab--selected",
                                     children=[
                                         html.Div(children=[
                                             html.Div(children=[
                                                 # dropdown used for selecting the feature to plot on the X axis
                                                 dbc.Label("X-Axis:",
                                                           style={"margin": "1vw 0vw 0.5vw 0vw"}),
                                                 dcc.Dropdown(id="scatter-x-axis",
                                                              style={"font-size": "95%"},
                                                              optionHeight=35,
                                                              multi=False,
                                                              searchable=True,
                                                              clearable=True,
                                                              placeholder="Select Feature"
                                                              ),
                                             ],
                                                 style={"display": "inline-block",
                                                        "width": "30%",
                                                        # "margin-left": "1vw",
                                                        "margin": "auto"}
                                             ),

                                             html.Div(children=[
                                                 # dropdown used for selecting the feature to plot on the Y axis
                                                 dbc.Label("Y-Axis:",
                                                           style={"margin": "1vw 0vw 0.5vw 0vw"}
                                                            ),
                                                 dcc.Dropdown(id="scatter-y-axis",
                                                              style={"font-size": "95%"},
                                                              optionHeight=35,
                                                              multi=False,
                                                              searchable=True,
                                                              clearable=True,
                                                              placeholder="Select Feature"),
                                             ],
                                                 style={"display": "inline-block",
                                                        "width": "30%",
                                                        # "margin-left": "2vw",
                                                        "margin": "auto"}
                                             ),
                                         ],
                                             className="row",
                                             style={"margin": "auto"}
                                         ),

                                         # TODO: Add the third dropdown for group by color

                                         html.Div(id="scatter_plot",
                                                  style={"margin": "1vw 1vw 1vw 1vw"})
                                     ]
                                     ),
                         ],
                         style={}
                         ),

                # div used for displaying the data preprocessing errors
                # html.Div(id="data_alerts_modal", children=[
                #
                #     dbc.Button(id="data_alerts_button",
                #                 children=["x"],
                #                 style={"line-height": "3vw",
                #                        "height": "3vw",
                #                        "width": "2vw",
                #                        "float": "right",
                #                        "border-color": "transparent"}
                #                 ),
                #
                #     html.Label(children=["Error:"],
                #                className="row",
                #                style={"font-size": "110%",
                #                       "margin": "2vw 2vw 1vw 2vw"}
                #                ),
                #
                #     html.Div(id="data_alerts_messages",
                #              className="row",
                #              style={"font-size": "100%",
                #                     "margin": "0vw 2vw 0.5vw 2vw"}
                #              ),
                #
                # ], className="data-modal"),

            ],
                     style={"display": "none"}),

            # div used for displaying the clustering output
            html.Div(id="cluster_output", children=[

                dcc.Tabs(value="cluster_tab1",
                         parent_className="cluster-tabs",
                         className="custom-tabs-container",
                         children=[
                             # first tab
                             dcc.Tab(label="Clustered Data",
                                     value="cluster_tab1",
                                     className="data-tab",
                                     selected_className="data-tab--selected",
                                     children=[
                                         html.Br(),
                                         html.Div(id="display_cluster_table"),
                                         html.A(id="cluster_data_link",
                                                target="_blank",
                                                children=[
                                                    dbc.Button(children=["Download CSV"],
                                                                id="cluster_data_button",
                                                                style={"margin": "1vw 0vw 0vw 50vw",
                                                                       "vertical-align": "middle",
                                                                       "cursor": "pointer"}
                                                                ),
                                                ],
                                                # className="row",
                                                style={"display": "inline-block",
                                                       "width": "15%"}
                                                ),
                                     ]
                                     ),
                             # second tab
                             dcc.Tab(label="PCA Plot",
                                     value="cluster_tab2",
                                     className="data-tab",
                                     selected_className="data-tab--selected",
                                     children=[
                                         html.Div(id="scree_plot",
                                                  style={"margin": "1vw 1vw 1vw 1vw"}
                                                  ),
                                         dbc.Alert("We recommend to select the number of dimension/components to be "
                                                   "the highest cumulative explained variance ratio",
                                                   style={"margin": "1vw 1vw 1vw 1vw"}),
                                     ]),
                             # TODO
                             dcc.Tab(label="Cluster Visualization",
                                     value="cluster_tab3",
                                     className="data-tab",
                                     selected_className="data-tab--selected",
                                     children=[
                                         html.Div(id="visualization_output",
                                                  children=[
                                                      # Update part
                                                      html.Div(children=[
                                                          # radio buttons used for selecting the dimension reduction technique
                                                          html.Div(
                                                              children=[html.Label("Dimension Reduction:",
                                                                                   style={"margin": "1vw 0vw 0.5vw 1vw"}),
                                                                        dbc.RadioItems(id="plot_dimension_reduction",
                                                                                       value="pca",
                                                                                        options=[{"label": "PCA",  "value": "pca"},
                                                                                                  {"label": "T-SNE", "value": "tsne"},
                                                                                                  {"label": "None", "value": "none"}],
                                                                                        style={"margin-left": "1vw"},
                                                                                        labelStyle={"font-size": "95%",
                                                                                                     "display": "inline-block",
                                                                                                     "margin": "0vw 0.5vw 0vw 0vw"}
                                                                                       ),
                                                                        # numeric input used for entering the number of components
                                                          html.Label("Number of Components:",
                                                                     style={"margin": "1vw 0vw 0vw 1vw"}),
                                                          html.P("If using PCA or T-SNE, "
                                                                 "enter the number of components.",
                                                                 style={"font-size": "80%",
                                                                        "margin": "0vw 2vw 0vw 1vw",
                                                                        "text-align": "justify"}),
                                                          dbc.Input(id="plot_components",
                                                                    type="number",
                                                                    placeholder=3,
                                                                    value=3,
                                                                    min=1,
                                                                    style={"margin-left": "1vw",
                                                                           "width": "60%"})
                                                                        ],
                                                              style={"display": "none"}
                                                          ),

                                                          # radio buttons used for selecting the plot dimensions
                                                          html.Label("Plot Dimensions:",
                                                                     style={"margin": "1vw 0vw 0.5vw 1vw"}),
                                                          dbc.RadioItems(id="plot_dimensions",
                                                                         value="2d",
                                                                         options=[{"label": "2-D", "value": "2d"},
                                                                                  {"label": "3-D", "value": "3d"}],
                                                                         style={"margin-left": "1vw"},
                                                                         labelStyle={"font-size": "95%",
                                                                                     "display": "inline-block",
                                                                                     "margin": "0vw 0.5vw 0vw 0vw"}
                                                                         ),
                                                          # run button used for updating the plot
                                                          html.Div(children=[
                                                              html.Label("Update Plot:",
                                                                         style={"margin": "1vw 0vw 0.3vw 0vw"}),
                                                              dbc.Button(id="plot_button",
                                                                         n_clicks=0,
                                                                         children=["update"],
                                                                         style={"background-color": "#3288BD",
                                                                                "font-size": "80%",
                                                                                "margin-left": "1vw",
                                                                                "font-weight": "500",
                                                                                "text-align": "center",
                                                                                "width": "30%",
                                                                                "color": "white"}),
                                                          ],
                                                              style={"margin": "0vw 0vw 0vw 1vw"}
                                                          ),
                                                      ]),
                                                      # Display part
                                                      html.Div(children=[
                                                            # dropdown used for selecting the feature to plot on the X axis
                                                            html.Div(children=[

                                                                 dbc.Label("X-Axis:",
                                                                           style={"margin": "1vw 0vw 0.5vw 0vw"}),
                                                                 dcc.Dropdown(id="x-axis",
                                                                              style={"font-size": "95%"},
                                                                              optionHeight=35,
                                                                              multi=False,
                                                                              searchable=True,
                                                                              clearable=True,
                                                                              placeholder="Select Feature"),
                                                             ],
                                                                style={"display": "inline-block",
                                                                       "width": "30%",
                                                                       "margin-left": "1vw"}
                                                            ),
                                                            # dropdown used for selecting the feature to plot on the Y axis

                                                            html.Div(children=[
                                                                 dbc.Label("Y-Axis:",
                                                                            style={"margin": "1vw 0vw 0.5vw 0vw"}),
                                                                 dcc.Dropdown(id="y-axis",
                                                                              style={"font-size": "95%"},
                                                                              optionHeight=35,
                                                                              multi=False,
                                                                              searchable=True,
                                                                              clearable=True,
                                                                              placeholder="Select Feature"),
                                                             ],
                                                                style={"display": "inline-block",
                                                                       "width": "30%",
                                                                       "margin-left": "2vw"}
                                                            ),
                                                            # dropdown used for selecting the feature to plot on the Z axis

                                                            html.Div(children=[
                                                                 dbc.Label("Z-Axis:",
                                                                            style={"margin": "1vw 0vw 0.5vw 0vw"}),
                                                                 dcc.Dropdown(id="z-axis",
                                                                              style={"font-size": "95%"},
                                                                              optionHeight=35,
                                                                              multi=False,
                                                                              searchable=True,
                                                                              clearable=True,
                                                                              placeholder="Select Feature"),
                                                             ],
                                                                style={"display": "inline-block",
                                                                       "width": "30%",
                                                                       "margin-left": "2vw"}
                                                            ),
                                                        ],
                                                          className="row",
                                                          style={"margin": "auto"}
                                                      ),
                                                      html.Div(id="cluster_plot",
                                                               style={"margin": "1vw 1vw 1vw 1vw"}),
                                                      # TODO add switch and radioitems for 2d/3d
                                                      ]
                                                  )
                                     ]),
                         ]),
                ],
                     style={"display": "none"}),
        ],

            style={"display": "inline-block",
                   "vertical-align": "top",
                   "width": "63vw", "height": "70vw",
                   "margin": "0vw 1vw 0vw 3vw",
                   "background-color": "white"}
        )
    ],
        className="row",
        style={"display": "flex"}
    ),

    # hidden divs used for storing the data shared across callbacks
    html.Div(id="uploaded_data", style={"display": "none"}),
    html.Div(id="raw_data", style={"display": "none"}),
    html.Div(id="processed_data", style={"display": "none"}),
    html.Div(id="clustered_data", style={"display": "none"}),
    html.Div(id="plot_data", style={"display": "none"}),
])


@app.callback([Output("alert_output", "style"),
               Output("data_output", "style"),
               Output("cluster_output", "style")],
              [Input("uploaded_data", "children"),
               Input("control_tab", "value")])
def render_switch(uploaded_data, tab):

    if uploaded_data is None:

        return [{"display": "block"}, {"display": "none"}, {"display": "none"}]

    elif tab == "tab1":

        return [{"display": "none"}, {"display": "block"}, {"display": "none"}]

    elif tab == "tab2":

        return [{"display": "none"}, {"display": "none"}, {"display": "block"}]


@app.callback(Output("uploaded_data", "children"),
              [Input("uploaded_file", "contents")],
              [State("uploaded_file", "filename")])
def load_file(contents, file_name):
    if contents is not None:
        df_json = parse_contents(contents, file_name)
        return df_json


@app.callback([Output("data_features", "options"),
               # Output("data_weights", "options"),
               Output("raw_data", "children")],
              [Input("uploaded_data", "children")])
def load_data(selected_file):

    if selected_file is not None:

        # load the data from the hidden div
        df = pd.read_json(selected_file)

        # process the missing values
        df[df == "-"] = np.nan
        df[df == "?"] = np.nan
        df[df == "."] = np.nan
        df[df == " "] = np.nan

        # include the indices in the first columns
        # TODO add user customized index options
        # TODO This one may need further modification to deal with some special cases
        df.rename(columns={"Unnamed: 0": "index"}, inplace=True)

        # save the raw data in the hidden div
        raw_data = df.to_json(orient="split")

        # transform the categorical variables into dummy variables
        df = pd.get_dummies(df, dummy_na=False, drop_first=True)

        # transform the indices to integers
        df["index"] = df.index

        # extract all features
        features = list(df.columns)
        features.remove("index")

        # create the list of features to be shown in the dropdown menu
        features_options = [{"value": features[0], "label": features[0]}]

        if len(features) > 1:
            for j in range(1, len(features)):
                features_options.append({"value": features[j], "label": features[j]})

        # create the list of weights to be shown in the dropdown menu
        # weights_options = [{"value": 0, "label": 0}]

        # for j in range(1, 1000):
        #     weights_options.append({"value": j, "label": j})

        return [features_options, {"raw_data": raw_data}]

    else:

        features_options = []
        weights_options = []
        raw_data = []

        return [features_options, {"raw_data": raw_data}]


@app.callback([Output("processed_data", "children"),
               Output("correlation_features", "options"),
               Output("correlation_features", "value"),
               Output("histogram_features", "options"),
               Output("histogram_features", "value"),
               Output("scatter-x-axis", "options"),
               Output("scatter-y-axis", "options"),
               # Output("scatter-x-axis", "value"),
               # Output("scatter-y-axis", "value"),
               Output("cluster_features", "options"),
               Output("display_table", "children"),
               Output("stats_data_table", "data"),
               Output("stats_data_table", "columns"),
               Output("data_alerts_messages", "children"),
               Output("data_alerts_button", "n_clicks")],
              [Input("data_button", "n_clicks"),
               Input("raw_data", "children")],
              [State("data_features", "value"),
               State("data_weights", "value"),
               # State("data_transformation", "value")
               ])
def data_preprocessing(clicks, raw_data, selected_features, data_weights, data_transformation):
    # TODO set index options
    # Missing Numerical and Categorical Value Processing Options in new_ui version 1
    # load the raw data from the hidden div
    df = pd.read_json(raw_data["raw_data"], orient="split")

    if len(df) != 0:

        # create the table containing the raw data
        raw_data_rows = df.to_dict(orient="records")
        raw_data_columns = [{"id": x, "name": x} for x in list(df.columns)]

        # process the missing numerical values
        num = df.loc[:, np.logical_and(df.dtypes != "object", df.dtypes != "category")]

        num.dropna(inplace=True)
        df = df.iloc[num.index, :]
        df.reset_index(inplace=True, drop=True)

        # process the missing categorical values
        cat = df.loc[:, np.logical_or(df.dtypes == "object", df.dtypes == "category")]

        cat.dropna(inplace=True)
        df = df.iloc[cat.index, :]
        df = pd.get_dummies(df, dummy_na=False, drop_first=True)
        df.reset_index(inplace=True, drop=True)

        # process the features selection
        if selected_features is not None:

            features = list(set(selected_features))

            if len(features) > 0:

                columns = ["index"]
                columns.extend(features)

            else:

                columns = list(df.columns)

        else:

            columns = list(df.columns)

        # extract the selected features
        df = df[columns]

        # drop the indices
        indices = df["index"]
        df.drop("index", axis=1, inplace=True)
        columns.remove("index")

        # apply the weights
        if data_weights is not None:

            if len(data_weights) == df.shape[1]:

                data_weights = data_weights / np.sum(data_weights)
                data_alert, n_clicks = None, None

                for i in range(df.shape[1]):

                    df.iloc[:, i] = df.iloc[:, i] * data_weights[i]

            else:

                data_alert = ["The number of weights is different from the number of features. No weights applied."]
                n_clicks = 0

        else:

            data_alert, n_clicks = None, None

        # apply the selected transformation
        if data_transformation == "log":

            df = pd.DataFrame(data=FunctionTransformer(np.log1p, validate=True).fit_transform(df),
                              columns=df.columns, index=df.index)
            df = df.astype(float).round(4)

        elif data_transformation == "z-score":

            df = pd.DataFrame(data=StandardScaler().fit_transform(df), columns=df.columns, index=df.index)
            df = df.astype(float).round(4)

        elif data_transformation == "minmax":

            df = pd.DataFrame(data=MinMaxScaler().fit_transform(df), columns=df.columns, index=df.index)
            df = df.astype(float).round(4)

        # create the table containing the descriptive statistics
        stats = df.describe().transpose()
        stats = stats.astype(float).round(4)
        stats["feature"] = stats.index
        names = ["feature"]
        names.extend(list(stats.columns[stats.columns != "feature"]))
        stats = stats[names]
        stats.reset_index(drop=True, inplace=True)

        stats_data_rows = stats.to_dict(orient="records")
        stats_data_columns = [{"id": x, "name": x} for x in list(stats.columns)]

        # create the lists of features to be shown in the dropdown menus
        features_list = [{"value": columns[0], "label": columns[0]}]

        if len(columns) > 1:
            for j in range(1, len(columns)):
                features_list.append({"value": columns[j], "label": columns[j]})

        correlation_features = features_list
        histogram_features = features_list
        cluster_features = features_list
        scatter_x_axis = features_list
        scatter_y_axis = features_list
        # TODO
        # scatter_x_axis_default = feature_list[0]
        # scatter_y_axis_default = feature_list[1]

        # create the lists of features to be initially displayed in the plots
        correlation_selection = columns[:10]
        histogram_selection = columns[0]

        # add back the indices
        names = ["index"]
        names.extend(list(df.columns))
        # TODO fix index issue

        df["index"] = indices
        # the final processed df
        df = df[names]

        display_df = df.head(10)
        display_table = html.Div(dt.DataTable(columns=[{"name": i, "id": i} for i in display_df.columns],
                                              data=display_df.to_dict('records'),
                                              editable=False,
                                              style_as_list_view=False,
                                              style_data_conditional=[{"if": {"row_index": "odd"},
                                                                       "background-color": "#ffffd2"},
                                                                      {"if": {"column_id": "feature"},
                                                                       "text-align": "left"}],
                                              style_table={"display": "block",
                                                           "max-height": "60vw",
                                                           "max-width": "97%",
                                                           "overflow-y": "scroll",
                                                           "overflow-x": "scroll",
                                                           "margin": "2vw 1vw 2vw 1vw"},
                                              style_cell={"text-align": "center",
                                                          "font-family": "Open Sans",
                                                          "font-size": "90%", "height": "2vw"},
                                              style_header={"background-color": "#3288BD",
                                                            "text-align": "center",
                                                            "color": "white",
                                                            "text-transform": "uppercase",
                                                            "font-family": "Open Sans",
                                                            "font-size": "85%",
                                                            "font-weight": "500",
                                                            "height": "2vw"}
                                              ),
                                 style={"overflow": "auto"}
                                 )

        # create the table containing the processed data
        # processed_data_rows = df.to_dict(orient="records")
        # processed_data_columns = [{"id": x, "name": x} for x in list(df.columns)]

        # save the processed data in the hidden div
        processed_data = df.to_json(orient="split")

        return [{"processed_data": processed_data}, correlation_features, correlation_selection, histogram_features,
                histogram_selection, cluster_features, scatter_x_axis, scatter_y_axis, display_table, stats_data_rows,
                stats_data_columns, data_alert, n_clicks]

    else:

        processed_data = []
        correlation_features = []
        correlation_selection = []
        histogram_features = []
        histogram_selection = []
        scatter_x_axis = []
        scatter_y_axis = []
        cluster_features = []
        processed_data_rows = []
        processed_data_columns = []
        stats_data_rows = []
        stats_data_columns = []
        data_alert = None
        n_clicks = None

        return [{"processed_data": processed_data}, correlation_features, correlation_selection, histogram_features,
                histogram_selection, cluster_features, scatter_x_axis, scatter_y_axis, processed_data_rows,
                processed_data_columns, stats_data_rows, stats_data_columns, data_alert, n_clicks]


@app.callback(Output("correlation_plot", "children"),
              [Input("processed_data", "children"),
              Input("correlation_features", "value")])
def update_correlation_matrix(processed_data, correlation_features):

    df = pd.read_json(processed_data["processed_data"], orient="split")

    if len(df) != 0:

        df.drop("index", axis=1, inplace=True)

        if correlation_features is not None:

            if len(correlation_features) > 0:

                sigma = df[correlation_features].corr()

            else:

                sigma = df.iloc[:, :10].corr()

        else:

            sigma = df.iloc[:, :10].corr()

        y = list(sigma.index)
        x = list(sigma.columns)
        z = np.nan_to_num(sigma.values)

        annotations = []
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                annotations.append(dict(x=x[i], y=y[j], text=str(np.round(z[i, j], 2)), showarrow=False))

        layout = dict(annotations=annotations,
                      xaxis=dict(tickangle=45),
                      yaxis=dict(tickangle=0),
                      font=dict(family="Open Sans", size=9),
                      margin=dict(t=5, l=5, r=5, b=5, pad=0))

        traces = [go.Heatmap(z=z, x=x, y=y, zmin=-1, zmax=1, colorscale="Spectral")]

        figure = go.Figure(data=traces, layout=layout).to_dict()

        correlation_plot = dcc.Graph(figure=figure,
                                     config={"responsive": True,
                                             "autosizable": True,
                                             "showTips": True,
                                             "displaylogo": False}
                                     )

        return correlation_plot

    else:

        return []


@app.callback(Output("histogram_plot", "children"),
              [Input("processed_data", "children"),
              Input("histogram_features", "value")])
def update_histogram(processed_data, histogram_features):

    df = pd.read_json(processed_data["processed_data"], orient="split")

    if len(df) != 0:

        df.drop("index", axis=1, inplace=True)

        if histogram_features is not None:

            if len(histogram_features) > 0:

                data = df[histogram_features].values
                name = histogram_features

            else:

                data = list(df.iloc[:, 0].values)
                name = df.columns[0]

        else:

            data = list(df.iloc[:,0].values)
            name = df.columns[0]

        layout = dict(plot_bgcolor="white",
                      paper_bgcolor="white",
                      showlegend=False,
                      font=dict(family="Open Sans", size=9),
                      margin=dict(t=5, l=5, r=5, b=5, pad=0),
                      xaxis=dict(zeroline=False,
                                 showgrid=True,
                                 mirror=True,
                                 linecolor="#d9d9d9",
                                 tickangle=0),
                      yaxis=dict(zeroline=False,
                                 showgrid=True,
                                 mirror=True,
                                 linecolor="#d9d9d9",
                                 tickangle=0)
                      )

        traces = [go.Histogram(x=data,
                               name=name,
                               text=name,
                               hoverinfo="text+x+y",
                               marker=dict(color="#98c3de",
                                           line=dict(color="#3288BD", width=1))
                               )
                  ]

        figure = go.Figure(data=traces, layout=layout).to_dict()

        histogram_plot = [

            dbc.Label(children=["Histogram of " + name],
                       style={"margin": "0vw 0vw 0.5vw 0vw",
                              "text-align": "center"}),
            dcc.Graph(figure=figure,
                      config={"responsive": True,
                              "autosizable": True,
                              "showTips": True,
                              "displaylogo": False},
                      style={"height": "28vw",
                             "width": "38vw"}),

        ]

        return histogram_plot

    else:

        return []


@app.callback(Output("scatter_plot", "children"),
              [Input("processed_data", "children"),
               Input("scatter-x-axis", "value"),
               Input("scatter-y-axis", "value")])
def update_scatter_plot(processed_data, x_axis, y_axis):
    # TODO
    df = pd.read_json(processed_data["processed_data"], orient="split")

    # x_axis = df.column[0]
    # y_axis = df.column[1]

    if len(df) != 0:

        if x_axis is None or y_axis is None:
            layout = dict(paper_bgcolor="white",
                          plot_bgcolor="white",
                          showlegend=False,
                          margin=dict(t=20, b=20, r=20, l=20),
                          font=dict(family="Open Sans", size=9),
                          xaxis=dict(zeroline=False,
                                     showgrid=False,
                                     mirror=True,
                                     linecolor="#d9d9d9",
                                     tickangle=0,
                                     title_text=df.columns[0]
                                     ),
                          yaxis=dict(zeroline=False,
                                     showgrid=False,
                                     mirror=True,
                                     linecolor="#d9d9d9",
                                     tickangle=0,
                                     title_text=df.columns[1]
                                     )
                          )

            traces = [go.Scatter(x=list(df.iloc[x_axis]),
                                 y=list(df.iloc[y_axis]),
                                 mode="markers",
                                 hoverinfo="text",
                                 text=["ID: " + str(x) for x in list(df["index"])],
                                 marker=dict(color=list(df["index"]),
                                             colorscale="Spectral",
                                             size=9,
                                             line=dict(width=1)))
                      ]
            figure = go.Figure(data=traces, layout=layout).to_dict()

        else:
            layout = dict(paper_bgcolor="white",
                          plot_bgcolor="white",
                          showlegend=False,
                          margin=dict(t=20, b=20, r=20, l=20),
                          font=dict(family="Open Sans", size=9),
                          xaxis=dict(zeroline=False,
                                     showgrid=False,
                                     mirror=True,
                                     linecolor="#d9d9d9",
                                     tickangle=0,
                                     title_text=x_axis),
                          yaxis=dict(zeroline=False,
                                     showgrid=False,
                                     mirror=True,
                                     linecolor="#d9d9d9",
                                     tickangle=0,
                                     title_text=y_axis)
                          )
            traces = [go.Scatter(x=list(df[x_axis]),
                                 y=list(df[y_axis]),
                                 mode="markers",
                                 hoverinfo="text",
                                 text=["ID: " + str(x) for x in list(df["index"])],
                                 marker=dict(
                                     # color=list(df["index"]),
                                             # colorscale="Spectral",
                                             size=9,
                                             line=dict(width=1)))
                      ]

            figure = go.Figure(data=traces, layout=layout).to_dict()

        scatter_plot = dcc.Graph(figure=figure, config={"responsive": True,
                                                        "autosizable": True,
                                                        "showTips": True,
                                                        "displaylogo": False},
                                 style={"height": "30vw", "width": "60vw"}
                                 )

    else:

        scatter_plot = []

    return scatter_plot


@app.callback(Output("scree_plot", "children"),
              [Input("processed_data", "children")])
def update_scree_plot(processed_data):

    df = pd.read_json(processed_data["processed_data"], orient="split")

    if len(df) != 0:

        df.drop("index", axis=1, inplace=True)

        pca = PCA(n_components=np.min([10, df.shape[1]]), random_state=0).fit(df)
        # placeholder for calculating cumulative explained variance
        cum_explained_var = []
        for i in range(0, len(pca.explained_variance_ratio_)):
            if i == 0:
                cum_explained_var.append(pca.explained_variance_ratio_[i])
            else:
                cum_explained_var.append(pca.explained_variance_ratio_[i] +
                                         cum_explained_var[i - 1])

        y = list(cum_explained_var)
        x = [z + 1 for z in range(np.min([10, df.shape[1]]))]

        layout = dict(plot_bgcolor="white",
                      paper_bgcolor="white",
                      showlegend=False,
                      font=dict(family="Open Sans", size=9),
                      margin=dict(t=20, l=20, r=20, b=20),
                      xaxis=dict(zeroline=False,
                                 showgrid=False,
                                 mirror=True,
                                 linecolor="#d9d9d9",
                                 tickangle=0,
                                 tickmode="array",
                                 tickvals=x,
                                 title_text="Number of Components"),
                      yaxis=dict(range=[-0.05, 1.05],
                                 zeroline=False,
                                 showgrid=False,
                                 mirror=True,
                                 linecolor="#d9d9d9",
                                 tickangle=0,
                                 tickmode="array",
                                 tickvals=list(np.linspace(0, 1, 11)),
                                 title_text="Cumulative Explained Variance Ratio"))

        traces = [go.Scatter(x=x,
                             y=y,
                             mode="lines+markers",
                             hoverinfo="x+y",
                             marker=dict(size=10,
                                         color="#cbe1ee",
                                         line=dict(color="#3288BD",
                                                   width=1)),
                             line=dict(color="#3288BD", width=2))
                  ]

        figure = go.Figure(data=traces, layout=layout).to_dict()

        scree_plot = dcc.Graph(figure=figure,
                               config={"responsive": True,
                                       "autosizable": True,
                                       "showTips": True,
                                       "displaylogo": False},
                               style={"height": "30vw", "width": "60vw"})

        return scree_plot

    else:

        return []


@app.callback([Output("display_cluster_table", "children"),
               Output("clustered_data", "children")],
              [Input("cluster_button", "n_clicks"),
               Input("processed_data", "children")],
              [State("cluster_random_sampling", "value"),
               State("cluster_sample_size", "value"),
               State("cluster_dimension_reduction", "value"),
               State("cluster_components", "value"),
               State("cluster_algorithm", "value"),
               State("cluster_number", "value"),
               State("cluster_size", "value")])
def cluster_analysis(clicks, processed_data, random_sampling, sample_size, dimension_reduction, num_components,
                     cluster_algorithm, num_clusters, cluster_size):

    # load the processed data from the hidden div
    df = pd.read_json(processed_data["processed_data"], orient="split")

    if len(df) != 0:

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

                sample_message = "Sampled " + str(df.shape[0]) + " records."

            else:

                sample_message = "Sample size not selected. Random sampling not performed."

        else:

            sample_message = "Random sampling not performed."

        # drop the indices
        indices = df["index"]
        df.drop("index", axis=1, inplace=True)

        # run the dimension reduction algorithm
        if dimension_reduction == "pca":

            if 0 < num_components <= df.shape[1]:

                df = PCA(n_components=np.int(num_components), random_state=0).fit_transform(df)
                df = pd.DataFrame(data=df, columns=["Component " + str(x) for x in range(1, num_components + 1)])

                reduction_message = "Extracted " + str(num_components) + " components with PCA."

            else:

                reduction_message = "Invalid number of components. PCA not performed."

        elif dimension_reduction == "tsne":

            if 0 < num_components <= df.shape[1]:

                df = TSNE(n_components=np.int(num_components), random_state=0).fit_transform(df)
                df = pd.DataFrame(data=df, columns=["Component" + str(x) for x in range(1, num_components + 1)])

                reduction_message = "Extracted " + str(num_components) + " components with TSNE."

            else:

                reduction_message = "Invalid number of components. TSNE not performed."

        else:

            reduction_message = "Dimension reduction not performed."

        # run the clustering algorithm
        if cluster_algorithm == "kmeans":

            if 0 < num_clusters <= df.shape[0]:

                algo = KMeans(n_clusters=num_clusters, random_state=0).fit(df)

            else:

                algo = KMeans(n_clusters=3).fit(df)

            algo_message = "Performed K-means clustering."

        elif cluster_algorithm == "hdbscan":

            if 1 < cluster_size <= df.shape[0]:

                algo = hdbscan.HDBSCAN(min_cluster_size=cluster_size).fit(df)

            else:

                algo = hdbscan.HDBSCAN(min_cluster_size=2).fit(df)

            algo_message = "Performed HDBSCAN clustering."

        # round all values to 2 digits
        df = df.astype(float).round(2)

        # add the cluster labels
        df["cluster labels"] = algo.labels_
        df["cluster labels"] = 1 + df["cluster labels"]

        cluster_message = "Found " + str(len(df["cluster labels"].unique())) + " clusters."
        n_clicks = 0

        # add back the indices
        df["index"] = indices
        df = df[["index", "cluster labels"]]
        df = pd.merge(left=df, right=df_copy, on="index", how="left")

        # save the results in the hidden div
        cluster_data = df.to_json(orient="split")

        # display the results in the table
        cluster_data_rows = df.to_dict("records")
        cluster_data_columns = [{"id": x, "name": x} for x in list(df.columns)]

        display_cluster_data = df.head(10)
        display_cluster_table = html.Div(
            dt.DataTable(columns=[{"name": i, "id": i} for i in display_cluster_data.columns],
                         data=display_cluster_data.to_dict('records'),
                         editable=False,
                         style_as_list_view=False,
                         style_data_conditional=[{"if": {"row_index": "odd"},
                                                  "background-color": "#ffffd2"},
                                                 {"if": {"column_id": "feature"},
                                                  "text-align": "left"}],
                         style_table={"display": "block",
                                      "max-height": "60vw",
                                      "max-width": "97%",
                                      "overflow-y": "scroll",
                                      "overflow-x": "scroll",
                                      "margin": "2vw 1vw 2vw 1vw"},
                         style_cell={"text-align": "center",
                                     "font-family": "Open Sans",
                                     "font-size": "90%", "height": "2vw"},
                         style_header={"background-color": "#3288BD",
                                       "text-align": "center",
                                       "color": "white",
                                       "text-transform": "uppercase",
                                       "font-family": "Open Sans",
                                       "font-size": "85%",
                                       "font-weight": "500",
                                       "height": "2vw"}
                         ),
            style={"overflow": "auto"}
        )

        return [display_cluster_table, {"clustered_data": cluster_data}]

    else:

        display_cluster_table = []
        cluster_data = []
        sample_message = []
        reduction_message = []
        algo_message = []
        cluster_message = []
        n_clicks = None

        return [display_cluster_table, {"clustered_data": cluster_data}]


@app.callback([Output("x-axis", "options"),
               Output("y-axis", "options"),
               Output("z-axis", "options"),
               Output("plot_data", "children")],
              [Input("clustered_data", "children"),
               Input("plot_button", "n_clicks")],
              [State("plot_dimension_reduction", "value"),
               State("plot_components", "value")])
def update_cluster_plot_data(clustered_data, plot_button, plot_dimension_reduction, plot_components):

    # load the clustering results from the hidden div
    df = pd.read_json(clustered_data["clustered_data"], orient="split")

    if len(df) != 0:

        # drop the indices and the cluster labels
        indices = df["index"]
        labels = df["cluster labels"]
        df.drop(["index", "cluster labels"], axis=1, inplace=True)

        # run the dimension reduction algorithm
        if plot_dimension_reduction == "pca":

            if 0 < plot_components <= df.shape[1]:

                df = PCA(n_components=np.int(plot_components), random_state=0).fit_transform(df)
                df = pd.DataFrame(data=df, columns=["Component " + str(x) for x in range(1, plot_components + 1)])

        elif plot_dimension_reduction == "tsne":

            if 0 < plot_components <= df.shape[1]:

                df = TSNE(n_components=np.int(plot_components), random_state=0).fit_transform(df)
                df = pd.DataFrame(data=df, columns=["Component" + str(x) for x in range(1, plot_components + 1)])

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
        plot_data = df.to_json(orient="split")

        return [x_axis_options, y_axis_options, z_axis_options, {"plot_data": plot_data}]

    else:

        x_axis_options = []
        y_axis_options = []
        z_axis_options = []
        plot_data = []

        return [x_axis_options, y_axis_options, z_axis_options, {"plot_data": plot_data}]


@app.callback(Output("cluster_plot", "children"),
              [Input("plot_data", "children"),
               Input("x-axis", "value"),
               Input("y-axis", "value"),
               Input("z-axis", "value")],
              [State("plot_dimensions", "value")])
def update_cluster_plot(plot_data, x_axis, y_axis, z_axis, plot_dimensions):

    df = pd.read_json(plot_data["plot_data"], orient="split")

    if len(df) != 0:

        if plot_dimensions == "2d":

            if x_axis is None or y_axis is None:

                layout = dict(paper_bgcolor="white",
                              plot_bgcolor="white",
                              showlegend=False,
                              margin=dict(t=20, b=20, r=20, l=20),
                              font=dict(family="Open Sans", size=9),
                              xaxis=dict(zeroline=False,
                                         showgrid=False,
                                         mirror=True,
                                         linecolor="#d9d9d9",
                                         tickangle=0,
                                         title_text=df.columns[0]),
                              yaxis=dict(zeroline=False,
                                         showgrid=False,
                                         mirror=True,
                                         linecolor="#d9d9d9",
                                         tickangle=0,
                                         title_text=df.columns[1])
                              )

                traces = [go.Scatter(x=list(df.iloc[:, 0]), y=list(df.iloc[:, 1]), mode="markers", hoverinfo="text",
                                     text=["Cluster " + str(x) + " (Index " + str(y) + ")" for x, y in
                                           zip(list(df["cluster labels"]), list(df["index"]))],
                                     marker=dict(color=list(df["cluster labels"]), colorscale="Spectral", size=9,
                                                 line=dict(width=1)))]

                figure = go.Figure(data=traces, layout=layout).to_dict()

            else:

                layout = dict(paper_bgcolor="white",
                              plot_bgcolor="white",
                              showlegend=False,
                              margin=dict(t=20, b=20, r=20, l=20),
                              font=dict(family="Open Sans", size=9),
                              xaxis=dict(zeroline=False,
                                         showgrid=False,
                                         mirror=True,
                                         linecolor="#d9d9d9",
                                         tickangle=0,
                                         title_text=x_axis),
                              yaxis=dict(zeroline=False,
                                         showgrid=False,
                                         mirror=True,
                                         linecolor="#d9d9d9",
                                         tickangle=0,
                                         title_text=y_axis)
                              )

                traces = [go.Scatter(x=list(df[x_axis]), y=list(df[y_axis]), mode="markers", hoverinfo="text",
                                     text=["Cluster " + str(x) + " (Index " + str(y) + ")" for x, y in
                                           zip(list(df["cluster labels"]), list(df["index"]))],
                                     marker=dict(color=list(df["cluster labels"]), colorscale="Spectral", size=9,
                                                 line=dict(width=1)))]

                figure = go.Figure(data=traces, layout=layout).to_dict()

        elif plot_dimensions == "3d":

            if x_axis is None or y_axis is None or z_axis is None:

                layout = dict(paper_bgcolor="white",
                              plot_bgcolor="white",
                              margin=dict(t=0, l=0, r=0, b=0, pad=0),
                              font=dict(family="Open Sans", size=9),
                              scene=dict(xaxis=dict(zeroline=False,
                                                    showgrid=True,
                                                    mirror=True,
                                                    linecolor="#d9d9d9",
                                                    tickangle=0,
                                                    title_text=df.columns[0]),
                                         yaxis=dict(zeroline=False,
                                                    showgrid=True,
                                                    mirror=True,
                                                    linecolor="#d9d9d9",
                                                    tickangle=0,
                                                    title_text=df.columns[1]),
                                         zaxis=dict(zeroline=False,
                                                    showgrid=True,
                                                    mirror=True,
                                                    linecolor="#d9d9d9",
                                                    tickangle=0,
                                                    title_text=df.columns[2]))
                              )

                traces = [
                    go.Scatter3d(x=list(df.iloc[:, 0]), y=list(df.iloc[:, 1]), z=list(df.iloc[:, 2]), mode="markers",
                                 hoverinfo="text",
                                 text=["Cluster " + str(x) + " (Index " + str(y) + ")" for x, y in
                                       zip(list(df["cluster labels"]), list(df["index"]))],
                                 marker=dict(color=list(df["cluster labels"]), colorscale="Spectral", size=7,
                                             line=dict(width=2)))]

                figure = go.Figure(data=traces, layout=layout).to_dict()

            else:

                layout = dict(paper_bgcolor="white",
                              plot_bgcolor="white",
                              margin=dict(t=0, l=0, r=0, b=0, pad=0),
                              font=dict(family="Open Sans", size=9),
                              scene=dict(xaxis=dict(zeroline=False,
                                                    showgrid=True,
                                                    mirror=True,
                                                    linecolor="#d9d9d9",
                                                    tickangle=0,
                                                    title_text=x_axis),
                                         yaxis=dict(zeroline=False,
                                                    showgrid=True,
                                                    mirror=True,
                                                    linecolor="#d9d9d9",
                                                    tickangle=0,
                                                    title_text=y_axis),
                                         zaxis=dict(zeroline=False,
                                                    showgrid=True,
                                                    mirror=True,
                                                    linecolor="#d9d9d9",
                                                    tickangle=0,
                                                    title_text=z_axis))
                              )

                traces = [go.Scatter3d(x=list(df[x_axis]), y=list(df[y_axis]), z=list(df[z_axis]), mode="markers",
                                       hoverinfo="text",
                                       text=["Cluster " + str(x) + " (Index " + str(y) + ")" for x, y in
                                             zip(list(df["cluster labels"]), list(df["index"]))],
                                       marker=dict(color=list(df["cluster labels"]), colorscale="Spectral", size=7,
                                                   line=dict(width=2)))]

                figure = go.Figure(data=traces, layout=layout).to_dict()

        scatter_plot = dcc.Graph(figure=figure,
                                 config={"responsive": True,
                                         "autosizable": True,
                                         "showTips": True,
                                         "displaylogo": False},
                                 style={"height": "30vw",
                                        "width": "60vw"}
                                 )

    else:

        scatter_plot = []

    return scatter_plot


@app.callback(Output("data_alerts_modal", "style"), [Input("data_alerts_messages", "children"),
              Input("data_alerts_button", "n_clicks")])
def data_messages(messages, n_clicks):

    if messages is not None and n_clicks == 0:

        return {"display": "block"}

    elif messages is not None and n_clicks != 0:

        return {"display": "none"}

    else:

        return {"display": "none"}


@app.callback(Output("cluster_alerts_modal", "style"), [Input("cluster_alerts_button", "n_clicks")])
def cluster_messages(n_clicks):

    if n_clicks == 0:

        return {"display": "block"}

    else:

        return {"display": "none"}


@app.callback([Output("cluster_data_link", "href"),
               Output("cluster_data_link", "download")],
              [Input("cluster_data_button", "n_clicks"),
               Input("clustered_data", "children")])
def download_file(n_clicks, clustered_data):

    # load the clustering results from the hidden div
    df = pd.read_json(clustered_data["clustered_data"], orient="split")

    # convert the table to csv
    csv = df.to_csv(index=False, encoding="utf-8")

    # create the file for download
    file_for_download = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv)

    # use the current time as the file name
    file_name = "clustering_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"

    return file_for_download, file_name


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


if __name__ == "__main__":
    app.run_server(port=8080, debug=False)