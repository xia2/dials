from __future__ import annotations

import copy
from collections import OrderedDict

from scipy.cluster import hierarchy

from dials.algorithms.clustering.plots import scipy_dendrogram_to_plotly_json


def linkage_matrix_to_dict(linkage_matrix):
    """
    Convert a linkage matrix to a dictionary

    Args:
      linkage_matrix (numpy.ndarray): Linkage matrix describing the
        dendrogram-style linkage from a distance matrix
    """

    tree = hierarchy.to_tree(linkage_matrix, rd=False)

    d = {}

    # http://w3facility.org/question/scipy-dendrogram-to-json-for-d3-js-tree-visualisation/
    # https://gist.github.com/mdml/7537455

    def add_node(node):
        if node.is_leaf():
            return
        cluster_id = node.get_id() - len(linkage_matrix) - 1
        row = linkage_matrix[cluster_id]
        d[cluster_id + 1] = {
            "datasets": [i + 1 for i in sorted(node.pre_order())],
            "height": row[2],
        }

        # Recursively add the current node's children
        if node.left:
            add_node(node.left)
        if node.right:
            add_node(node.right)

    add_node(tree)

    return OrderedDict(sorted(d.items()))


def to_plotly_json(
    correlation_matrix, linkage_matrix, labels=None, matrix_type="correlation"
):

    """
    Prepares a plotly-style plot of the heatmap corresponding to the input matrix
    with dendrograms on the top and left sides.
    Args:
      correlation_matrix (numpy.ndarray): correlation matrix relating datasets
      linkage_matrix (numpy.ndarray): Linkage matrix describing the
        dendrogram-style linkage from a distance matrix
      labels (list): list of dataset labels
      matrix_type (str): either "correlation" or "cos_angle"
    """

    assert matrix_type in ("correlation", "cos_angle")

    ddict = hierarchy.dendrogram(
        linkage_matrix,
        color_threshold=0.05,
        labels=labels,
        show_leaf_counts=False,
        no_plot=True,
    )

    y2_dict = scipy_dendrogram_to_plotly_json(
        ddict, "Dendrogram", xtitle="Individual datasets", ytitle="Ward distance"
    )  # above heatmap
    x2_dict = copy.deepcopy(y2_dict)  # left of heatmap, rotated
    for d in y2_dict["data"]:
        d["yaxis"] = "y2"
        d["xaxis"] = "x2"

    for d in x2_dict["data"]:
        x = d["x"]
        y = d["y"]
        d["x"] = y
        d["y"] = x
        d["yaxis"] = "y3"
        d["xaxis"] = "x3"

    D = correlation_matrix
    index = ddict["leaves"]
    D = D[index, :]
    D = D[:, index]
    ccdict = {
        "data": [
            {
                "name": "%s_matrix" % matrix_type,
                "x": list(range(D.shape[0])),
                "y": list(range(D.shape[1])),
                "z": D.tolist(),
                "type": "heatmap",
                "colorbar": {
                    "title": (
                        "Correlation coefficient"
                        if matrix_type == "correlation"
                        else "cos(angle)"
                    ),
                    "titleside": "right",
                    "xpad": 0,
                },
                "colorscale": "YIOrRd",
                "xaxis": "x",
                "yaxis": "y",
            }
        ],
        "layout": {
            "autosize": False,
            "bargap": 0,
            "height": 1000,
            "hovermode": "closest",
            "margin": {"r": 20, "t": 50, "autoexpand": True, "l": 20},
            "showlegend": False,
            "title": "Dendrogram Heatmap",
            "width": 1000,
            "xaxis": {
                "domain": [0.2, 0.9],
                "mirror": "allticks",
                "showgrid": False,
                "showline": False,
                "showticklabels": True,
                "tickmode": "array",
                "ticks": "",
                "ticktext": y2_dict["layout"]["xaxis"]["ticktext"],
                "tickvals": list(range(len(y2_dict["layout"]["xaxis"]["ticktext"]))),
                "tickangle": 300,
                "title": "",
                "type": "linear",
                "zeroline": False,
            },
            "yaxis": {
                "domain": [0, 0.78],
                "anchor": "x",
                "mirror": "allticks",
                "showgrid": False,
                "showline": False,
                "showticklabels": True,
                "tickmode": "array",
                "ticks": "",
                "ticktext": y2_dict["layout"]["xaxis"]["ticktext"],
                "tickvals": list(range(len(y2_dict["layout"]["xaxis"]["ticktext"]))),
                "title": "",
                "type": "linear",
                "zeroline": False,
            },
            "xaxis2": {
                "domain": [0.2, 0.9],
                "anchor": "y2",
                "showgrid": False,
                "showline": False,
                "showticklabels": False,
                "zeroline": False,
            },
            "yaxis2": {
                "domain": [0.8, 1],
                "anchor": "x2",
                "showgrid": False,
                "showline": False,
                "zeroline": False,
            },
            "xaxis3": {
                "domain": [0.0, 0.1],
                "anchor": "y3",
                "range": [max(max(d["x"]) for d in x2_dict["data"]), 0],
                "showgrid": False,
                "showline": False,
                "tickangle": 300,
                "zeroline": False,
            },
            "yaxis3": {
                "domain": [0, 0.78],
                "anchor": "x3",
                "showgrid": False,
                "showline": False,
                "showticklabels": False,
                "zeroline": False,
            },
        },
    }
    d = ccdict
    d["data"].extend(y2_dict["data"])
    d["data"].extend(x2_dict["data"])

    d["clusters"] = linkage_matrix_to_dict(linkage_matrix)

    return d
