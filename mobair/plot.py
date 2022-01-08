import osmnx as ox
import altair as alt
from altair_saver import save


def streetDraw(place, G, attribute_to_plot, schemeColor='yelloworangered', save_fig=False):
    """ Plots a road network with the roads colored with respect to an attribute.

    Parameters
    ----------
    place : str
        the name of the place that is being plotted (only used for saving).

    G : networkx.MultiDiGraph
        the road network to be plotted.

    attribute_to_plot : str
        the name of the feature to be plotted. It must be present as an attribute of the graph's edges.

    schemeColor : str
        the name of a Vega's color scheme (https://vega.github.io/vega/docs/schemes/)

    save_fig
        whether to save (in .svg and .png) the resulting figure or not.
    Returns
        the Altair chart.
    -------

    Credits to Daniele Fadda.
    """

    widthCanvas = 600
    heightCanvas = 600

    nodes, edges = ox.graph_to_gdfs(G)
    df = edges.reset_index()
    df.sort_values(by=attribute_to_plot, na_position='first', ascending=True, inplace=True)

    lines = alt.Chart(df[df[attribute_to_plot].isnull()]).mark_geoshape(
        filled=False,
        strokeWidth=1,
        color='lightgray'
    )

    speedLines = alt.Chart(df[df[attribute_to_plot].isnull() == False]).mark_geoshape(
        filled=False,
        strokeWidth=1.5
    ).encode(
        alt.Color(
            attribute_to_plot + ':Q',
            # legend=None,
            scale=alt.Scale(scheme=schemeColor, type='symlog')
        )
    ).properties(
        width=widthCanvas,
        height=heightCanvas
    )
    # water = waterDraw(place)

    chart = lines + speedLines  # + water

    if save_fig:
        city = place.split(',')[0]
        chart.save(f'{city}_map_{attribute_to_plot}.png')
        chart.save(f'{city}_map_{attribute_to_plot}.svg')
        print(f'{city} images saved')

    return chart

