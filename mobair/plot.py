import osmnx as ox
import altair as alt


def streetDraw(place, G, attribute_to_plot, normalize_by_length=False, show_legend=True, schemeColor='yelloworangered',
               size=(600, 600), save_fig=False):
    """ Plots a road network with the roads colored with respect to an attribute.

    Parameters
    ----------
    place : str
        the name of the place that is being plotted (only used for saving).

    G : networkx.MultiDiGraph
        the road network to be plotted.

    attribute_to_plot : str
        the name of the feature to be plotted. It must be present as an attribute of the graph's edges.

    normalize_by_length : boolean
        whether to normalize the value of the attribute with the edge's length.

    show_legend : boolean
        whether to show the legend (colorbar) or not.

    schemeColor : str
        the name of a Vega's color scheme (https://vega.github.io/vega/docs/schemes/)

    size : tuple
        the size of the chart as (widthCanvas, heightCanvas).

    save_fig
        whether to save (in .svg and .png) the output chart or not.

    Returns
        the Altair chart.
    -------

    Credits to Daniele Fadda.
    """

    widthCanvas = size[0]
    heightCanvas = size[1]

    nodes, edges = ox.graph_to_gdfs(G)
    df = edges.reset_index()
    df.sort_values(by=attribute_to_plot, na_position='first', ascending=True, inplace=True)

    if normalize_by_length:
        attribute_to_plot_old = attribute_to_plot
        attribute_to_plot = attribute_to_plot + ' normalized'
        df[attribute_to_plot] = df[attribute_to_plot_old] / df['length']
        plot_label = '%s (grams per meter of road)' % attribute_to_plot_old
    else:
        plot_label = '%s' % attribute_to_plot
    #
    first_nonzero = [x for x in df[attribute_to_plot].dropna().sort_values() if x != 0][0]
    #
    if show_legend:
        legend = alt.Legend(title=plot_label)
    else:
        legend = None
    #

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
            legend=legend,
            scale=alt.Scale(scheme=schemeColor, type='symlog', constant=first_nonzero)
        )
    ).properties(
        width=widthCanvas,
        height=heightCanvas
    )

    chart = lines + speedLines

    if save_fig:
        from altair_saver import save
        city = place.split(',')[0]
        attribute = attribute_to_plot.replace(' ', '_')
        save(chart, f'{city}_map_{attribute}.png')
        save(chart, f'{city}_map_{attribute}.svg')
        print(f'{city} images saved')

    return chart

