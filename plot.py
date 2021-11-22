#!/usr/bin/env python

"""Plot the Coding Game trajectories."""

import re
from typing import Dict, Union

from plotly.subplots import make_subplots

from code_busters import PATH, HOME

COLOR_SHORT = dict(
    r="red",
    g="green",
    b="blue",
    y="yellow",
    v="violet",
    o="orange",
    c="cyan",
    m="magenta",
    k="black",
    w="white",
)
COLOR_REPLACEMENT = dict(
    darkyellow="gold",
    darkblack="black",
    darkwhite="lightslategrey",
    lightred="indianred",
    lightviolet="palevioletred",
    lightorange="orangered",
    lightmagenta="lightcoral",
    lightblack="darkslategrey",
    lightwhite="white",
)

def look(
    desc: str = "", *, y2: bool = False
) -> Dict[str, Union[str, Dict[str, Union[str, int]]]]:
    f"""
    Helper function to format a plotly scatter plot.

    Parameters
    ----------
    desc: str, optional
        A simplified description of the plot. Can contain a line shape (default
        to marker only), a color (line and/or marker), and/or a width/size.
        Line: _ or . or - or -- or -. or --. (solid, dotted, dash, longdash...)
        Color: all the plotly colors
               + {', '.join([f'{k} ({v})' for k, v in COLOR_SHORT.items()])}
                 with optional prefix l (light) or d (dark)
        Width (for lines) / Size (for markers):
                0 to 6

    y2: int, optional, keyword-only
        If True, set the option 'secondary_y' to True

    Examples
    --------
    >>> look()
    {{'mode': 'markers'}}
    >>> look("-dr1")
    {{'mode': 'lines', 'line': {{'dash': 'dash', 'color': 'darkred', 'width': 1}} }}
    >>> look(y2=True)
    {{'mode': 'markers', 'secondary_y': True}}
    >>> look("--.lb")
    {{'mode': 'lines', 'line': {{'dash': 'longdashdot', 'color': 'lightblue'}} }}
    >>> look("white3")
    {{'mode': 'markers', 'marker': {{'color': 'white', 'size': 3}} }}

    Returns
    -------
    Dict[str, Union[str, Dict[str, Union[str, int]]]]
    """
    kwargs = dict(mode="markers")

    if y2 is True:
        kwargs['secondary_y'] = True

    if not desc:
        return kwargs

    # noinspection RegExpAnonymousGroup
    reg_desc = re.match(
        r'^(?P<dash>_|(--\.)|(--)|(-\.)|\.|-)?(?P<color>[a-z]+)?(?P<width>\d)?$', desc
    )
    if not reg_desc:
        print(f"Can not match look description: {desc}")
        return kwargs

    if reg_desc['dash']:
        kwargs['mode'] = "lines"
        dash = ""
        if '_' in reg_desc['dash']:
            dash += "solid"
        elif '--' in reg_desc['dash']:
            dash += "longdash"
        elif '-' in reg_desc['dash']:
            dash += "dash"
        if '.' in reg_desc['dash']:
            dash += "dot"
        kwargs['line'] = dict(dash=dash)

    if reg_desc['color']:
        cols = "".join(COLOR_SHORT.keys())
        reg_col = re.match(fr'^(?P<prefix>[ld])?(?P<col>[{cols}])$', reg_desc['color'])
        if reg_col:
            color = ""
            if reg_col['prefix']:
                color = "light" if 'l' in reg_col['prefix'] else "dark"
            color += COLOR_SHORT[reg_col['col']]
        else:
            color = reg_desc['color']
        color = COLOR_REPLACEMENT.get(color, color)
        if kwargs['mode'] == "lines":
            kwargs.setdefault('line', {})['color'] = color
        else:
            kwargs.setdefault('marker', {})['color'] = color

    if reg_desc['width']:
        if kwargs['mode'] == "lines":
            # noinspection PyTypeChecker
            kwargs.setdefault('line', {})['width'] = int(reg_desc['width'])
        else:
            # noinspection PyTypeChecker
            kwargs.setdefault('marker', {})['size'] = int(reg_desc['width'])

    return kwargs

if __name__ == '__main__':
    fig = make_subplots()
    for num, point_list in PATH.items():
        x = [HOME.x]
        y = [HOME.y]
        for p in point_list:
            x.append(p.x)
            y.append(p.y)
        fig.add_scatter(x=x, y=y, name=num)
    fig.show()
