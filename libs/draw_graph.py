# This script are inspired by the blog:
# http://matthiaseisen.com/articles/graphviz/
import graphviz as gv
import functools
graph = functools.partial(gv.Graph, format='svg')
digraph = functools.partial(gv.Digraph, format='svg')


def add_nodes(graph, nodes):
    for n in nodes:
        if isinstance(n, tuple):
            graph.node(n[0], **n[1])
        else:
            graph.node(n)
    return graph


def add_edges(graph, edges):
    for e in edges:
        if isinstance(e[0], tuple):
            graph.edge(*e[0], **e[1])
        else:
            graph.edge(*e)
    return graph


# Find the colors here: https://coolors.co/c5d5ea-dc602e-759eb8-b2c9ab-7392b7
styles1 = {
    'graph': {
        'label': 'Convolution Module',
        'fontsize': '16',
        'fontcolor': 'black',
        'bgcolor': 'white',
        'rankdir': 'BT',
        'splines': 'curved',
        'ratio': '1'
    },
    'nodes': {
        'fontname': 'Helvetica',
        'shape': 'ellipse',
        'fontcolor': 'white',
        'color': 'white',
        'style': 'filled',
        'fillcolor': '#F37748',
    },
    'edges': {
        'style': 'dashed',
        'color': '#381D2A',
        'arrowhead': 'normal',
        'fontname': 'Courier',
        'fontsize': '12',
        'fontcolor': 'black',
    }
}


def apply_styles(graph, styles):
    graph.graph_attr.update(('graph' in styles and styles['graph']) or {})
    graph.node_attr.update(('nodes' in styles and styles['nodes']) or {})
    graph.edge_attr.update(('edges' in styles and styles['edges']) or {})
    return graph


g1 = add_edges(
    add_nodes(digraph(), [('Input', {
        'label': 'Input',
        'fillcolor': '#30323D',
        'shape': 'circle',
        'width': '1'
    }), ('By-pass', {
        'label': 'By-pass'
    }), ('Squeeze', {
        'label': 'Squeeze'
    }), ('Expand 1', {
        'label': 'Expand 1'
    }), ('Expand 2', {
        'label': 'Expand 2',
        'fillcolor': '#16DB93'
    }),
                          ('Concat', {
                              'label': 'Concat',
                              'fillcolor': '#662E9B',
                              'fontcolor': 'white',
                              'shape': 'rect',
                              'width': '1'
                          }),
                          ('Output', {
                              'label': 'Output',
                              'fillcolor': '#067BC2',
                              'fontcolor': 'white',
                              'shape': 'circle',
                              'width': '1'
                          })]), [(('Input', 'Squeeze'), {
                              'label': 'Edge 1'
                          }), (('Squeeze', 'Expand 1'), {
                              'label': 'Edge 2'
                          }), (('Squeeze', 'Expand 2'), {
                              'label': 'Edge 3'
                          }), (('Expand 1', 'Concat'), {
                              'label': 'Edge 4'
                          }), (('Expand 2', 'Concat'), {
                              'label': 'Edge 5'
                          }), (('Concat', 'Output'), {
                              'label': 'Edge 6'
                          }), (('Input', 'By-pass'), {
                              'label': 'Edge 7'
                          }), (('By-pass', 'Output'), {
                              'label': 'Edge 8'
                          })])
g1 = apply_styles(g1, styles1)
g1.subgraph
g1.render('img/g1')

# Find the colors here: https://coolors.co/c5d5ea-dc602e-759eb8-b2c9ab-7392b7
styles2 = {
    'graph': {
        'label': 'Deonvolution Module',
        'fontsize': '16',
        'fontcolor': 'black',
        'bgcolor': 'white',
        'rankdir': 'BT',
        'splines': 'curved',
        'ratio': '1'
    },
    'nodes': {
        'fontname': 'Helvetica',
        'shape': 'ellipse',
        'fontcolor': 'white',
        'color': 'white',
        'style': 'filled',
        'fillcolor': '#F37748',
    },
    'edges': {
        'style': 'dashed',
        'color': '#381D2A',
        'arrowhead': 'normal',
        'fontname': 'Courier',
        'fontsize': '12',
        'fontcolor': 'black',
    }
}


def apply_styles(graph, styles):
    graph.graph_attr.update(('graph' in styles and styles['graph']) or {})
    graph.node_attr.update(('nodes' in styles and styles['nodes']) or {})
    graph.edge_attr.update(('edges' in styles and styles['edges']) or {})
    return graph


g2 = add_edges(
    add_nodes(digraph(), [('Input', {
        'label': 'Input',
        'fillcolor': '#30323D',
        'shape': 'circle',
        'width': '1'
    }), ('Squeeze', {
        'label': 'Squeeze'
    }), ('Expand 1', {
        'label': 'Expand 1'
    }), ('Expand 2', {
        'label': 'Expand 2',
        'fillcolor': '#16DB93'
    }),
                          ('Concat', {
                              'label': 'Concat',
                              'fillcolor': '#662E9B',
                              'fontcolor': 'white',
                              'shape': 'rect',
                              'width': '1'
                          }),
                          ('Output', {
                              'label': 'Output',
                              'fillcolor': '#067BC2',
                              'fontcolor': 'white',
                              'shape': 'circle',
                              'width': '1'
                          })]), [(('Input', 'Expand 1'), {
                              'label': 'Edge 1'
                          }), (('Input', 'Expand 2'), {
                              'label': 'Edge 2'
                          }), (('Expand 1', 'Concat'), {
                              'label': 'Edge 3'
                          }), (('Expand 2', 'Concat'), {
                              'label': 'Edge 4'
                          }), (('Concat', 'Squeeze'), {
                              'label': 'Edge 5'
                          }), (('Squeeze', 'Output'), {
                              'label': 'Edge 6'
                          })])
g2 = apply_styles(g2, styles2)
g2.render('img/g2')
