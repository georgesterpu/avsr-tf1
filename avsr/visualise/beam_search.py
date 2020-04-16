import networkx as nx
from networkx.readwrite import json_graph
from string import Template
import json
from os import makedirs, path
from shutil import copy


r'''
Code forked from:
https://github.com/OpenNMT/VisTools

Main changes (including the .css and .js):
=========================================
1. node fill colour proportional to beam negative log likelihood
2. aesthetic updates
    2.1 no more text overlap
    2.2 place characters inside the circles
3. copy css and js to root outdir and reference them in the html
4. display the ground truth transcription along with the graph
'''

r'''
TODO
====
*. properly normalise the scores
*. toggle scores on / off
*. URL to dataset dir - play example
*. colour legend
*. display likelihood of the ground truth sequence
*. highlight best translation - stronger line width and different colour
*. update to latest d3.js version
'''


HTML_TEMPLATE = Template("""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Beam Search</title>
    <link rel="stylesheet" type="text/css" href="${WALK}tree.css">
    <script src="http://d3js.org/d3.v3.min.js"></script>
    <script src="https://d3js.org/d3-color.v1.min.js"></script>
    <script src="https://d3js.org/d3-interpolate.v1.min.js"></script>
    <script src="https://d3js.org/d3-scale-chromatic.v1.min.js"></script>
  </head>
  <body>
    <script>
      var treeData = $DATA;
      const transcript = "$transcript";
    </script>
    <script src="${WALK}tree.js"></script>
  </body>
</html>""")


def _add_graph_level(graph, level, parent_ids, names, scores):
    """Adds a levelto the passed graph"""
    for i, parent_id in enumerate(parent_ids):
        new_node = (level, i)
        parent_node = (level - 1, parent_id)
        score_str = '%.3f' % float(scores[i]) if scores[i] is not None else '-inf'
        graph.add_node(new_node)
        graph.node[new_node]["name"] = names[i]
        graph.node[new_node]["score"] = score_str
        graph.node[new_node]["size"] = 100
        # Add an edge to the parent
        graph.add_edge(parent_node, new_node)


def create_graph(predicted_ids, parent_ids, scores, vocab=None):
    def get_node_name(pred):
        return vocab[pred] if vocab else pred

    seq_length = len(predicted_ids)
    graph = nx.DiGraph()
    for level in range(seq_length):
        names = [get_node_name(pred) for pred in predicted_ids[level]]
        _add_graph_level(graph, level + 1, parent_ids[level], names, scores[level])
    graph.node[(0, 0)]["name"] = "START"
    return graph


def create_html(predicted_ids, parent_ids, scores, labels_ids, vocab, filename, output_dir):
    graph = create_graph(
        predicted_ids=predicted_ids,
        parent_ids=parent_ids,
        scores=scores,
        vocab=vocab)

    json_str = json.dumps(json_graph.tree_data(graph, (0, 0)), ensure_ascii=True)

    transcript = [vocab[sym] for sym in labels_ids]
    transcript.remove('EOS')
    transcript = ''.join(transcript)

    output_fname = path.join(output_dir, filename + '.html')
    num_subdirs = filename.count('/')  # too hacky ?

    makedirs(path.dirname(output_fname), exist_ok=True)

    html_str = HTML_TEMPLATE.substitute(DATA=json_str, WALK='../'*num_subdirs, transcript=transcript)

    with open(output_fname, 'w') as f:
        f.write(html_str)


def copy_headers(out_dir):
    css_path = './avsr/visualise/tree.css'
    js_path = './avsr/visualise/tree.js'
    makedirs(out_dir, exist_ok=True)
    copy(css_path, out_dir)
    copy(js_path, out_dir)
