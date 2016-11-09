For each node, initialize a boolean array of whether it has been "touched". The array will be of length 256. Only the root node has the first level set to 1.

We put all the nodes in a big array, and we also put all the edges in a big array.

The nodes have an position in the array, as well as the edges.

Example:

IN this example node 3 is root

Nodes:
	[0] Node 0 [false, false, false, false, ...]
	[1] Node 1 [false, false, false, false, ...]
	[2] Node 2 [false, false, false, false, ...]
	[3] Node 3 [true, false, false, false, ...]

Edges (note they are bidirectional)
	[0] 0 - 3
	[1] 1 - 2
	[2] 3 - 0
	[3] 0 - 2
	[4] 2 - 3

The algorithm is as follows.

In every GPU iteration for each work item, we:
1. Iterate over the corresponding edges for the work item
2. If the node is visited, visit all of its children (how do you get the children of a node?)
3. When visiting a child, simply set its "visited" bool to true.
4. Finish when noone reports any more new visited arrays

In the example above, after the first iteration, things will look like this:

Nodes:
	[0] Node 0 [false, true, false, false, ...]
	[1] Node 1 [false, false, false, false, ...]
	[2] Node 2 [false, true, false, false, ...]
	[3] Node 3 [true, false, false, false, ...]