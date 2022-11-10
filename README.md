# MSci Project on Higher-order Social Dynamics

Work based upon "Group interactions modulate critical mass dynamics in social convention" [link](https://doi.org/10.1038/s42005-022-00845-y).

**Current capabilities**

* Generate a non-time-resolved edge list from Sociopatterns data. [link](http://www.sociopatterns.org/)

Given an edge list:
* Simulate the higher order naming game using the "unanimous" and "union" rules
* Supports committed agents
* Supports multiple (>2) opinions/names
* Includes a measure of social influence exhibited by each agent

**Next steps**
* Consider allowing multiple copies of each word to exist in a given vocabulary, this would increase the chances of the word being broadcast the next time that node is chosen as speaker. Possible limitation of the study above as it has vocabularies going from a very steady position (one word known), to a very precarious one in only one interaction. This could almost mimic continuous opinion dynamics.
* Adapt reconstruction process to time-resolve the hyperedges and therefore incorporate some causality into the model.
* Think about the proper long-time limit, where the absorbing state with all B is reached. We could maybe consider working out the expected length of time required to reach that state on a randomly seeded, completely connected hypergraph/simplicial complex. It would be a very large number, but we could maybe test it for very small hypergraphs.
* Investigate the dynamics of multiple (>2) opinions existing on the network.
* Give the nodes a social influence parameter $\beta$ that is related to their importance or centrality.
