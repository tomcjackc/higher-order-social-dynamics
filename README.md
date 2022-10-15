# MSci Project on Higher-order Social Dynamics

**Current capabilities**

Given a time-resolved edge list:
* Simulate the higher order naming game using the "unanimous rule"
* Supports committed agents
* Supports multiple (>2) opinions/names

**Next steps**
* Include support for more rules, not just unanimous
* Include a measure of social influence exhibited by each agent
* Rewrite in object-oriented form - requires digging through documentation
* Refine what one timestep involves, look at "Group interactions modulate critical mass dynamics in social convention" [link](https://doi.org/10.1038/s42005-022-00845-y)
* Consider allowing multiple copies of each word to exist in a given vocabulary, this would increase the chances of the word being broadcast the next time that node is chosen as speaker. Possible limitation of the study above as it has vocabularies going from a very steady position (one word known), to a very precarious one in only one interaction.