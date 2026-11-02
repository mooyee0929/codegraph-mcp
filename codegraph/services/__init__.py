"""Cross-service structure that no single-file parser can see.

The post-index passes in :mod:`codegraph.passes` use these helpers to find HTTP routes
and the client calls that reach them, message channels and who publishes or listens
on them, near-duplicate functions, and how files change together in git.

Detection works on raw source text with per-language regular expressions rather than
on parser output, for two reasons: the interesting facts (a decorator's path literal,
a topic string) are lexical anyway, and it lets route and channel detection run for
languages whose parsers have not landed yet. Graph nodes are only used afterwards, to
attach what was found to the function that owns it.
"""
