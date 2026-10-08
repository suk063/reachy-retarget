"""Catalog generators, one module per family (``python -m reachy_retarget.acquire.generators.<family>``).

They read publisher metadata (HF tree API, Box file pages, tar headers by HTTP range,
package tables) at pinned revisions and write ``acquire/catalog/<family>.yaml`` plus its
row tables. Network access happens only when a generator is run explicitly; importing
this package does nothing.
"""
