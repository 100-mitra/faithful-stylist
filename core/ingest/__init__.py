"""Ingestion: synthetic catalog generator (always works) and the responsible
rate-limited real-brand scraper (optional, Phase 1 step 14).

The whole pipeline must run end-to-end on either source. Synthetic rows are clearly
marked ``source="synthetic"``; scraped rows record their ``source``/``source_url``.
"""
