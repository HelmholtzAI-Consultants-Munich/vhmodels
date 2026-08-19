# Intentionally empty: vhmodels.models.discovery is imported by every
# isolated model worker, including dependency-free ones (see
# tests/fixtures/persistent_worker). Adding a Pydantic-based re-export here
# (e.g. Registry) would import pydantic as a side effect of importing
# vhmodels.models at all -- see docs/manifest.md.
