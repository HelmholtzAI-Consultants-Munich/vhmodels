# Shared constants for the parent<->child handover protocol.
# Both the parent (ModelProxy in factory.py) and the child
# (vhmodels.vh_checker.embed) import RESULT_MARKER so the framing cannot drift.
#
# The child writes its JSON result framed between two RESULT_MARKERs on stdout.
# Everything else on stdout (progress bars, library logging, warnings) is ignored
# by the parser, so the result channel is immune to that noise.

RESULT_MARKER = "===VHMODELS_RESULT==="
