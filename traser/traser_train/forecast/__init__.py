"""Scene graph forecasting on top of TRASER (TRASER-F).

Given the first H seconds of a clip and the mask trajectories of its objects, predict the
relationship triplets of the next F seconds and where every object will be. See
docs/FORECAST.md for the design and the benchmark protocol.
"""
