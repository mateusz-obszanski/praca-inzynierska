# General concept

Results are obtained by calling API. API has CLI (minimal version) and web
interface (future). Results from each run are stored in separate directories,
each consists of run config, path to initial results and solutions (not optimized)
and also files with optimized solutions and results. Every optimized result can
be deterministically regenerated from run config. Run config consists of path to
environment config and simulation config and also of optimization config file path.
There are scripts for faster access to those configs, solutions and results from run config.
Paths are local (minimal version) or can be URLs (future). Computation is local (minimal version)
or remote, on cluster (future).

## Results

There is API for generating static results visualization (minimal version) and
dynamic visualization as a video file (future).

## GUI (future)

Implemented as Internet page server (Plotly's Python Dash library).

Enables:
- **configuration creation** - based on templates and previous run configs
- **simulation execution** - with real-time progress logging
- **viewing results** - charts, dynamic charts (interpolated simulation frames)
    and viewing simulation video
- **adding descriptions and comments**

## Optimalizators
- tabu search
- ant algorithms

## Simulation types

### Discrete (minimal version)
- TSP
- TSP with loss function based on environment dynamics
- Dynamic TSP (wind)
- DynTSP with loss function based on environment dynamics

## Continuous (future)
- Agents move continuously, real-time swarm algoritms

## Project roadmap
- simple simulation type - with simple mathematical model
- heuristic, randomized initial solution generation
- solution simulation
- results visualization
- simple optimizer
- results difference visualization
- simple GUI for convenience and debugging

Then repeat the cycle for subsequent simulation types and optimization algorithms.