# Numerical Methods for Option Pricing

Numerical methods for pricing options.

Includes: 

A `EuropeanOption` class (found in `./src/options.py`. This class consists of:
- A number if implementations of the Crank Nicolson method to price European Options without dividends (found in `./src/fdm.py`). Each implementation is slighty optimised relative to its direct previous ancestor. Speed is compared in `./figures/comparing_cn_implementations.png`, while ability to deal with oscillations is highlighted in `./figures/effect_of_rannacher_smoothing.png`
- Implementations of some monte carlo methods, namely euler-maryuma, milstein, exact integration (found in `./src/monte_carlo.py`)


An `AmericanOption` class. This class currently only consists of:
- A single implementations of the Crank Nicolson method to price American style options without dividends (found in `./src/fdm.py`). The implementation mostly uses optimsations from the most optimised crank nicolson implementation for European style options.

`sweep.py` for generating data, which is stored in `.\data`. There already exists `.\data\test_run.parquet` which is used in `compare_cn_methods.ipynb`
