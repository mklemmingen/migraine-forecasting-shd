"""
Defines a metric print to txt contract (how the result.txt look inside) as a two-way.

getContract() can be called to get a line by line view of the strings expected before the numeric values.
-> as a eval script, contract should be kept with header, line strings and footer
-> as the result aggregator, the contract should be used to dynamically find the metric numeric values to use
for any kind of html visualisation of results.

getContract_cv() can be also called to get a line by line view for the more tabular cross validation.

The Contracts should be updated if they do happen to change. document in readme under metrics first, then here,
then in all evals depending.

fyi: the aggregate result runner will fill out N/A under values not found, and will give any user running it a red text
in their terminal where supported.
"""

getContract()
    getHeaderContract()
    getMetricsContract()
    getFooterContract()

getContract_cv()
    getHeaderContract
    getMetricsContract_CV()
    getFooterContract()

_getHeaderContract()

_getFooterContract()

_getMetricsContract()

_getMetricsContract_CV()