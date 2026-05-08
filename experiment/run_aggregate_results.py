"""
run script that aggregates any and all results/ *LATEST* entries into one interactive table in html.

Uses the predefined structure of the folders as described in the Readme of:
`experiment/<NrAddition>/<feature_set>/<architecture>/<*dataSplit>/<*SplitType>`

to name the current approach in the tabular views.

Though to the multidimensional state of the experiments, namely (as of 26-05-08):
feature set
architecture
<version where applicable>
split in fractions
split in type

the tabular views are sorted by these 2 dimensions, allowing selections of the lower dimension to move to the
next applied tabular view, and so on, until the metrics as defined in the Readme at the lowest level are shown.

The complexity of the resulting html is scaled dynamically based on results/ folders found with valid last result and paths mapped.

Expects the prints in the results txt to be based as defined by sharedMetricPrinter.
-> this code checks the contract template by calling getContract() on it, to ensure any change in format can be
centralized in sharedMetricPrinter.py

The results are metadated by time in the last view.
the overall html has a human readable iso timestamp and a uuid to not allow accidental overwrites by sudden reruns.

fyi: the aggregate result runner will fill out N/A under values not found that were defined by contract,
but not in the results/ folders latest result txt.  - and will give any user running it a red text
in their terminal where supported.
"""

# imports all packages needed for parsing and interactive html creation

# getContract and getContract_CV

# create dynamic html with ascending and jumping table to table views until results



