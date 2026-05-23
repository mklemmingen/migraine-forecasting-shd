"""Optional measured-weather covariates for Addition 6 (headache target only).

The SHD diary carries only a self-reported ``weather_change`` flag, not measured
weather. Smartphone-app evidence links low barometric pressure, higher humidity,
and rainfall to headache occurrence [katsuki2023weather, p. 585]. Because the
diary records dates and the cohort attended two known Korean hospital sites, a
historical daily meteorological series can be joined and ablated against the
diary-only model.

Scope decision (docs Section 3.4): HEADACHE target only. The migraine
full_features cell is already EPV 3.9 (docs/dataset.md); adding weather columns
worsens overfitting and shrinkage cannot create information the ~201 positive
migraine days do not contain [martin2025samplesize, p. 2]. Report as an ablation
(with vs without weather) on discrimination and net benefit, not a new headline.
"""


def join_weather(diary_df, station_by_site, weather_source):
    """Left-join daily meteorological features onto the diary by site + date.

    TODO (external-data decision, docs Section 9): decide whether to source
    measured weather this iteration or record it as future work. If sourcing:
      - map each patient's hospital site to a KMA (Korea Meteorological
        Administration) station; the two sites are Uijeongbu and Dongtan.
      - pull daily barometric pressure, pressure change, humidity, rainfall for
        Sept 2014 - Jan 2015; document the source URL and licence here.
      - left-join on (site, date); features missing for a date stay NaN and are
        handled by the model's usual missing-value path.
    Return the diary with the added columns. ~20-30 lines plus the data fetch.
    """
    raise NotImplementedError("measured-weather join - external-data decision, see TODO")
