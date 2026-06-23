# Raw Data

This package does not duplicate the large raw datasets into Git. The locked Model 6 branch reads them from the retained local workspace rooted at:

- `/home/mircore/Desktop/diff_hydro`

The clean publication repo keeps only this manifest and uses `MODEL6_PUBLICATION_DATA_ROOT` to locate the retained data.

## Retained roots used by the locked branch

| Source product | Local path | Period | Variables used | Type | Depended on by |
| --- | --- | --- | --- | --- | --- |
| CAMELS-US forcing and attributes | `/home/mircore/Desktop/diff_hydro/Camels` | training window and test window | `prcp`, `tmean`, PET-Hargreaves, streamflow, 35 static attributes | raw | `model6/train_model6.py`, copied Model 6 source |
| Caravan-derived radiation support | `/home/mircore/Desktop/diff_hydro/global_data/Caravan_v1_5/extracted/usr/local/google/home/kratzert/Data/Caravan-Jan25-csv/timeseries/csv/camels` | daily overlap with train/test | net solar radiation, net thermal radiation | raw | copied LAI helper and ET/Rn diagnostics |
| NOAA AVHRR CDR gap-filled daily LAI | `/home/mircore/Desktop/diff_hydro/ECO_HYBRID/NOAA_LAI_CDR_671_DAILY/lai_cdr_daily_basin_671_gapfilled_rf.csv` | 1980-2010 | `lai_mean` | derived/cached | training and figure scripts |
| Independent validation downloads | `/home/mircore/Desktop/diff_hydro/outputs/IndependentValidationDownloads` | product-specific | MODIS MOD16 ET, GLEAM ET, ESA CCI SM and related caches | raw/cached | evaluation bundle and external validation figures |
| ECO input bundle | `/home/mircore/Desktop/diff_hydro/ECO_HYBRID/ECO_INPUTS_2020` | product-specific | minimally disturbed 455-basin subset list, GRACE/JPL tables, SWE support tables, ancillary validation tables | raw/derived | 455 subset mode, Rohini-style figures, TWSA and SWE diagnostics |

## Environment variable

If the retained raw data live somewhere else on your machine, point the wrappers there:

```bash
export MODEL6_PUBLICATION_DATA_ROOT=/path/to/original/diff_hydro_workspace
```

