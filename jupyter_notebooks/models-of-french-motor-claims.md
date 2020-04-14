---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: env_research
    language: python
    name: env_research
---

# Modelling motor insurance claim frequency
Using the French Motor Claims dataset as an example to work through predictive modelling considerations. It is designed to be run on a Kaggle Kernel here: <https://www.kaggle.com/btw78jt/models-of-french-motor-claims>

<!-- This table of contents is updated *manually* -->
## Contents
1. [Setup](#Setup)
1. [Load data](#Load-data): Small sample, Full dataset
1. [Resulting preprocessing](#Resulting-preprocessing)
1. [Data subsets for modelling](#Data-subsets-for-modelling)
1. [Exploratory analysis](#Exploratory-analysis): 
    - [Identifier](#Identifier)
    - [Exposure](#Exposure)
    - [Explanatory variables: duplicates](#Explanatory-variables:-duplicates)
    - [Response](#Response)
    - [One-ways](#One-ways)
1. [Geographic factors](#Geographic-factors)


<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

## Setup

```python
# Import built-in modules
import sys
import platform
import os
from pathlib import Path
import warnings

# Import external modules
from IPython import __version__ as IPy_version
import IPython.display as ipyd
import numpy as np
import pandas as pd
from sklearn import __version__ as skl_version
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
from bokeh import __version__ as bk_version
import geopandas as gpd
import geoviews as gv

# Check they have loaded and the versions are as expected
assert platform.python_version_tuple() == ('3', '6', '6')
print(f"Python version:\t\t{sys.version}")
assert IPy_version == '6.4.0'
print(f'IPython version:\t{IPy_version}')
assert np.__version__ == '1.16.2'
print(f'numpy version:\t\t{np.__version__}')
assert pd.__version__ == '0.23.4'
print(f'pandas version:\t\t{pd.__version__}')
assert skl_version == '0.20.3'
print(f'sklearn version:\t{skl_version}')
assert mpl.__version__ == '2.2.3'
print(f'matplotlib version:\t{mpl.__version__}')
assert bk_version == '1.0.4'
print(f'bokeh version:\t\t{bk_version}')
assert gpd.__version__ == '0.4.0'
print(f'geopandas version:\t{gpd.__version__}')
assert gv.__version__ == '1.6.1'
print(f'geoviews version:\t{gv.__version__}')
```

```python
# Set warning messages
# Show all warnings in IPython
warnings.filterwarnings('always')
# Ignore specific numpy warnings (as per <https://github.com/numpy/numpy/issues/11788#issuecomment-422846396>)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
# Additional warning that sometimes pops up
warnings.filterwarnings("ignore", message="`nbconvert.exporters.exporter_locator` is deprecated")
```

```python
# Load Bokeh for use in a notebook
from bokeh.io import output_notebook
output_notebook()
```

```python
# Initialise geoviews
gv.extension('bokeh')
```

```python
# Configuration variables
claims_data_filepath = Path('/kaggle/input/french-motor-claims-datasets-fremtpl2freq/freMTPL2freq.csv')
additional_data_folderpath = Path('/kaggle/input/additionaldataforfrenchmotorclaims')
```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

## Load data
### Small sample

```python
# Load data
nrows_sample = 1000
df_small_sample = pd.read_csv(claims_data_filepath, delimiter=',', nrows = nrows_sample)

# Check it has loaded OK
assert df_small_sample.shape == (1000, 12)
print("Correct: Shape of DataFrame is as expected")
```

```python
#  Look at the first few rows
df_small_sample.head()
```

```python
# Look at data types of the columns
dtypes_sample = df_small_sample.dtypes
print(dtypes_sample)  # Check these look reasonable
```

```python
assert df_small_sample.isna().sum().sum() == 0
print("Correct: There are no missing values in the sample dataset")
```

### Full dataset

```python
# Load data
df_raw = pd.read_csv(
    claims_data_filepath, 
    delimiter=',', 
    dtype={'IDpol': np.int64}  # Without this, IDpol is cast as a float, but not sure why
)
```

```python
# Check it has loaded OK
nRows, nCols = (678013, 12)
assert df_raw.shape == (nRows, nCols)
print(f"Correct: Shape of DataFrame is as expected: {nRows} rows, {nCols} cols")
assert (df_raw.dtypes == dtypes_sample).all()
print("Correct: Data types are as expected")
assert df_raw.isna().sum().sum() == 0
print("Correct: There are no missing values in the raw dataset")
```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

## Resulting preprocessing
This section creates a function that implements the decisions that are justified below. We put this section here (and not at the end) so that the resulting DataFrame can be used throughout the analysis.

```python
def get_df_extra(df):
    """
    Given a DataFrame of that contains the raw data columns (and possibly additional columns), 
    return the DataFrame with additional pre-processed columns
    """
    df_extra = df.copy()
    
    # Exposure rounded to 4 dps
    df_extra['Exp_4dps'] = df_extra.Exposure.round(4)
    
    # Calculate frequency per year on each row
    df_extra['freq_pyr'] = df_extra['ClaimNb'] / df_extra['Exp_4dps']
    
    return(df_extra)  
```

```python
# Run pre-processing to get a new DataFrame
df_extra = get_df_extra(df_raw)
```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

## Data subsets for modelling

```python
# Import modules specific for this section
from sklearn.model_selection import train_test_split
```

```python
# Get index sorted with ascending IDpol, just in case it is out or order
df_all = df_raw.sort_values('IDpol').reset_index(drop=True)

# Split out training data
df_train, df_not_train = train_test_split(
    df_all, test_size=0.3, random_state=51, shuffle=True
)
# Split remaining data between validation and holdout
df_validation, df_holdout = train_test_split(
    df_not_train, test_size=0.5, random_state=13, shuffle=True
)
```

```python
# Check resulting split looks reasonable
df_raw.assign(  
    # Add indicator columns for whether each row is in_train, in_validation, in_holdout
    in_train=df_raw.IDpol.isin(df_train.IDpol),
    in_validation=df_raw.IDpol.isin(df_validation.IDpol),
    in_holdout=df_raw.IDpol.isin(df_holdout.IDpol),
    # Add column of which subset each row is in
    subset=lambda x: np.select(
        [x.in_train, x.in_validation, x.in_holdout],
        ['train', 'validation', 'holdout'],
        default='no_subset')
).groupby(  # Group rows by which subset they are in
    ['in_train', 'in_validation', 'in_holdout', 'subset']
).agg({  # Calculate stats for each group
    'IDpol': 'size', 'Exposure': 'sum', 'ClaimNb': 'sum'
}).rename(columns={'IDpol': 'num_of_rows'}).assign(
    # Add additional stats
    Claim_freq_overall=lambda x: x.ClaimNb / x.Exposure,
    num_of_rows_prop=lambda x: x.num_of_rows / x.num_of_rows.sum(),
    Exposure_prop=lambda x: x.Exposure / x.Exposure.sum(),
    ClaimNb_prop=lambda x: x.ClaimNb / x.ClaimNb.sum(),
).pipe(lambda df: df.append(pd.DataFrame.from_dict({
    # Add totals row. It is the sum for every column except 'Claim_freq_overall'
    # where it is the overall claims frequency of the entire data set
    ('Total','','',''): [
        df.ClaimNb.sum() / df.Exposure.sum() if col_name == 'Claim_freq_overall'
        else df.loc[:,col_name].sum() for col_name in df
    ]}, orient='index', columns=df.columns
))) \
.style.format(  # Format the output so it looks reasonable when printed
    '{:.2%}'  # Default number format
).format({  # Specific number formats where we want to override the default
    **{col: '{:,.0f}' for col in ['num_of_rows', 'ClaimNb']},
    'Exposure': '{:,.1f}'
}).apply(  # Separate totals row by adding a line
    lambda x: ['border-top-style: double'] * x.shape[0], subset=pd.IndexSlice["Total",:]
)
```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

## Exploratory analysis


<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

### Identifier

```python
assert (df_raw.duplicated('IDpol') == False).shape[0] == df_raw.shape[0]
print("Correct: There are no duplicates of IDpol")
assert (df_raw.sort_values('IDpol').index == df_raw.index).all()
print("Correct: Rows are sorted in order of ascending IDpol")
```

### Exposure
`Exposure` = total exposure in yearly units

```python
df_raw.Exposure.plot.hist(bins=50)
plt.title("Histogram of Exposure (years)")
plt.show()
```

Observations: 
- A low proportion of policies have `Exposure` above 1. Could be poor data quality?
- A significant number of policies have low `Exposure`, i.e. close to zero.
- Range of values is reasonable.

```python
# In fact, the values close to 1 are all to 2 dps
# Look at the frequency close to 1. Does not look unreasonable
df_extra[(df_extra.Exposure > 0.95) & (df_raw.Exposure <= 1.05)].Exposure.value_counts().sort_index()
```

```python
# Not *all* policies have Exposure to 2 dps. The following shows that:
# - Values below 0.01 are split further
# - Values greater than or equal to 0.01 are given to 2 dps
df_extra.assign(Exp_100=lambda x: (x.Exposure * 100).round(6)).assign(
    Exp_100_int_part=lambda x: np.trunc(x.Exp_100)
).assign(
    Exp_100_frac_part=lambda x: x.Exp_100 - x.Exp_100_int_part,
    Exp_100_int_part_gte_1=lambda x: x.Exp_100_int_part >= 1
).groupby(['Exp_100_int_part_gte_1', 'Exp_100_frac_part']).size(
).to_frame('num_of_policies').reset_index().style.format(
    {'Exp_100_frac_part': '{:.8f}', 'num_of_policies': '{:,}'})
```

```python
def display_side_by_side(*args):
    """
    Print the display of multiple DataFrames side-by-side
    
    *args: DataFrames or Stylers to be displayed side-by-side
    Return: No return value
    Adapted from: <https://stackoverflow.com/a/44923103>
    """
    html_str=''
    for df_styler in args:
        if isinstance(df_styler, pd.DataFrame):
            df_styler = df_styler.style
        html_str += df_styler.set_table_attributes(
            "style='display:inline'"  # Side-by-side
        )._repr_html_()
    ipyd.display_html(html_str,raw=True)

# Example usage
# df1 = pd.DataFrame(np.arange(12).reshape((3,4)),columns=['A','B','C','D',])
# df2 = pd.DataFrame(np.arange(16).reshape((4,4)),columns=['A','B','C','D',])

# df1_styler = df1.style.set_table_attributes("style='display:inline'").set_caption('Caption table 1')
# df2_styler = df2.style.set_table_attributes("style='display:inline'").set_caption('Caption table 2')

# display_side_by_side(df1, df2_styler)
```

```python
# In fact, the following shows that some Exposure values that 
# are rounded (to 8 dps), and some that are not.
exposure_cnts_df = df_extra.Exposure.value_counts().sort_index(
).to_frame('num_of_policies').reset_index(
).rename(columns={'index': 'Exposure'}).assign(
    Exposure_to_8_dp=lambda x: x.Exposure.round(8),
)[['Exposure', 'Exposure_to_8_dp', 'num_of_policies']]  # Re-order columns

# We can compare these to the Exposure value calculated for a given number of days 
# (and assumed number of days in the year). By comparing these to the above, 
# we see that the Exp_100 values below one are highly likely to equate to a number of days. 
days_in_yr_df = pd.DataFrame({'num_of_days': np.arange(1, 5)}).assign(
    In_365d_yr=lambda x: x.num_of_days / 365,
    In_366d_yr=lambda x: x.num_of_days / 366,
)
num_dps = 15
col_format = f'{{:.{num_dps}f}}'

display_side_by_side(
    exposure_cnts_df[exposure_cnts_df.Exposure < 0.01].style.format(
        {'Exposure': '{:.20f}'}
    ).set_caption("Exposure values below 0.01 in the data").hide_index(),
    days_in_yr_df.style.format(
        {'In_365d_yr': col_format, 'In_366d_yr': col_format}
    ).set_caption("Calculated Exposure for given num of days").hide_index()
)
```

**Decision**: Therefore, we choose to round Exposure to 4 dps so that:
- The rounded and unrounded Exposure values round to the same value
- Exposure values that represent x days in a year of 365 or 366 days round to the same value

```python
def plot_exposure_hist_zoom(sers=df_extra.Exp_4dps, lims=(None, None), width=0.01):
    """Zoom in on a specific region of the histogram of Exposure"""
    lims = list(lims)
    if lims[0] is None:
        lims[0] = 0
    if lims[1] is None:
        lims[1] = sers.max()
    n_bins = int((lims[1] - lims[0]) / width)
    sers[(sers >= lims[0]) & (sers <= lims[1])].plot.hist(bins=n_bins)
    plt.title(f"Histogram of Exposure (years) between {lims[0]} and {lims[1]}")
```

```python
# No obvious pattern in the values above 1
plot_exposure_hist_zoom(lims=(1.05, None))
```

```python
# Some very low values
plot_exposure_hist_zoom(lims=(None, 0.3), width=0.002)
print(
    "The peak at 0.08 is equivalent to one month because:\n"
    f"  - Upper bound: 31 / 365 = {31 / 365 :.5f}\n"
    f"  - Lower bound: 28 / 366 = {28 / 366 :.5f}"
)
```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

### Explanatory variables: duplicates

```python
expl_var_names = df_raw.columns[3:].tolist()
print('Explanatory variables:\t' + '  '.join(expl_var_names))
```

```python
# Are there any duplicates? Yes, a lot!
dup_summary1 = df_extra.duplicated(
    expl_var_names, keep='first'  # Unique values are marked as False, so reverse this in the next operation
).ne(True).value_counts().to_frame('num_of_rows').rename_axis(
    'expl_vars_is_unique', axis=0
)
dup_summary1.style.format("{:,}")
```

```python
# But there are a lot fewer if we add Exp_4dps
df_extra.duplicated(
    ['Exp_4dps'] + expl_var_names, keep='first'
).ne(True).value_counts().to_frame('num_of_rows').rename_axis(
    'expl_vars_and_exp_is_unique', axis=0
).style.format("{:,}")
```

```python
# Add columns to identify duplicates
df_with_dups = df_extra.assign(
    is_dup=lambda x: x.duplicated(expl_var_names, keep=False)
).reset_index().rename(columns={'index': 'CSV_order'}).sort_values(
    expl_var_names + ['CSV_order']  # CSV_order ensures the order is unique in case of tie-break
).reset_index(drop=True).assign(
    occurence_1st=lambda x: ~x.duplicated(expl_var_names, keep='first'),
    d_CSV_order=lambda x: x.CSV_order - x.CSV_order.shift(1),  # change in CSV_order (current minus previous)
    IDpol_unique=lambda x: x.occurence_1st.cumsum(),
    d_CSV_order_dups=lambda x: np.where(~x.occurence_1st, x.d_CSV_order, 0).astype(int)
)
```

```python
# Check this matches the above
dup_summary2 = df_with_dups.groupby(['occurence_1st', 'is_dup']).size(
).to_frame('num_of_rows')
assert dup_summary1.loc[False][0] == dup_summary2.loc[False].sum()[0]
print(f'Correct: Agrees number of repeated explanatory variable combinations')
assert dup_summary1.loc[True][0] == dup_summary2.loc[True].sum()[0]
print(f'Correct: Agrees number of unique explanatory variable combinations')
dup_summary2.style.format("{:,}")
```

```python
# Many of the duplicated rows are consecutive IDpol numbers (i.e. d_CSV_order is 1)
df_with_dups.assign(
    d_CSV_order_dups_grpd=lambda x: np.where(x.d_CSV_order_dups >= 5, '5+', x.d_CSV_order_dups)
).query("is_dup == True").groupby(
    ['occurence_1st', 'd_CSV_order_dups_grpd']
).size().to_frame('num_of_rows').style.format(
    "{:,}").set_caption("Break down includes duplicated rows only")
```

```python
# Look at an example
CSV_order_gap = 5  # Try amending these to get another example
rand_example = 4
view_range = 2

IDpol_unique_example = df_with_dups.IDpol_unique[
    df_with_dups.d_CSV_order_dups == CSV_order_gap].iloc[rand_example,]
CSV_order_range = df_with_dups.CSV_order[df_with_dups.IDpol_unique == IDpol_unique_example].values

display(
    df_with_dups.query(
        "IDpol_unique >= @IDpol_unique_example - @view_range & IDpol_unique <= @IDpol_unique_example + @view_range"
    ).style.apply(lambda row_sers: np.repeat(
        np.select(
            [row_sers.IDpol_unique == IDpol_unique_example], 
            ['background-color: yellow'], 
            default=''
        ), row_sers.shape[0]
    ), axis=1).set_caption("Example highlighted from table ordered by duplicates")
)

display(
    df_extra.loc[CSV_order_range[0]-2:CSV_order_range[1]+2,:].style.apply(
        lambda row_sers: np.repeat(
            np.select([
                (row_sers.name == CSV_order_range[0]) | (row_sers.name == CSV_order_range[1]),
                (row_sers.name >= CSV_order_range[0]) & (row_sers.name <= CSV_order_range[1])
            ],[
                'background-color: yellow',
                'background-color: red'
            ], default=''
            ), row_sers.shape[0]
        ), axis=1
    ).set_caption("Same example from table ordered by IDpol")
)
```

```python
# Get DataFrame with duplicates aggregated into one row each
df_de_duped = df_with_dups.groupby(expl_var_names + ['IDpol_unique']).agg({
    'IDpol': 'size',
    'Exp_4dps': 'sum',
    'ClaimNb': 'sum',
}).rename(columns={'IDpol': 'num_of_rows'}).reset_index().set_index('IDpol_unique')

# Check it worked
assert df_de_duped.shape[0] == dup_summary1.loc[True][0]
print("Correct: Number of rows matches number of unique explanatory variable combinations")
assert (df_de_duped.Exp_4dps.sum() - df_extra.Exp_4dps.sum()) < 1e-6
print("Correct: Sum of exposure agrees")
assert df_de_duped.ClaimNb.sum() == df_extra.ClaimNb.sum()
print("Correct: Sum of number of claims agrees")
```

```python
# Look at first few rows
df_de_duped.head()
```

```python
# Look at the Exposure histogram now
df_de_duped.Exp_4dps.plot.hist(bins=50)
plt.title("Histogram of Exposure (years)")
plt.show()
```

```python
# In fact, if we look back at a graph of the original (not aggregated) data
# and compare it to the de-duped data, we see the peak at 0.08 (= one month)
# has been largely removed, but little else noticeable has changed.
fig, _ = plt.subplots(2, 2, figsize=(12, 4))
plt.subplot(1, 2, 1)
plot_exposure_hist_zoom(width=0.01, lims=(None, 2))
plt.subplot(1, 2, 2)
plot_exposure_hist_zoom(df_de_duped.Exp_4dps, width=0.01, lims=(None, 2))
```

```python
# That change is even more pronounced when you zoom in on values below 1
fig, _ = plt.subplots(2, 2, figsize=(12, 4))
plt.subplot(1, 2, 1)
plot_exposure_hist_zoom(width=0.01, lims=(None, 0.98))
plt.subplot(1, 2, 2)
plot_exposure_hist_zoom(df_de_duped.Exp_4dps, width=0.01, lims=(None, 0.98))
```

```python
# Above 1, neither have very many policies, although the aggregated data 
# clearly has a peak at 2.
fig, _ = plt.subplots(2, 2, figsize=(12, 4))
plt.subplot(1, 2, 1)
plot_exposure_hist_zoom(width=0.01, lims=(1.02, None))
plt.subplot(1, 2, 2)
plot_exposure_hist_zoom(df_de_duped.Exp_4dps, width=0.01, lims=(1.02, 3))
```

```python
# How many ID_pols have gone in to each IDpol_unique? 
# This shows the distribution. We see there is a sharp drop after 12.
df_de_duped.groupby('num_of_rows').agg({
    'Area': 'size',
    'Exp_4dps': ['sum', 'mean', 'min', 'max'],
    'ClaimNb': ['sum', 'mean', 'min', 'max']
}).rename(columns={'Area': 'num_of_IDpol_unique'}).assign(
    Per_row_exp=lambda x: x[('Exp_4dps', 'mean')] / x.index,
    Per_row_ClaimNb=lambda x: x[('ClaimNb', 'mean')] / x.index,
).style.bar(subset=['Per_row_exp', 'Per_row_ClaimNb'], color='violet')
```

**Question**: Are there underlying individuals who have multiple `IDpol`s?

**Answer**: The data is inconclusive. For instance:
- **In favour**: The Exposure distribution in the aggregated data shows a higher peak at 1, and less of a peak at one month. That might suggest that the 
- **Against**: The Exposure distribution in the aggregated data shows a higher peak at 2, but the explanatory variables include `DrivAge`, and you'd expect that to increase 

**Decision**: Decided *not* to use the aggregate data, and to stick with the original data, as identified by `IDpol`. However, when modelling, we should consider that multiple `IDpol`s could belong to the same individual, so they may not be independent samples. In fact, each `IDpol` should be considered a *policy period*, rather than a policy. 

*Further idea*: It might be interesting to test any model produced to see how it is affected by considering original (`IDpol`) or aggregated (`ÌDpol_unique`) data.

```python
# Look at ClaimNb distribution in the aggregated data
df_de_duped.ClaimNb.value_counts().sort_index(
).to_frame("num_of_IDpol_unique").rename_axis("ClaimNb").style.format('{:,}')
```

```python
# Investigate high number of claims on these particular aggregations
df_de_duped.query("num_of_rows == 15")
```

```python
# Look at the one example that shows a *lot* of claims
IDpol_unique_example = 313686
df_with_dups.loc[
    df_with_dups.IDpol_unique == IDpol_unique_example,
    [col for col in df_with_dups.columns if col not in expl_var_names]
].style.bar(subset=['ClaimNb'], color='violet')
```

This is a *lot* of claims in one unique bucket. Even if you took out the one row with 16 claims, there is still 6, 8 and 9, which are very extreme values in the data. That is, even excluding the 16 row, we'd still have 32 in this bucket.

```python
# The next most extreme example
df_de_duped.query("ClaimNb == 24")
```

```python
# Look at this example
IDpol_unique_example = 314815
df_with_dups.loc[
    df_with_dups.IDpol_unique == IDpol_unique_example,
    [col for col in df_with_dups.columns if col not in expl_var_names]
].style.bar(subset=['ClaimNb'], color='violet')
```

**Summary**: The two `IDpol_unique`s (of `313686` and `314815`) are certainly outliers, and look very suspiciously like either poor data or fraudulent activity. Given the fields in the input data, we'll never be able to fully ascertain the true cause. The options are:
1. Leave the data and hope any resulting model is unaffected. It may be a question of selecting a model specifically to be immune to this type of data (e.g. model claim propensity, and then number of claims). 
1. Remove one or both from the data.
1. Keep them but manually alter the data, e.g. cap the number of claims on any `IDpol` row at 3 or 4, say.

**Decision**: For the time being, I will leave them unaltered in the data (opion 1 above), but bear in mind when I come to look at frequency and considering model techniques. In fact, it'll be interesting to fit and assess any model with and without these extreme observations, to see what difference it makes.


<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

### Response

```python
# Look at stats of number of claims and frequency per year for IDpol
display_side_by_side(
    df_extra[['ClaimNb', 'freq_pyr']].describe(
        percentiles=[.5, .9, .99, .999]).style.set_caption("All IDpol rows"),
    df_extra.query("ClaimNb > 0")[['ClaimNb', 'freq_pyr']].describe(
        percentiles=[.5, .9, .99, .999]).style.set_caption("Only rows with claims"),
    df_extra.ClaimNb.value_counts().sort_index(
    ).to_frame("num_of_rows").rename_axis("ClaimNb").style.format('{:,}'),
)
```

```python
# Look at claim frequency
assert (df_extra.freq_pyr == 0).sum() == 643953  # Check the number of zeros is consistent
df_extra.query("freq_pyr > 0.02").freq_pyr.plot.hist(bins=50)
plt.show()
```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

### One-ways

```python
def get_bin_width_rdd(
    lims_exact, nbins_target=30,
    split_prop_candidates=None
):
    """Get a sensible width of intervals to split a range given a target number of bins
    lims_exact: tuple (min, max) of the range (with min < max)
    nbins_target: positive int, number of intervals to target
    split_prop_candidates: Defines what are 'sensible' width sizes. Can be:
        - str: 'strict' or 'expanded' to choose pre-defined version
        - None: same as 'strict'
        - np.array of values between 0 and 1
    Returns: The calculated sensible width
    """
    split_prop_candidates_dict = {
        'one_dp': np.array([.1, .2, .5]),
        'strict': np.array([.1, .2, .25, .5]),
        'expanded': np.concatenate((
            np.arange(10, 40, 5) / 100, 
            np.arange(4, 6, 1) / 10, 
            np.array([.6, .8, 1.])
        ))
    }
    if split_prop_candidates is None:
        split_prop_candidates = 'strict'
    if isinstance(split_prop_candidates, str):
        split_prop_candidates = split_prop_candidates_dict[split_prop_candidates]
    width_exact = (lims_exact[1] - lims_exact[0]) / float(nbins_target)
    width_oom = np.ceil(np.log10(width_exact)).astype(np.int64)  # oom = order of magnitude
    split_prop_best = split_prop_candidates[(np.abs(
        split_prop_candidates - np.around(  # Attempt to avoid rounding instability
            width_exact / (10. ** width_oom), 6
        )
    )).argmin()]
    return(split_prop_best * (10. ** width_oom))

def get_breaks_rdd(
    lims_exact, nbins_target=30, 
    boundary=0.,
    split_prop_candidates=None
):
    """Split a range by a sensible width intervals given a target number of bins
    boundary: float, all breaks will be `boundary + width * n` for some n
    Other arguments: see get_bin_width_rdd()
    Returns: Equally spaced breaks
    """
    width_rdd = get_bin_width_rdd(lims_exact, nbins_target, split_prop_candidates)
    lims_expd = (
        np.floor((lims_exact[0] - boundary) / width_rdd) * width_rdd + boundary,
        np.ceil((lims_exact[1] - boundary) / width_rdd) * width_rdd + boundary
    )    
    return(np.arange(lims_expd[0], lims_expd[1] + width_rdd / 2, width_rdd))
```

```python
df_extra.head()
```

```python
x_col = 'DrivAge'
weight_col = 'Exp_4dps'
line_cols = ['freq_pyr']
```

```python
x_sers = df_extra[x_col]
break_points = get_breaks_rdd(
    (x_sers.min(), x_sers.max()), nbins_target=30, split_prop_candidates='one_dp')
bin_labels = [f'[{break_points[0]}, {break_points[1]}]'] + [
    f'({lwr}, {upr}]' for lwr,upr in zip(break_points[1:-1], break_points[2:])]
cut_obj = pd.cut(
    x_sers, bins=break_points,
    right=True, include_lowest=True, labels=None
)
```

```python
cols_needed = [x_col, weight_col] + line_cols
df_plot = df_extra.copy()[cols_needed].rename(columns={
    weight_col: 'Weight'
}).assign(num_of_rows=1)
```

```python
agg_commands = {**{
    'num_of_rows': 'size',
    'Weight': 'sum',
}, **{
    line_col: ['mean', 'max', 'min'] for line_col in line_cols
}}
df_plot_agg = df_plot.groupby(cut_obj).agg(agg_commands).set_index(
    pd.IntervalIndex.from_breaks(break_points)
).assign(
    bin_labels=bin_labels,
    left=lambda x: x.index.left,
    right=lambda x: x.index.right,
    mid=lambda x: x.index.mid,
)
df_plot_agg.head()
```

```python
# Import specific modules for this section
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show, output_notebook
from bokeh.models.ranges import Range1d
from bokeh.models.axes import LinearAxis
```

```python
bkplt = figure(
    title="title", x_axis_label=x_col, y_axis_label="Exposure (yrs)", 
    tools="reset,box_zoom,pan,wheel_zoom,save", background_fill_color="#fafafa",
    plot_width=800, plot_height=500
)
bkplt.quad(
    top=df_plot_agg.Weight['sum'], bottom=0, left=df_plot_agg.left, right=df_plot_agg.right,
    fill_color="khaki", line_color="white"#, alpha=0.5
)
bkplt.y_range=Range1d(0, df_plot_agg.Weight['sum'].max() / 0.5)

y_range2_name = 'y_range2_name'
bkplt.extra_y_ranges[y_range2_name] = Range1d(0, 0.6)
ax_new = LinearAxis(y_range_name=y_range2_name, axis_label="Response")
bkplt.add_layout(ax_new, 'right')

for line_col in line_cols:
    bkplt.circle(
        df_plot_agg.mid, df_plot_agg[line_col]['mean'], 
        color="purple", size=4,
        y_range_name=y_range2_name,
    )
    
bkplt.grid.grid_line_color="white"
show(bkplt)
```

**NOT COMPLETE**


<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

## Geographic factors

```python
deps_sf = gpd.read_file(additional_data_folderpath / 'departements-version-simplifiee.geojson')
assert deps_sf.shape == (96, 3)
print("Correct: Shape of the DataFrame is as expected")
```

```python
# Look at first few rows
deps_sf.head()
```

```python
# Note that accents and hats on characters have loaded correctly
deps_sf.loc[[
    region_name in ["Bouches-du-Rhône", "Pyrénées-Orientales", "Finistère"] 
    for region_name in deps_sf.nom
],:]
```

```python
# Check data types look OK
deps_sf.applymap(type).groupby(deps_sf.columns.tolist(), sort=False).size(
).to_frame("num_of_rows").rename_axis("data_type", 1)
```

```python
# In a notebook, a Polygon object automatically prints out
deps_sf.geometry[27]
```

```python
regs_sf = gpd.read_file(additional_data_folderpath / 'regions-avant-redecoupage-2015.geojson')
assert regs_sf.shape == (22, 3)
print("Correct: Shape of the DataFrame is as expected")
assert regs_sf.code.duplicated().sum() == 0
print("Correct: All values in the field 'code' are unique")
```

```python
# Look at first few rows
regs_sf.head()
```

```python
# Load lookup tables for data
reg_lookup_df = pd.read_csv(additional_data_folderpath / 'regions.csv', sep=';', encoding='latin1')
assert reg_lookup_df.shape == (22, 3)
print("Correct: Shape of the DataFrame is as expected")
assert reg_lookup_df.isna().sum().sum() == 0
print("Correct: There are no missing values")
assert reg_lookup_df.Region.duplicated().sum() == 0
print("Correct: All values in the field 'Region' are unique")
```

```python
# Look at first few rows
reg_lookup_df.head()
```

```python
# Merge the Regions lookup and sf tables
reg_combined_df = regs_sf.merge(
    reg_lookup_df.assign(code=lambda x: x.Region.str[1:]),
    how='outer', on=['code']
)
assert reg_combined_df.isna().sum().sum() == 0
print("Correct: Every row has been matched exactly once")
```

```python
# We also see that any differences in `Name` and `nom` are explainable
reg_combined_df.query("Name != nom")[['Region', 'Name', 'nom']]
```

```python
# Data fields available to color the chloropleth
agg_region_df = df_extra.assign(num_of_rows=1).groupby('Region').agg({
    'num_of_rows': 'size',
    'Exp_4dps': 'sum',
    'ClaimNb': 'sum',
    'Density': 'mean',
    'DrivAge': 'mean'
}).reset_index()
agg_region_df.head()
```

```python
# Plot chloropleth
regs_chloro_plt = gv.Polygons(
    reg_combined_df.merge(agg_region_df, on='Region'), 
    vdims=['nom','DrivAge']
)
regs_chloro_plt.opts(
    width=520, height=500, toolbar='above', color_index='DrivAge',
    colorbar=True, tools=['hover']#, aspect='equal'
)
```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>
