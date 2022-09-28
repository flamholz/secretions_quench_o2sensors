import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import uncertainties

from matplotlib import pyplot as plt
from os import path
from uncertainties.unumpy import uarray

# In a typical experiment we add 11 microliter sensor to 100 ul sample
# to get 111, then removed 11. Hence this dilution factor.
DEFAULT_DILUTION_FACTOR = (1.0-11/111.0)


def _calc_blank_by_date(pre_df, colname):
    mask = pre_df.name == 'BLANK'
    blank_mean = pre_df[mask].groupby('date').mean()
    blank_std = pre_df[mask].groupby('date').std()
    
    blanks_vals = uarray(blank_mean[colname].values,
                         blank_std[colname].values)
    
    return pd.Series(blanks_vals, blank_mean.index)


def _calc_unquenched_by_date(post_df, colname, sensor_name):
    mask = post_df.name == sensor_name
    uq_mean = post_df[mask].groupby('date').mean()
    uq_std = post_df[mask].groupby('date').std()
    
    uq_vals = uarray(uq_mean[colname].values,
                     uq_std[colname].values)
    return pd.Series(uq_vals, uq_mean.index)


def calc_quenched_flour(pre_df, post_df, fluor_col, sensor_name,
                        dilution_factor=DEFAULT_DILUTION_FACTOR):
    """Infer the fluorescence of a sensor/fluorophore in the presence of quencher.
    
    Uses measurements of quencher fluorescence prior to addition of sensor (pre_df) and
    after (post_df) to infer the signal in the presence of quencher. The pre_df contains
    blank wells labelled BLANK and the post_df contains reference measurements of 
    the unquenched sensor in buffer. 
    
    Args:
        pre_df: long-form DF with fluorescence prior to addition of sensor.
        post_df: long-form DF with fluorescence after addition of sensor.
        colname: the name of the column containing fluorescence data.
        sensor_name: the value of the "name" column for sensor in buffer data.
        dilution_factor: multiplicative change in concentration upon sensor addition. 
    """
    # Calculate the blanks and reference values per-date in case there's some systematic
    # variation. This only matters in one case where we combine measurements from > 1 day. 
    blank_vals = _calc_blank_by_date(pre_df, fluor_col)
    unquenched_vals = _calc_unquenched_by_date(post_df, fluor_col, sensor_name)
    
    # Blank both pre- and post-data
    blanked_fluor_col = "blanked_{0}".format(fluor_col)
    pre_df[blanked_fluor_col] = pre_df[fluor_col].values - blank_vals.loc[pre_df.date].values
    post_df[blanked_fluor_col] = post_df[fluor_col].values - blank_vals.loc[post_df.date].values

    # Save the fluorescence expected for each well based on the blank and the
    # measurement for that well prior to adding the dye.
    expected_fluor_col = "expected_blanked_{0}".format(fluor_col)
    post_df[expected_fluor_col] = (
        dilution_factor*pre_df[blanked_fluor_col] + unquenched_vals.loc[pre_df.date].values)

    # We want to calculate F0/F, i.e. the ratio ratio of fluorescence intensity without quencher
    # to intensity with quencher. The problem is that the quencher might fluoresce in the same channel
    # as the sensor (e.g. RTDP) so we need to subtract off the flourescence measured before RTDP was added
    # after correcting for the small addition of volume (1 ul in 100)
    estimated_fluor_col = "estimated_true_{0}".format(fluor_col)
    post_df[estimated_fluor_col] = (
        post_df[blanked_fluor_col] - dilution_factor*pre_df[blanked_fluor_col])

    # We can now calculate the F0/F ratio by dividing
    # This is useful for fitting to Stern-Vollmer type models.
    post_df['F0_F_ratio'] = unquenched_vals.loc[post_df.date].values / post_df[estimated_fluor_col]
    post_df['F_F0_ratio'] = post_df[estimated_fluor_col] / unquenched_vals.loc[post_df.date].values
    
    # Make new pre- and post-DFs with uncertainties in a separate column.
    # Makes all the subsequent operations play nicer with Pandas and Statsmodels.
    out_pre_df = pre_df.copy()
    ucol = 'u_{0}'.format(blanked_fluor_col)
    out_pre_df[ucol] = [v.std_dev for v in pre_df[blanked_fluor_col]]
    out_pre_df[blanked_fluor_col] = [v.nominal_value for v in pre_df[blanked_fluor_col]]
    
    out_post_df = post_df.copy()
    uncertain_cols = [
        blanked_fluor_col, expected_fluor_col,
        estimated_fluor_col, 'F0_F_ratio', 'F_F0_ratio']
    for col in uncertain_cols:
        ucol = 'u_{0}'.format(col)
        out_post_df[ucol] = [v.std_dev for v in post_df[col]]
        out_post_df[col] = [v.nominal_value for v in post_df[col]]
    
    return out_pre_df, out_post_df, blank_vals, unquenched_vals


def well_mean_with_error_propagation(my_df, fluor_col, unquenched_val):
    """Takes by-well mean of numerical values in a long-form DF and propoagate error.
    
    Wells are identified by grouping with [name, well, plate, date].
    
    Individual values have associated error due to variation in the blanks, for example.
    We want to preserve this error while also recording the variation in the mean. Assuming
    uncorrelated errors, the SDs add in quadrature. 
    
    Args:
        my_df: a long-form dataframe. uncertainty cols have u_ prefix.
        unquenched_val: ufloat of the unquenched fluorescence of the sensor. 
    
    Returns:
        A new dataframe with means and errors appropriately propagated.
    """
    cols2group = 'name,well,plate,date'.split(',')
    means = my_df.groupby(cols2group).mean().drop('time_s', axis=1)
    stds = my_df.groupby(cols2group).std().drop('time_s', axis=1)
    
    blanked_fluor_col = "blanked_{0}".format(fluor_col)
    expected_fluor_col = "expected_blanked_{0}".format(fluor_col)
    estimated_fluor_col = "estimated_true_{0}".format(fluor_col)
    
    # For the following columns, the corresponding u_ values are identical for replicates
    # of the same well because they derive from additive error propagation from blanks and other 
    # reference measurements. So we combine the SD of measurements with a single estimate
    # of measurement error (since all the values are identical, the mean equals those values).
    cols = [blanked_fluor_col, expected_fluor_col, estimated_fluor_col]
    for c in cols:
        if c not in means.columns:
            continue
            
        uc = 'u_{0}'.format(c)
        means[uc] = np.sqrt(np.power(stds[c], 2) + np.power(means[uc], 2))
        
    # Return now if ratio data not in this DF, e.g. if it's a pre-sensor-addition DF.
    if 'F0_F_ratio' not in means.columns:
        return means
    
    # These ratio columns have error estimates that depend on the scale of the values measured.
    # This is due to propagation of error in division/multiplication. We will just re-calculate
    # the error of these ratios using the combined error calculated above. 
    true_fluor_uarray = uarray(means[estimated_fluor_col],
                               means["u_{0}".format(estimated_fluor_col)])
    if isinstance(unquenched_val, pd.Series):
        dates = means.reset_index().date.values
        uq_vals = unquenched_val.loc[dates].values
    else:
        n = means[estimated_fluor_col].size
        uq_vals = uarray([unquenched_val.nominal_value]*n, [unquenched_val.std_dev]*n)
    
    f0_f = uq_vals/true_fluor_uarray    
    means['F0_F_ratio'] = [v.nominal_value for v in f0_f]
    means['u_F0_F_ratio'] = [v.std_dev for v in f0_f]
    
    f_f0 = true_fluor_uarray/uq_vals
    means['F_F0_ratio'] = [v.nominal_value for v in f_f0]
    means['u_F_F0_ratio'] = [v.std_dev for v in f_f0]
    
    return means


def fit_quenching_linear_unitless(post_df, names):
    """Fits the inferred quenched fluorescence ratios to SV model. 
    
    See Gehlen J. Photochem. Photobio. 2020 for summary and Szabo J Phys Chem 1989
    for derivations of the linear and higher-order models.
    
    Args:
        post_df: a data-frame containing the post-sensor-addition measurements and
            inferred fluorescence ratios as processed by above methods. 
        names: the names of the quenchers to calculate KSV for. 
    """
    quencher_dict = {
        'name':[],
        'K_SV': [],
        'K_SV err': [],
        'R': [],
        'N_concs': [],
    }
    for qname in names:
        print('Fitting {0}'.format(qname))
        mask = post_df.name == qname
        qdf = post_df[mask]

        corr = qdf.corr().loc['concentration', 'F0_F_ratio']
        quencher_dict['R'].append(corr)
        print('\tF0/F ~ [{0}] with R = {1:.3f}'.format(qname, corr))
        n_concs = qdf.concentration.unique().size
        print('\tData for {0} concentrations'.format(n_concs))
        quencher_dict['N_concs'].append(n_concs)

        # Do a linear fit to the canonical SV model and record the fit slope.
        concs = qdf.concentration.values.copy()
        X1 = concs.T
        Y = qdf.F0_F_ratio - 1
        
        # Fit with weights if we have measurement error (SD)
        if 'u_F0_F_ratio' in qdf.columns:
            # WLS is a weighted least-squares, assumes weights are proportional to 1/variance.
            # Since we have SD of measurements, we take w = 1/SD^2 and set fixed-scale. 
            w = 1/np.power(qdf.u_F0_F_ratio, 2)
            res1 = sm.WLS(Y, X1, weights=w).fit(cov_type='fixed scale')
        else:
            # Otherwise ordinary least-squares
            res1 = sm.OLS(Y, X1).fit()

        quencher_dict['name'].append(qname)
        quencher_dict['K_SV'].append(res1.params[0])

        # sm.OLS.fit() gives symmetric 95% CIs by default.
        CIs = res1.conf_int()
        quencher_dict['K_SV err'].append(res1.params[0] - CIs.loc['x1', 0])
        
    return pd.DataFrame(quencher_dict)


def fit_quenching_linear(post_df, names):
    """Fits the Stern-Volmer coefficient KSV for the passed names.
    
    Uses a linear model, i.e. without correction for higher order effects.
    Assumes that the input concentrations are in micromolar units, as they 
    are for the pure molecule quencher experiments but not the supernatant exps.
    
    Args:
        post_df: the DataFrame with normalized fluorescence data F0_F_ratio.
        names: the names to fit KSV for.  
        
    Returns:
        A DataFrame with all the fits. 
    """
    qdf = fit_quenching_linear_unitless(post_df, names)
    
    # Convert to /M units knowing that the concentrations are uM. 
    qdf['K_SV (/M)'] = qdf['K_SV'] * 1e6
    qdf['K_SV err (/M)'] = qdf['K_SV err'] * 1e6
    
    return qdf


def mark_non_quenchers(qdf, min_corr=0.4, min_KSV=None, min_effect_size=None):
    """Mark which quenchers are real in a DF of fit KSV values.
    
    Call something a non-quencher if the lower end of the credible interval is < min_KSV 
    or the Pearson correlation of F0/F with concentration is < min_corr. 
    
    Args:
        qdf: DataFrame of fit KSVs produced by one of the functions above. 
        min_corr: lowest correlation between normalized fluorescence and
            concentration to call a molecule a quencher.
        min_KSV: lowest fit KSV value to call a molecule a quencher.
        min_effect_size: lowest effect size to call a quencher.
    """
    large_effect = np.ones(qdf.index.size)
    if min_effect_size:
        # If unitless, we calculate the effect size
        lb = (qdf['K_SV'] - qdf['K_SV err'])
        effect_size = 1-(1/(1+lb))
        large_effect = effect_size > min_effect_size
        
    # If units, we compare the KSV to the lower bound
    large_KSV = np.ones(qdf.index.size)
    if min_KSV is not None:
        lb = (qdf['K_SV (/M)'] - qdf['K_SV err (/M)'])
        large_KSV = lb > min_KSV
        
    high_corr = qdf['R'].abs() > min_corr
    qdf['quencher'] = np.logical_and(
        np.logical_and(large_effect, large_KSV), high_corr)
    return qdf
    