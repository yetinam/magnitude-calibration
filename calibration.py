import numpy as np
import pandas as pd
import gurobipy as solver
from gurobipy import GRB
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
import matplotlib.pyplot as plt
import xgboost as xgb

D2M = 111194.92664455874 / 1000


def estimate_attenuation_correction(data,
                                    metric,
                                    x_ref=(np.linspace(0, 500, 50), np.linspace(0, 200, 20)),
                                    hypocentral=False,
                                    lambda_r=10e3,
                                    lambda_d=1e2,
                                    knn_correction=False,
                                    n_neighbours=10,
                                    sampling=0.1,
                                    depth_factor=3.0,
                                    lambda_nn=1e-2,
                                    lambda_mw=1e-1,
                                    mw_limits=(5.0, 6.0),
                                    noise_limit=None):
    """
    Estimates the correction functions for a given feature
    :param data: pandas dataframe - for details on the format please consult the jupyter notebook and the example file
    :param metric: name of the column containing the feature to build the scale from
    :param x_ref: 2-tuple consisting of gridpoints for distance and depth (assumes linear spacing of points)
    :param hypocentral: use hypocental instead of epicentral distance
    :param lambda_r: hyperparameter lambda_r
    :param lambda_d: lambda_d
    :param knn_correction: enable source correction (using k nearest neighbors)
    :param n_neighbours: number of neighbors for source correction
    :param sampling: probability of each event to be assigned a source correction term
    :param depth_factor: rescale factor for depth
    :param lambda_nn: hyperparameter lambda_l
    :param lambda_mw: hyperparameter lambda_mw
    :param mw_limits: limits for considering mw for normalization
    :param noise_limit: minimum SNR - requires a column names NOISE_{metric} containing the noise level
    :return: If knn_correction: station correction, distance depth correction, source correction
    :return: If not knn_correction: station correction, distance depth correction
    """
    np.random.seed(42)

    stations = sorted(list(set(data['STATION'].values)))
    m = solver.Model()

    m.setParam('OutputFlag', False)
    # m.setParam('Threads', 64)

    s = {}  # Station corrections
    for station in stations:
        s[station] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f's_{station}')
    m.addConstr(solver.quicksum(s.values()) == 0)  # Set mean of station corrections to zero

    g = []  # Attenuation function
    for dist in x_ref[0]:
        g += [[]]
        for depth in x_ref[1]:
            g[-1] += [m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'g_{dist:.0f}_{depth:.0f}')]

    if knn_correction:
        nn = {}
        nn_vars = {}
        mask = np.random.random(len(data)) < sampling
        for station, data_station in data.iloc[mask].groupby('STATION'):
            tmp_n_neighbours = n_neighbours
            if n_neighbours > len(data_station):
                tmp_n_neighbours = len(data_station) // 2
                print(f'Insufficient data for station {station}. Reducing n_neighbors to {tmp_n_neighbours}.')
            nn[station] = NearestNeighbors(n_neighbors=tmp_n_neighbours)
            X = data_station[['LAT', 'LON', 'DEPTH']].values
            X[:, 0:2] *= D2M
            X[:, 2] *= depth_factor
            nn[station].fit(X)
            nn_vars[station] = m.addVars(len(X), lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'nn_{station}')

    # Difference between measurements and expected values
    eps = []
    eps_mw = []

    l_terms = {}
    for station in stations:
        l_terms[station] = []

    for event, event_df in data.groupby('EVENT'):
        mag = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'm_{event}')
        if mw_limits[0] < event_df.iloc[0]['M_EXT'] < mw_limits[1]:
            eps_mw += [(mag - event_df.iloc[0]['M_EXT']) * (mag - event_df.iloc[0]['M_EXT'])]
        for _, row in event_df.iterrows():
            station = row['STATION']
            dist = row['DIST']
            lat = row['LAT']
            lon = row['LON']
            depth = row['DEPTH']
            val = row[metric]
            if np.isnan(val):
                continue
            if noise_limit is not None:
                noise = row[f'NOISE_{metric}']
                # Second check catches logarithms from missing data
                if val - noise < noise_limit or val < -100:
                    continue
            eps += [m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'eps_{event}_{station}')]
            if hypocentral:
                dist = np.sqrt(dist ** 2 + depth ** 2)
            g_term = linear_grid_combination((dist, depth), x_ref, g)
            if knn_correction:
                neighbours = nn[station].kneighbors([[lat * D2M, lon * D2M, depth * depth_factor]],
                                                    return_distance=False)[0]
                l_term = solver.quicksum([nn_vars[station][neigh] for neigh in neighbours]) / n_neighbours
            else:
                l_term = 0
            l_terms[station] += [l_term]
            m.addConstr(val - mag == g_term + s[station] + l_term + eps[-1])

    obj = solver.QuadExpr()
    obj.addTerms(np.ones(len(eps)), eps, eps)
    obj /= len(eps)

    obj_mw = solver.quicksum(eps_mw)
    obj_mw /= len(eps_mw)
    obj_mw *= lambda_mw

    for station in stations:
        m.addConstr(solver.quicksum(l_terms[station]) == 0)

    # Penalty term with the squares of the second order derivatives
    pen = []
    for j in range(len(g)):
        for i in range(len(g[j])):
            if i == 0 or i == len(g[j]) - 1:
                d2_depth = 0
            else:
                d2_depth = (g[j][i - 1] - 2 * g[j][i] + g[j][i + 1]) / ((x_ref[1][i] - x_ref[1][i - 1]) * (
                        x_ref[1][i + 1] - x_ref[1][i]))  # Second order derivative in depth direction
            if j == 0 or j == len(g) - 1:
                d2_dist = 0
            else:
                d2_dist = (g[j - 1][i] - 2 * g[j][i] + g[j + 1][i]) / ((x_ref[0][j] - x_ref[0][j - 1]) * (
                        x_ref[0][j + 1] - x_ref[0][j]))  # Second order derivative in distance direction
            pen += [lambda_r * d2_depth * d2_depth + lambda_d * d2_dist * d2_dist]
    pen = solver.quicksum(pen)
    pen /= len(x_ref[0]) * len(x_ref[1])

    # Penalty term with L2 norm of l term
    terms = []
    if knn_correction:
        for station in stations:
            terms += nn_vars[station].values()
    if knn_correction:
        pen_l = solver.QuadExpr()
        pen_l.addTerms(np.ones(len(terms)), terms, terms)
        pen_l /= len(terms)
        pen_l *= lambda_nn
    else:
        pen_l = 0

    m.setObjective(obj + obj_mw + pen + pen_l, solver.GRB.MINIMIZE)

    print('Optimizing...')
    m.optimize()

    # print(f'Objective terms: {len(eps)}\nMw terms: {len(eps_mw)}')
    # print(f'Objective:\t{obj.getValue()}\tPenalty:\t{pen.getValue()}')

    for station in s.keys():
        s[station] = s[station].X
    for i in range(len(g)):
        for j in range(len(g[i])):
            g[i][j] = g[i][j].X

    if knn_correction:
        for station in stations:
            for key in nn_vars[station]:
                nn_vars[station][key] = nn_vars[station][key].X
            nn_vars[station] = dict(nn_vars[station])
        return s, np.array(g), (nn, nn_vars)
    else:
        return s, np.array(g)


def linear_grid_combination(p, x_ref, g, linspaced=True):
    """
    Helper function to calculate bilinear interpolations on the grid
    Linspaced improves performance by assuming the grid is linearly spaced. Disable for non-linear spaced grids.
    """
    if len(p) == 0:
        return g
    tmp_p = p[0]
    tmp_x_ref = x_ref[0]

    if tmp_p <= tmp_x_ref[0]:
        return linear_grid_combination(p[1:], x_ref[1:], g[0], linspaced)

    if tmp_p >= tmp_x_ref[-1]:
        return linear_grid_combination(p[1:], x_ref[1:], g[-1], linspaced)

    if linspaced:
        inc = tmp_x_ref[1] - tmp_x_ref[0]
        idx = min(int((tmp_p - tmp_x_ref[0]) / inc), len(tmp_x_ref) - 1)
    else:
        idx = int(np.max(np.where(tmp_x_ref <= tmp_p)))

    alpha = (tmp_p - tmp_x_ref[idx]) / (tmp_x_ref[idx + 1] - tmp_x_ref[idx])
    return alpha * linear_grid_combination(p[1:], x_ref[1:], g[idx + 1], linspaced) + \
           (1 - alpha) * linear_grid_combination(p[1:], x_ref[1:], g[idx], linspaced)


def add_prediction(data,
                   metric,
                   s,
                   g,
                   nn=None,
                   x_ref=None,
                   hypocentral=False,
                   depth_factor=3.0,
                   noise_limit=None):
    """
    Adds predictions to a dataframe
    For parameter specification consult calculate_attenuation_correction
    """
    pred = []
    for _, line in data.iterrows():
        if noise_limit is not None:
            if line[metric] - line[f'NOISE_{metric}'] < noise_limit or line[metric] < -100:
                pred += [np.nan]
                continue
        if hypocentral:
            dist = np.sqrt(line['DIST'] ** 2 + (line['DEPTH']) ** 2)
        else:
            dist = line['DIST']
        g_term = linear_grid_combination((dist, line['DEPTH']), x_ref, g)
        if nn:
            station = line['STATION']
            lat = line['LAT']
            lon = line['LON']
            depth = line['DEPTH']
            if station in nn[0]:
                neighbours = nn[0][station].kneighbors([[lat * D2M, lon * D2M, depth * depth_factor]],
                                                       return_distance=False)[0]
                l_term = np.mean([nn[1][station][neigh] for neigh in neighbours])
            else:
                l_term = 0
        else:
            l_term = 0

        if line['STATION'] in s:
            s_term = s[line['STATION']]
        else:
            s_term = 0
        pred += [line[metric] - s_term - g_term - l_term]

    data.loc[:, f'PRED_{metric}'] = pred


def calc_means(data, metric):
    """
    Calculates mean magnitude for each event and residuals for each measurement
    """
    metric = f'PRED_{metric}'
    mean = {event: val[metric] for event, val in data.groupby('EVENT').mean().iterrows()}
    res = [line[metric] - mean[line['EVENT']] for _, line in data.iterrows()]
    means = [mean[line['EVENT']] for _, line in data.iterrows()]
    data[f'RESIDUAL_{metric}'] = res
    data[f'MEAN_{metric}'] = means


def distance_depth_correction(x_ref, g, vmin=None, vmax=None, ax=None, cmap='viridis'):
    """
    Plot distance and depth correction
    :param x_ref: See estimate_attenuation_correction
    :param g: See estimate_attenuation_correction
    :param vmin: Minimum value for plot
    :param vmax: Maximum value for plot
    :param ax: matplotlib axis
    :param cmap: Colormap
    :return: Colorbar to be added to figure
    """
    if not ax:
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111)

    if vmin is None:
        vmin = np.min(g)
    if vmax is None:
        vmax = np.max(g)
    x, y = np.meshgrid(*x_ref)
    levels = np.linspace(vmin, vmax, 50)
    cb = ax.contourf(x / 1000, y, np.array(g).T, levels=levels, cmap=cmap)
    ax.invert_yaxis()
    ax.set_ylabel('Depth in km')
    ax.set_xlabel('Distance in km')

    return cb


def remove_outliers(data, metric, max_dev=2.0):
    """
    Removes outliers from the data set
    :param data: Data set
    :param metric: Name of feature column
    :param max_dev: Detection threshold for outlier in terms of number of STDs
    :return: Data without outliers
    """
    rmse = np.sqrt(np.mean(data[f'RESIDUAL_{metric}'] ** 2))
    outliers = (np.abs(data[f'RESIDUAL_{metric}'].values) > (max_dev * rmse))
    outliers[np.isnan(outliers)] = True
    for ind in data.groupby('EVENT').indices.values():
        if sum(~outliers[ind]) < 3:
            outliers[ind] = False
    print(f'Found {100*sum(outliers)/len(data):.2f}% outliers')
    return data.loc[~outliers]


def create_boosting_scale(data, metric, epochs=100):
    """
    Creates a homogeneous boosting tree scale by combining three boosting trees with different training sets
    :param data: Data set
    :param metric: Name of feature column
    :param epochs: Number of epochs for the boosting tree
    :return: Data set with new BOOST column
    """
    events = data['EVENT'].unique()
    target = f'MEAN_PRED_{metric}'
    rand = np.random.random(len(events))

    ev1 = events[rand < 1 / 3]
    ev2 = events[np.logical_and(1 / 3 < rand, rand < 2 / 3)]
    ev3 = events[2 / 3 < rand]

    df1 = data.loc[data['EVENT'].isin(ev1)]
    df2 = data.loc[data['EVENT'].isin(ev2)]
    df3 = data.loc[data['EVENT'].isin(ev3)]

    keys = [x for x in data.columns if x[:5] == 'PRED_']
    _, pred1, _, _ = predict_ml_boost((pd.concat((df2, df3)), df1, None), keys, target, depth=11,
                                               epochs=epochs, splitted=True, verbosity=0)
    _, pred2, _, _ = predict_ml_boost((pd.concat((df1, df3)), df2, None), keys, target, depth=11,
                                               epochs=epochs, splitted=True, verbosity=0)
    _, pred3, _, _ = predict_ml_boost((pd.concat((df1, df2)), df3, None), keys, target, depth=11,
                                               epochs=epochs, splitted=True, verbosity=0)

    new_preds = np.zeros(len(data))
    new_preds[data['EVENT'].isin(ev1)] = pred1
    new_preds[data['EVENT'].isin(ev2)] = pred2
    new_preds[data['EVENT'].isin(ev3)] = pred3
    data[f'PRED_BOOST_{metric}'] = new_preds
    calc_means(data=data, metric=f'BOOST_{metric}')

    return data


def predict_ml_boost(data, keys, target, depth=12, epochs=100, gpu=False, splitted=False, verbosity=1):
    """
    Trains a boosting tree scale on the test set
    :param data: Data set
    :param keys: Keys to use for regression
    :param target: Gold standard magnitude
    :param depth: Depth of the boosting tree
    :param epochs: Number of epochs for the boosting tree
    :param gpu: Use GPU
    :param splitted: Enable if data is given as a tuple of tree data sets for training, dev and test
    :param verbosity: 1 - verbose output    0 - silent
    :return:
    """
    if splitted:
        df_train = data[0]
        df_dev = data[1]
        df_test = data[2]
    else:
        data = data[~np.isnan(data[target])]
        df_train = data[data['SPLIT'] == 'TRAIN']
        df_dev = data[data['SPLIT'] == 'DEV']
        df_test = data[data['SPLIT'] == 'TEST']

    d_train = xgb.DMatrix(df_train[keys].values, df_train[target].values)
    d_dev = xgb.DMatrix(df_dev[keys].values, df_dev[target].values)

    if df_test is None:
        d_test = None
    else:
        d_test = xgb.DMatrix(df_test[keys].values, df_test[target].values)

    if gpu:
        tree_method = 'gpu_hist'
    else:
        tree_method = 'auto'
    param = {'max_depth': depth, 'eta': 0.1, 'objective': 'reg:linear', 'eval_metric': 'rmse',
             'tree_method': tree_method, 'base_score': 0, 'silent': 1 - verbosity}
    num_round = epochs

    bst = xgb.train(param, d_train, num_round, evals=[(d_train, 'Train'), (d_dev, 'Dev')], verbose_eval=100)

    if d_test is None:
        return bst.predict(d_train), bst.predict(d_dev), None, bst
    else:
        return bst.predict(d_train), bst.predict(d_dev), bst.predict(d_test), bst