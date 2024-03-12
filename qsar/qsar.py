import numpy as np
from sklearn.model_selection import RepeatedKFold, train_test_split
from scipy.stats import pearsonr
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path


# name = 'EGFR'
# name = 'HTR1A'
name = 'S1PR1'
X = np.load('npy/{}_X.npy'.format(name))
y = np.load('npy/{}_y.npy'.format(name))
print(X.shape)
print(y.shape)
Ntree = 500
nBatches = 16
nCpU = 16
pH = 7.4
rkf = RepeatedKFold(n_splits=5, n_repeats=4, random_state=128)  # 20 cross validation


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10)
print(y_train.shape)
print(y_test.shape)


model = lgb.LGBMRegressor(n_estimators=Ntree, n_jobs=nCpU, subsample=0.8, colsample_bytree=0.8,
                                  subsample_freq=3)
model.fit(X_train, y_train)

Y_pred_test = model.predict(X_test)
R_test = pearsonr(y_test, Y_pred_test)[0]
mse_test = mean_squared_error(y_test,Y_pred_test)
rmse_test = np.sqrt(mse_test)
print(mse_test)
print(R_test)

# save the model to disk
model.booster_.save_model('model/{}.txt'.format(name))

# X_rnn = np.load('./npy/{}_RNN_X.npy'.format(name))
# Y_rnn_pred = model.predict(X_rnn)
# np.save('./npy/y/{}_RNN_Y.npy'.format(name),Y_rnn_pred)

# X_trans = np.load('./npy/{}_Transformer_X.npy'.format(name))
# Y_trans_pred = model.predict(X_trans)
# np.save('./npy/y/{}_Transformers_Y.npy'.format(name),Y_trans_pred)

npy_folder = Path('../generated_molecules/npy').resolve()

# X_unconditional = np.load(npy_folder / 'Unconditional_X.npy')
# Y_unconditional_pred = model.predict(X_unconditional)
# np.save(npy_folder / f'{name}_Unconditional_Y.npy', Y_unconditional_pred)

# X_trans = np.load(npy_folder / f'{name}_X.npy')
# Y_trans_pred = model.predict(X_trans)
# np.save(npy_folder / f'{name}_Conditional_Y.npy', Y_trans_pred)

# Assessment for multi-target conditional generation
for agg in ('AVG', 'SUM', 'MAX'):
    # X_multi = np.load(npy_folder / f'EGFR_HTR1A_{agg}_X.npy')
    # Y_multi_pred = model.predict(X_multi)
    # np.save(npy_folder / f'EGFR_HTR1A_{agg}_Y_{name}.npy', Y_multi_pred)

    X_multi = np.load(npy_folder / f'EGFR_S1PR1_{agg}_X.npy')
    Y_multi_pred = model.predict(X_multi)
    np.save(npy_folder / f'EGFR_S1PR1_{agg}_Y_{name}.npy', Y_multi_pred)

    X_multi = np.load(npy_folder / f'HTR1A_S1PR1_{agg}_X.npy')
    Y_multi_pred = model.predict(X_multi)
    np.save(npy_folder / f'HTR1A_S1PR1_{agg}_Y_{name}.npy', Y_multi_pred)

# Assessment for target exclusion
# X_multi = np.load(npy_folder / 'Unconditional-EFGR-DIFF_X.npy')
# Y_multi_pred = model.predict(X_multi)
# np.save(npy_folder / 'Unconditional-EFGR-DIFF_Y.npy', Y_multi_pred)