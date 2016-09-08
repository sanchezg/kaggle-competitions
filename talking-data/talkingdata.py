import os
import numpy as np
import pandas as pd
from datetime import datetime
from slugify import slugify
from sklearn.cross_validation import StratifiedKFold
from models import PhoneBrandEstimator


def run(model, dataset, target_column='group', valid=None, dump_path=None):
    """
    Run the indicated model using the data passed by arguments. Returned
    results might contain the predictions for the validation dataset if is
    provided.
    """
    predictions = []
    pvalids = []
    actuals = []
    devices = []

    date_slug = datetime.now().strftime('%Y%m%d%H%M%S')
    kf = StratifiedKFold(
        dataset[target_column], n_folds=4, shuffle=True, random_state=42)
    columns = [c for c in dataset.columns if not c == target_column]

    for fold_number, (train_index, test_index) in enumerate(kf):
        X_train = dataset.loc[train_index, columns]
        X_test = dataset.loc[test_index, columns]
        y_train = dataset.loc[train_index, target_column]
        y_test = dataset.loc[test_index, target_column]
        m = model()
        m.fit(X_train, y_train)
        y_hat = m.predict_proba(X_test)
        predictions.append(y_hat)
        actuals.append(y_test.values)
        devices.append(X_test['device_id'].values)

        if valid is not None:
            pvalid = m.predict_proba(valid)
            pvalids.append(pvalid)

        if dump_path is not None:
            model_name = '{}_{}_fold{}.pickle'.format(date_slug,
                                                      slugify(m.name),
                                                      fold_number)
            m.dump(os.path.join(dump_path, model_name))

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    out = pd.DataFrame(predictions, columns=m.get_classes())
    out.insert(0, 'device_id', dataset['device_id'])
    out.insert(1, 'actual', actuals)
    if pvalids != []:
        return out, np.concatenate(pvalids)
    return out


if __name__ == '__main__':
    print("Please don't call this module directly")
