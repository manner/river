from anomaly import MStream
from river.stream import iter_pandas
from river import metrics
import pandas as pd

unswnumeric_size = 39
unswcateg_size = 7
feature_types = []
headers = []
feature_i = 0
for i in range(0, unswnumeric_size):
    feature_types.append(0)
    headers.append(str(feature_i))
    feature_i += 1

for i in range(0, unswcateg_size):
    feature_types.append(1)
    headers.append(str(feature_i))
    feature_i += 1

headers.append("timestamp")
dataset = "elec"

def print_progress(sample_id, acc, kappa):
    print(f'Samples processed: {sample_id}')
    print(acc)
    print(kappa)

def evaluate(stream, model, n_wait=1000, verbose=False):
    acc = metrics.Accuracy()
    acc_rolling = metrics.Rolling(metric=metrics.Accuracy(), window_size=n_wait)
    kappa = metrics.CohenKappa()
    kappa_rolling = metrics.Rolling(metric=metrics.CohenKappa(), window_size=n_wait)
    raw_results = []
    model_name = "mstream"
    for i, (x, y) in enumerate(stream):
        # Predict
        print(x)
        y_pred = model.score_one(x)
        # Update metrics and results
        acc.update(y_true=y, y_pred=y_pred)
        acc_rolling.update(y_true=y, y_pred=y_pred)
        kappa.update(y_true=y, y_pred=y_pred)
        kappa_rolling.update(y_true=y, y_pred=y_pred)
        if i % n_wait == 0 and i > 0:
            if verbose:
                print_progress(i, acc, kappa)
            raw_results.append([model_name, i, acc.get(), acc_rolling.get(), kappa.get(), kappa_rolling.get()])
        # Learn (train)
        model.learn_one(x)
    print_progress(i, acc, kappa)
    return pd.DataFrame(raw_results, columns=['model', 'id', 'acc', 'acc_roll', 'kappa', 'kappa_roll'])

# Load the data, will be transformed into a stream later
df_categ = pd.read_csv("./test_data/unswcateg.txt", header=None)
df_numeric = pd.read_csv("./test_data/unswnumeric.txt", header=None)
df_time = pd.read_csv("./test_data/unswtime.txt", header=None)
X = pd.concat([df_numeric, df_categ, df_time], axis=1)
X.columns = headers
Y = pd.read_csv("./test_data/unsw_label.txt")

# Initialize models
mstream = MStream(feature_types)

results = evaluate(stream=iter_pandas(X=X, y=Y), model=mstream)