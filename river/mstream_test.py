from anomaly import MStream
from river.stream import iter_pandas
from river import metrics
import pandas as pd
import time
from river.utils import numpy2dict

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
    print(f'Accuracy: {acc}')
    print(f'Kappa: {kappa}')


def evaluate(f, stream, model, n_wait=1000, verbose=False):
    # acc = metrics.Accuracy()
    # acc_rolling = metrics.Rolling(metric=metrics.Accuracy(), window_size=n_wait)
    # kappa = metrics.CohenKappa()
    # kappa_rolling = metrics.Rolling(metric=metrics.CohenKappa(), window_size=n_wait)
    # raw_results = []

    start = time.time()

    model_name = "mstream"
    for i, (x, y) in enumerate(stream):
        # Predict
        # print('\n')
        # print(i)
        model.learn_one(x)
        y_pred = model.score_one(x)
        # print(y_pred)
        f.write(str(y_pred) + '\n')
        # Update metrics and results
        y_value = list(y.values())[0]
        # acc.update(y_true=y_value, y_pred=y_pred)
        # acc_rolling.update(y_true=y_value, y_pred=y_pred)
        # kappa.update(y_true=y_value, y_pred=y_pred)
        # kappa_rolling.update(y_true=y_value, y_pred=y_pred)
        # if i % n_wait == 0 and i > 0:
        # if verbose:
        # print_progress(i, acc, kappa)
        # raw_results.append([model_name, i, acc.get(), acc_rolling.get(),
        #    kappa.get(), kappa_rolling.get()])
        # Learn (train)
    end = time.time()
    print(end - start)
    # print_progress(i, acc, kappa)
    # return pd.DataFrame(raw_results, columns=['model', 'id', 'acc', 'acc_roll', 'kappa', 'kappa_roll'])


# Load the data, will be transformed into a stream later
df_categ = pd.read_csv("./test_data_small/unswcateg_100000.txt", header=None)
df_numeric = pd.read_csv("./test_data_small/unswnumeric_100000.txt", header=None)
df_time = pd.read_csv("./test_data_small/unswtime_100000.txt", header=None)
X = pd.concat([df_numeric, df_categ, df_time], axis=1)
X.columns = headers
Y = pd.read_csv("./test_data_small/unsw_label_100000.txt", header=None)

# Initialize models
mstream = MStream(feature_types, factor=0.4, timestamp_key='timestamp')
model = mstream

# results = evaluate(f, stream=iter_pandas(X=X, y=Y), model=mstream)
XSS = X.to_dict('records')
YSS = Y.to_dict('records')
# print(XSS)
start = time.time()
results = []
auc = metrics.ROCAUC()
print(YSS)
print(len(YSS))
for i, x in enumerate(XSS):
    # x = numpy2dict(xs)
    # print(xs)
    # print(x)
    y_pred = model.score_one(x)
    model.learn_one(x)
    results.append(y_pred)
    auc = auc.update(YSS[i], y_pred)
end = time.time()
print(end - start)
print(auc)
f = open("scores.txt", 'w')
for result in results:
    f.write(str(result) + '\n')
f.close()
