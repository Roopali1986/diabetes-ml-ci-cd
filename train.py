from azureml.core import Workspace, Dataset
from sklearn.linear_model import Ridge
import joblib

ws = Workspace.from_config()
dataset = Dataset.get_by_name(ws, name='Diabetes_1')
df = dataset.to_pandas_dataframe()

X = df.drop("target", axis=1)
y = df["target"]

model = Ridge(alpha=0.5)
model.fit(X, y)
joblib.dump(model, 'outputs/model.pkl')
