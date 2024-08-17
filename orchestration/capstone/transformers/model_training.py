import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df: pd.DataFrame, *args, **kwargs):
    categorical_columns = ['PULocationID', 'DOLocationID']
    target = 'duration'
    
    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(df[categorical_columns].to_dict(orient="records"))
    y_train = df[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    print(f"The intercept is {lr.intercept_}")
    return lr, dv


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
