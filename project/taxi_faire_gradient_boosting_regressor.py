from metaflow import FlowSpec, step, card, conda_base, current, Parameter, Flow, project, retry
from metaflow.cards import Markdown, Table, Image, Artifact

URL = "https://outerbounds-datasets.s3.us-west-2.amazonaws.com/taxi/latest.parquet"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# @trigger(events=["s3"])   # Disable trigger for this project.
@project(name="fullstack")  # <-- Add project
@conda_base(
    libraries={
        "pandas": "1.4.2",
        "pyarrow": "11.0.0",
        "scikit-learn": "1.1.2",
    }
)
class TaxiFarePrediction(FlowSpec):
    data_url = Parameter("data_url", default=URL)

    def transform_features(self, df):
        idx = (df.fare_amount > 0)
        idx &= (df.trip_distance <= 100)
        idx &= (df.trip_distance > 0)
        idx &= (df.tip_amount >= 0)
        df = df[idx]
        return df

    @step
    def start(self):
        import pandas as pd
        df = pd.read_parquet(self.data_url)
        self.df = self.transform_features(df)
        self.X = self.df["trip_distance"].values.reshape(-1, 1)
        self.y = self.df["total_amount"].values
        self.next(self.gradient_boosting_regressor_model)

    @step
    def gradient_boosting_regressor_model(self):
        "Fit a single variable, linear model to the data."
        from sklearn.ensemble import GradientBoostingRegressor
        self.model_type = "GradientBoostingRegressor"
        self.model = GradientBoostingRegressor()
        self.next(self.validate)

    @card(type="corise")
    @retry(times=2) 
    @step
    def validate(self):
        from sklearn.model_selection import cross_val_score
        # Get CV scores
        self.scores = cross_val_score(self.model, self.X, self.y, cv=5)

        # Fit the model with all of the data
        self.model.fit(self.X, self.y)
        self.next(self.end)

    @step
    def end(self):
        print("Success!")


if __name__ == "__main__":
    TaxiFarePrediction()
