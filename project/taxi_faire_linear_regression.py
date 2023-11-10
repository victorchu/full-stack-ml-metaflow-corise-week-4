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
        from sklearn.model_selection import train_test_split
        df = pd.read_parquet(self.data_url)
        self.df = self.transform_features(df)
        self.X = self.df["trip_distance"].values.reshape(-1, 1)
        self.y = self.df["total_amount"].values
        self.next(self.linear_model)

    @step
    def linear_model(self):
        "Fit a single variable, linear model to the data."
        from sklearn.linear_model import LinearRegression
        self.model_type = "LinearRegression"
        self.model = LinearRegression()
        self.next(self.validate)

    @card(type="corise")
    @retry(times=2) 
    @step
    def validate(self):
        from sklearn.model_selection import cross_val_score
        # Get CV scores
        self.scores = cross_val_score(self.model, self.X, self.y, cv=5)
        # We still need to fit the model
        self.model.fit(self.X, self.y)

        # Cards
        current.card.append(Markdown("# Taxi Fare Prediction Results"))
        current.card.append(Artifact(self.model_type, name="model_type"))
        current.card.append(Artifact(self.model, name="model"))
        current.card.append(Artifact(self.scores, name="CV scores"))

        self.next(self.end)

    # @card
    @step
    def end(self):
        print("Success!")


if __name__ == "__main__":
    TaxiFarePrediction()
