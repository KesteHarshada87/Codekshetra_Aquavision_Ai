import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib


def build_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Create a supervised learning dataset from the clean_dwlr time series.

    Each row uses the previous 4 observations of currentlevel for a station
    plus the current month/year and station id to predict the currentlevel.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.sort_values(["station_name", "date"])

    feature_rows = []
    target = []

    for station, g in df.groupby("station_name"):
        g = g.dropna(subset=["currentlevel"]).reset_index(drop=True)
        levels = g["currentlevel"].astype(float).values
        months = g["date"].dt.month.values
        years = g["date"].dt.year.values

        # For each time step t, use t-4..t-1 levels to predict level at t
        for t in range(4, len(g)):
            last4 = levels[t - 4 : t]
            feature_rows.append(
                {
                    "l1": last4[-1],
                    "l2": last4[-2],
                    "l3": last4[-3],
                    "l4": last4[-4],
                    "month": months[t],
                    "year": years[t],
                    "station_name": station,
                }
            )
            target.append(levels[t])

    X = pd.DataFrame(feature_rows)
    y = np.array(target, dtype=float)
    return X, y


def main():
    print("Loading clean_dwlr.csv ...")
    df = pd.read_csv("clean_dwlr.csv")

    print("Building training frame ...")
    X, y = build_training_frame(df)

    print(f"Training samples: {len(X)}")
    if len(X) == 0:
        raise RuntimeError("No training samples could be constructed from clean_dwlr.csv")

    # Encode station names
    encoder = LabelEncoder()
    X["station_encoded"] = encoder.fit_transform(X["station_name"])
    X_model = X[["l1", "l2", "l3", "l4", "month", "year", "station_encoded"]].values

    X_train, X_val, y_train, y_val = train_test_split(
        X_model, y, test_size=0.2, random_state=42
    )

    print("Training RandomForestRegressor ...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    val_score = model.score(X_val, y_val)
    print(f"Validation R^2: {val_score:.3f}")

    print("Saving model to forecast_model.pkl ...")
    joblib.dump(model, "forecast_model.pkl")

    print("Saving station encoder to station_encoder.pkl ...")
    joblib.dump(encoder, "station_encoder.pkl")

    print("Done.")


if __name__ == "__main__":
    main()

