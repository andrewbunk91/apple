"""Train an LSTM model to predict stock prices from historical CSV data."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
except ImportError as exc:  # pragma: no cover - import guarded for environments without TF
    raise ImportError(
        "TensorFlow is required to run this script. Install it with `pip install tensorflow`."
    ) from exc


SequenceData = Tuple[np.ndarray, np.ndarray]
PreparedData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, pd.Index]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the training script."""

    parser = argparse.ArgumentParser(
        description="Train an LSTM model on stock price data and generate predictions.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("apple_stock_data_2010.csv"),
        help="CSV file containing the historical stock data (default: %(default)s)",
    )
    parser.add_argument(
        "--column",
        default="Adj Close",
        help="Price column to model (default: %(default)s)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Number of past observations included in each training sample (default: %(default)s)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of the dataset used for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducibility (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to store a CSV with predictions and actual prices.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print Keras training progress.",
    )

    return parser.parse_args()


def load_price_series(csv_path: Path, column: str) -> pd.Series:
    """Load a price series from *csv_path* using the specified *column*."""

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file '{csv_path}' was not found.")

    data_frame = pd.read_csv(csv_path)

    if "Date" in data_frame.columns:
        data_frame["Date"] = pd.to_datetime(data_frame["Date"], errors="coerce")
        data_frame = data_frame.set_index("Date").sort_index()

    if column not in data_frame.columns:
        available = ", ".join(sorted(data_frame.columns))
        raise KeyError(
            f"Column '{column}' was not found in the CSV file. Available columns: {available}."
        )

    series = data_frame[column].astype(float)

    if series.isna().any():
        raise ValueError("The selected column contains missing values. Clean the data first.")

    return series


def create_sequences(dataset: np.ndarray, window: int) -> SequenceData:
    """Create input/output sequences for LSTM training from *dataset*."""

    features, targets = [], []

    for index in range(window, len(dataset)):
        features.append(dataset[index - window : index, 0])
        targets.append(dataset[index, 0])

    if not features:
        raise ValueError(
            "Not enough observations to create sequences. Try reducing the window size."
        )

    return np.array(features), np.array(targets)


def prepare_datasets(
    series: pd.Series,
    window: int,
    train_split: float,
) -> PreparedData:
    """Scale *series* and create training and test datasets for the LSTM model."""

    if not 0 < train_split < 1:
        raise ValueError("train_split must be within the interval (0, 1).")

    scaled_values = series.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(scaled_values)

    train_size = int(len(scaled_data) * train_split)
    if train_size <= window:
        raise ValueError(
            "The training portion is too small for the requested window size. "
            "Adjust --train-split or --window."
        )

    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - window :]

    x_train, y_train = create_sequences(train_data, window)
    x_test, y_test = create_sequences(test_data, window)

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    test_index = series.index[train_size:]

    return x_train, y_train, x_test, y_test, scaler, test_index


def build_lstm_model(window: int) -> Sequential:
    """Construct a simple LSTM network for univariate time-series forecasting."""

    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(window, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model


def train_model(
    model: Sequential,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    verbose: bool,
) -> tf.keras.callbacks.History:
    """Train *model* on the provided training dataset."""

    validation_split = 0.1 if len(x_train) >= 10 else 0.0

    callbacks = [
        EarlyStopping(
            monitor="val_loss" if validation_split > 0 else "loss",
            patience=5,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1 if verbose else 0,
        callbacks=callbacks,
        shuffle=False,
    )

    return history


def evaluate_model(
    model: Sequential,
    x_test: np.ndarray,
    y_test: np.ndarray,
    scaler: MinMaxScaler,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Generate predictions with *model* and compute evaluation metrics."""

    predicted_scaled = model.predict(x_test, verbose=0)
    predicted = scaler.inverse_transform(predicted_scaled).flatten()

    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = float(np.sqrt(np.mean(np.square(predicted - actual))))
    mae = float(np.mean(np.abs(predicted - actual)))

    return predicted, actual, rmse, mae


def build_results_frame(
    index: Iterable[pd.Timestamp],
    actual: np.ndarray,
    predicted: np.ndarray,
) -> pd.DataFrame:
    """Create a DataFrame comparing *actual* prices to *predicted* ones."""

    results_index = pd.Index(index, name="Date")
    return pd.DataFrame({"actual": actual, "predicted": predicted}, index=results_index)


def main() -> None:
    """Train the LSTM model using the provided arguments and report metrics."""

    args = parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    price_series = load_price_series(args.csv, args.column)
    x_train, y_train, x_test, y_test, scaler, test_index = prepare_datasets(
        price_series,
        args.window,
        args.train_split,
    )

    model = build_lstm_model(args.window)
    train_model(
        model,
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    predicted, actual, rmse, mae = evaluate_model(model, x_test, y_test, scaler)
    results = build_results_frame(test_index, actual, predicted)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print("Last 5 predictions:")
    print(results.tail())

    if args.output is not None:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path)
        print(f"Saved predictions to {output_path.resolve()}")


if __name__ == "__main__":
    main()
