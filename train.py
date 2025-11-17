import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


class SalesModelTrainer:
    def __init__(self, products_path: str, sales_path: str):

        # file paths for the csv
        self.products_path = products_path
        self.sales_path = sales_path

        # dataframes
        self.products = None
        self.sales = None
        self.df = None

        # models
        self.lr_model = None
        self.rf_model = None
        self.prophet_models = {}

    # load product and sales dataframe from the csv files
    def load_data(self):
        self.products = pd.read_csv(self.products_path)
        self.sales = pd.read_csv(self.sales_path, parse_dates=["date"])

    # basic data cleaning and make the date to datetime for futher manipulation
    def clean_data(self):
        self.sales = self.sales.dropna(subset=["product_id", "count", "unit_price"])
        self.sales = self.sales[self.sales["count"] > 0]
        self.sales["date"] = pd.to_datetime(self.sales["date"])

    # extra features for analysis like 7 days and 30 days trends
    def prepare_features(self):
        
        # daily product sales
        daily = (
            self.sales.groupby(["product_id", "date"])["count"]
            .sum()
            .reset_index()
            .sort_values(["product_id", "date"])
        )

        # 7 days and 30 days trend features
        daily["trend_7d"] = (
            daily.groupby("product_id")["count"]
            .rolling(7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        daily["trend_30d"] = (
            daily.groupby("product_id")["count"]
            .rolling(30, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # extra date features
        daily["day_of_week"] = daily["date"].dt.dayofweek
        daily["month"] = daily["date"].dt.month
        daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)

        # merge metadata
        self.df = daily.merge(
            self.products.rename(columns={"id": "product_id"}),
            on="product_id",
            how="left",
        )

        # product date for further imporvement if needed :)
        launch = (
            self.sales.groupby("product_id")["date"]
            .min()
            .reset_index()
            .rename(columns={"date": "launch_date"})
        )
        self.df = self.df.merge(launch, on="product_id", how="left")
        self.df["product_age_days"] = (
            self.df["date"] - self.df["launch_date"]
        ).dt.days

        # categories
        self.df["category"] = self.df["category"].fillna("unknown")
        self.df = pd.get_dummies(self.df, columns=["category"], prefix="cat")

        # price bucket
        self.df["price_bucket"] = pd.qcut(
            self.df["unit_price"], q=5, duplicates="drop", labels=False
        )
        self.df = pd.get_dummies(self.df, columns=["price_bucket"], prefix="price")

        # remove NaN from trends
        self.df = self.df.dropna(subset=["trend_7d", "trend_30d", "product_age_days"])

    # train model linear regression prediction
    def train_linear_regression(self):
        base_cols = [
            "trend_7d",
            "trend_30d",
            "day_of_week",
            "month",
            "is_weekend",
            "product_age_days",
        ]
        cat_cols = [c for c in self.df.columns if c.startswith(("cat_", "price_"))]
        feature_cols = base_cols + cat_cols

        X = self.df[feature_cols].fillna(0)
        y = self.df["count"]

        # split (80% train)
        split_idx = int(len(self.df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # train
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        pred = lr.predict(X_test)

        # metric
        metrics = {
            "mae": mean_absolute_error(y_test, pred),
            "rmse": np.sqrt(mean_squared_error(y_test, pred)),
            "r2": r2_score(y_test, pred),
        }
        print(
            f"Regression linéaire: MAE={metrics['mae']:.2f} | RMSE={metrics['rmse']:.2f} | R²={metrics['r2']:.3f}"
        )

        self.lr_model = {"model": lr, "metrics": metrics, "features": feature_cols}
        return metrics

    # random forest model training
    def train_random_forest(self):

        base_cols = [
            "trend_7d",
            "trend_30d",
            "day_of_week",
            "month",
            "is_weekend",
            "product_age_days",
        ]
        cat_cols = [c for c in self.df.columns if c.startswith(("cat_", "price_"))]
        feature_cols = base_cols + cat_cols

        X = self.df[feature_cols].fillna(0)
        y = self.df["count"]

        # split (80% train)
        split_idx = int(len(self.df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # traing model
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)

        # metric
        metrics = {
            "mae": mean_absolute_error(y_test, pred),
            "rmse": np.sqrt(mean_squared_error(y_test, pred)),
            "r2": r2_score(y_test, pred),
        }
        print(
            f"Random Forest MAE={metrics['mae']:.2f} | RMSE={metrics['rmse']:.2f} | R²={metrics['r2']:.3f}"
        )

        self.rf_model = {"model": rf, "metrics": metrics, "features": feature_cols}
        return metrics

    # train and create prophet per product
    def train_prophet_per_product(self, min_obs: int = 30):
        prophet_metrics = {"mae": [], "rmse": [], "r2": []}
        count_trained = 0

        for pid in self.df["product_id"].unique():
            prod = self.df[self.df["product_id"] == pid][["date", "count"]].copy()
            
            if len(prod) < min_obs:
                continue

            prod = prod.rename(columns={"date": "ds", "count": "y"})
            
            # temporal split
            split = int(len(prod) * 0.8)
            train_df = prod.iloc[:split]
            test_df = prod.iloc[split:]

            if len(test_df) == 0:
                continue

            try:
                # train prophet model
                m = Prophet(
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                    growth="flat",
                    seasonality_mode="multiplicative",
                )
                m.fit(train_df)

                # prediction
                future = test_df[["ds"]].copy()
                forecast = m.predict(future)
                y_pred = forecast["yhat"].clip(lower=0).values
                y_true = test_df["y"].values

                # metric
                prophet_metrics["mae"].append(mean_absolute_error(y_true, y_pred))
                prophet_metrics["rmse"].append(
                    np.sqrt(mean_squared_error(y_true, y_pred))
                )
                prophet_metrics["r2"].append(r2_score(y_true, y_pred))

                self.prophet_models[pid] = m
                count_trained += 1

            except Exception:
                # Ignorer les produits problématiques
                continue

        if count_trained == 0:
            return {}

        # mean metric
        agg = {
            "mae": np.mean(prophet_metrics["mae"]),
            "rmse": np.mean(prophet_metrics["rmse"]),
            "r2": np.mean(prophet_metrics["r2"]),
            "trained": count_trained,
        }
        print(
            f"Prophet MAE={agg['mae']:.2f} | RMSE={agg['rmse']:.2f} | R²={agg['r2']:.3f}"
        )
        return agg

    # export models to file using pickling and unpickling technique with pickle
    def save_models(self, lr_metrics, rf_metrics, prophet_agg):
        # remove all previous data first
        if os.path.exists("models"):
            print("Nettoyage du dossier models/...")
            import shutil
            shutil.rmtree("models")
        
        os.makedirs("models", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Linear regression model with timestamp
        lr_path = f"models/lr_global_{ts}.pkl"
        with open(lr_path, "wb") as f:
            pickle.dump(self.lr_model, f)
        print(f"{lr_path}")

        # Random Forest model with timestamp
        rf_path = f"models/rf_global_{ts}.pkl"
        with open(rf_path, "wb") as f:
            pickle.dump(self.rf_model, f)
        print(f"{rf_path}")

        # prophet models (per product)
        if self.prophet_models:
            prophet_dir = f"models/prophet_{ts}"
            os.makedirs(prophet_dir, exist_ok=True)
            for pid, model in self.prophet_models.items():
                with open(f"{prophet_dir}/{pid}.pkl", "wb") as f:
                    pickle.dump(model, f)
        
        # cleaned dataset
        dataset_path = f"models/cleaned_dataset_{ts}.csv"
        self.df.to_csv(dataset_path, index=False)
        print(f"Dataset: {dataset_path}")

        # training and analysis report
        report_path = f"models/report_{ts}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("DONNÉES\n")
            f.write("-" * 70 + "\n")
            f.write(f"Produits : {len(self.products)}\n")
            f.write(f"Transactions : {len(self.sales):,}\n")
            f.write(f"Période : {self.sales['date'].min().date()} → {self.sales['date'].max().date()}\n")
            f.write(f"Dataset nettoyé : {len(self.df)} lignes\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("COMPARAISON DES MODÈLES\n")
            f.write("-" * 70 + "\n\n")
            
            f.write(f"{'Modèle':<25} {'MAE':<12} {'RMSE':<12} {'R²':<10}\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Régression Linéaire':<25} {lr_metrics['mae']:<12.2f} {lr_metrics['rmse']:<12.2f} {lr_metrics['r2']:<10.3f}\n")
            f.write(f"{'Random Forest':<25} {rf_metrics['mae']:<12.2f} {rf_metrics['rmse']:<12.2f} {rf_metrics['r2']:<10.3f}\n")
            
            if prophet_agg:
                f.write(f"{'Prophet (moyenne)':<25} {prophet_agg['mae']:<12.2f} {prophet_agg['rmse']:<12.2f} {prophet_agg['r2']:<10.3f}\n")
            
            f.write("\n")
            
            # top model
            best_model = min(
                [("Régression Linéaire", lr_metrics['mae']),
                 ("Random Forest", rf_metrics['mae']),
                 ("Prophet", prophet_agg.get('mae', float('inf')))],
                key=lambda x: x[1]
            )
            f.write(f"Meilleur modèle (MAE): {best_model[0]}\n\n")
            
            if prophet_agg:
                f.write("-" * 70 + "\n")
                f.write("PROPHET (PAR PRODUIT)\n")
                f.write("-" * 70 + "\n")
                f.write(f"Modèles entraînés : {prophet_agg.get('trained', 0)}\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write("FICHIERS GÉNÉRÉS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Modèle LR : lr_global_{ts}.pkl\n")
            f.write(f"Modèle RF : rf_global_{ts}.pkl\n")
            if self.prophet_models:
                f.write(f"Modèles Prophet : prophet_{ts}/ ({len(self.prophet_models)} fichiers)\n")
            f.write(f"Dataset nettoyé : cleaned_dataset_{ts}.csv\n")
            f.write(f"Rapport : report_{ts}.txt\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"Rapport d'analyse: {report_path}")
        return ts

    # training pipeline
    def run(self):
        self.load_data()
        self.clean_data()
        self.prepare_features()
        
        lr_metrics = self.train_linear_regression()
        rf_metrics = self.train_random_forest()
        prophet_agg = self.train_prophet_per_product()
        
        self.save_models(lr_metrics, rf_metrics, prophet_agg)
        
        print("\n" + "=" * 70)
        print("ENTRAÎNEMENT TERMINÉ - 3 MODÈLES")
        print("=" * 70)
        print(f"   • Régression Linéaire (global)")
        print(f"   • Random Forest (global)")
        print(f"   • Prophet ({len(self.prophet_models)} modèles par produit)")
        print("\nLancez maintenant: streamlit run app.py\n")


def main():
    PRODUCTS_PATH = "./data/products.csv"
    SALES_PATH = "./data/sales.csv"

    trainer = SalesModelTrainer(PRODUCTS_PATH, SALES_PATH)
    try:
        trainer.run()
    except Exception as e:
        print(f"\nErreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()