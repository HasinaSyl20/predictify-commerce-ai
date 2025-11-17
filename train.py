import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')


class SalesModelTrainer:
    def __init__(self, products_path, sales_path):
        self.products_path = products_path
        self.sales_path = sales_path
        self.sales = None
        self.products = None
        self.df = None
        
    def load_data(self):
        """Charger les données"""
        print("📥 Chargement des données...")
        self.products = pd.read_csv(self.products_path)
        self.sales = pd.read_csv(self.sales_path, parse_dates=["date"])
        print(f"   ✓ {len(self.products)} produits")
        print(f"   ✓ {len(self.sales)} transactions")
        
    def clean_data(self):
        """Nettoyer les données"""
        print("\n🧹 Nettoyage...")
        initial = len(self.sales)
        
        self.sales = self.sales.dropna(subset=["product_id", "count", "unit_price"])
        self.sales = self.sales[self.sales["count"] > 0]
        self.sales["date"] = pd.to_datetime(self.sales["date"])
        
        print(f"   ✓ {initial - len(self.sales)} lignes supprimées")
        
    def prepare_features(self):
        """Préparer les features pour l'entraînement"""
        print("\n🔧 Préparation des features...")
        
        # Agrégation par produit et date
        daily = (
            self.sales.groupby(["product_id", "date"])["count"]
            .sum()
            .reset_index()
        )
        
        # Calculer les tendances par produit
        daily = daily.sort_values(["product_id", "date"])
        
        # Tendances sur différentes fenêtres
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
        
        # Features temporelles
        daily["day_of_week"] = daily["date"].dt.dayofweek
        daily["month"] = daily["date"].dt.month
        daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
        
        # Fusionner avec produits
        self.df = daily.merge(self.products, left_on="product_id", right_on="id", how="left")
        
        # Supprimer les lignes avec tendances NaN au début
        self.df = self.df.dropna(subset=["trend_7d", "trend_30d"])
        
        print(f"   ✓ {len(self.df)} lignes préparées")
        
    def train_models(self):
        """Entraîner les 3 modèles"""
        print("\n🚀 Entraînement des modèles...")
        
        # Features pour ML
        feature_cols = ["trend_7d", "trend_30d", "day_of_week", "month", "is_weekend"]
        X = self.df[feature_cols].fillna(0)
        y = self.df["count"]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        results = {}
        
        # 1. Régression Linéaire
        print("   [1/3] Régression Linéaire...", end=" ")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        
        results["linear_regression"] = {
            "model": lr_model,
            "mae": mean_absolute_error(y_test, lr_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, lr_pred)),
            "r2": r2_score(y_test, lr_pred)
        }
        print("✓")
        
        # 2. Random Forest
        print("   [2/3] Random Forest...", end=" ")
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        results["random_forest"] = {
            "model": rf_model,
            "mae": mean_absolute_error(y_test, rf_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, rf_pred)),
            "r2": r2_score(y_test, rf_pred),
            "feature_importance": dict(zip(feature_cols, rf_model.feature_importances_))
        }
        print("✓")
        
        # 3. Prophet
        print("   [3/3] Prophet...", end=" ")
        prophet_df = self.df[["date", "count"]].rename(columns={"date": "ds", "count": "y"})
        
        train_size = int(len(prophet_df) * 0.8)
        prophet_train = prophet_df[:train_size]
        prophet_test = prophet_df[train_size:]
        
        prophet_model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
        prophet_model.fit(prophet_train)
        
        forecast = prophet_model.predict(prophet_test[["ds"]])
        prophet_pred = forecast["yhat"].values
        prophet_actual = prophet_test["y"].values
        
        results["prophet"] = {
            "model": prophet_model,
            "mae": mean_absolute_error(prophet_actual, prophet_pred),
            "rmse": np.sqrt(mean_squared_error(prophet_actual, prophet_pred)),
            "r2": r2_score(prophet_actual, prophet_pred)
        }
        print("✓")
        
        return results
    
    def save_models(self, results):
        """Sauvegarder les modèles et rapport"""
        print("\n💾 Sauvegarde...")
        
        import os
        os.makedirs("models", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarder chaque modèle
        for name, data in results.items():
            model_path = f"models/{name}_{timestamp}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(data["model"], f)
            print(f"   ✓ {name}")
        
        # Sauvegarder le dataset préparé
        dataset_path = f"models/dataset_{timestamp}.csv"
        self.df.to_csv(dataset_path, index=False)
        print(f"   ✓ Dataset sauvegardé")
        
        return timestamp
    
    def generate_report(self, results, timestamp):
        """Générer le rapport de comparaison"""
        print("\n📋 Génération du rapport...")
        
        report_path = f"models/rapport_analyse_{timestamp}.txt"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("RAPPORT D'ANALYSE PRÉDICTIVE - VENTES E-COMMERCE\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Données
            f.write("-" * 70 + "\n")
            f.write("1. DONNÉES\n")
            f.write("-" * 70 + "\n\n")
            f.write(f"Période: {self.sales['date'].min().date()} à {self.sales['date'].max().date()}\n")
            f.write(f"Produits: {len(self.products)}\n")
            f.write(f"Transactions: {len(self.sales):,}\n\n")
            
            # Performance
            f.write("-" * 70 + "\n")
            f.write("2. PERFORMANCE DES MODÈLES\n")
            f.write("-" * 70 + "\n\n")
            f.write(f"{'Modèle':<25} {'MAE':<12} {'RMSE':<12} {'R²':<10}\n")
            f.write("-" * 60 + "\n")
            
            model_names = {
                "linear_regression": "Régression Linéaire",
                "random_forest": "Random Forest",
                "prophet": "Prophet"
            }
            
            best_model = None
            best_mae = float('inf')
            
            for key, name in model_names.items():
                mae = results[key]["mae"]
                rmse = results[key]["rmse"]
                r2 = results[key]["r2"]
                f.write(f"{name:<25} {mae:<12.2f} {rmse:<12.2f} {r2:<10.3f}\n")
                
                if mae < best_mae:
                    best_mae = mae
                    best_model = name
            
            # Importance des features (Random Forest)
            f.write("\n" + "-" * 70 + "\n")
            f.write("3. IMPORTANCE DES VARIABLES (Random Forest)\n")
            f.write("-" * 70 + "\n\n")
            
            importance = results["random_forest"]["feature_importance"]
            for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                f.write(f"   {feature:<20}: {score:.3f}\n")
            
            # Recommandations
            f.write("\n" + "-" * 70 + "\n")
            f.write("4. RECOMMANDATIONS\n")
            f.write("-" * 70 + "\n\n")
            f.write(f"✓ Meilleur modèle: {best_model}\n")
            f.write(f"✓ Utiliser ce modèle pour les prédictions de stock\n")
            f.write(f"✓ Réentraîner mensuellement avec nouvelles données\n\n")
            
            f.write("=" * 70 + "\n")
        
        print(f"   ✓ {report_path}")


def main():
    print("\n" + "=" * 70)
    print("ENTRAÎNEMENT DES MODÈLES - VENTES E-COMMERCE")
    print("=" * 70 + "\n")
    
    # Chemins
    PRODUCTS_PATH = "./data/products.csv"
    SALES_PATH = "./data/sales.csv"
    
    # Entraînement
    trainer = SalesModelTrainer(PRODUCTS_PATH, SALES_PATH)
    
    try:
        trainer.load_data()
        trainer.clean_data()
        trainer.prepare_features()
        results = trainer.train_models()
        timestamp = trainer.save_models(results)
        trainer.generate_report(results, timestamp)
        
        print("\n" + "=" * 70)
        print("✓ ENTRAÎNEMENT TERMINÉ")
        print("=" * 70)
        print("\n💡 Lancez maintenant: streamlit run streamlit_app.py\n")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()