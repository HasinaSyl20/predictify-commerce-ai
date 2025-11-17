import glob
import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# load models from ./models/* (Linear regression and random forest)
def load_latest_models():
    # Linear Regression
    lr_files = glob.glob("models/lr_global_*.pkl")
    lr_model, lr_features = None, None
    if lr_files:
        latest_lr = max(lr_files, key=os.path.getctime)
        with open(latest_lr, "rb") as f:
            payload = pickle.load(f)
            lr_model = payload["model"]
            lr_features = payload["features"]
    
    # Random Forest
    rf_files = glob.glob("models/rf_global_*.pkl")
    rf_model, rf_features = None, None
    if rf_files:
        latest_rf = max(rf_files, key=os.path.getctime)
        with open(latest_rf, "rb") as f:
            payload = pickle.load(f)
            rf_model = payload["model"]
            rf_features = payload["features"]
    
    return lr_model, lr_features, rf_model, rf_features


# load dataframes from imported csv files
def load_data(products_file, sales_file):
    try:
        products = pd.read_csv(products_file)
        sales = pd.read_csv(sales_file, parse_dates=["date"])
    except Exception as e:
        st.error(f"Erreur de lecture CSV : {e}")
        return None, None

    # basic data cleaning
    sales = sales.dropna(subset=["product_id", "count"])
    sales = sales[sales["count"] > 0]
    sales["date"] = pd.to_datetime(sales["date"])
    
    return products, sales


# extra attributes or features for a given product, like a features engineering
def build_features_for_product(sales, products, product_id, ref_date):
    prod_sales = sales[sales["product_id"] == product_id].copy()
    
    if prod_sales.empty:
        # Cold start: utiliser les données de la catégorie
        cat_row = products[products["id"] == product_id]
        if cat_row.empty:
            return None, 1.0
        
        cat = cat_row["category"].iloc[0]
        cat_ids = products[products["category"] == cat]["id"]
        cat_sales = sales[sales["product_id"].isin(cat_ids)]
        
        if cat_sales.empty:
            return None, 1.0
        
        daily = cat_sales.groupby("date")["count"].sum().reset_index()
        trend_7d = daily["count"].rolling(7, min_periods=1).mean().iloc[-1]
        trend_30d = daily["count"].rolling(30, min_periods=1).mean().iloc[-1]
        age = 0
        boost = 2.0  # new product, boost
    else:
        # product history
        daily = (
            prod_sales.groupby("date")["count"]
            .sum()
            .reset_index()
            .sort_values("date")
        )
        trend_7d = daily["count"].rolling(7, min_periods=1).mean().iloc[-1]
        trend_30d = daily["count"].rolling(30, min_periods=1).mean().iloc[-1]
        launch = daily["date"].min()
        age = (ref_date - launch).days
        
        # boost
        last_sale = daily["date"].max()
        boost = 1.5 if (ref_date - last_sale).days < 7 else 1.0

    # build trend
    row = {
        "trend_7d": trend_7d,
        "trend_30d": trend_30d,
        "day_of_week": ref_date.weekday(),
        "month": ref_date.month,
        "is_weekend": int(ref_date.weekday() >= 5),
        "product_age_days": age,
    }
    
    return row, boost


# 7 days trend prediction based (Random forest and linear regression only)
def predict_next_7_days_ml(model, features_dict, feature_names, boost):
    future_sum = 0
    base_date = datetime.today()
    
    for i in range(1, 8):
        d = base_date + timedelta(days=i)
        row = {
            "trend_7d": features_dict["trend_7d"],
            "trend_30d": features_dict["trend_30d"],
            "day_of_week": d.weekday(),
            "month": d.month,
            "is_weekend": int(d.weekday() >= 5),
            "product_age_days": features_dict["product_age_days"] + i,
        }
        
        # create data frame and align features
        X = pd.DataFrame([row])
        X = X.reindex(columns=feature_names, fill_value=0)
        
        pred = model.predict(X)[0]
        future_sum += max(0, pred * boost)
    
    return future_sum


# prophets'models prediction based
def predict_next_7_days_prophet(prophet_model):
    try:
        future = prophet_model.make_future_dataframe(periods=7)
        forecast = prophet_model.predict(future)
        return forecast.iloc[-7:]["yhat"].clip(lower=0).sum()
    except Exception:
        return None


def main():
    st.set_page_config(
        page_title="Tableau de bord",
        layout="wide"
    )
    
    st.title("Recommendation de produits selon la tendance")
    st.markdown("### Prédictions avec 3 modèles: LR + RF + Prophet")
    
    # sidemenu
    st.sidebar.header("Importer les fichiers csv")
    products_file = st.sidebar.file_uploader("products.csv", type=["csv"])
    sales_file = st.sidebar.file_uploader("sales.csv", type=["csv"])
    
    if not (products_file and sales_file):
        st.markdown("""
            ### Format des fichiers CSV avec exemple(s)
            
            **products.csv:**
            ```
            id,name,category,unit_price
            1,Product A,Electronics,299.99
            ```
            
            **sales.csv:**
            ```
            id,product_id,unit_price,count,ttc,date
            1,1,299.99,5,1499.95,2024-01-01
            ```
            """
        )
        st.stop()
    
    # load data
    products, sales = load_data(products_file, sales_file)
    if products is None:
        st.stop()
    
    st.sidebar.success("Données chargées avec succès")
    st.sidebar.metric("Produits", len(products))
    st.sidebar.metric("Transactions", len(sales))
    
    # load model (regression lineaire, random forest)
    lr_model, lr_features, rf_model, rf_features = load_latest_models()
    
    if lr_model is None and rf_model is None:
        st.error("Aucun modèle trouvé. Lancez d'abord: `python train.py`")
        st.stop()
    
    # show available models
    st.sidebar.markdown("---")
    st.sidebar.subheader("Modèles chargés")
    if lr_model:
        st.sidebar.success("Régression Linéaire")
    if rf_model:
        st.sidebar.success("Random Forest")
    
    # load prophet models
    prophet_dirs = glob.glob("models/prophet_*")
    if prophet_dirs:
        latest_prophet_dir = max(prophet_dirs, key=os.path.getctime)
        n_prophet = len(glob.glob(f"{latest_prophet_dir}/*.pkl"))
        st.sidebar.success(f"Prophet ({n_prophet} produits)")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    total_units = sales["count"].sum()
    revenue = (sales["count"] * sales["unit_price"]).sum()
    period_days = max((sales["date"].max() - sales["date"].min()).days, 1)
    
    col1.metric("Unités vendues", f"{total_units:,.0f}")
    col2.metric("Revenu", f"{revenue:,.0f} €")
    col3.metric("Période", f"{period_days} jours")
    col4.metric("Ventes/jour", f"{total_units / period_days:.0f}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Statistiques", "Suggestions de produit(s)", "Rapport d'analyse"])
    
    with tab1:
        st.header("Visualisations")
        
        # daily sales total
        st.subheader("Évolution des Ventes")
        daily_total = sales.groupby("date")["count"].sum().reset_index()
        
        fig = px.line(
            daily_total,
            x="date",
            y="count",
            title="Ventes Quotidiennes Totales",
            labels={"count": "Unités", "date": "Date"},
        )
        fig.update_traces(line_color="#1f77b4", line_width=2)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 10 products
            st.subheader("Top 10 Produits")
            top10 = (
                sales.groupby("product_id")["count"]
                .sum()
                .nlargest(10)
                .reset_index()
            )
            top10 = top10.merge(products[["id", "name"]], left_on="product_id", right_on="id")
            
            fig = px.bar(
                top10,
                x="count",
                y="name",
                orientation="h",
                title="Par Volume de Ventes",
                labels={"count": "Ventes", "name": "Produit"},
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # daily sales per week day
            st.subheader("Ventes par Jour")
            dow_sales = sales.copy()
            dow_sales["dow"] = dow_sales["date"].dt.day_name()
            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_agg = dow_sales.groupby("dow")["count"].sum().reindex(order).reset_index()
            
            fig = px.bar(
                dow_agg,
                x="dow",
                y="count",
                title="Saisonnalité Hebdomadaire",
                labels={"count": "Ventes", "dow": "Jour"},
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # specifi product trend
        st.subheader("Tendance d'un Produit")
        prod_options = products.set_index("id")["name"].to_dict()
        chosen = st.selectbox(
            "Choisir un produit",
            options=list(prod_options.keys()),
            format_func=lambda x: prod_options[x]
        )
        
        prod_hist = (
            sales[sales["product_id"] == chosen]
            .groupby("date")["count"]
            .sum()
            .reset_index()
            .sort_values("date")
        )
        prod_hist["ma7"] = prod_hist["count"].rolling(7, min_periods=1).mean()
        prod_hist["ma30"] = prod_hist["count"].rolling(30, min_periods=1).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prod_hist["date"],
            y=prod_hist["count"],
            mode="markers",
            name="Ventes réelles",
            marker=dict(size=5, color="lightblue")
        ))
        fig.add_trace(go.Scatter(
            x=prod_hist["date"],
            y=prod_hist["ma7"],
            mode="lines",
            name="Tendance 7j",
            line=dict(color="orange", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=prod_hist["date"],
            y=prod_hist["ma30"],
            mode="lines",
            name="Tendance 30j",
            line=dict(color="red", width=2)
        ))
        fig.update_layout(
            title=f"Tendances - {prod_options[chosen]}",
            xaxis_title="Date",
            yaxis_title="Ventes"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # suggestion tab
    with tab2:
        st.header("Suggestions de Produits")
        st.markdown("*Prédictions des ventes sur 7 jours avec les 3 modèles*")
        
        top_n = st.slider("Nombre de suggestions", 5, 25, 10)
        
        # Sélection du modèle à utiliser
        model_choice = st.radio(
            "Modèle de prédiction",
            ["Prophet (prioritaire)", "Random Forest", "Régression Linéaire", "Moyenne des 3"],
            horizontal=True
        )
        
        # compute prediction
        ref_date = datetime.today()
        suggestions = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, pid in enumerate(products["id"]):
            progress_bar.progress((idx + 1) / len(products))
            status_text.text(f"Analyse en cours... {idx + 1}/{len(products)}")
            
            # build features
            features, boost = build_features_for_product(sales, products, pid, ref_date)
            if features is None:
                continue
            
            # each model predictions
            pred_lr = None
            pred_rf = None
            pred_prophet = None
            
            # LR
            if lr_model and lr_features:
                pred_lr = predict_next_7_days_ml(lr_model, features, lr_features, boost)
            
            # RF
            if rf_model and rf_features:
                pred_rf = predict_next_7_days_ml(rf_model, features, rf_features, boost)
            
            # Prophet
            prophet_path = glob.glob(f"models/prophet_*/{pid}.pkl")
            if prophet_path:
                try:
                    with open(prophet_path[0], "rb") as f:
                        m = pickle.load(f)
                    pred_prophet = predict_next_7_days_prophet(m)
                except Exception:
                    pass
            
            # show prediction per chosen model
            if model_choice == "Prophet (prioritaire)":
                final_pred = pred_prophet if pred_prophet is not None else (pred_rf or pred_lr)
                source = "Prophet" if pred_prophet is not None else ("RF" if pred_rf else "LR")
            elif model_choice == "Random Forest":
                final_pred = pred_rf if pred_rf is not None else (pred_lr or pred_prophet)
                source = "RF"
            elif model_choice == "Régression Linéaire":
                final_pred = pred_lr if pred_lr is not None else (pred_rf or pred_prophet)
                source = "LR"
            else:  # Moyenne
                preds = [p for p in [pred_lr, pred_rf, pred_prophet] if p is not None]
                final_pred = np.mean(preds) if preds else None
                source = "Moyenne"
            
            if final_pred is None:
                continue
            
            suggestions.append({
                "product_id": pid,
                "name": products[products["id"] == pid]["name"].iloc[0],
                "predicted_7d": final_pred,
                "source": source,
                "lr": pred_lr,
                "rf": pred_rf,
                "prophet": pred_prophet,
            })
        
        progress_bar.empty()
        status_text.empty()
        
        # create and sort dataframe
        suggestions_df = pd.DataFrame(suggestions)
        suggestions_df = suggestions_df.sort_values("predicted_7d", ascending=False).head(top_n)
        
        # show suggestions
        st.subheader(f"Top {top_n} Produits à Promouvoir")
        
        for idx, row in suggestions_df.iterrows():
            rank = list(suggestions_df.index).index(idx) + 1
            rank_icon = "(1)" if rank == 1 else "(2)" if rank == 2 else "(3)" if rank == 3 else "-"
            
            with st.expander(
                f"{rank_icon} #{rank} - {row['name']} ({row['source']})",
                expanded=(rank <= 3)
            ):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Prédiction Finale", f"{row['predicted_7d']:.1f} unités")
                
                with col2:
                    st.metric("Modèle utilisé", row['source'])
                
                with col3:
                    # Afficher les 3 prédictions
                    st.write("**Comparaison:**")
                    if row['lr']:
                        st.write(f"LR: {row['lr']:.1f}")
                    if row['rf']:
                        st.write(f"RF: {row['rf']:.1f}")
                    if row['prophet']:
                        st.write(f"Prophet: {row['prophet']:.1f}")
                
                # Barre de progression
                prog = row['predicted_7d'] / suggestions_df['predicted_7d'].max()
                st.progress(prog)
        
        # 
        st.subheader("Visualisation des Prédictions")
        
        fig = px.bar(
            suggestions_df,
            x="name",
            y="predicted_7d",
            color="source",
            title="Prédictions de Ventes (7 jours)",
            labels={"predicted_7d": "Unités prévues", "name": "Produit"},
        )
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # report
    with tab3:
        st.header("Rapport d'analyse des modèles")
        
        # load report file
        reports = glob.glob("models/report_*.txt")
        if reports:
            latest_report = max(reports, key=os.path.getctime)
            with open(latest_report, "r", encoding="utf-8") as f:
                report_content = f.read()
            
            st.subheader("Rapport d'Entraînement")
            st.text(report_content)
        else:
            st.info("Aucun rapport disponible. Lancez `python train.py` pour générer le rapport.")


if __name__ == "__main__":
    main()