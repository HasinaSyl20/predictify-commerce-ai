import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_data(products_file, sales_file):
    """Charger les fichiers uploadés"""
    products = pd.read_csv(products_file)
    sales = pd.read_csv(sales_file, parse_dates=["date"])
    
    # Nettoyage basique
    sales = sales.dropna(subset=["product_id", "count"])
    sales = sales[sales["count"] > 0]
    sales["date"] = pd.to_datetime(sales["date"])
    
    return products, sales


def calculate_trends(sales):
    """Calculer les tendances par produit"""
    # Agrégation quotidienne
    daily = (
        sales.groupby(["product_id", "date"])["count"]
        .sum()
        .reset_index()
        .sort_values(["product_id", "date"])
    )
    
    # Tendances 7 et 30 jours
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
    
    return daily


def get_product_suggestions(sales, products, top_n=10):
    """Suggérer les produits basés sur les tendances"""
    # Calculer les tendances
    trends = calculate_trends(sales)
    
    # Prendre les 30 derniers jours
    recent = trends.groupby("product_id").tail(30)
    
    # Calculer le score de tendance
    suggestions = (
        recent.groupby("product_id")
        .agg({
            "count": "sum",
            "trend_7d": "mean",
            "trend_30d": "mean"
        })
        .reset_index()
    )
    
    # Score composite: tendance récente + tendance longue
    suggestions["trend_score"] = (
        suggestions["trend_7d"] * 0.6 + suggestions["trend_30d"] * 0.4
    )
    
    # Ajouter les noms de produits
    suggestions = suggestions.merge(products[["id", "name"]], left_on="product_id", right_on="id")
    
    # Trier par score décroissant
    suggestions = suggestions.sort_values("trend_score", ascending=False).head(top_n)
    
    return suggestions


def main():
    st.set_page_config(
        page_title="Suggestions de produits par tendance",
        layout="wide"
    )
    
    # Titre
    st.title("Dashboard Analyse Ventes E-commerce")
    st.markdown("### Visualisation et suggestions de produits")
    st.markdown("---")
    
    # Sidebar - Upload
    st.sidebar.header("Importer les données")
    
    products_file = st.sidebar.file_uploader("Products.csv", type=["csv"])
    sales_file = st.sidebar.file_uploader("Sales.csv", type=["csv"])
    
    if products_file and sales_file:
        # Charger les données
        products, sales = load_data(products_file, sales_file)
        
        st.sidebar.success("Données chargées")
        st.sidebar.metric("Produits", len(products))
        st.sidebar.metric("Transactions", len(sales))
        
        # KPIs en haut
        st.header("Indicateurs Clés")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sales = sales["count"].sum()
            st.metric("Ventes Totales", f"{total_sales:,.0f} unités")
        
        with col2:
            if "unit_price" in sales.columns:
                revenue = (sales["count"] * sales["unit_price"]).sum()
                st.metric("Revenu Total", f"{revenue:,.0f} €")
            else:
                st.metric("Revenu", "N/A")
        
        with col3:
            period = (sales["date"].max() - sales["date"].min()).days
            st.metric("Période", f"{period} jours")
        
        with col4:
            avg_daily = total_sales / max(period, 1)
            st.metric("Ventes/jour", f"{avg_daily:.0f}")
        
        st.markdown("---")
        
        # Tabs
        tab1, tab2 = st.tabs(["Graphiques", "Suggestions"])
        
        with tab1:

            # Ventes dans le temps
            st.subheader("Évolution des Ventes")
            daily_total = sales.groupby("date")["count"].sum().reset_index()
            
            fig = px.line(
                daily_total,
                x="date",
                y="count",
                title="Ventes Quotidiennes Totales",
                labels={"count": "Ventes", "date": "Date"}
            )
            fig.update_traces(line_color='#1f77b4', line_width=2)
            st.plotly_chart(fig, use_container_width=True)
            
            # Deux colonnes
            col1, col2 = st.columns(2)
            
            with col1:
                # Top 10 produits
                st.subheader("Top 10 Produits")
                top_products = (
                    sales.groupby("product_id")["count"]
                    .sum()
                    .nlargest(10)
                    .reset_index()
                )
                top_products = top_products.merge(products[["id", "name"]], left_on="product_id", right_on="id")
                
                fig = px.bar(
                    top_products,
                    x="count",
                    y="name",
                    orientation="h",
                    title="Par Volume de Ventes",
                    labels={"count": "Ventes", "name": "Produit"}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Ventes par jour de la semaine
                st.subheader("Ventes par Jour")
                sales_copy = sales.copy()
                sales_copy["day_name"] = sales_copy["date"].dt.day_name()
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                
                by_day = (
                    sales_copy.groupby("day_name")["count"]
                    .sum()
                    .reindex(day_order)
                    .reset_index()
                )
                
                fig = px.bar(
                    by_day,
                    x="day_name",
                    y="count",
                    title="Saisonnalité Hebdomadaire",
                    labels={"count": "Ventes", "day_name": "Jour"}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Tendances des produits
            st.subheader("Tendances des Produits")
            
            # Sélection produit
            product_options = products.set_index("id")["name"].to_dict()
            selected_product = st.selectbox(
                "Choisir un produit",
                options=list(product_options.keys()),
                format_func=lambda x: product_options[x]
            )
            
            # Filtrer et calculer tendances
            product_sales = sales[sales["product_id"] == selected_product].copy()
            product_daily = (
                product_sales.groupby("date")["count"]
                .sum()
                .reset_index()
                .sort_values("date")
            )
            
            # Calculer moving average
            product_daily["trend_7d"] = product_daily["count"].rolling(7, min_periods=1).mean()
            product_daily["trend_30d"] = product_daily["count"].rolling(30, min_periods=1).mean()
            
            # Graphique
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=product_daily["date"],
                y=product_daily["count"],
                mode="markers",
                name="Ventes réelles",
                marker=dict(size=4, color="lightblue")
            ))
            fig.add_trace(go.Scatter(
                x=product_daily["date"],
                y=product_daily["trend_7d"],
                mode="lines",
                name="Tendance 7j",
                line=dict(color="orange", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=product_daily["date"],
                y=product_daily["trend_30d"],
                mode="lines",
                name="Tendance 30j",
                line=dict(color="red", width=2)
            ))
            
            fig.update_layout(
                title=f"Tendances - {product_options[selected_product]}",
                xaxis_title="Date",
                yaxis_title="Ventes"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("*Produits recommandés basés sur les tendances de ventes récentes*")
            
            # Slider pour nombre de suggestions
            top_n = st.slider("Nombre de suggestions", 5, 20, 10)
            
            # Calculer les suggestions
            suggestions = get_product_suggestions(sales, products, top_n)
            
            # Afficher
            st.subheader(f"Top {top_n} Produits à Promouvoir")
            
            for i, row in suggestions.iterrows():
                rank = list(suggestions.index).index(i) + 1
                emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "📦"
                
                with st.expander(f"{emoji} #{rank} - {row['name']}", expanded=(rank <= 3)):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Ventes (30j)", f"{row['count']:.0f}")
                    
                    with col2:
                        st.metric("Tendance 7j", f"{row['trend_7d']:.1f}")
                    
                    with col3:
                        st.metric("Score", f"{row['trend_score']:.1f}")
                    
                    # Barre de progression
                    score_pct = (row['trend_score'] / suggestions['trend_score'].max()) * 100
                    st.progress(score_pct / 100)
                    
                    if rank <= 3:
                        st.success("⭐ Produit hautement recommandé pour promotion")
            
            # Graphique
            st.subheader("Visualisation des Suggestions")
            
            fig = px.bar(
                suggestions,
                x="name",
                y="trend_score",
                title="Score de Tendance par Produit",
                labels={"name": "Produit", "trend_score": "Score"},
                color="trend_score",
                color_continuous_scale="Viridis"
            )
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Export
            st.markdown("---")
            csv = suggestions[["name", "count", "trend_7d", "trend_30d", "trend_score"]].to_csv(index=False)
        
    else:
        # Message d'accueil
        st.info("Veuillez charger les fichiers CSV dans la barre latérale")
        
        st.markdown("""
        ### Instructions
        
        1. **Importer vos données** dans la barre latérale:
           - `products.csv` - Catalogue de produits
           - `sales.csv` - Historique des ventes
        
        2. **Explorez 3 onglets**:
           - **Graphiques** - Visualisations des ventes
           - **Suggestions** - Produits recommandés (basé sur tendances)
        
        ### Format des fichiers
        
        **products.csv:**
        ```
        id,name,category,price
        1,Product A,Electronics,299.99
        ```
        
        **sales.csv:**
        ```
        product_id,date,count,unit_price
        1,2024-01-01,5,299.99
        ```
        """)


if __name__ == "__main__":
    main()