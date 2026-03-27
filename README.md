# Retail Intelligence Platform

An end-to-end retail analytics project built with Python, FastAPI, Streamlit, ETL pipelines, SQL-ready data modeling, and machine learning forecasting.

## Overview
This project combines two retail datasets with different structures into a unified analytics-ready pipeline. It demonstrates data cleaning, transformation, API development, dashboarding, and forecasting in one modular repository.

## Tech Stack
- Python
- Pandas
- FastAPI
- Streamlit
- SQL
- Prophet
- GitHub

## Project Structure
retail-intelligence-platform/
- backend/ → FastAPI backend
- frontend/ → Streamlit dashboard
- etl/ → ETL pipeline
- ml/ → forecasting scripts
- sql/ → schema and analysis queries
- data/ → raw and processed data
- screenshots/ → project screenshots

## Features
- Cleans and standardizes two retail datasets
- Merges data into a unified analytics-ready format
- Serves KPI and trend data through REST API endpoints
- Displays analytics in an interactive dashboard
- Forecasts future monthly sales using Prophet

## Architecture
Raw CSV → ETL Pipeline → Processed Data → FastAPI Backend → Streamlit Frontend → Forecasting

## API Endpoints
- `/`
- `/kpi`
- `/top-products`
- `/sales-by-country`
- `/sales-trend`
- `/forecast`

## How to Run

### Run ETL
```bash
python etl/etl_retail.py