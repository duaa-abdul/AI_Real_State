# AI-Powered Real Estate Application

An AI-powered real estate web application built with Python, FastAPI, Machine Learning, Generative AI, and Retrieval-Augmented Generation (RAG).

## Features
- Real estate data analysis and dashboard
- House price prediction using Random Forest
- Property recommendations based on budget and preferences
- AI real estate assistant using Groq API and ChromaDB
- Responsive web interface built with HTML, CSS, and JavaScript

## Technologies
Python, Pandas, Scikit-learn, FastAPI, Groq, Sentence Transformers, ChromaDB, HTML, CSS, JavaScript.

## Setup
1. Create and activate a Python virtual environment.
2. Install packages using `pip install -r requirements.txt`.
3. Add your Groq API key to `.env`.
4. Start the application using `python -m uvicorn main:app --reload`.
5. Open `http://127.0.0.1:8000`.

The application generates a sample property dataset locally and trains its price prediction model automatically.

**Note:** This is an academic demonstration using synthetic property data. Predictions are not verified market valuations.