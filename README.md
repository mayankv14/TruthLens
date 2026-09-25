# TruthLens

TruthLens is a web-based fake news analysis and fact-checking application that combines machine learning with external fact-checking evidence to analyze textual news claims.

The application uses a TF-IDF + Logistic Regression model for news classification and the Google Fact Check Tools API to retrieve relevant external fact-checking information.

## Live Demo

🔗 **[Open TruthLens](https://truthlens-nexq.onrender.com/)**

## GitHub Repository

🔗 **[GitHub - TruthLens](https://github.com/mayankv14/TruthLens)**

---

## Features

- Machine learning based news classification
- Claim-level analysis for statements containing multiple claims
- External fact-checking using Google Fact Check Tools API
- Evidence-aware overall results
- User registration and login
- JWT-based authentication
- Password hashing
- Prediction history
- MongoDB Atlas database
- Responsive web interface
- Production deployment using Render

---

## How TruthLens Works

A user submits a news statement through the application.

```text
User Input
    ↓
Text Processing
    ↓
Claim Decomposition
    ↓
┌───────────────────────┐
│                       │
▼                       ▼
ML Classification    Fact Checking
TF-IDF +              Google Fact
Logistic Regression   Check Tools API
│                       │
└───────────┬───────────┘
            ↓
     Evidence Analysis
            ↓
      Overall Result
            ↓
      Save to MongoDB
            ↓
       Display Result
