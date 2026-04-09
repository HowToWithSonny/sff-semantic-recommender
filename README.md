# SFF-Semantic-Lab: An Integrated System for Movie Recommendation and Spatial-Temporal Scheduling

### *A Computational Approach to Movie Festival Logistics via Semantic Embeddings and Heuristic Optimization*

---

## Project Overview
During the **27th Shanghai International Film Festival (SIFF)**, attendees faced significant "information overload"—selecting from over 200 films with limited descriptions and complex geographic scheduling conflicts. 

This project develops a comprehensive **Computational Modeling** pipeline to bridge the gap between abstract cinematic content and practical decision-making. By combining **Natural Language Processing (NLP)** and **Operations Research**, the system automates data acquisition, performs semantic clustering, and generates optimal viewing itineraries under real-world constraints.

## Scientific Methodology
This project implements a multi-layered mathematical approach as detailed in my research:

### 1. Semantic Feature Extraction (NLP Layer)
* **Vector Space Modeling**: Implemented a dual-model approach to map movie synopses into computable formats:
    * **Statistical (TF-IDF)**: Used for high-interpretability keyword indexing.
    * **Neural (Sentence-BERT)**: Utilized the `paraphrase-multilingual-MiniLM-L12-v2` transformer to generate **384-dimensional embeddings**, capturing latent semantic nuances that traditional keyword matching misses.
* **Similarity Computation**: Applied **Cosine Similarity** to measure the angular proximity between movie vectors, effectively solving the "Cold Start" problem for niche festival films.

### 2. Unsupervised Learning (Clustering Layer)
* **Theme Identification**: Used **K-Means Clustering** (k=6) to automatically categorize films into thematic groups (e.g., "Urban Social Margins," "Family & Relationships").
* **Dimensionality Reduction**: Employed **Principal Component Analysis (PCA)** to project high-dimensional embeddings into 2D/3D space for cluster validation and visualization.

### 3. Heuristic Scheduling (Optimization Layer)
* **Constraint Satisfaction Problem (CSP)**: Designed a scheduling engine to handle:
    * **Temporal Constraints**: Screening windows and overlapping durations.
    * **Spatial Constraints**: Cinema-to-cinema commuting time across 50+ venues in Shanghai.
* **Visualization**: Developed an automated **Gantt Chart** generator to visualize optimized daily itineraries.

## 🛠 Technical Stack
* **Language**: Python 3.13
* **Libraries**: 
    * *Modeling*: `scikit-learn`, `sentence-transformers`, `pandas`, `numpy`
    * *Visualization*: `matplotlib`, `plotly`, `wordcloud`, `seaborn`
    * *Deployment*: `gradio`
* **Data Source**: Custom-built scraper for **TMDB (The Movie Database) API**.

## Live Interactive Demo
Experience the semantic recommendation engine live:
👉 **[SFF Semantic Lab on Hugging Face Spaces](https://huggingface.co/spaces/howtowithsonny/sff-smart-scheduler)** *(Note: Use movie titles like "Blue Velvet" or "Pride and Prejudice" to test the vector similarity.)*

## Repository Structure
* `app.py`: The production Gradio application.
* `recommend_app_BERT.py`: Core logic for transformer-based embedding generation.
* `tmdb_scraper.py`: Automated data acquisition and cleaning pipeline.
* `clustering_analysis.py`: Logic for K-Means clustering and PCA visualization.
* `movie_data.csv`: Curated dataset of the 27th SIFF.

## Academic Context
This research was conducted as part of the **"Big Data & Intelligent Information Processing"** curriculum at **Nanjing Normal University**. It represents a synthesis of my undergraduate focus on **Applied Mathematics** and my interest in **Computational Science**—translating cultural data into structured, computable models.

---
**Author:** Songning (Sonny) Li  
**Target Program:** M.Sc. Computational Modeling and Simulation, TU Dresden  
**Contact:** [sonnylee030730@gmail.com]
