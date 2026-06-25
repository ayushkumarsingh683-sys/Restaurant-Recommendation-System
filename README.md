# 🍽️ Restaurant Recommendation System

## 📌 What is this Project?
This project is built for my Data Science Internship at **Cognifyz Technologies**. 

The goal of this project is to help users find the best restaurants based on their personal choices. When a user tells the system their **City**, preferred **Cuisines** (food types), and **Price Range**, the system quickly searches the dataset and recommends the top 3 best matching restaurants.

---

## 📂 Files in this Repository
* **`Dataset .csv`** — The raw data sheet containing all the restaurant records.
* **`restaurant_recommender.py`** — The main Python script that cleans data and runs the recommendations.
* **`top_cuisines.png`** — An automated chart showing the most popular food types in the dataset.
* **`recommendation_results.png`** — An automated chart showing the match scores for our test case.
* **`README.md`** — This documentation homepage.

---

## 🛠️ Tools and Libraries Used
We used the Python programming language along with these basic data science tools:
* **Pandas & NumPy:** For loading and managing the dataset table.
* **Scikit-Learn:** For scaling data and calculating matching scores (**Cosine Similarity**).
* **Matplotlib & Seaborn:** For drawing the color charts and graphs.

---

## 🚀 How the System Works

### 1. Cleaning the Data (Preprocessing)
* **Fixing Missing Data:** Some restaurants did not have cuisines listed. We filled those blank spaces with the word `'Unknown'` so the code does not crash.
* **Turning Words into Numbers (Encoding):** Computers do not understand words like "Italian" or "Mexican", they only understand numbers. We converted the text columns (`Cuisines` and `Price range`) into binary columns of $0$s and $1$s.
* **Normalizing Ratings:** We resized the restaurant ratings so they all sit perfectly on a standard scale from $0$ to $1$.

### 2. Finding the Best Match (Cosine Similarity)
* The system turns the user's favorite choices into a **User Vector** (a list of numbers).
* It uses a formula called **Cosine Similarity** to compare the user's choices against every restaurant in the chosen city. 
* A score of **1.0** means a perfect 100% match. The system sorts all options and outputs the top 3 closest matches.

---

## 📊 Automated Chart Previews
When you run the code, it automatically saves two high-quality images in your folder:

1. **Top Cuisines Distribution:** Displays a bar chart tracking the most common food varieties available in the dataset.
2. **Recommendation Match Scores:** Visualizes the final top choices matching our test case query (`City: New Delhi`, `Cuisines: Italian, Continental`, `Price Range: 3`).

---

## ⚡ How to Setup and Run This Project

### 1. Install Required Packages
Open your terminal or command prompt and run this command to install the necessary tools:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
