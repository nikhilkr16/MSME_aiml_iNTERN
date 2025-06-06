# Titanic Dataset - Exploratory Data Analysis (EDA)

## 📊 Project Overview

This project performs comprehensive Exploratory Data Analysis (EDA) on the famous Titanic dataset to understand passenger demographics, survival patterns, and key insights through statistical analysis and data visualizations.

## 🎯 Objectives

- Understand the Titanic dataset through descriptive statistics
- Identify patterns and relationships in passenger data
- Analyze survival rates across different demographics
- Create meaningful visualizations to communicate findings
- Generate insights about factors affecting passenger survival

## 🛠️ Tools & Technologies

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Static plotting and visualizations
- **Seaborn** - Advanced statistical visualizations
- **Plotly** - Interactive visualizations

## 📁 Project Structure

```
AIML-INTERN/
├── archive/
│   └── Titanic-Dataset.csv          # Original dataset
├── task2.py                         # Main EDA script
├── basic_distributions.png          # Age, fare, survival, class distributions
├── correlation_matrix.png           # Correlation heatmap
├── boxplots.png                     # Box plots by passenger class
├── age_fare_survival.html           # Interactive age vs fare scatter plot
├── survival_by_class.html           # Interactive survival rate bar chart
├── titanic_processed.csv            # Processed dataset
└── README.md                        # Project documentation
```

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd AIML-INTERN
   ```

2. **Install required packages:**
   ```bash
   pip install pandas matplotlib seaborn plotly
   ```

3. **Run the analysis:**
   ```bash
   python task2.py
   ```

## 📈 Analysis Results

### Dataset Overview
- **Total Passengers:** 891
- **Survival Rate:** 38.4%
- **Average Age:** ~30 years
- **Average Fare:** $32.20

### Missing Data Analysis
| Column | Missing Values |
|--------|----------------|
| Age | 177 (19.9%) |
| Cabin | 687 (77.1%) |
| Embarked | 2 (0.2%) |

### Key Insights

#### 🔍 **Survival Patterns**
- **Overall survival rate:** 38.4%
- **Gender impact:** Significant difference in survival rates between male and female passengers
- **Class impact:** Higher class passengers had better survival rates

#### 💰 **Fare Analysis**
- Wide range of ticket prices from $0 to $512.33
- Clear correlation between passenger class and fare paid
- Higher fares generally associated with better survival rates

#### 👥 **Demographics**
- Age range: 0.42 to 80 years
- Most passengers traveled alone (low SibSp and Parch values)
- Third class was the most populated passenger class

## 📊 Visualizations Generated

### Static Plots (PNG)
1. **Basic Distributions** - Histograms showing:
   - Age distribution of passengers
   - Fare distribution
   - Survival count (survived vs died)
   - Passenger class distribution

2. **Correlation Matrix** - Heatmap showing relationships between numerical variables

3. **Box Plots** - Age and fare distributions segmented by passenger class

### Interactive Plots (HTML)
1. **Age vs Fare Scatter Plot** - Interactive visualization colored by survival status
2. **Survival Rate by Class** - Interactive bar chart showing class-based survival rates

## 🔍 Statistical Summary

### Numerical Variables Summary
- **PassengerId:** 1-891 (passenger identification)
- **Survived:** 0 (died) or 1 (survived)
- **Pclass:** 1 (first), 2 (second), 3 (third class)
- **Age:** 0.42-80 years
- **SibSp:** 0-8 siblings/spouses aboard
- **Parch:** 0-6 parents/children aboard
- **Fare:** $0-$512.33 ticket price

## 📝 Code Structure

The `task2.py` script performs the following analyses:

1. **Data Loading & Basic Statistics**
   - Load dataset and display basic info
   - Generate descriptive statistics for numerical and categorical columns

2. **Missing Values Analysis**
   - Identify and count missing values in each column

3. **Static Visualizations**
   - Create distribution plots for key variables
   - Generate correlation matrix heatmap
   - Create box plots for class-based analysis

4. **Interactive Visualizations**
   - Age vs Fare scatter plot with survival coloring
   - Survival rate bar chart by passenger class

5. **Additional Insights**
   - Calculate survival rates by gender and class
   - Compute average fares by passenger class

## 🚀 Future Enhancements

- [ ] Implement data preprocessing and cleaning
- [ ] Add machine learning models for survival prediction
- [ ] Create more advanced visualizations (e.g., survival heatmaps)
- [ ] Perform feature engineering
- [ ] Add statistical significance testing
- [ ] Create dashboard with all visualizations

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit your changes (`git commit -am 'Add new analysis'`)
4. Push to the branch (`git push origin feature/new-analysis`)
5. Open a Pull Request

## 📞 Contact

For questions or suggestions, please open an issue in this repository.

---

**Note:** This project is part of an AI/ML internship program and demonstrates exploratory data analysis techniques on the classic Titanic dataset. 