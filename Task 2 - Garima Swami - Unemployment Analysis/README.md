# Task 2: Unemployment Analysis in India

## 📋 Project Overview

This project provides a **comprehensive data analysis** of unemployment trends in India from **May 2019 to June 2020**, with special focus on the **COVID-19 pandemic's impact** on employment. The analysis examines unemployment rates, employment levels, and labour participation across different states and between rural and urban areas.

### 🎯 Objectives

1. Analyze unemployment trends before and after COVID-19 lockdown
2. Compare rural vs urban unemployment patterns
3. Identify states most severely impacted by the pandemic
4. Understand labour force participation changes
5. Provide actionable insights from employment data

---

## 📊 Dataset Information

### Data Overview
- **Time Period:** May 2019 - June 2020 (14 months)
- **Geographic Coverage:** 28 states and Union Territories of India
- **Area Classification:** Rural and Urban
- **Total Records:** 400+ monthly observations
- **Frequency:** Monthly data collection

### Key Features
1. **Region** - State or Union Territory
2. **Date** - Month of observation (DD-MM-YYYY)
3. **Frequency** - Data collection frequency (Monthly)
4. **Estimated Unemployment Rate (%)** - Percentage of unemployed in labour force
5. **Estimated Employed** - Number of employed individuals
6. **Estimated Labour Participation Rate (%)** - % of population in labour force
7. **Area** - Rural or Urban classification

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization

---

## 📁 Project Structure

```
Task 2 - Garima Swami - Unemployment Analysis/
├── Unemployment_in_India.csv              # Dataset file
├── unemployment_analysis.py                # Main Python script
├── README.md                               # Project documentation
├── unemployment_analysis.png               # Main analysis visualizations
├── unemployment_heatmap.png                # State-time heatmaps
└── unemployment_distributions.png          # Distribution plots
```

---

## 🚀 How to Run

### Prerequisites
Install required packages:
```bash
pip install pandas numpy matplotlib seaborn
```

### Execution
```bash
python unemployment_analysis.py
```

### Expected Output
The script will:
1. Load and explore the unemployment dataset
2. Clean and preprocess the data
3. Perform comprehensive statistical analysis
4. Analyze COVID-19's impact on employment
5. Generate multiple visualizations
6. Display detailed summary report

---

## 📈 Analysis Sections

### 1. Data Loading & Exploration
- Dataset shape and structure
- Column analysis
- Missing value detection
- Data type validation
- Rural vs Urban data distribution

### 2. Statistical Analysis
- Descriptive statistics for all metrics
- Rural vs Urban comparison
- Regional unemployment patterns
- Top/Bottom performing states
- Trend identification

### 3. COVID-19 Impact Analysis
- Pre-COVID period (May 2019 - March 2020) baseline
- COVID-19 period (April 2020 onwards) comparison
- Unemployment rate changes
- Employment level impact
- Labour participation rate impact
- State-wise impact assessment

### 4. Visualization Analysis
- Unemployment rate trends (Rural vs Urban)
- Regional comparison charts
- Employment level tracking
- Labour participation trends
- Pre vs COVID period box plots
- State-wise COVID impact comparison
- Heatmaps showing state and temporal patterns
- Distribution analysis

---

## 🔍 Key Findings

### COVID-19 Impact
- **Unemployment Spike:** Sharp increase in unemployment rate from April 2020
- **Rural Impact:** Rural areas experienced higher unemployment rise
- **Most Affected States:** Bihar, Jharkhand, Tamil Nadu show highest impact
- **Employment Drop:** Millions lost employment during lockdown period
- **Participation Decline:** Labour force participation decreased significantly

### Regional Insights
- **Eastern India:** Higher unemployment rates overall
- **Southern India:** More resilient but still affected
- **Northern India:** Mixed impact across states
- **Union Territories:** Variable impact depending on economic structure

### Rural vs Urban
- **Rural Unemployment:** Average ~10% pre-COVID, spiked to ~30% during COVID
- **Urban Unemployment:** Average ~8% pre-COVID, rose to ~25% during COVID
- **Differential Impact:** Rural areas showed more severe disruption

---

## 📊 Visualizations Generated

### 1. unemployment_analysis.png
**6 comprehensive subplots:**
- Unemployment trend comparison (Rural vs Urban)
- Top 10 states by unemployment rate
- Employment level trends
- Labour participation rate trends
- Pre vs COVID unemployment box plot
- Most affected states comparison

### 2. unemployment_heatmap.png
**State-Time heatmaps:**
- Rural unemployment by state and date
- Urban unemployment by state and date
- Easily identifies hotspots and trends

### 3. unemployment_distributions.png
**4 distribution analysis plots:**
- Unemployment rate distribution (histogram)
- Employment level distribution (histogram)
- Data distribution pie chart (Rural vs Urban)
- Unemployment vs Labour participation scatter plot

---

## 💡 Insights & Learnings

### Data Analysis Skills Demonstrated
1. **Data Cleaning** - Handling missing values and data quality
2. **Exploratory Analysis** - Understanding data structure and distribution
3. **Statistical Analysis** - Computing and interpreting key metrics
4. **Time Series Analysis** - Tracking trends over time
5. **Comparative Analysis** - Rural vs Urban, Pre vs COVID
6. **Geospatial Analysis** - State-wise patterns and rankings
7. **Visualization** - Creating meaningful, informative charts
8. **Report Generation** - Summarizing findings clearly

### Business Insights
- Need for targeted relief in severely affected states
- Importance of rural employment support programs
- Labour force participation monitoring
- Regional economic recovery planning
- Policy intervention effectiveness measurement

---

## 🔧 Customization

### Analyze Specific States
Modify the analysis to focus on particular states:
```python
specific_state = analyzer.data[analyzer.data['Region'] == 'Bihar']
```

### Change Time Periods
Analyze different date ranges:
```python
pre_covid = analyzer.data[analyzer.data['Date'] < '30-04-2020']
```

### Add More Metrics
Extend the analysis with additional calculations:
```python
analyzer.data['Unemployment_per_capita'] = analyzer.data['Estimated Unemployment Rate (%)'] / analyzer.data['Estimated Labour Participation Rate (%)']
```

---

## 📝 Notes

- Data represents government estimates and actual numbers may vary
- COVID-19 period starts from April 2020 (lockdown announcement)
- Some states have incomplete data for certain periods
- Rural and Urban classifications follow government definitions
- Labour participation rate indicates economically active population

---

## 📚 References

- [Ministry of Statistics & Programme Implementation](https://mospi.gov.in/)
- [Periodic Labour Force Survey (PLFS)](https://plfs.mospi.gov.in/)
- [COVID-19 Impact on Indian Economy](https://www.worldbank.org/en/news)
- [India Labour Statistics](https://labourbureau.gov.in/)

---

## 👨‍💻 Author

**Garima Swami**  
*Data Science Internship Project*  
*Date: 2026*

---

## ✨ Conclusion

This analysis successfully demonstrates the devastating impact of COVID-19 on India's employment sector. The data clearly shows:

1. **Massive Job Losses** - Millions became unemployed overnight
2. **Regional Disparities** - Eastern states suffered more severely
3. **Urban-Rural Divide** - Rural areas more vulnerable
4. **Systemic Impact** - Labour participation dropped significantly
5. **Recovery Signs** - June 2020 shows early recovery indicators

The insights from this analysis can inform policy makers in designing targeted recovery programs and employment generation schemes.

**Data-Driven Decision Making at Work!** 📊
