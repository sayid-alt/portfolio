# Student Performance Factor Analysis

# Table of Contents

- [📂 Dataset Source](##dataset-source)
- [🎯 Objective](##Objective)
- [🥅 Goals](#-goals)
- [💡 Purpose](#-purpose)
- [🗂️ Project Structure](#️-project-structure)
- [🌟 Potential Applications](#-potential-applications)
- [📝 Analysis](#-analysis)
  - [Data Understanding](#data-understanding)
    - [Variables of Data](#variables-of-data)
  - [Data Preparation](#data-preparation)
    - [1. Data Cleansing](#1-data-cleansing)
    - [2. Distribution and Outliers Handling](#2-distribution-and-outliers-handling)
    - [3. Encoding the Categorical Features](#3-encoding-the-categorical-features)
    - [4. Data Scaling](#4-data-scaling)
- [📊 Build Clustering Model](#-build-clustering-model)

## 📂 Dataset Source
The dataset for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors).

## 🎯 Objective
The primary goal of this project is to perform clustering on a student dataset and assign each student to a well-defined cluster. By applying clustering techniques, we aim to segment students based on relevant characteristics, providing a deeper understanding of patterns, behaviors, or distinct groupings among the students.

## 🥅 Goals

1. **🏷️ Cluster Labeling**:
   - Label each student based on their assigned cluster, providing insights into the distinct features or traits associated with each group.
2. **🔍 Cluster Prediction**: 
   - Using clustering algorithms, identify the underlying groups within the dataset and predict the cluster each student belongs to.

## 💡 Purpose
This analysis aims to identify potential factors influencing student behavior or performance. The insights gained can assist educational institutions in tailoring their approaches to better meet the diverse needs of students.

## 🗂️ Project Structure
- **🧹 Data Preprocessing**: Prepare and clean the dataset for clustering.
- **🔗 Clustering Analysis**: Apply clustering algorithms to segment students.
- **🏷️ Cluster Labeling**: Assign meaningful labels to each cluster based on observed characteristics.
- **📊 Result Interpretation**: Analyze and interpret the clusters to understand the factors influencing student performance and behavior.

## 🌟 Potential Applications
This clustering analysis can aid educational institutions in:
- 📌 Identifying groups of students with similar needs or behaviors.
- 🎯 Developing targeted interventions to support different student clusters.
- 📈 Enhancing resource allocation based on student groupings.

## 📝 Analysis

### Data Understanding
---
Dataset is containing information of student performances, behaviors, and family background that indicates the special characteristics of each student and various factors effecting the student exam scores.

#### **Variables of Data**
Variables has been described on kaggle datasets [here](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)

| **Attribute**                  | **Description**                                                               |
| ------------------------------ | ----------------------------------------------------------------------------- |
| **Hours_Studied**              | Number of hours spent studying per week.                                      |
| **Attendance**                 | Percentage of classes attended.                                               |
| **Parental_Involvement**       | Level of parental involvement in the student's education (Low, Medium, High). |
| **Access_to_Resources**        | Availability of educational resources (Low, Medium, High).                    |
| **Extracurricular_Activities** | Participation in extracurricular activities (Yes, No).                        |
| **Sleep_Hours**                | Average number of hours of sleep per night.                                   |
| **Previous_Scores**            | Scores from previous exams.                                                   |
| **Motivation_Level**           | Student's level of motivation (Low, Medium, High).                            |
| **Internet_Access**            | Availability of internet access (Yes, No).                                    |
| **Tutoring_Sessions**          | Number of tutoring sessions attended per month.                               |
| **Family_Income**              | Family income level (Low, Medium, High).                                      |
| **Teacher_Quality**            | Quality of the teachers (Low, Medium, High).                                  |
| **School_Type**                | Type of school attended (Public, Private).                                    |
| **Peer_Influence**             | Influence of peers on academic performance (Positive, Neutral, Negative).     |
| **Physical_Activity**          | Average number of hours of physical activity per week.                        |
| **Learning_Disabilities**      | Presence of learning disabilities (Yes, No).                                  |
| **Parental_Education_Level**   | Highest education level of parents (High School, College, Postgraduate).      |
| **Distance_from_Home**         | Distance from home to school (Near, Moderate, Far).                           |
| **Gender**                     | Gender of the student (Male, Female).                                         |
| **Exam_Score**                 | Final exam score.                                                             |

### **Data Preparation**
Machine learning model needs proper data format to be inputed to the model. unprepared data will causes problem or unexpected bias for the model to predict or inference. For instance, the missing values will causes the unreliable model prediction which model learn from undefined information of data. Hence, since the most original data might consumes those problem, here's the steps would be implemented.
#### **1. Data Cleansing**: 
The data cleansing steps might follows some practical doing. Such as formatting data which in this case, the maximum columns `exam_score` which has a `101` score. thus, we consider it as a fault, and we'll change to `100` when score `101` might consider as a typo or false inputed values.  Another step to be passed is handling missing values. Which in this case we have `3.56%` of missing values. In our case, when the portion is not too high, we simply drop those data for stable learning. Another one is remove the duplicates which in our case does not found any, thus, we can skip this step.
#### **2. Distribution and Outliers Handling**
Checking up of distribution and correlation is another way to preparing data for learning process. As we found in our case, The `tutoring_sessions` and the `exam_scores` shows the higher gap between median of 75% data or third quartile and the max. This indicates the skewed distribution of the data. Either the data between Q3 and max are very small distribution which the possibilities might happened as follow:

- The `tutoring scores` are tend to be small distribution as the value get higher, means that the students are more likely to have a short amount of tutoring sessions.
- The `exam_scores` are also tend to be small distribution as the value get higher, means that the students that get the higher score is a little amount. This might have a correlation between student whom take more tutoring sessions.

These problem we called as an outlier, there some to fix these, one of the easiest way is dropping or removing the outliers data. But the trade-off must be losing a bunch of information of the data. On this analysis, which we consider it has a better approach is as follows:

- Convert the numerical features to the categorical features with binning method. Reasons for this, that the information we need is just the cluster the spesific characteristics of students, not the detail of their scores. so we can group the students with score in range of 70-79 as `B` indicator, and so on.
- Here we'll use the For the `exam_score` `cut` method. to binning the numerical features to the categorical features.
    - exam_score 80 – 100 = A
    - exam_score 70 – 79 = B
    - exam_score 60 – 69 = C
    - exam_score 60 = D
- For the `hours_studied` we'll use the `cut` metdoh as well. And we'll divide them as follows:
    - hours_studided < 10 = low
    - hours_studided < 30 = medium
    - hours_studided > 30 = high
- We'll use for the `tutoring_sessions` cut method as well. And we'll divide them as follows:
    - tutoring_sessions < 2 = low
    - tutoring_sessions < 6 = medium
    - tutoring_sessions > 6 = high
- The result of the binning would become the category datatype, we should convert it back to object datatype for unified data type.

Those steps are applied in python code below:
```python
# set binning and store in new columns labeled as indocator
df_cleaned['exam_score_indicator'] = pd.cut(df_cleaned['exam_score'], bins=[-1, 60, 69, 79, 100], labels=['D', 'C', 'B', 'A'])

df_cleaned['hours_studied_indicator'] = pd.cut(df_cleaned['hours_studied'], bins=[-1, 10, 30, 60], labels=['Low', 'Medium', 'High'])

df_cleaned['tutoring_sessions_indicator'] = pd.cut(df_cleaned['tutoring_sessions'], bins=[-1, 2, 6, 10], labels=['Low', 'Medium', 'High'])

```

However, doing all the steps above, does not remove the root problems, which are the outliers of numerical columns still exist in the data. Hence, the choice of the algorithm that robust to thee outliers such as `DBSCAN`, `k-medoids`  might handle it properly for clustering analysis.
#### **3. Encoding the categorical features**
Model of machine learning can accept only numerical values. However, the data that contains categorical values should be encoded to the numerical data types. The steps will be passed as the follows:

- For ordinal type such as `parental_involvement`, `access_to_resources`, `motivational_level`, `family_income`, `teacher_quality`, `parental_education_level`, `distance_from_home`, `hours_studied_indicator`, `exam_score_indicator`, `tutoring_sessions_indicator` would be an ordinal categories type. we would use `OrdinalEncoder`, because it encodes the order of the data.
- For nominal type such as `Extracurricular_activities`, `internet_access`, `school_type`, `peer_influence`, `learning_disabilities`, `gender` would be considered as nominal categories types. we would use `OneHotEncode`, because it incicates the category of the data.

The ordinal encoding should be done by the order, so that we have to declare the order of the categories. Each ordinal feature has their unique order, so that we have to declare the order for every ordinal features as follows:

1. **Class one**: ordinal features of`parental_involvement` ,`access_to_resources`, `motivation_level`, `family_income`, `teacher_quality`, `hours_studied_indicator`, `tutoring_sessions_indicator` would be an order of Low < Medium < High
    
2. **Class two**: The ordinal features of `parental_education_level` would be ordered by High School < College < Postgraduate.
    
3. **Class three**: The ordinal features of `distance_from_home` would be ordered by Near < Moderate < Far
    
4. **Class Four**: The ordinal features of `exam_score_indicator` would be ordered by D < C < B < A

Here's the implementation on python
```python
# store each ordinal column classes

class_one_ordinal = [col for col in ordinal_features if 'Medium' in df_encoded[col].unique()]
class_two_ordinal = [col for col in ordinal_features if 'College' in df_encoded[col].unique()]
class_three_ordinal = [col for col in ordinal_features if 'Moderate' in df_encoded[col].unique()]
class_four_ordinal = [col for col in ordinal_features if 'B' in df_encoded[col].unique()]

ordinal_classes = [class_one_ordinal, class_two_ordinal, class_three_ordinal, class_four_ordinal]

# set categories in each ordinal class
categories = [
    ['Low', 'Medium', 'High'],
    ['High School', 'College', 'Postgraduate'],
    ['Near', 'Moderate', 'Far'],
    ['D', 'C', 'B', 'A']
]

# Encode the classes 
for i, class_i in enumerate(ordinal_classes): 
	enc = OrdinalEncoder(categories=[categories[i]]*len(class_i), dtype=np.int64) df_encoded[class_i] = enc.fit_transform(df_encoded[class_i])
```
#### **4. Data Scaling**
By bringing data into the same scale will offers several beneficial of model training. Such as, Improving algorithm performance, faster convergence, prevention of numerical instability, equal features contribution, enhanced model stability and robustness. 

In this analysis, the min-max scaling method will be applied to the data. It does scale the data into range values between `0` to `1`. In essence, data scaling optimizes the data for machine learning algorithms, leading to better performance, faster training, and more reliable results.

Simply put the code just like this;
```python
scaler = MinMaxScaler()
df_scaled[numerical_cols] = scaler.fit_transform(df_cleaned[numerical_cols])
```

### **Build Clustering Model**
The algorithm will be trained is DBSCAN and Kmeans for clsutering:
- `DBSCAN` has a capability for cluster the dataset without effected by the outliers values. To be considered that our dataset has a lot of outliers
- `Kmeans` Another alogrithm that will be compared to DBSCAN. To recognize which the best solution

#### **Model Evaluation**
To determine the optimal number of clusters in a clustering model, we can use the **Elbow** or **Silhouette Score** method.

This method helps us find the number of clusters that provide the best separation between data groups, so that the model built can be more effective. Here are the the steps.

1. Use the Silhouette Score and Elbow Method to determine the optimal number of clusters.
2. Calculate the Silhouette Score as a measure of cluster quality.

The evaluation got `-0.168` of Silhouette Score for `DBSCAN` and `0.0621` for kmeans. Which they are sill far from our target which `0.55` minimum score.

#### **Feature Selections**
We will select the most relevant feautures for our model, and we drop the unrelevant columns for good cluster prediction

To do feature selection we can try the following strategies:
1. Drop Featrues with low variance
2. PCA (principle component analysis)

However, following table describes the summary of Silhouette Score of each strategies.

|               | dbscore    |                 | kmeans     |                    |
| ------------- | ---------- | --------------- | ---------- | ------------------ |
|               | Best Score | Number of Clust | Best Score | Number of Clusters |
| High Variance | 1          | 9               | 1          | 9                  |
| PCA           | 0.7411     | 8               | 0.7432     | 9                  |

And for the visualization of the clusters as follows:

<img src="https://raw.githubusercontent.com/sayid-alt/student-performance-factor-analysis/refs/heads/main/img/cluster_viz.png" />

### Cluster Results Analysis and Interpretation

1. Cluster 0:
    - ****Family_income**** = Low
    - ****Parental_education_level**** = Postgraduate
    - **Rata-rata exam score** = 66,496
    - **Rata-rata attendance** = 79,925
    - **Frekuensi** = 1286
    - **Analisis**: Kluster 1 memiliki nilai **family_income** yang rendah dan **parental_education_level** yang tinggi (postgraduate). Kemungkinan demografi pada siswa ini memungkinan tidak seimbangnya antara akonomi dan pencapaian pada pendidikannya, atau seorang orang tua yang lebih memfokuskan edukasi yang tinggi dari pada income yang tinggi.

2. Cluster 1:
    - ****Family_income**** = Medium
    - ****Parental_education_level**** = High School
    - **Rata-rata exam score** = 67.900
    - **Rata-rata attendance** = 79.706
    - **Frekuensi** = 521
    - **Analisis** = Kluster 2 memiliki nilai **family_income** yang tinggi dan **parental_education_level** yang tinggi (high school). Siswa ini memiliki keluarga yang mungkin memiliki kemampuan teknikal yang dapat meningkatkan income dan kestabilan ekonominya.

3. Cluster 2:
    - ****Family_income**** = High
    - **Parental_education_level** = College
    - **Rata-rata exam score** = 67.045
    - **Rata-rata attendance** = 79.644
    - **Frekuensi** = 1254
    - **Analisis** =  Kluster ini merepresentasikan keluarga yang memiliki income tinggi dan kemampuan intelektual yang tinggi. Sehingga dapat pekerjaan yang layak sesuai dari jurusan pendidikan yang diampu, seperti: management, engineering, atau spesialis.

4. Cluster 3:
    - **Family_income** = Low
    - **Parental_education_level** = High school
    - **Rata-rata exam score** = 66.860
    - **Rata-rata attendance** = 79.888
    - **Frekuensi** = 784
    - **Analisis** =  Kluster yang mempresentasikan keluarga dengan pemasukan yang rendah dan tingkat pendidikan yang rendah. Hal itu dapat mengindikasikan kelemahan ekonomi dan terbatasnya lapangan pekerjaan yang tersedia.

5. Cluster 4:

    - **Family_income** = Medium
    - **Parental_education_level** = College
    - **Rata-rata exam score** = 67.417
    - **Rata-rata attendance** = 79.591
    - **Frekuensi** = 619
    - **Analisis** = Kluster merepresentasikan potensial dari middle-class dimana yang dapat gaji yang stabil, pekerjaan ini seperti, guru atau Professional tingkat pemula

6. Cluster 5:
    - **Family_income** = Low
    - **Parental_education_level** = College
    - **Rata-rata exam score** = 67.740
    - **Rata-rata attendance** = 81.461
    - **Frekuensi** = 512
    - **Analisis**  Kluster ini menggambarkan, walau orang tua yang memiliki pendidikan yang tinggi, kluster ini memungkinkan underemployment, dimana pekerjaan yang mereka dapat tidak sesuai dengan gelar pendidikan yang didapatkan, atau mereka tinggal di wilayah yang pendapatan daerahnya rendah.

7. Cluster 6:
    - **Family_income** = High
    - **Parental_education_level** = High School
    - **Rata-rata exam score** = 67.539
    - **Rata-rata attendance** = 80.351
    - **Frekuensi** = 791
    - **Analisis** Kluster ini meskipun mempunyai pendidikan yang rendah, akan tetapi mempunyai income yang tinggi, ini memungkinkan seorang pengusaha, mempunyai warisan, atau kemampuan dalam berdagang yang tidak dimiliki oleh seorang yang pendidikan tinggi

8. Cluster 7:
    - **Family_income** = Medium
    - **Parental_education_level** = Postgraduate
    - **Rata-rata exam score** = 68.041
    - **Rata-rata attendance** = 80.148
    - **Frekuensi** = 364
    - **Analisis** = Kelompok ini mencakup keluarga dengan pendapatan sedang dan pendidikan pascasarjana. Ini mungkin menunjukkan profesi di bidang pendidikan, penelitian, atau layanan publik, yang sering kali memerlukan gelar lanjutan tetapi tidak selalu menawarkan gaji tinggi.

9. Cluster 8:
    - **Family_income** = High
    - **Parental_education_level** = Postgraduate
    - **Rata-rata exam score** = 68.607
    - **Rata-rata attendance** = 80.360
    - **Frekuensi** = 247
    - **Analisis** = Kelompok ini kemungkinan besar terdiri dari keluarga yang berpendidikan tinggi dan berpenghasilan tinggi. Orang tua dalam kelompok ini mungkin memiliki profesi bergengsi atau bergaji tinggi, seperti dokter, pengacara, atau eksekutif perusahaan.
