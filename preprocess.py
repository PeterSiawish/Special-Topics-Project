# Import important libraries
import pandas as pd

# Import the .csv file as a pandas dataframe
df = pd.read_csv("./personality_dataset.csv")

# Display first 15 rows of the dataset
df.head(15)

# Display info about the dataframe
df.info()

# Determine how many nulls are in each column
df.isnull().sum()

# Step 1: Deal with the numerical data
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

# Loop through the numerical columns, and fill null values with the median of that column
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# Step 2: Deal with the categorical data
categorical_cols = ['Stage_fear', 'Drained_after_socializing', 'Personality']

# Loop through the categorical columns, and fill null values with the mode of that column
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Make sure we have no null values
df.info()

# Step 3: Map the categorical data to an appropriate numerical value
binary_map = {"Yes": 1, "No": 0, "Introvert": 0, "Extrovert": 1}

df['Stage_fear'] = df['Stage_fear'].map(binary_map)
df['Drained_after_socializing'] = df['Drained_after_socializing'].map(binary_map)
df['Personality'] = df['Personality'].map(binary_map)

# Step 4: Normalize the value from 0-1
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Before saving the cleaned dataset, make sure everything is fine
print(df.info())
print(df.describe())

# Step 5: Save the clean data as a new .csv file
df.to_csv("cleaned_personality_dataset.csv", index=False)