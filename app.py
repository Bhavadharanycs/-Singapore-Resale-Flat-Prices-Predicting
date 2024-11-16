# Preprocessing the dataset
import pandas as pd

# Load the dataset
file_path = 'resale data.csv'
df = pd.read_csv(file_path)

# Convert 'remaining_lease' to total months
def parse_remaining_lease(lease_str):
    try:
        years, months = lease_str.split('years')
        months = months.strip().split(' ')[0] if 'month' in months else 0
        return int(years.strip()) * 12 + int(months)
    except:
        return None

df['remaining_lease_months'] = df['remaining_lease'].apply(parse_remaining_lease)

# Calculate lease age
current_year = 2024
df['lease_age'] = current_year - df['lease_commence_date']

# Drop unnecessary columns
df = df.drop(['month', 'block', 'street_name', 'remaining_lease'], axis=1)

# Encode categorical variables
df = pd.get_dummies(df, columns=['town', 'flat_type', 'storey_range', 'flat_model'], drop_first=True)

# Handle missing values
df = df.dropna()

# Save cleaned data
df.to_csv('/mnt/data/cleaned_resale_data.csv', index=False)
