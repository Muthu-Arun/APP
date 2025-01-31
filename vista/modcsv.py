import pandas as pd

# Read the existing CSV file into a DataFrame
df = pd.read_csv('new_predictions.csv')

# Generate a set of all possible image_ids from 0 to 11999
all_image_ids = set(range(12000))

# Extract the existing image_ids from the CSV
existing_image_ids = set(df['image_id'])

# Find the missing image_ids
missing_image_ids = all_image_ids - existing_image_ids

# Create a DataFrame for the missing image_ids with label 0.5
missing_df = pd.DataFrame({'image_id': list(missing_image_ids), 'label': [0.5] * len(missing_image_ids)})

# Append the missing rows to the original DataFrame
df = pd.concat([df, missing_df])

# Sort the DataFrame by image_id
df = df.sort_values(by='image_id').reset_index(drop=True)

# Write the updated DataFrame back to a CSV file
df.to_csv('updated_file.csv', index=False)

print(f"Added {len(missing_image_ids)} missing image_ids with label 0.5.")
