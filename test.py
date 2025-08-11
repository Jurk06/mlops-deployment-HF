# test.py
import pandas as pd

# Load dataset
dataset = pd.read_csv('iris.csv')

# Define both possible schemas
original_cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'target']
clean_cols    = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']

# Detect which naming is present
if all(col in dataset.columns for col in original_cols):
    cols = original_cols
    sl, sw, pl, pw = cols[0], cols[1], cols[2], cols[3]
elif all(col in dataset.columns for col in clean_cols):
    cols = clean_cols
    sl, sw, pl, pw = cols[0], cols[1], cols[2], cols[3]
else:
    cols = None

# Test 1: Feature Check
feature_check = cols is not None

# Test 2: Value Ranges (only if features exist)
if feature_check:
    sepal_length_test = dataset[sl].between(4, 7).all()
    sepal_width_test  = dataset[sw].between(2, 5).all()
    petal_length_test = dataset[pl].between(1, 6).all()
    petal_width_test  = dataset[pw].between(0, 3).all()
else:
    sepal_length_test = sepal_width_test = petal_length_test = petal_width_test = False

# Test 3: Schema check (should have expected number of columns)
expected_columns = 5
actual_columns = dataset.shape[1]
schema_test = (actual_columns == expected_columns)

# Write test results
with open("test.txt", 'w') as outfile:
    outfile.write("Feature Test: %s\n" % ("Passed ✅" if feature_check else "Failed ❌"))
    outfile.write("Schema Test: %s\n" % ("Passed ✅" if schema_test else "Failed ❌"))
    outfile.write("Sepal Length Range Test: %s\n" % ("Passed ✅" if sepal_length_test else "Failed ❌"))
    outfile.write("Sepal Width Range Test: %s\n" % ("Passed ✅" if sepal_width_test else "Failed ❌"))
    outfile.write("Petal Length Range Test: %s\n" % ("Passed ✅" if petal_length_test else "Failed ❌"))
    outfile.write("Petal Width Range Test: %s\n" % ("Passed ✅" if petal_width_test else "Failed ❌"))

print("✅ Dataset validation complete. Results saved to test.txt")
