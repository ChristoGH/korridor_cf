import pandas as pd

ZM = pd.read_csv('data/ZM.csv')
unique_combinations = ZM[['ProductId', 'BranchId', 'Currency']].drop_duplicates()
ZM.query("ProductId"==18,"BranchId"==11,"Currency"=='USD')
# Display the unique combinations

ZM.query('ProductId=18')
print(unique_combinations)

filtered_ZM = ZM.query("ProductId == 18 & BranchId == 11 & Currency == 'USD'")
filtered_ZM[[ 'EffectiveDate', 'Demand',]].to_csv('data/filtered_18_11_USD.csv', index=False)
list(filtered_ZM.columns)