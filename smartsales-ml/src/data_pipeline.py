import pandas as pd

df = pd.read_excel(r'C:\Users\mamun\OneDrive\Desktop\BDREN-Project\smartsales-ml\data\row\online_retail_II.xlsx')

# Remove missing customers
df = df[df['Customer ID'].notna()]

# Remove cancelled invoices
df = df[~df['Invoice'].astype(str).str.startswith('C')]

# Remove invalid quantities
df = df[df['Quantity'] > 0]

# Remove invalid prices
df = df[df['Price'] > 0]

# Convert date
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Create revenue column
df['Revenue'] = df['Quantity'] * df['Price']

print(df.shape)
print(df.head())