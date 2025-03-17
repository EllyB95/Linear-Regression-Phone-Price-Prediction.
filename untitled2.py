#%%
import pandas as pd

# Sample exchange rates (update these with live rates if needed)
exchange_rates = {
    'INR': 0.012,  # 1 INR to USD
    'PKR': 0.0036, # 1 PKR to USD
    'CNY': 0.14,   # 1 CNY to USD
    'AED': 0.27,   # 1 AED to USD
    'USD': 1.0     # 1 USD to USD
}
#%%
def convert_column_to_usd(df, column_name, currency):
    """
    Converts the given currency column to USD and adds a new column with '_in_USD' suffix.
    """
    if currency in exchange_rates:
        df[column_name + '_in_USD'] = df[column_name] * exchange_rates[currency]
    return df

#%%
# Convert all currency columns to USD
for currency in exchange_rates.keys():
    if currency in df.columns:
        df = convert_column_to_usd(df, currency, currency)


#%%
