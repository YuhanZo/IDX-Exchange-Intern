import mysql.connector
import pandas as pd
import os

#connect the database
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='root',
    database='real_estate'
)

#extract the data from the database
query = """
SELECT L_ListingID, L_Address, L_City, L_Keyword2 as beds,
       LM_Dec_3 as baths, L_SystemPrice as price, L_Remarks as remarks
FROM rets_property
WHERE L_Remarks IS NOT NULL AND LENGTH(L_Remarks) > 50
ORDER BY RAND() LIMIT 1000
"""

print("Extracting data from the database...")
df = pd.read_sql(query, conn)
conn.close()

#ensure the output directory exists
os.makedirs('data/processed', exist_ok=True)

#save the extracted data to a CSV file
df.to_csv('data/processed/listing_sample.csv', index=False)
print("Data extraction complete. Saved to data/processed/listing_sample.csv")
print(f"Extracted {len(df)} records.")
print(f"it has been saved to data/processed/listing_sample.csv")