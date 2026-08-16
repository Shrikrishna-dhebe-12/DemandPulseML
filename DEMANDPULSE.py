import numpy as np
import pandas as pd

np.random.seed(42)

# Date range: 3 years of daily data
dates = pd.date_range(start="2023-01-01", end="2025-12-31", freq="D")
n_rows = len(dates)

# Multiple Amazon-style products
products = [
    {"asin": "B073496Y7S", "title": "boAt Rockerz Wireless Bluetooth Earbuds", "mrp": 2990.00},
    {"asin": "B08N5WRWNW", "title": "Noise ColorFit Smartwatch", "mrp": 4999.00},
    {"asin": "B07X9Y3Q4S", "title": "Mi Power Bank 10000mAh", "mrp": 1299.00},
    {"asin": "B09V4D4N5T", "title": "Samsung Galaxy M13 Smartphone", "mrp": 13999.00},
    {"asin": "B07HGH8D2R", "title": "HP Wireless Mouse", "mrp": 799.00}
]

data_list = []

for i, date in enumerate(dates):
    day_of_week = date.dayofweek
    month = date.month
    is_weekend = 1 if day_of_week in [5, 6] else 0

    # Pick random product
    product = np.random.choice(products)
    product_id = product["asin"]
    product_title = product["title"]
    mrp = product["mrp"]

    # Sale season logic
    is_sale_season = 1 if (month in [10, 11] and date.day <= 15) or (month == 9 and date.day >= 23) or (
                month == 1 and date.day == 26) else 0

    # Discount logic
    discount_percent = np.random.choice([40, 50, 60, 65, 70],
                                        p=[0.2, 0.4, 0.2, 0.1, 0.1]) if is_sale_season else np.random.choice(
        [20, 30, 40, 50], p=[0.4, 0.3, 0.2, 0.1])
    selling_price = round(mrp * (1 - discount_percent / 100), 2)

    # Ratings & reviews
    customer_rating = round(np.random.uniform(3.8, 4.5), 1)
    review_count = int(np.random.randint(5000, 15000) + (i * 3))

    # Marketing
    ad_spend = np.random.randint(2000, 8000) if is_sale_season else np.random.randint(500, 2000)
    page_views = int(ad_spend * np.random.uniform(4, 7) + np.random.randint(1000, 5000))

    # Demand calculation
    base_demand = 30
    sale_boost = 100 if is_sale_season else 0
    discount_boost = discount_percent * 1.2
    traffic_effect = page_views * 0.006
    rating_effect = (customer_rating - 3.0) * 12
    noise = np.random.normal(0, 8)

    actual_demand = max(5, int(base_demand + sale_boost + discount_boost + traffic_effect + rating_effect + noise))

    # Inventory & stockout
    inventory_level = np.random.randint(actual_demand // 2, actual_demand * 2)
    is_stockout = 1 if inventory_level < actual_demand else 0
    units_sold = actual_demand if is_stockout == 0 else inventory_level

    data_list.append({
        "date": date.strftime("%Y-%m-%d"),
        "asin": product_id,
        "product_name": product_title,
        "mrp": mrp,
        "selling_price": selling_price,
        "discount_percent": discount_percent,
        "customer_rating": customer_rating,
        "total_reviews": review_count,
        "page_views": page_views,
        "ad_spend_inr": ad_spend,
        "is_sale_season": is_sale_season,
        "is_weekend": is_weekend,
        "inventory_level": inventory_level,
        "units_sold": units_sold,
        "is_stockout": is_stockout,
        "actual_demand": actual_demand
    })

df = pd.DataFrame(data_list)
file_name = "amazon_multi_product_demand.csv"
df.to_csv(file_name, index=False)
print(f"✅ Success! Dataset with {len(df)} rows & multiple products saved to '{file_name}'.")
