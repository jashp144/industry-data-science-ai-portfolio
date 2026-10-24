import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DatabaseManager:
    def __init__(self, db_path: str = None):
        if db_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(project_root, "data", "enterprise_analytics.db")
        self.db_path = db_path

    def get_connection(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        return sqlite3.connect(self.db_path)

    def initialize_and_seed_database(self, random_state: int = 42):
        """Creates tables and populates with rich enterprise relational sales data."""
        np.random.seed(random_state)
        conn = self.get_connection()
        cursor = conn.cursor()

        # Drop existing tables
        cursor.executescript("""
            DROP TABLE IF EXISTS order_items;
            DROP TABLE IF EXISTS orders;
            DROP TABLE IF EXISTS products;
            DROP TABLE IF EXISTS customers;

            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                region TEXT NOT NULL,
                segment TEXT NOT NULL,
                signup_date DATE NOT NULL
            );

            CREATE TABLE products (
                product_id INTEGER PRIMARY KEY,
                product_name TEXT NOT NULL,
                category TEXT NOT NULL,
                cost_price REAL NOT NULL,
                retail_price REAL NOT NULL
            );

            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                order_date DATE NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
            );

            CREATE TABLE order_items (
                item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                unit_price REAL NOT NULL,
                discount REAL NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders (order_id),
                FOREIGN KEY (product_id) REFERENCES products (product_id)
            );
        """)

        # 1. Seed Customers (100 customers)
        regions = ["North America", "Europe", "Asia Pacific", "Latin America"]
        segments = ["Enterprise", "Mid-Market", "SMB", "Consumer"]
        cust_records = []
        for i in range(1, 101):
            signup = datetime(2023, 1, 1) + timedelta(days=int(np.random.randint(0, 365)))
            cust_records.append((
                i,
                f"Client_{i:03d}",
                np.random.choice(regions),
                np.random.choice(segments),
                signup.strftime("%Y-%m-%d")
            ))
        cursor.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", cust_records)

        # 2. Seed Products (15 products across 3 categories)
        products = [
            (1, "Cloud Server Small", "Compute", 45.0, 75.0),
            (2, "Cloud Server Large", "Compute", 150.0, 240.0),
            (3, "GPU Accelerator H100", "Compute", 800.0, 1400.0),
            (4, "NVMe Storage 1TB", "Storage", 20.0, 45.0),
            (5, "Cold Storage 10TB", "Storage", 60.0, 110.0),
            (6, "High-Speed CDN Pool", "Networking", 35.0, 65.0),
            (7, "Enterprise Firewall", "Networking", 120.0, 210.0),
            (8, "Managed DB Instance", "Database", 90.0, 160.0),
            (9, "Vector Search Cluster", "Database", 200.0, 350.0),
            (10, "DevOps Pipeline Runner", "Developer Tools", 25.0, 50.0)
        ]
        cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)", products)

        # 3. Seed Orders & Order Items (500 orders across 2024)
        order_records = []
        order_item_records = []
        statuses = ["Delivered", "Delivered", "Delivered", "Processing", "Cancelled"]

        start_date = datetime(2024, 1, 1)
        for order_id in range(1, 501):
            cust_id = int(np.random.randint(1, 101))
            order_date = start_date + timedelta(days=int(np.random.randint(0, 360)))
            status = np.random.choice(statuses, p=[0.75, 0.10, 0.05, 0.05, 0.05])
            order_records.append((order_id, cust_id, order_date.strftime("%Y-%m-%d"), status))

            # Items in order
            n_items = int(np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))
            chosen_prods = np.random.choice(range(len(products)), size=n_items, replace=False)
            for p_idx in chosen_prods:
                p_id, _, _, _, r_price = products[p_idx]
                qty = int(np.random.randint(1, 5))
                disc = float(np.random.choice([0.0, 0.05, 0.10, 0.15], p=[0.7, 0.15, 0.1, 0.05]))
                order_item_records.append((order_id, p_id, qty, r_price, disc))

        cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", order_records)
        cursor.executemany("INSERT INTO order_items (order_id, product_id, quantity, unit_price, discount) VALUES (?, ?, ?, ?, ?)", order_item_records)

        conn.commit()
        conn.close()
        print(f"Database seeded successfully at {self.db_path} with 100 customers, 10 products, 500 orders.")

    def execute_query(self, query: str) -> pd.DataFrame:
        """Executes read-only query and returns a pandas DataFrame."""
        conn = self.get_connection()
        try:
            df = pd.read_sql_query(query, conn)
            return df
        finally:
            conn.close()
