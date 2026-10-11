"""Seed the PostgreSQL warehouse with a deterministic, multi-table e-commerce dataset.

Run:  python data/seed.py

13 tables (up from the original 5) — deliberately more than a schema-retrieval
step needs to "cheat" by just showing everything: app/text2sql/schema_retriever.py
is expected to select a handful of relevant tables per question out of this
catalog, not the whole thing.
"""
import asyncio
import os
import random
from datetime import date, timedelta

import asyncpg

DATABASE_URL = os.getenv("DATABASE_ADMIN_URL") or os.getenv(
    "DATABASE_URL", "postgresql://text2sql:text2sql@localhost:5432/text2sql"
)

FIRST = ["Aarav", "Diya", "Vihaan", "Ananya", "Ishaan", "Meera", "Kabir", "Sara",
         "Liam", "Emma", "Noah", "Olivia", "Mateo", "Sofia", "Yuki", "Chen",
         "Fatima", "Omar", "Elena", "Lucas"]
LAST = ["Sharma", "Patel", "Iyer", "Khan", "Reddy", "Gupta", "Smith", "Garcia",
        "Kim", "Tanaka", "Muller", "Rossi", "Silva", "Nair", "Das", "Bose"]
COUNTRIES = ["India", "United States", "Germany", "Japan", "Brazil",
             "United Kingdom", "Australia", "United Arab Emirates"]
SEGMENTS = ["consumer", "small_business", "enterprise"]
CHANNELS = ["web", "mobile_app", "marketplace"]
ORDER_STATUS = ["completed", "shipped", "processing", "cancelled", "returned"]
PAY_METHODS = ["credit_card", "upi", "paypal", "bank_transfer", "wallet"]
PAY_STATUS = ["captured", "pending", "failed", "refunded"]
SHIP_CARRIERS = ["DHL", "FedEx", "UPS", "BlueDart", "Aramex"]
SHIP_STATUS = ["delivered", "in_transit", "returned_to_sender"]
RETURN_REASONS = ["defective", "wrong_item", "no_longer_needed", "damaged_in_transit"]
EMP_ROLES = ["sales_rep", "account_manager"]
CAMPAIGN_CHANNELS = ["email", "search_ads", "social", "affiliate"]

CATALOG = {
    "Electronics": [("Wireless Earbuds", 79.99), ("Mechanical Keyboard", 129.0), ("4K Monitor", 349.0),
                    ("USB-C Hub", 45.0), ("Smart Speaker", 99.0), ("Webcam Pro", 89.0),
                    ("Portable SSD 1TB", 119.0), ("Noise-Cancel Headphones", 249.0)],
    "Home & Kitchen": [("Espresso Machine", 199.0), ("Air Fryer", 149.0), ("Chef Knife Set", 89.0),
                       ("Robot Vacuum", 299.0), ("Cast Iron Pan", 39.0), ("Blender Max", 79.0)],
    "Sports": [("Yoga Mat", 29.0), ("Running Shoes", 119.0), ("Dumbbell Set", 149.0),
               ("Cycling Helmet", 59.0), ("Resistance Bands", 19.0)],
    "Books": [("Data Engineering 101", 42.0), ("The Startup Playbook", 24.0),
              ("Deep Learning Illustrated", 55.0), ("Mystery of the Nile", 14.0)],
    "Beauty": [("Vitamin C Serum", 25.0), ("Sunscreen SPF50", 18.0), ("Hair Dryer Ionic", 69.0)],
}

START = date(2024, 1, 1)
END = date(2026, 6, 30)
DAYS = (END - START).days

SCHEMA_SQL = """
DROP SCHEMA IF EXISTS public CASCADE;
CREATE SCHEMA public;

CREATE TABLE customers (
    customer_id  SERIAL PRIMARY KEY,
    name         VARCHAR NOT NULL,
    email        VARCHAR NOT NULL,
    country      VARCHAR NOT NULL,
    segment      VARCHAR NOT NULL,
    signup_date  DATE NOT NULL
);

CREATE TABLE suppliers (
    supplier_id  SERIAL PRIMARY KEY,
    name         VARCHAR NOT NULL,
    country      VARCHAR NOT NULL
);

CREATE TABLE warehouses (
    warehouse_id SERIAL PRIMARY KEY,
    name         VARCHAR NOT NULL,
    country      VARCHAR NOT NULL
);

CREATE TABLE employees (
    employee_id  SERIAL PRIMARY KEY,
    name         VARCHAR NOT NULL,
    role         VARCHAR NOT NULL,
    region       VARCHAR NOT NULL
);

CREATE TABLE products (
    product_id   SERIAL PRIMARY KEY,
    name         VARCHAR NOT NULL,
    category     VARCHAR NOT NULL,
    unit_price   NUMERIC(10,2) NOT NULL,
    supplier_id  INTEGER REFERENCES suppliers(supplier_id)
);

CREATE TABLE orders (
    order_id      SERIAL PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    sales_rep_id  INTEGER REFERENCES employees(employee_id),
    order_date    DATE NOT NULL,
    status        VARCHAR NOT NULL,
    channel       VARCHAR NOT NULL,
    region        VARCHAR NOT NULL
);

CREATE TABLE order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL,
    unit_price    NUMERIC(10,2) NOT NULL,
    discount      NUMERIC(4,3) NOT NULL DEFAULT 0
);

CREATE TABLE payments (
    payment_id  SERIAL PRIMARY KEY,
    order_id    INTEGER NOT NULL REFERENCES orders(order_id),
    method      VARCHAR NOT NULL,
    status      VARCHAR NOT NULL,
    amount      NUMERIC(10,2) NOT NULL,
    paid_date   DATE
);

CREATE TABLE shipments (
    shipment_id   SERIAL PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    warehouse_id  INTEGER REFERENCES warehouses(warehouse_id),
    carrier       VARCHAR NOT NULL,
    shipped_date  DATE,
    status        VARCHAR NOT NULL
);

CREATE TABLE returns (
    return_id       SERIAL PRIMARY KEY,
    order_item_id   INTEGER NOT NULL REFERENCES order_items(order_item_id),
    reason          VARCHAR NOT NULL,
    refund_amount   NUMERIC(10,2) NOT NULL,
    return_date     DATE NOT NULL
);

CREATE TABLE product_reviews (
    review_id    SERIAL PRIMARY KEY,
    product_id   INTEGER NOT NULL REFERENCES products(product_id),
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    rating       INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    review_date  DATE NOT NULL
);

CREATE TABLE marketing_campaigns (
    campaign_id  SERIAL PRIMARY KEY,
    name         VARCHAR NOT NULL,
    channel      VARCHAR NOT NULL,
    start_date   DATE NOT NULL,
    end_date     DATE NOT NULL,
    budget       NUMERIC(12,2) NOT NULL
);

CREATE TABLE campaign_attribution (
    order_id     INTEGER NOT NULL REFERENCES orders(order_id),
    campaign_id  INTEGER NOT NULL REFERENCES marketing_campaigns(campaign_id),
    PRIMARY KEY (order_id, campaign_id)
);
"""


def _rand_date(rng: random.Random) -> date:
    return START + timedelta(days=rng.randint(0, DAYS))


def _region_for_country(country: str) -> str:
    return "india" if country == "India" else ("us" if country == "United States" else "row")


async def main(database_url: str = DATABASE_URL) -> None:
    rng = random.Random(42)
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(SCHEMA_SQL)

        # ---- customers ----
        customers = [
            (f"{rng.choice(FIRST)} {rng.choice(LAST)}", None, rng.choice(COUNTRIES),
             rng.choice(SEGMENTS), _rand_date(rng))
            for _ in range(300)
        ]
        customers = [(n, f"{n.lower().replace(' ', '.')}{i}@example.com", c, s, d)
                     for i, (n, _, c, s, d) in enumerate(customers)]
        await conn.executemany(
            "INSERT INTO customers (name, email, country, segment, signup_date) "
            "VALUES ($1,$2,$3,$4,$5)", customers)

        # ---- suppliers / warehouses / employees ----
        suppliers = [(f"{rng.choice(LAST)} Supply Co.", rng.choice(COUNTRIES)) for _ in range(10)]
        await conn.executemany("INSERT INTO suppliers (name, country) VALUES ($1,$2)", suppliers)

        warehouses = [(f"{c} Fulfillment Center", c)
                      for c in ["India", "United States", "Germany", "United Arab Emirates"]]
        await conn.executemany("INSERT INTO warehouses (name, country) VALUES ($1,$2)", warehouses)

        employees = [(f"{rng.choice(FIRST)} {rng.choice(LAST)}", rng.choice(EMP_ROLES),
                     rng.choice(["us", "india", "row"])) for _ in range(15)]
        await conn.executemany(
            "INSERT INTO employees (name, role, region) VALUES ($1,$2,$3)", employees)

        supplier_ids = list(range(1, len(suppliers) + 1))
        products = []
        for category, items in CATALOG.items():
            for name, price in items:
                products.append((name, category, price, rng.choice(supplier_ids)))
        await conn.executemany(
            "INSERT INTO products (name, category, unit_price, supplier_id) "
            "VALUES ($1,$2,$3,$4)", products)

        product_rows = await conn.fetch("SELECT product_id, unit_price FROM products")
        customer_rows = await conn.fetch("SELECT customer_id, country FROM customers")
        employee_ids = list(range(1, len(employees) + 1))
        warehouse_ids = list(range(1, len(warehouses) + 1))

        # ---- orders + order_items + payments + shipments ----
        orders, order_items, payments, shipments = [], [], [], []
        order_id = 0
        for customer_id, country in [(r["customer_id"], r["country"]) for r in customer_rows]:
            n_orders = rng.choices([0, 1, 2, 3, 4], weights=[10, 30, 30, 20, 10])[0]
            for _ in range(n_orders):
                order_id += 1
                order_date = _rand_date(rng)
                status = rng.choices(ORDER_STATUS, weights=[55, 20, 10, 10, 5])[0]
                region = _region_for_country(country)
                orders.append((customer_id, rng.choice(employee_ids), order_date, status,
                               rng.choice(CHANNELS), region))

                n_items = rng.randint(1, 4)
                item_total = 0.0
                for _ in range(n_items):
                    prod = rng.choice(product_rows)
                    qty = rng.randint(1, 3)
                    discount = rng.choice([0, 0, 0, 0.1, 0.15, 0.2])
                    order_items.append((order_id, prod["product_id"], qty,
                                        float(prod["unit_price"]), discount))
                    item_total += qty * float(prod["unit_price"]) * (1 - discount)

                pay_status = "refunded" if status == "returned" else (
                    "failed" if status == "cancelled" else "captured")
                payments.append((order_id, rng.choice(PAY_METHODS), pay_status,
                                 round(item_total, 2),
                                 order_date + timedelta(days=rng.randint(0, 2))
                                 if pay_status == "captured" else None))

                if status in ("completed", "shipped", "returned"):
                    shipments.append((order_id, rng.choice(warehouse_ids), rng.choice(SHIP_CARRIERS),
                                      order_date + timedelta(days=rng.randint(1, 5)),
                                      rng.choice(SHIP_STATUS)))

        await conn.executemany(
            "INSERT INTO orders (customer_id, sales_rep_id, order_date, status, channel, region) "
            "VALUES ($1,$2,$3,$4,$5,$6)", orders)
        await conn.executemany(
            "INSERT INTO order_items (order_id, product_id, quantity, unit_price, discount) "
            "VALUES ($1,$2,$3,$4,$5)", order_items)
        await conn.executemany(
            "INSERT INTO payments (order_id, method, status, amount, paid_date) "
            "VALUES ($1,$2,$3,$4,$5)", payments)
        await conn.executemany(
            "INSERT INTO shipments (order_id, warehouse_id, carrier, shipped_date, status) "
            "VALUES ($1,$2,$3,$4,$5)", shipments)

        # ---- returns (a fraction of order_items) ----
        item_rows = await conn.fetch("SELECT order_item_id FROM order_items")
        returns = []
        for row in item_rows:
            if rng.random() < 0.05:
                returns.append((row["order_item_id"], rng.choice(RETURN_REASONS),
                                round(rng.uniform(10, 200), 2), _rand_date(rng)))
        await conn.executemany(
            "INSERT INTO returns (order_item_id, reason, refund_amount, return_date) "
            "VALUES ($1,$2,$3,$4)", returns)

        # ---- product reviews ----
        reviews = []
        for _ in range(500):
            prod = rng.choice(product_rows)
            cust = rng.choice(customer_rows)
            reviews.append((prod["product_id"], cust["customer_id"],
                            rng.choices([1, 2, 3, 4, 5], weights=[5, 5, 15, 35, 40])[0],
                            _rand_date(rng)))
        await conn.executemany(
            "INSERT INTO product_reviews (product_id, customer_id, rating, review_date) "
            "VALUES ($1,$2,$3,$4)", reviews)

        # ---- marketing campaigns + attribution ----
        campaigns = []
        for i in range(8):
            start = _rand_date(rng)
            campaigns.append((f"{rng.choice(CAMPAIGN_CHANNELS).title()} Push {i + 1}",
                              rng.choice(CAMPAIGN_CHANNELS), start,
                              start + timedelta(days=rng.randint(14, 60)),
                              round(rng.uniform(5000, 50000), 2)))
        await conn.executemany(
            "INSERT INTO marketing_campaigns (name, channel, start_date, end_date, budget) "
            "VALUES ($1,$2,$3,$4,$5)", campaigns)

        campaign_ids = list(range(1, len(campaigns) + 1))
        attribution = [(oid, rng.choice(campaign_ids))
                       for oid in range(1, order_id + 1) if rng.random() < 0.35]
        await conn.executemany(
            "INSERT INTO campaign_attribution (order_id, campaign_id) VALUES ($1,$2)",
            list({(o, c) for o, c in attribution}))

        print(f"Seeded: {len(customers)} customers, {order_id} orders, "
              f"{len(order_items)} order_items, {len(products)} products across "
              f"{len(CATALOG)} categories, 13 tables total.")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
