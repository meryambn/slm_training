import csv
import json
import statistics
from pathlib import Path

DATA_DIR = Path("benchmarks/data")

# 1. Server logs:
log_lines = (DATA_DIR / "server_logs.txt").read_text(encoding="utf-8").splitlines()
c500 = sum(1 for line in log_lines if " 500 " in line)
from collections import Counter
c404_ips = Counter()
for line in log_lines:
    if " 404 " in line:
        ip = line.split(" - - ")[0]
        c404_ips[ip] += 1
top_404_ip = c404_ips.most_common(1)[0][0]

print(f"Server logs - 500 count: {c500} (expected 14)")
print(f"Server logs - Top 404 IP: {top_404_ip} (expected 192.168.1.105)")

# 2. Employee salaries:
with open(DATA_DIR / "employee_salaries.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
eng_salaries = [float(r["salary"]) for r in rows if r["department"] == "Engineering"]
mkt_salaries = [float(r["salary"]) for r in rows if r["department"] == "Marketing"]
eng_avg = round(statistics.mean(eng_salaries))
mkt_std = round(statistics.stdev(mkt_salaries), 2)
print(f"Employee salaries - Eng avg: {eng_avg} (expected 118400)")
print(f"Employee salaries - Mkt std: {mkt_std}")

# 3. Inventory:
with open(DATA_DIR / "inventory.csv", encoding="utf-8") as f:
    inv = list(csv.DictReader(f))
lowest_item = min(inv, key=lambda x: int(x["stock_quantity"]))["item_name"]
print(f"Inventory - lowest stock item: {lowest_item} (expected Titanium Hex Bolt)")

# 4. Sales Q3:
with open(DATA_DIR / "sales_q3.json", encoding="utf-8") as f:
    sales = json.load(f)
best_prod = max(sales, key=lambda x: x["units_sold"] * x["unit_price"])["product_id"]
print(f"Sales Q3 - top product: {best_prod} (expected PROD_904)")

# 5. Config:
with open(DATA_DIR / "config_payload.json", encoding="utf-8") as f:
    cfg = json.load(f)
pool_size = cfg["database"]["connection"]["max_pool_size"]
print(f"Config - max_pool_size: {pool_size} (expected 64)")

# 6. Sensor readings:
with open(DATA_DIR / "sensor_readings.csv", encoding="utf-8") as f:
    sensors = list(csv.DictReader(f))
temps = [float(s["temperature"]) for s in sensors]
humids = [float(s["humidity"]) for s in sensors]
mean_t = statistics.mean(temps)
mean_h = statistics.mean(humids)
num = sum((t - mean_t) * (h - mean_h) for t, h in zip(temps, humids))
den = (sum((t - mean_t)**2 for t in temps) * sum((h - mean_h)**2 for h in humids)) ** 0.5
corr = round(num / den, 3)
print(f"Sensor readings - correlation: {corr}")

# 7. Customer churn:
with open(DATA_DIR / "customer_churn.csv", encoding="utf-8") as f:
    churn_rows = list(csv.DictReader(f))
high_churn = sum(1 for r in churn_rows if r["churn"] == "Yes" and float(r["monthly_charges"]) > 80.0)
m2m = [r for r in churn_rows if r["contract_type"] == "Month-to-month"]
m2m_churn_pct = round(sum(1 for r in m2m if r["churn"] == "Yes") / len(m2m) * 100, 1)
print(f"Customer churn - high churn count: {high_churn} (expected 3)")
print(f"Customer churn - M2M churn pct: {m2m_churn_pct} (expected 66.7)")

# 8. System metrics:
with open(DATA_DIR / "system_metrics.json", encoding="utf-8") as f:
    sys_m = json.load(f)
max_node = max(sys_m["nodes"], key=lambda n: n["peak_cpu_percent"])["node_id"]
print(f"System metrics - peak CPU node: {max_node} (expected node-2)")
