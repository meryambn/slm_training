"""
Generator for curated GAIA benchmark data files.
Ensures exact, verified ground truths for reproducible evaluation.
"""
from pathlib import Path
import csv
import json
import statistics

DATA_DIR = Path("benchmarks/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_server_logs():
    lines = []
    # 14 lines with 500 status code
    for i in range(14):
        lines.append(f"10.0.0.{i+1} - - [10/Sep/2026:08:{i:02d}:15 +0000] \"GET /api/v1/checkout HTTP/1.1\" 500 482")
    # 8 lines with 404 for 192.168.1.105
    for i in range(8):
        lines.append(f"192.168.1.105 - - [10/Sep/2026:09:{i:02d}:30 +0000] \"GET /missing/resource_{i} HTTP/1.1\" 404 154")
    # Other 404s with fewer counts
    for i in range(3):
        lines.append(f"192.168.1.110 - - [10/Sep/2026:09:15:00 +0000] \"GET /bad/path HTTP/1.1\" 404 154")
    # Normal 200 logs
    for i in range(35):
        lines.append(f"172.16.0.{i+1} - - [10/Sep/2026:10:{i:02d}:00 +0000] \"GET /index.html HTTP/1.1\" 200 2410")
    
    (DATA_DIR / "server_logs.txt").write_text("\n".join(lines), encoding="utf-8")
    print("Generated server_logs.txt")

def generate_employee_salaries():
    sal_m = [71342, 81000, 91000, 104658]
    rows = [
        ["id", "name", "department", "salary"],
        ["E101", "Alice Smith", "Engineering", "110000"],
        ["E102", "Bob Jones", "Engineering", "115000"],
        ["E103", "Charlie Brown", "Engineering", "118400"],
        ["E104", "Diana Prince", "Engineering", "120600"],
        ["E105", "Evan Wright", "Engineering", "128000"],
        ["M201", "Fiona Gallagher", "Marketing", str(sal_m[0])],
        ["M202", "George Clark", "Marketing", str(sal_m[1])],
        ["M203", "Hannah Abbott", "Marketing", str(sal_m[2])],
        ["M204", "Ian Malcolm", "Marketing", str(sal_m[3])],
        ["S301", "Julia Roberts", "Sales", "95000"],
        ["S302", "Kevin Bacon", "Sales", "102000"]
    ]
    with open(DATA_DIR / "employee_salaries.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print("Generated employee_salaries.csv")

def generate_inventory():
    rows = [
        ["sku", "item_name", "category", "stock_quantity", "reorder_level"],
        ["SKU-001", "Carbon Steel Washer", "Fasteners", "1250", "200"],
        ["SKU-002", "Brass Bushing 10mm", "Bearings", "85", "30"],
        ["SKU-003", "Titanium Hex Bolt", "Fasteners", "4", "25"],
        ["SKU-004", "Ceramic Insulator", "Electrical", "320", "50"],
        ["SKU-005", "Neoprene O-Ring", "Seals", "450", "100"],
        ["SKU-006", "Aluminum Extrusion 1m", "Structural", "18", "10"]
    ]
    with open(DATA_DIR / "inventory.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print("Generated inventory.csv")

def generate_sensor_readings():
    temps = [20.0, 21.5, 23.0, 24.5, 26.0, 27.5, 29.0, 31.0, 32.5, 34.0]
    humids = [68.0, 65.0, 63.0, 58.0, 56.0, 50.0, 48.0, 42.0, 45.0, 38.0]
    rows = [["timestamp", "temperature", "humidity"]]
    for i, (t, h) in enumerate(zip(temps, humids)):
        rows.append([f"2026-09-10T12:{i:02d}:00Z", f"{t:.1f}", f"{h:.1f}"])
    with open(DATA_DIR / "sensor_readings.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print("Generated sensor_readings.csv")

def generate_customer_churn():
    # 12 customer records, 5 churned ('Yes'), 3 churned with monthly_charges > 80.00
    rows = [
        ["customer_id", "contract_type", "tenure_months", "monthly_charges", "churn"],
        ["CUST-001", "Month-to-month", "2", "85.50", "Yes"],
        ["CUST-002", "Two year", "45", "64.20", "No"],
        ["CUST-003", "One year", "18", "92.10", "Yes"],
        ["CUST-004", "Month-to-month", "6", "45.00", "No"],
        ["CUST-005", "Month-to-month", "1", "78.00", "Yes"],
        ["CUST-006", "Two year", "60", "110.00", "No"],
        ["CUST-007", "Month-to-month", "3", "89.90", "Yes"],
        ["CUST-008", "One year", "24", "55.40", "No"],
        ["CUST-009", "Month-to-month", "12", "72.00", "Yes"],
        ["CUST-010", "Two year", "36", "68.50", "No"],
        ["CUST-011", "One year", "14", "49.00", "No"],
        ["CUST-012", "Month-to-month", "8", "95.00", "No"]
    ]
    with open(DATA_DIR / "customer_churn.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print("Generated customer_churn.csv")

def generate_system_metrics():
    data = {
        "cluster_id": "us-east-compute-04",
        "nodes": [
            {"node_id": "node-1", "cpu_cores": 32, "memory_gb": 128, "peak_cpu_percent": 74.2, "status": "active"},
            {"node_id": "node-2", "cpu_cores": 64, "memory_gb": 256, "peak_cpu_percent": 91.8, "status": "active"},
            {"node_id": "node-3", "cpu_cores": 32, "memory_gb": 128, "peak_cpu_percent": 45.0, "status": "idle"},
            {"node_id": "node-4", "cpu_cores": 64, "memory_gb": 256, "peak_cpu_percent": 88.5, "status": "active"}
        ],
        "gateway": {
            "rate_limit_rps": 15000,
            "ssl_enabled": True
        }
    }
    (DATA_DIR / "system_metrics.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("Generated system_metrics.json")

if __name__ == "__main__":
    generate_server_logs()
    generate_employee_salaries()
    generate_inventory()
    generate_sensor_readings()
    generate_customer_churn()
    generate_system_metrics()
