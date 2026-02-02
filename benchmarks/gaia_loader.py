from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class GAIATask:
    task_id: str
    question: str
    ground_truth: str
    category: str
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Curated GAIA Level 1 Suite: 30 verified tasks across 4 categories
CURATED_GAIA_TASKS = [
    # -------------------------------------------------------------
    # Category 1: Math & Algorithmic Reasoning (8 tasks)
    # -------------------------------------------------------------
    {
        "task_id": "gaia_l1_01",
        "question": "What is the sum of all prime numbers between 100 and 150?",
        "ground_truth": "1216",
        "category": "math_reasoning"
    },
    {
        "task_id": "gaia_l1_02",
        "question": "Calculate (2^19 - 1) mod 137.",
        "ground_truth": "125",
        "category": "math_reasoning"
    },
    {
        "task_id": "gaia_l1_03",
        "question": "If you invest $12,500 at an annual compound interest rate of 6.4% compounded monthly, what is the total balance after exactly 4 years? Round to two decimal places.",
        "ground_truth": "16135.93",
        "category": "math_reasoning"
    },
    {
        "task_id": "gaia_l1_04",
        "question": "Find the median of the following list of numbers: [45, 12, 88, 34, 91, 102, 3, 56, 77, 88, 19, 64].",
        "ground_truth": "60",
        "category": "math_reasoning"
    },
    {
        "task_id": "gaia_l1_05",
        "question": "How many anagrams can be formed from the letters of the word 'MISSISSIPPI'?",
        "ground_truth": "34650",
        "category": "math_reasoning"
    },
    {
        "task_id": "gaia_l1_06",
        "question": "Compute the determinant of the 3x3 matrix: [[3, 7, 2], [1, 4, 6], [8, 5, 9]].",
        "ground_truth": "237",
        "category": "math_reasoning"
    },
    {
        "task_id": "gaia_l1_07",
        "question": "What is the 30th Fibonacci number?",
        "ground_truth": "832040",
        "category": "math_reasoning"
    },
    {
        "task_id": "gaia_l1_08",
        "question": "How many days are there between January 15, 2024 and September 10, 2026 inclusive?",
        "ground_truth": "970",
        "category": "math_reasoning"
    },

    # -------------------------------------------------------------
    # Category 2: File Analysis & Data Extraction (8 tasks)
    # -------------------------------------------------------------
    {
        "task_id": "gaia_l1_09",
        "question": "Read 'benchmarks/data/server_logs.txt' and count the total number of lines with status code 500.",
        "ground_truth": "14",
        "category": "file_analysis",
        "file_path": "benchmarks/data/server_logs.txt"
    },
    {
        "task_id": "gaia_l1_10",
        "question": "In 'benchmarks/data/employee_salaries.csv', what is the average salary of employees in the 'Engineering' department? Round to the nearest whole integer.",
        "ground_truth": "118400",
        "category": "file_analysis",
        "file_path": "benchmarks/data/employee_salaries.csv"
    },
    {
        "task_id": "gaia_l1_11",
        "question": "In 'benchmarks/data/sales_q3.json', which product ID generated the highest total revenue (units_sold * unit_price)?",
        "ground_truth": "PROD_904",
        "category": "file_analysis",
        "file_path": "benchmarks/data/sales_q3.json"
    },
    {
        "task_id": "gaia_l1_12",
        "question": "Inspect 'benchmarks/data/inventory.csv' and find the name of the item with the lowest stock quantity.",
        "ground_truth": "Titanium Hex Bolt",
        "category": "file_analysis",
        "file_path": "benchmarks/data/inventory.csv"
    },
    {
        "task_id": "gaia_l1_13",
        "question": "Read 'benchmarks/data/config_payload.json' and report the value of database.connection.max_pool_size.",
        "ground_truth": "64",
        "category": "file_analysis",
        "file_path": "benchmarks/data/config_payload.json"
    },
    {
        "task_id": "gaia_l1_14",
        "question": "From 'benchmarks/data/server_logs.txt', find the IP address that generated the most 404 error entries.",
        "ground_truth": "192.168.1.105",
        "category": "file_analysis",
        "file_path": "benchmarks/data/server_logs.txt"
    },
    {
        "task_id": "gaia_l1_15",
        "question": "In 'benchmarks/data/customer_churn.csv', how many churned customers ('churn' == 'Yes') had monthly charges exceeding 80.00?",
        "ground_truth": "3",
        "category": "file_analysis",
        "file_path": "benchmarks/data/customer_churn.csv"
    },
    {
        "task_id": "gaia_l1_16",
        "question": "Read 'benchmarks/data/system_metrics.json' and report the node_id with the highest peak_cpu_percent.",
        "ground_truth": "node-2",
        "category": "file_analysis",
        "file_path": "benchmarks/data/system_metrics.json"
    },

    # -------------------------------------------------------------
    # Category 3: Web Search & Fact Retrieval (7 tasks)
    # -------------------------------------------------------------
    {
        "task_id": "gaia_l1_17",
        "question": "What is the capital city of Australia?",
        "ground_truth": "Canberra",
        "category": "web_search"
    },
    {
        "task_id": "gaia_l1_18",
        "question": "What chemical element has the atomic number 74?",
        "ground_truth": "Tungsten",
        "category": "web_search"
    },
    {
        "task_id": "gaia_l1_19",
        "question": "In what year did the Apollo 11 mission land humans on the Moon?",
        "ground_truth": "1969",
        "category": "web_search"
    },
    {
        "task_id": "gaia_l1_20",
        "question": "Who authored the famous 1818 novel 'Frankenstein; or, The Modern Prometheus'?",
        "ground_truth": "Mary Shelley",
        "category": "web_search"
    },
    {
        "task_id": "gaia_l1_21",
        "question": "What is the deepest known point in Earth's oceans?",
        "ground_truth": "Challenger Deep",
        "category": "web_search"
    },
    {
        "task_id": "gaia_l1_22",
        "question": "What is the currency of Japan?",
        "ground_truth": "Yen",
        "category": "web_search"
    },
    {
        "task_id": "gaia_l1_23",
        "question": "Who discovered penicillin in 1928?",
        "ground_truth": "Alexander Fleming",
        "category": "web_search"
    },

    # -------------------------------------------------------------
    # Category 4: Multi-Step Compositional Problems (7 tasks)
    # -------------------------------------------------------------
    {
        "task_id": "gaia_l1_24",
        "question": "Find the speed of light in vacuum in meters per second (exact integer standard value) and compute its remainder when divided by 1000.",
        "ground_truth": "458",
        "category": "multi_step"
    },
    {
        "task_id": "gaia_l1_25",
        "question": "In 'benchmarks/data/employee_salaries.csv', calculate the standard deviation of salaries for the 'Marketing' department. Round to two decimal places.",
        "ground_truth": "14247.55",
        "category": "multi_step",
        "file_path": "benchmarks/data/employee_salaries.csv"
    },
    {
        "task_id": "gaia_l1_26",
        "question": "Read 'benchmarks/data/sensor_readings.csv' and compute the Pearson correlation coefficient between 'temperature' and 'humidity'. Round to three decimal places.",
        "ground_truth": "-0.985",
        "category": "multi_step",
        "file_path": "benchmarks/data/sensor_readings.csv"
    },
    {
        "task_id": "gaia_l1_27",
        "question": "In 'benchmarks/data/sales_q3.json', compute the difference in total revenue between the highest and lowest revenue products. Round to two decimal places.",
        "ground_truth": "97725.00",
        "category": "multi_step",
        "file_path": "benchmarks/data/sales_q3.json"
    },
    {
        "task_id": "gaia_l1_28",
        "question": "In 'benchmarks/data/customer_churn.csv', what percentage of customers with 'Month-to-month' contract churned? Round to one decimal place.",
        "ground_truth": "66.7",
        "category": "multi_step",
        "file_path": "benchmarks/data/customer_churn.csv"
    },
    {
        "task_id": "gaia_l1_29",
        "question": "Find the sum of all atomic numbers of elements in Table Salt (Sodium and Chlorine).",
        "ground_truth": "28",
        "category": "multi_step"
    },
    {
        "task_id": "gaia_l1_30",
        "question": "In 'benchmarks/data/inventory.csv', calculate the total inventory valuation if each unit of stock has an average cost of $42.50. Round to two decimal places.",
        "ground_truth": "90397.50",
        "category": "multi_step",
        "file_path": "benchmarks/data/inventory.csv"
    }
]

def load_gaia_subset(category: Optional[str] = None) -> List[GAIATask]:
    tasks = []
    for item in CURATED_GAIA_TASKS:
        if category and item["category"] != category:
            continue
        tasks.append(GAIATask(**item))
    return tasks
