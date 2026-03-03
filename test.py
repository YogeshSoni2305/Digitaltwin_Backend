import json
import random
import hashlib
from pathlib import Path

# -------------------------
# CONFIGURATION
# -------------------------

TOTAL_EMPLOYEES = 500
OUTPUT_FILE = "company.json"
SEED = 42
random.seed(SEED)

# Salary Bands (USD)
SALARY_BANDS = {
    "Executive": (250000, 400000),
    "VP": (180000, 250000),
    "Director": (140000, 190000),
    "Manager": (110000, 150000),
    "Engineer": (80000, 140000),
    "Sales": (70000, 130000),
    "Operations": (70000, 120000),
    "Product": (90000, 140000),
    "Finance": (80000, 130000),
    "HR": (60000, 110000),
}

# Name Pools
FIRST_NAMES = [
    "Aarav","Vivaan","Arjun","Rohan","Kabir","Neha","Ananya","Priya","Kavya","Meera",
    "Michael","Daniel","Sophia","Olivia","Ethan","Liam","Emma","Ava","Noah","Lucas",
    "Isabella","Mason","Logan","Amelia","Elijah","James","Charlotte","Benjamin","Harper","Henry",
    "Alexander","Scarlett","Sebastian","Victoria","Matthew","Grace","Jack","Chloe","David","Luna"
]

LAST_NAMES = [
    "Sharma","Patel","Verma","Singh","Mehta","Johnson","Brown","Smith","Davis","Wilson",
    "Garcia","Martinez","Anderson","Thomas","Taylor","Moore","Jackson","Martin","Lee","Perez"
]

# Skills by Department
DEPARTMENT_SKILLS = {
    "Engineering": ["backend", "frontend", "devops", "data", "cloud", "architecture"],
    "Sales": ["negotiation", "client_acquisition", "enterprise_sales", "communication"],
    "Operations": ["process_optimization", "delivery_management", "vendor_management"],
    "Product": ["roadmap_planning", "stakeholder_alignment", "analytics"],
    "Finance": ["financial_modeling", "budgeting", "forecasting"],
    "HR": ["talent_acquisition", "performance_management"]
}

# Department Distribution (Realistic Services Company)
DEPARTMENT_COUNTS = {
    "Engineering": 240,
    "Sales": 80,
    "Operations": 70,
    "Product": 60,
    "Finance": 30,
    "HR": 20
}

# -------------------------
# UTILITIES
# -------------------------

def generate_name(index):
    first = FIRST_NAMES[index % len(FIRST_NAMES)]
    last = LAST_NAMES[(index * 7) % len(LAST_NAMES)]
    return f"{first} {last}"

def random_salary(role):
    low, high = SALARY_BANDS[role]
    return random.randint(low, high)

def generate_skills(department, seniority_level):
    skills = {}
    skill_pool = DEPARTMENT_SKILLS.get(department, [])
    selected = random.sample(skill_pool, min(3, len(skill_pool)))

    for skill in selected:
        base = 0.6 if seniority_level in ["Manager", "Director", "VP", "Executive"] else 0.5
        skills[skill] = round(random.uniform(base, 0.95), 2)

    if seniority_level in ["Director", "VP", "Executive"]:
        skills["leadership"] = round(random.uniform(0.8, 0.95), 2)
        skills["strategy"] = round(random.uniform(0.75, 0.9), 2)

    return skills

def create_employee(emp_id, name, role, department, reports_to):
    return {
        "id": emp_id,
        "name": name,
        "role": role,
        "department": department,
        "salary": random_salary(role),
        "skills": generate_skills(department, role),
        "capacity_hours_per_week": 40,
        "reports_to": reports_to,
        "productivity_multiplier": 1.0
    }

# -------------------------
# BUILD COMPANY
# -------------------------

employees = []
edges = []

emp_counter = 1

def next_id():
    global emp_counter
    eid = f"EMP{emp_counter:04d}"
    emp_counter += 1
    return eid

# 1️⃣ EXECUTIVE LAYER
exec_roles = ["CEO", "CTO", "CFO", "COO", "CHRO", "CRO"]
exec_ids = {}

for role in exec_roles:
    eid = next_id()
    emp = create_employee(
        eid,
        generate_name(emp_counter),
        "Executive",
        "Executive",
        None if role == "CEO" else exec_ids.get("CEO")
    )
    employees.append(emp)
    exec_ids[role] = eid

# 2️⃣ VP LAYER
vp_ids = []

for dept, count in DEPARTMENT_COUNTS.items():
    num_vps = max(1, count // 60)
    for _ in range(num_vps):
        eid = next_id()
        reports_to = exec_ids.get("CTO") if dept == "Engineering" else exec_ids.get("COO")
        emp = create_employee(eid, generate_name(emp_counter), "VP", dept, reports_to)
        employees.append(emp)
        vp_ids.append(eid)

# 3️⃣ DIRECTORS
director_ids = []

for vp in vp_ids:
    for _ in range(3):
        eid = next_id()
        vp_data = next((e for e in employees if e["id"] == vp), None)
        if not vp_data:
            raise ValueError(f"VP {vp} not found in employee list")
        vp_dept = vp_data["department"]

        emp = create_employee(eid, generate_name(emp_counter), "Director", vp_dept, vp)
        employees.append(emp)
        director_ids.append(eid)

# 4️⃣ MANAGERS
manager_ids = []

for director in director_ids:
    for _ in range(2):
        eid = next_id()
        director_data = next((e for e in employees if e["id"] == director), None)
        if not director_data:
            raise ValueError(f"Director {director} not found in employee list")
        dept = director_data["department"]

        emp = create_employee(eid, generate_name(emp_counter), "Manager", dept, director)
        employees.append(emp)
        manager_ids.append(eid)

# 5️⃣ INDIVIDUAL CONTRIBUTORS
remaining = TOTAL_EMPLOYEES - len(employees)

for i in range(remaining):
    manager = random.choice(manager_ids)
    manager_data = next((e for e in employees if e["id"] == manager), None)
    if not manager_data:
        raise ValueError(f"Manager {manager} not found in employee list")
    dept = manager_data["department"]


    role = dept if dept in SALARY_BANDS else "Engineer"

    eid = next_id()
    emp = create_employee(eid, generate_name(emp_counter), role, dept, manager)
    employees.append(emp)

# -------------------------
# OUTPUT
# -------------------------

company_data = {
    "metadata": {
        "company_name": "Nexora Digital Solutions",
        "industry": "Technology Consulting",
        "total_employees": len(employees),
        "seed": SEED
    },
    "employees": employees
}

Path("data").mkdir(exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(company_data, f, indent=2)

print("✅ Company generated successfully!")
print(f"Total Employees: {len(employees)}")
print(f"Saved to: {OUTPUT_FILE}")