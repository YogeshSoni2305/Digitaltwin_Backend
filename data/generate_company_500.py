import json
import random
from pathlib import Path

# -------------------------
# CONFIGURATION
# -------------------------

TOTAL_EMPLOYEES = 20
OUTPUT_PATH = Path("data/company_20.json")
SEED = 42
random.seed(SEED)

# -------------------------
# SALARY BANDS (USD)
# -------------------------

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

# -------------------------
# NAME POOLS
# -------------------------

FIRST_NAMES = [
    "Aarav","Vivaan","Arjun","Rohan","Kabir","Neha","Ananya","Priya","Kavya","Meera",
    "Michael","Daniel","Sophia","Olivia","Ethan","Liam","Emma","Ava","Noah","Lucas"
]

LAST_NAMES = [
    "Sharma","Patel","Verma","Singh","Mehta","Johnson","Brown","Smith","Davis","Wilson"
]

# -------------------------
# SKILLS BY DEPARTMENT
# -------------------------

DEPARTMENT_SKILLS = {
    "Engineering": ["backend", "frontend", "devops", "data", "cloud", "architecture"],
    "Sales": ["negotiation", "client_acquisition", "enterprise_sales", "communication"],
    "Operations": ["process_optimization", "delivery_management", "vendor_management"],
    "Product": ["roadmap_planning", "stakeholder_alignment", "analytics"],
    "Finance": ["financial_modeling", "budgeting", "forecasting"],
    "HR": ["talent_acquisition", "performance_management"]
}

emp_counter = 1

def next_id():
    global emp_counter
    eid = f"EMP{emp_counter:04d}"
    emp_counter += 1
    return eid

def generate_name(index):
    first = FIRST_NAMES[index % len(FIRST_NAMES)]
    last = LAST_NAMES[(index * 3) % len(LAST_NAMES)]
    return f"{first} {last}"

def random_salary(role):
    low, high = SALARY_BANDS[role]
    return random.randint(low, high)

def generate_skills(department, role):
    skills = {}
    pool = DEPARTMENT_SKILLS.get(department, [])
    selected = random.sample(pool, min(3, len(pool)))

    for skill in selected:
        base = 0.6 if role in ["Manager","Director","VP","Executive"] else 0.5
        skills[skill] = round(random.uniform(base, 0.95), 2)

    if role in ["Director","VP","Executive"]:
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
# BUILD COMPANY (20 TOTAL)
# -------------------------

employees = []

# 1️⃣ CEO
ceo_id = next_id()
employees.append(create_employee(
    ceo_id,
    generate_name(emp_counter),
    "Executive",
    "Executive",
    None
))

# 2️⃣ VPs (Engineering + Operations)
departments = ["Engineering", "Sales", "Operations", "Product"]
vp_ids = {}

for dept in departments:
    eid = next_id()
    employees.append(create_employee(
        eid,
        generate_name(emp_counter),
        "VP",
        dept,
        ceo_id
    ))
    vp_ids[dept] = eid

# 3️⃣ Managers under each VP
manager_ids = []

for dept, vp_id in vp_ids.items():
    eid = next_id()
    employees.append(create_employee(
        eid,
        generate_name(emp_counter),
        "Manager",
        dept,
        vp_id
    ))
    manager_ids.append(eid)

# 4️⃣ Individual Contributors (fill until 20)
while len(employees) < TOTAL_EMPLOYEES:
    manager = random.choice(manager_ids)
    dept = next(e["department"] for e in employees if e["id"] == manager)

    role = "Engineer" if dept == "Engineering" else dept

    eid = next_id()
    employees.append(create_employee(
        eid,
        generate_name(emp_counter),
        role,
        dept,
        manager
    ))

# -------------------------
# SAVE TO FILE
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

OUTPUT_PATH.parent.mkdir(exist_ok=True)

with open(OUTPUT_PATH, "w") as f:
    json.dump(company_data, f, indent=2)

print("✅ Company generated successfully!")
print(f"Total Employees: {len(employees)}")
print(f"Saved to: {OUTPUT_PATH}")