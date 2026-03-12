-- Workforce Digital Twin: Enterprise Persistence Schema (PostgreSQL)
-- Optimized for time-series organizational snapshots and simulation audit trails.

-- 1. Organizations
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Employees (Organizational Snapshot)
CREATE TABLE employees (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID REFERENCES organizations(id),
    external_id VARCHAR(100), -- ID from original system (SaaS integration)
    name VARCHAR(255) NOT NULL,
    role VARCHAR(100),
    department VARCHAR(100),
    salary DECIMAL(15, 2),
    capacity_hours_per_week DECIMAL(5, 2),
    reports_to UUID REFERENCES employees(id),
    productivity_multiplier DECIMAL(3, 2) DEFAULT 1.0,
    skills JSONB, -- { "skill_name": proficiency_level }
    is_active BOOLEAN DEFAULT TRUE,
    snapshot_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Projects & Tasks
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID REFERENCES organizations(id),
    name VARCHAR(255) NOT NULL,
    base_revenue DECIMAL(15, 2),
    decay_rate DECIMAL(5, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),
    name VARCHAR(255) NOT NULL,
    required_skill VARCHAR(100),
    required_level DECIMAL(3, 2),
    estimated_hours DECIMAL(10, 2),
    dependencies UUID[] -- Array of task UUIDs
);

-- 4. Simulation Audit Trail
CREATE TABLE simulation_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID REFERENCES organizations(id),
    strategy_key VARCHAR(50),
    target_employee_id UUID REFERENCES employees(id),
    input_params JSONB, -- { "seed": 42, "iterations": 50, "price_delta": 0.1 }
    results_summary JSONB, -- { "mean_profit": 500.2, "risk_probability": 0.12 }
    full_payload_url TEXT, -- Link to JSON blob store (S3/GCS) for high-fidelity metrics
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indices for performance
CREATE INDEX idx_emp_org ON employees(org_id);
CREATE INDEX idx_proj_org ON projects(org_id);
CREATE INDEX idx_sim_org ON simulation_history(org_id);
CREATE INDEX idx_emp_reports ON employees(reports_to);
