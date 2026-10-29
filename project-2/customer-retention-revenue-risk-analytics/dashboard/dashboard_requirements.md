# Power BI / Tableau Dashboard Requirements & Architecture

## Portfolio Project: Customer Retention & Revenue Risk Analytics

### 1. Dashboard Objective & Executive Scope
The **Customer Retention & Revenue Risk Dashboard** provides business leadership, customer success managers, and retention teams with interactive analytics to track monthly churn rates, quantify revenue exposure ($), and prioritize high-value customer accounts for immediate intervention.

---

## 2. Dashboard Page Layout & Visual Architecture

### Page 1: Executive KPI & Revenue Exposure Overview
- **Top KPI Cards**:
  - `Total Active MRR`: Total monthly recurring revenue ($456.1K)
  - `Overall Churn Rate`: Current customer churn percentage (26.54%)
  - `Monthly Revenue Lost`: Historical revenue leakage ($139.1K/mo)
  - `Expected Revenue at Risk`: ML-predicted monthly revenue exposure ($207.8K)
  - `High-Risk High-Value Accounts`: Count of Tier 1 retention targets (1,150 accounts)
- **Visuals**:
  - *Monthly Revenue Loss by Contract Type* (Bar Chart)
  - *Customer Risk Matrix 4-Quadrant Distribution* (Scatter Plot / Treemap)
  - *Tenure Group Churn Breakdown* (Horizontal Bar Chart)
  - *Monthly Revenue at Risk by Internet Service Tier* (Donut Chart)

### Page 2: Customer Segmentation & Risk Prioritization
- **Visuals**:
  - *Top 20 Priority At-Risk Accounts Table*:
    - Columns: `CustomerID`, `Contract`, `Tenure`, `MonthlyCharges`, `ChurnProbability`, `ExpectedRevenueAtRisk`, `RiskSegment`, `RetentionPriority`
  - *Churn Probability Distribution vs Value Score* (Density Plot)
  - *Tech Support & Security Service Penetration vs Churn* (Stacked Bar Chart)
  - *Payment Method Churn Exposure* (Clustered Column Chart)

---

## 3. Data Model Schema & Field Specifications

### Primary Table: `fact_customer_churn_risk`
| Field Name | Data Type | Description |
| :--- | :--- | :--- |
| `CustomerID` | String (PK) | Unique Customer Identifier |
| `Gender` | String | Gender (Male / Female) |
| `SeniorCitizen` | Integer | Senior Status (0 = No, 1 = Yes) |
| `Tenure` | Integer | Tenure in Months |
| `TenureGroup` | String | Lifecycle Group (0-6m, 7-12m, 13-24m, 25-48m, 49+m) |
| `Contract` | String | Contract Type (Month-to-month, One year, Two year) |
| `InternetService` | String | Internet Service Type (DSL, Fiber optic, No) |
| `TechSupport` | String | Technical Support Subscription (Yes, No, No internet) |
| `PaymentMethod` | String | Payment Method |
| `MonthlyCharges` | Decimal ($) | Monthly Subscription Fee |
| `TotalCharges` | Decimal ($) | Lifetime Total Charges |
| `Churn` | String | Historical Churn Status (Yes / No) |
| `Churn_Numeric` | Integer | Binary Target (1 = Churned, 0 = Active) |
| `ChurnProbability` | Decimal (0-1) | ML Predicted Churn Probability |
| `ExpectedRevenueAtRisk` | Decimal ($) | `MonthlyCharges * ChurnProbability` |
| `RiskSegment` | String | 4-Quadrant Risk Category |
| `PriorityScore` | Decimal | Risk Ranking Score |
| `RetentionPriority` | String | Action Tier (Tier 1 to Tier 4) |

---

## 4. Key DAX Measures (Power BI)

```dax
-- 1. Overall Churn Rate (%)
Overall Churn Rate = 
DIVIDE(
    CALCULATE(COUNT(fact_customer_churn_risk[CustomerID]), fact_customer_churn_risk[Churn_Numeric] = 1),
    COUNT(fact_customer_churn_risk[CustomerID]),
    0
)

-- 2. Total Expected Monthly Revenue at Risk ($)
Total Revenue At Risk = 
SUM(fact_customer_churn_risk[ExpectedRevenueAtRisk])

-- 3. High-Value High-Risk MRR ($)
High Value High Risk Revenue = 
CALCULATE(
    SUM(fact_customer_churn_risk[ExpectedRevenueAtRisk]),
    fact_customer_churn_risk[RiskSegment] = "High Risk + High Value"
)

-- 4. Tier 1 Retention Priority Count
Tier 1 Target Count = 
CALCULATE(
    COUNT(fact_customer_churn_risk[CustomerID]),
    fact_customer_churn_risk[RetentionPriority] = "Tier 1 - Immediate Intervention"
)
```

---

## 5. Interactive Slicers & Filters
- **Contract Type**: All, Month-to-month, One year, Two year
- **Risk Segment**: High Risk + High Value, High Risk + Low Value, Low Risk + High Value, Low Risk + Low Value
- **Tenure Group**: 0-6m, 7-12m, 13-24m, 25-48m, 49+m
- **Internet Service**: Fiber optic, DSL, No Internet
