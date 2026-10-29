-- SQL Script 2: Comprehensive Churn Rate & Customer Segmentation Analysis
-- Database Target: SQLite / PostgreSQL

-- 1. Churn Rate by Contract Type
SELECT 
    Contract,
    COUNT(CustomerID) AS TotalCustomers,
    SUM(CASE WHEN Churn_Numeric = 1 THEN 1 ELSE 0 END) AS ChurnedCustomers,
    SUM(CASE WHEN Churn_Numeric = 0 THEN 1 ELSE 0 END) AS ActiveCustomers,
    ROUND(100.0 * SUM(CASE WHEN Churn_Numeric = 1 THEN 1 ELSE 0 END) / COUNT(CustomerID), 2) AS ChurnRatePct,
    ROUND(SUM(MonthlyCharges), 2) AS TotalMonthlyRevenue,
    ROUND(SUM(CASE WHEN Churn_Numeric = 1 THEN MonthlyCharges ELSE 0 END), 2) AS HistoricalMonthlyRevenueLost
FROM customers
GROUP BY Contract
ORDER BY ChurnRatePct DESC;

-- 2. Churn Rate by Tenure Group
SELECT 
    TenureGroup,
    COUNT(CustomerID) AS TotalCustomers,
    SUM(CASE WHEN Churn_Numeric = 1 THEN 1 ELSE 0 END) AS ChurnedCustomers,
    ROUND(100.0 * SUM(CASE WHEN Churn_Numeric = 1 THEN 1 ELSE 0 END) / COUNT(CustomerID), 2) AS ChurnRatePct,
    ROUND(AVG(MonthlyCharges), 2) AS AvgMonthlyCharges
FROM customers
GROUP BY TenureGroup
ORDER BY ChurnRatePct DESC;

-- 3. Churn Rate by Payment Method
SELECT 
    PaymentMethod,
    COUNT(CustomerID) AS TotalCustomers,
    SUM(CASE WHEN Churn_Numeric = 1 THEN 1 ELSE 0 END) AS ChurnedCustomers,
    ROUND(100.0 * SUM(CASE WHEN Churn_Numeric = 1 THEN 1 ELSE 0 END) / COUNT(CustomerID), 2) AS ChurnRatePct,
    ROUND(SUM(MonthlyCharges), 2) AS MonthlyRevenue
FROM customers
GROUP BY PaymentMethod
ORDER BY ChurnRatePct DESC;

-- 4. Churn Rate by Revenue Segment (Low, Medium, High)
SELECT 
    MonthlyChargeGroup,
    COUNT(CustomerID) AS TotalCustomers,
    SUM(CASE WHEN Churn_Numeric = 1 THEN 1 ELSE 0 END) AS ChurnedCustomers,
    ROUND(100.0 * SUM(CASE WHEN Churn_Numeric = 1 THEN 1 ELSE 0 END) / COUNT(CustomerID), 2) AS ChurnRatePct,
    ROUND(AVG(Tenure), 1) AS AvgTenureMonths
FROM customers
GROUP BY MonthlyChargeGroup
ORDER BY ChurnRatePct DESC;
