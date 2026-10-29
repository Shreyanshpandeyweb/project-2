-- SQL Script 1: Customer Demographic and Service Subscriptions Analysis
-- Database Target: SQLite / PostgreSQL

-- 1. Customer Breakdown by Demographics (Gender, Senior Status, Partner, Dependents)
SELECT 
    Gender,
    SeniorCitizen,
    Partner,
    Dependents,
    COUNT(CustomerID) AS TotalCustomers,
    ROUND(AVG(Tenure), 1) AS AvgTenureMonths,
    ROUND(AVG(MonthlyCharges), 2) AS AvgMonthlyCharges,
    ROUND(SUM(MonthlyCharges), 2) AS TotalMonthlyRevenue
FROM customers
GROUP BY Gender, SeniorCitizen, Partner, Dependents
ORDER BY TotalCustomers DESC;

-- 2. Service Adoption Distribution
SELECT 
    InternetService,
    Contract,
    PaymentMethod,
    COUNT(CustomerID) AS CustomerCount,
    ROUND(AVG(ServiceCount), 2) AS AvgServicesSubscribed,
    ROUND(AVG(SupportServiceCount), 2) AS AvgSupportServices
FROM customers
GROUP BY InternetService, Contract, PaymentMethod
ORDER BY CustomerCount DESC;

-- 3. Tech Support & Security Service Penetration
SELECT 
    OnlineSecurity,
    TechSupport,
    COUNT(CustomerID) AS CustomerCount,
    SUM(CASE WHEN Churn_Numeric = 1 THEN 1 ELSE 0 END) AS ChurnedCount,
    ROUND(100.0 * SUM(CASE WHEN Churn_Numeric = 1 THEN 1 ELSE 0 END) / COUNT(CustomerID), 2) AS ChurnRatePct
FROM customers
GROUP BY OnlineSecurity, TechSupport
ORDER BY ChurnRatePct DESC;
