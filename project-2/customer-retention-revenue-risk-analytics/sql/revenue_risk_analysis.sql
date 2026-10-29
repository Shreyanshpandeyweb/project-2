-- SQL Script 3: Production Revenue Risk & Prioritization Analytics (10 Business Queries)
-- Database Target: SQLite / PostgreSQL

-- ============================================================================
-- QUERY 1: Overall Customer Churn Rate & Core Revenue KPIs
-- Purpose: Provide executive baseline metrics on churn percentage and revenue loss.
-- ============================================================================
SELECT 
    COUNT(CustomerID) AS TotalCustomers,
    SUM(CASE WHEN Churn_Numeric = 0 THEN 1 ELSE 0 END) AS ActiveCustomers,
    SUM(CASE WHEN Churn_Numeric = 1 THEN 1 ELSE 0 END) AS ChurnedCustomers,
    ROUND(100.0 * SUM(CASE WHEN Churn_Numeric = 1 THEN 1 ELSE 0 END) / COUNT(CustomerID), 2) AS ChurnRatePct,
    ROUND(AVG(MonthlyCharges), 2) AS AvgMonthlyCharges,
    ROUND(AVG(Tenure), 1) AS AvgTenureMonths,
    ROUND(SUM(MonthlyCharges), 2) AS TotalMonthlyMRR,
    ROUND(SUM(CASE WHEN Churn_Numeric = 1 THEN MonthlyCharges ELSE 0 END), 2) AS HistoricalMonthlyRevenueLost
FROM customers;


-- ============================================================================
-- QUERY 2: Churn Rate and Revenue Lost by Contract Type
-- Purpose: Analyze structural contract risk and historical revenue leakage.
-- ============================================================================
SELECT 
    Contract,
    COUNT(CustomerID) AS CustomerCount,
    SUM(CASE WHEN Churn_Numeric = 1 THEN 1 ELSE 0 END) AS ChurnedCount,
    ROUND(100.0 * SUM(CASE WHEN Churn_Numeric = 1 THEN 1 ELSE 0 END) / COUNT(CustomerID), 2) AS ChurnRatePct,
    ROUND(AVG(MonthlyCharges), 2) AS AvgMonthlyCharges,
    ROUND(SUM(CASE WHEN Churn_Numeric = 1 THEN MonthlyCharges ELSE 0 END), 2) AS MonthlyRevenueLost
FROM customers
GROUP BY Contract
ORDER BY ChurnRatePct DESC;


-- ============================================================================
-- QUERY 3: Tenure Groups with Highest Churn Risk
-- Purpose: Identify customer lifecycle stages where churn spikes.
-- ============================================================================
SELECT 
    TenureGroup,
    COUNT(CustomerID) AS TotalCustomers,
    SUM(CASE WHEN Churn_Numeric = 1 THEN 1 ELSE 0 END) AS ChurnedCustomers,
    ROUND(100.0 * SUM(CASE WHEN Churn_Numeric = 1 THEN 1 ELSE 0 END) / COUNT(CustomerID), 2) AS ChurnRatePct,
    ROUND(SUM(MonthlyCharges), 2) AS GroupMonthlyRevenue
FROM customers
GROUP BY TenureGroup
ORDER BY ChurnRatePct DESC;


-- ============================================================================
-- QUERY 4: Average Monthly Revenue by Customer Segment (Contract & Internet)
-- Purpose: Understand ARPU (Average Revenue Per User) variations across product tiers.
-- ============================================================================
SELECT 
    Contract,
    InternetService,
    COUNT(CustomerID) AS CustomerCount,
    ROUND(AVG(MonthlyCharges), 2) AS AvgMonthlyRevenue,
    ROUND(MIN(MonthlyCharges), 2) AS MinMonthlyRevenue,
    ROUND(MAX(MonthlyCharges), 2) AS MaxMonthlyRevenue
FROM customers
GROUP BY Contract, InternetService
ORDER BY AvgMonthlyRevenue DESC;


-- ============================================================================
-- QUERY 5: High-Value Customers Who Churned (MonthlyCharges > $80)
-- Purpose: Audit top-tier revenue losses for exit-interview intervention.
-- ============================================================================
SELECT 
    CustomerID,
    Contract,
    InternetService,
    Tenure,
    PaymentMethod,
    MonthlyCharges,
    TotalCharges
FROM customers
WHERE Churn_Numeric = 1 AND MonthlyCharges > 80.0
ORDER BY MonthlyCharges DESC
LIMIT 20;


-- ============================================================================
-- QUERY 6: Rank Customer Segments by Historical Revenue Exposure
-- Purpose: Use RANK() window function to identify highest total revenue lost by segment.
-- ============================================================================
WITH SegmentExposure AS (
    SELECT 
        Contract,
        PaymentMethod,
        InternetService,
        COUNT(CustomerID) AS SegmentSize,
        SUM(CASE WHEN Churn_Numeric = 1 THEN 1 ELSE 0 END) AS ChurnedCount,
        ROUND(SUM(CASE WHEN Churn_Numeric = 1 THEN MonthlyCharges ELSE 0 END), 2) AS MonthlyRevenueLost
    FROM customers
    GROUP BY Contract, PaymentMethod, InternetService
)
SELECT 
    Contract,
    PaymentMethod,
    InternetService,
    SegmentSize,
    ChurnedCount,
    MonthlyRevenueLost,
    RANK() OVER (ORDER BY MonthlyRevenueLost DESC) AS ExposureRank
FROM SegmentExposure
ORDER BY ExposureRank ASC;


-- ============================================================================
-- QUERY 7: Calculate Monthly Revenue Contribution & Loss Share by Contract Type
-- Purpose: Benchmark revenue distribution using windowed SUM totals.
-- ============================================================================
WITH ContractTotals AS (
    SELECT 
        Contract,
        SUM(MonthlyCharges) AS ContractMRR,
        SUM(CASE WHEN Churn_Numeric = 1 THEN MonthlyCharges ELSE 0 END) AS ContractRevenueLost
    FROM customers
    GROUP BY Contract
)
SELECT 
    Contract,
    ROUND(ContractMRR, 2) AS ContractMRR,
    ROUND(100.0 * ContractMRR / SUM(ContractMRR) OVER(), 2) AS MRRContributionPct,
    ROUND(ContractRevenueLost, 2) AS ContractRevenueLost,
    ROUND(100.0 * ContractRevenueLost / SUM(ContractRevenueLost) OVER(), 2) AS RevenueLossSharePct
FROM ContractTotals
ORDER BY ContractMRR DESC;


-- ============================================================================
-- QUERY 8: Top 10 Highest-Risk Customer Segments by Churn Probability & Value
-- Purpose: Pinpoint multi-attribute segments with extreme churn rate and revenue concentration.
-- ============================================================================
SELECT 
    Contract,
    TenureGroup,
    InternetService,
    PaymentMethod,
    COUNT(CustomerID) AS TotalCustomers,
    SUM(CASE WHEN Churn_Numeric = 1 THEN 1 ELSE 0 END) AS ChurnedCustomers,
    ROUND(100.0 * SUM(CASE WHEN Churn_Numeric = 1 THEN 1 ELSE 0 END) / COUNT(CustomerID), 2) AS ChurnRatePct,
    ROUND(SUM(MonthlyCharges), 2) AS SegmentMonthlyRevenue
FROM customers
GROUP BY Contract, TenureGroup, InternetService, PaymentMethod
HAVING COUNT(CustomerID) >= 20
ORDER BY ChurnRatePct DESC, SegmentMonthlyRevenue DESC
LIMIT 10;


-- ============================================================================
-- QUERY 9: Rank Active Customers by Expected Revenue Risk (Post-ML Probabilities)
-- Purpose: Use DENSE_RANK() to prioritize active accounts by Expected Revenue Risk.
-- Note: Uses churn_predictions table produced by ML model pipeline.
-- ============================================================================
SELECT 
    CustomerID,
    Contract,
    Tenure,
    MonthlyCharges,
    ROUND(ChurnProbability, 4) AS ChurnProbability,
    ROUND(ExpectedRevenueAtRisk, 2) AS ExpectedRevenueAtRisk,
    RiskSegment,
    RetentionPriority,
    DENSE_RANK() OVER (ORDER BY ExpectedRevenueAtRisk DESC) AS RiskRank
FROM churn_predictions
WHERE Churn_Numeric = 0
ORDER BY RiskRank ASC
LIMIT 20;


-- ============================================================================
-- QUERY 10: Cumulative Expected Revenue Exposure Using Window Function
-- Purpose: Compute running sum of expected revenue risk across ranked active customers.
-- ============================================================================
WITH RankedActiveRisk AS (
    SELECT 
        CustomerID,
        Contract,
        MonthlyCharges,
        ChurnProbability,
        ExpectedRevenueAtRisk,
        PriorityScore
    FROM churn_predictions
    WHERE Churn_Numeric = 0
)
SELECT 
    CustomerID,
    Contract,
    MonthlyCharges,
    ROUND(ChurnProbability, 4) AS ChurnProbability,
    ROUND(ExpectedRevenueAtRisk, 2) AS ExpectedRevenueAtRisk,
    ROUND(SUM(ExpectedRevenueAtRisk) OVER (ORDER BY PriorityScore DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2) AS CumulativeRevenueRisk
FROM RankedActiveRisk
ORDER BY PriorityScore DESC
LIMIT 25;
