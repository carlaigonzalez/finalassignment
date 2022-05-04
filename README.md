# finalassignment
Analysis of synthetic telecommunications company to predict churn (cancel their subscription)

Analysis of synthetic IBM dataset modeling a telecommunications company. Each observation refers to a customer, and it our job as data analylist to predict churn. 
After analysing the data, we formed our two hypothesis: a) There is a relationship between customers churning and the type of contract they have and b)Customers are more likely to churn if they are not partnered and are female.

After testing our hypothesis with a logistic regression model, we concluded that:
  People with month-to-month contract have e1.6554457 times the odd of churn rather than: 
    one year contract customers who are e-0.041919 less likely to churn
    two year contract customers who are e-1.61333368 less likely to churn. 
  As we can see, it is much probable that Telco customers stay if they contract their services for two years.
  Females have e0.50023097 times the odd of churn rather than men, in the case of partner status,
  People with partners are e-0.18145626 less likely to churn rather than single customers.
