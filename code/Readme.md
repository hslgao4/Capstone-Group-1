## Time Series Data Generator 

### The goal is to create synthetic data for testing different models.

  - I. Data without Exogenous Input
    - Deterministic: customized function, sine, cosine, random, linear, exponential, random
    - Stachastic: AR, MA, ARMA, ARIMA, SARIMA, Multipicative, non-linear AR/MA
    

  - II. Data with one Exogenous Input 
    - I + random/sine/cosine/customized function ...
    - By Box-Jenkins model, generating u(t) first using I, then generating y(t)
    
       { y(t) = G(q)*u(t) + H(q)*e(t) }
    

  - III. Data with more than one Exogenous Inputs 
    - add more random/sine/cosine ... 
    - TBD

___
## Model 
