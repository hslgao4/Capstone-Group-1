## Time Series Data Generator 

### The goal is to create synthetic data for testing different models.

  - I. Data without Exogenous Input
    - By customized function 
    - sine, cosine, random, linear, exponential, random
    - By model: AR, MA, ARMA, ARIMA, SARIMA, Multipicative, non-linear AR/MA
    

  - II. Data with one Exogenous Input 
    - I + random/sine/cosine/cutomized function ...
    - By Box-Jenkins model, generating u(t) first using I, then generating y(t)
    
       { y(t) = G(q)*u(t) + H(q)*e(t) }
    

  - III. Data with more than one Exogenous Inputs 
    - add more random/sine/cosine ... 
    - TBD

___
## Model 
