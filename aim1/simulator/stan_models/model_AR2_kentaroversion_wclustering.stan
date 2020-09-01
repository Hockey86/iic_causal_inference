data {
    int W;  
    int N;   //number of patients
    int T;
    int T0;  // initialization time step
    int ND;
    int not_empty_num;
    int not_empty_ids[not_empty_num];
    int Eobs_flatten_nonan[not_empty_num];
    real sample_weights[not_empty_num];
    matrix[N,ND] D[T];
    matrix[N,T0] A_start;
    int NClust;
    matrix[N,NClust] cluster;
}

parameters {
    real mu_a0;
    real mu_a1;
    //vector<lower = 0>[ND] mu_b[NClust];
    vector<lower = 0>[ND] mu_b;
    
    real<lower=0> sigma_a0;
    real<lower=0> sigma_a1;
    real<lower=0> sigma_b;
    
    vector[N] a0;
    vector<lower=-0.95,upper=0.95>[N] a1;
    //vector[ND] b[NClust];
    
    matrix<lower = 0>[ NClust,ND] b;
}


model {
    matrix[N,T] A;
    matrix[N,T] p;
    vector[N*(T-T0)] p_flatten;
    vector[not_empty_num] p_flatten_nonan;
    vector[NClust] ones_b;
    
    
    
    a0 ~ normal(mu_a0, sigma_a0);
    a1 ~ normal(mu_a1, sigma_a1);
    
    //for (i in 1:NClust){
     //   for (j in 1:ND){
            //b[j,i] ~ normal(mu_b[i], sigma_b);
     //   }
    //}
    
    for (i in 1:NClust){
      b[i,:] ~   normal(mu_b, sigma_b);
    }
    
    
    for (t in 1:T0) { A[:,t] = A_start[:,t]; }


    ones_b  = rep_vector(1, NClust);
    
    for (t in T0+1:T) {
        A[:,t] = a0 + a1 .* A[:,t-1] - ((D[t] * b).*cluster)* ones_b;//
       // A[:,t] = a0 + a1 .* A[:,t-1] -() (D[t] * b);
        //A[:,t] = a0 + a1 .* A[:,t-1];
    }
    
    

    
    p = inv_logit(A);
    p_flatten = to_vector(p[:,T0+1:]'); // to_vector is flattening p in column-major order, but we need row-major order, so transpose it
    p_flatten_nonan = p_flatten[not_empty_ids];
    
    
    Eobs_flatten_nonan ~ binomial(W, p_flatten_nonan);
    //for (i in 1:not_empty_num) {
    //    target += binomial_lpmf(Eobs_flatten_nonan[i] | W, p_flatten_nonan[i]);
    //}
}

