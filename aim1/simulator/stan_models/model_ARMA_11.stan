//AR1 with ragged datatypes
//https://mc-stan.org/docs/2_24/stan-users-guide/ragged-data-structs-section.html
data {
    int W;
    int N;
    int T;
    int T0;  // initialization time step
    int ND;
    int not_empty_num;
    int total_nonnan_len;
    //int Eobs_flatten_nonan;
    //vector[total_nonnan_len] Eobs_flatten_nonan;
    int Eobs_flatten_nonan[total_nonnan_len];
    real sample_weights[not_empty_num];
    int not_empty_ids[not_empty_num];    
    //matrix[N,T] Pobs;
    int NClust;
    int cluster[N];
    int patient_nonnan_len[N];
    
    int cum_patient_nonnan_len[N];
    
    
}

parameters {
  
    real mu_a0;
    real mu_a1;
    real<lower=0> sigma_a0;
    real<lower=0> sigma_a1;
    
    
    
    vector[N] a0;
    vector<lower=-0.9,upper=0.9>[N] a1;
    
    
    //real mu_theta;
    //real<lower=0> sigma_theta;
    vector<lower= -1, upper = 1>[N] theta;
   
}


transformed parameters{
//  vector[total_nonnan_len] epsilon;
//  epsilon[1] = Eobs_flatten_nonan[1] - a0[1] + a1[1];
//  for ( i in 1: (N-1)){
//    for (j in 2:(patient_nonnan_len[i])){
    
            //Eobs_flatten_nonan[j+pos] ~ normal(a0[i] + a1[i] * Eobs_flatten_nonan[j+pos-1] ,sigma);  
//            epsilon[j+cum_patient_nonnan_len[i]] = (Eobs_flatten_nonan[j+cum_patient_nonnan_len[i]]  - a0[1] - theta[1] * epsilon[j+cum_patient_nonnan_len[i]-1]);
  //        }
 // }
  
  
  
  
}


model {
    matrix[N,T] A;
    matrix[N,T] p;
    vector[N*(T-T0)] p_flatten;
    vector[not_empty_num] p_flatten_nonan;
    int pos;
    
    vector[total_nonnan_len] err;
    vector[total_nonnan_len] nu;
    
    
    
    for (i in 1:N){
        
        a0[i] ~ normal(mu_a0,sigma_a0);
        a1[i] ~ normal(mu_a1,sigma_a1);
        //theta[i] ~ normal(mu_theta, sigma_theta);
    
        err[cum_patient_nonnan_len[i]+1] = a0[i] + a1[i] * Eobs_flatten_nonan[cum_patient_nonnan_len[i]+1];
      
      
    }
    
    
    
    
    pos = 0;
    for (i in 1:N){
          for (j in 2:(patient_nonnan_len[i])){
            //Eobs_flatten_nonan[j+pos] ~ normal(a0[i] + a1[i] * Eobs_flatten_nonan[j+pos-1] ,sigma);  
            //Eobs_flatten_nonan[j+pos] ~ normal(a0[i] + a1[i] * Eobs_flatten_nonan[j+pos-1] + theta[1] * epsilon[j+pos-1] ,sigma);  
            
            //print(j + pos - 1);
            
            //print( err);
            nu[j+pos] = a0[i] + a1[i] * Eobs_flatten_nonan[j+pos-1] + theta[1] * err[j+pos-1];
            //nu[j+pos] = a0[i] + theta[i] * err[j+pos-1];
            err[j+pos] = Eobs_flatten_nonan[j+pos]  - nu[j+pos];
            
            
            //err[j+pos] ~ normal(nu[j+pos], 1);
            
            //if (nu[j+pos] < -12){
            //  nu[j+pos] = -12;
            //}
            //target += binomial_lpmf(Eobs_flatten_nonan[j+pos] |W, inv_logit(nu[j+pos] ));
            target += binomial_logit_lpmf(Eobs_flatten_nonan[j+pos] |W, nu[j+pos] );
          }
          pos = pos + patient_nonnan_len[i];
    }
      
 
    
    //for (i in 1:not_empty_num) {
          //target += binomial_lpmf()
        //target += binomial_lpmf(Eobs_flatten_nonan[i] | W, p_flatten_nonan[i]) * sample_weights[i]; // not_empty_num;
        //target += normal_lpdf(err_flatten_nonan[i] | 0, sigma_err_flatten_nonan[i]) * sample_weights[i]; // not_empty_num;
    //}
}
    
    
    

