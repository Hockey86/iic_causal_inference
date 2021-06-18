data {
    int W;
    int N;
    int ND;
    int AR_p;  // AR order, but not used
    
    int total_len;
    int patient_lens[N];
    int Eobs[total_len];
    real sample_weights[total_len];
    
    vector[ND] Ddose[total_len];
    vector[ND] Dhalflife_mean;  // gamma = exp(log0.5 / half life)
    vector[ND] Dhalflife_lb;
    vector[ND] Dhalflife_ub;
       
    int NClust;
    int cluster[N];
}

parameters {
    // PK parameters
    vector<lower=0.001>[ND] sigma_halflife[NClust];
    vector<lower=0.001,upper=0.999>[ND] halflife[N];
    
    // PD parameters
    real<lower=0.001> sigma_alpha[NClust];
    vector<lower=0.001>[ND] sigma_beta[NClust];
    real<lower=0.001> sigma_lambda0[NClust];
    real<lower=0.001> sigma_t0[NClust];
    real<lower=0.001> sigma_sigma0[NClust];

    real<lower=0.001> sigma_err[NClust];
    
    vector<lower=0.01,upper=10>[N] lambda0;
    vector<lower=-144,upper=-0.01>[N] t0;
    vector<lower=0.01,upper=10>[N] sigma0;
    vector[N] alpha0;
    vector<lower=0.001,upper=0.999>[N] alpha1;
    vector<lower=0.001,upper=0.999>[N] alpha2_transformed;  // only works for AR=2
    vector<lower=0,upper=100>[ND] beta[N];
}

model {
    vector[ND] D[total_len];
    vector[total_len] A;
    vector[ND] gamma[N];
    vector[N] alpha2;
    vector[ND] ones_b;
    real tmp1;
    real tmp2;
    real tmp3;
    int pos;
    ones_b = rep_vector(1, ND);
    
    for (i in 1:N){
        alpha0[i] ~ normal(0, sigma_alpha[cluster[i]]);
        alpha1[i] ~ normal(0, sigma_alpha[cluster[i]]);
        alpha2_transformed[i] ~ normal(0, sigma_alpha[cluster[i]]);
        alpha2[i] = -alpha1[i]^2/4+alpha2_transformed[i]*(1-alpha1[i]+alpha1[i]^2/4);
        lambda0[i] ~ normal(0, sigma_lambda0[cluster[i]]);
        t0[i] ~ normal(0, sigma_t0[cluster[i]]);
        sigma0[i] ~ normal(0, sigma_sigma0[cluster[i]]);
    }
    for (j in 1:ND) {
        for (i in 1:N) {
            halflife[i,j] ~ normal((Dhalflife_mean[j]-Dhalflife_lb[j])/(Dhalflife_ub[j]-Dhalflife_lb[j]), sigma_halflife[cluster[i]][j]);
            gamma[i,j] = halflife[i,j]*(Dhalflife_ub[j]-Dhalflife_lb[j]) + Dhalflife_lb[j];
            gamma[i,j] = exp(log(0.5)/gamma[i,j]);
            beta[i,j] ~ normal(0, sigma_beta[cluster[i]][j]);
        }
    }
    
    pos = 0;
    for (i in 1:N){
        for (j in 1:patient_lens[i]){
            if (j<=1) {
                D[pos+j] = Ddose[pos+j];
                A[pos+j] = 0;
                continue;
            }
            
            // AR(p)
            tmp1 = alpha0[i];
            if (j-1>=1)
                tmp1 += alpha1[i]*A[pos+j-1];
            if (j-2>=1)
                tmp1 += alpha2[i]*A[pos+j-2];
            A[pos+j] = tmp1;
            //print("after AR, A[", i, ",", j, "] = ",A[pos+j]);
            
            // get D from PK
            D[pos+j] = (gamma[i] .* D[pos+j-1]) + Ddose[pos+j];
            
            if (Eobs[pos+j]>=0) {  // not miss data
                tmp1 = exp(-(log(j-1-t0[i]))^2/(2*sigma0[i]^2))*lambda0[i];  // lognormal
                tmp3 = tmp1 + A[pos+j];
                tmp2 = -(D[pos+j-1] .* beta[i])' * ones_b;   //drug
                if (tmp3>12)
                    tmp3 = 12;
                else if (tmp3<-12)
                    tmp3 = -12;
                if (tmp2>12)
                    tmp2 = 12;
                else if (tmp2<-12)
                    tmp2 = -12;
                tmp2 = inv_logit(tmp2)*2;
                target += binomial_lpmf(Eobs[pos+j] | W, inv_logit(tmp3)*tmp2) * sample_weights[pos+j];

                tmp2 = Eobs[pos+j]*1./W/tmp2;
                if (tmp2<0.001)
                    tmp2 = 0.001;
                else if (tmp2>0.999)
                    tmp2 = 0.999;
                tmp2 = logit(tmp2) - tmp1;
                target += cauchy_lpdf(tmp2 | A[pos+j], sigma_err[cluster[i]]) * sample_weights[pos+j];
            }
        }
        pos += patient_lens[i];
    }
}

generated quantities {
    vector[N] alpha2;
    
    for (i in 1:N)
        alpha2[i] = -alpha1[i]^2/4+alpha2_transformed[i]*(1-alpha1[i]+alpha1[i]^2/4);
}
