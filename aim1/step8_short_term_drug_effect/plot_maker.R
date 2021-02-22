cutoff = c(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)
ordinal_ATE = c(0.032,0.031,0.035,0.079,0.081,0.089,0.097,0.096,0.0986)
ordinal_std = c(0.0297,0.0282,0.0219,0.01934,0.02029,0.02745,0.01476,0.014,0.0144)

utility_ATE = c(-0.008,-0.008,0.005,0.041,0.0426,0.0543,0.0652,0.0697,0.06204)
utility_std = c(0.017,0.0185,0.0162,0.0159,0.01198,0.01257,0.0126,0.01248,0.00908)


df1 = data.frame(cutoff, ordinal_ATE, ordinal_std, utility_ATE, utility_std)

library(ggplot2)

p = ggplot(df1, aes(x = cutoff, y = ordinal_ATE, color = 'red')) + 
  geom_line() + 
  geom_point( ) + 
  geom_errorbar(aes(ymin = ordinal_ATE - ordinal_std, ymax = ordinal_ATE + ordinal_std)) + 
  labs(title= ' ATE', y = 'Probability MRS <= 3')  


print(p)



p = ggplot(df1, aes(x = cutoff, y = utility_ATE, color = 'red')) + 
  geom_line() + 
  geom_point( ) + 
  geom_errorbar(aes(ymin = utility_ATE - utility_std, ymax = utility_ATE + utility_std)) + 
  labs(title= ' ATE measured in Utility MRS', y = 'UMRS')  


print(p)


p = ggplot(df1, aes(x = ordinal_ATE, y = utility_ATE, color = as.factor(cutoff))) + 
  geom_point( ) + 
  labs(title= ' ATE measured in Utility MRS', y = 'UMRS')  
print(p)
