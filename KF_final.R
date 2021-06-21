set.seed(500)
T <- 100        
x_0 <- 0.0         
t_x <- matrix(0,nrow=T, ncol=1)  
y <- matrix(0,nrow=T, ncol=1)       
t_x[1,1] <- x_0      
  

a <- 0.7
b <- 1
c <- 1
d <- 0.3

in_V <- 1          

v <- rnorm(T)
w <- rnorm(T)


y[1,1] <- c*t_x[1,1]+d*w[1] 
for (t in seq(2,T)){
  t_x[t,1] <- a*t_x[t-1,1] + b*v[t]
  y[t,1] <- c*t_x[t,1]+d*w[t] 
}


RLM <- matrix(0, nrow=T, ncol=1)
RLV <- RLM


mu <- matrix(0, nrow=T, ncol=1)
S <- matrix(0, nrow=T, ncol=1)
mu[1,1] <- x_0
S[1,1] <- in_V

mu_pred <- matrix(0, nrow=T, ncol=1)
S_pred <- matrix(0, nrow=T, ncol=1)
mu_pred[1,1] <- a*x_0
S_pred[1,1] <- a^2*in_V + b^2

RLM[1,1] <- c*mu_pred[1,1]
RLV[1,1] <- c^2*S_pred[1,1]+d^2

for (t in seq(2,T)){
  mu_pred[t,1] <- a*mu[(t-1),]
  S_pred[t,1] <- a^2*S[(t-1),]+b^2
  
  RLM[t,1] <- c*mu_pred[t,1]
  RLV[t,1] <- c^2*S_pred[t,1]+d^2
  
  K <- c*S_pred[t,1]/RLV[t,1]
  S[t,1] <- (1-K*c)*S_pred[t,1]
  mu[t,1]=mu_pred[t,1]+K*(y[t,1]-RLM[t,1])
}
plot(t_x, mu, xlab='True value of X_n', ylab='Kalman Filter Estimation')
abline(0, 1, col = "red", lwd = 2)
dat = c(t_x, mu_pred, mu_pred +1.96*S_pred, mu_pred - 1.96*S_pred)
data = matrix(dat, nrow=T, ncol = 4)
matplot(data, type = c("l"), lty = c(1, 1, 2, 2), col = c("red", "blue", "green", "green"), pch=1, ylab = "Data", xlab = "Time")
dat = c(t_x, mu, mu +1.96*S, mu - 1.96*S)
data = matrix(dat, nrow=T, ncol = 4)
matplot(data, type = c("l"), lty = c(1, 1, 2, 2), col = c("red", "blue", "green", "green"), pch=1, ylab = "Data", xlab = "Time")
