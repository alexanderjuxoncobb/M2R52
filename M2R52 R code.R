library(latex2exp)
set.seed(1)
T <- 100         # Number of time steps.
x0 <- 0.0         # Initial state. UNIFORM(0,1)
true_x <- matrix(0,nrow=T, ncol=1)  # Hidden states.
y <- matrix(0,nrow=T, ncol=1)       # Observations.
true_x[1,1] <- x0      # Initial state. 


# LQG model parameters
a <- 0.7
b <- 1
c <- 1
d <- 0.3

initVar <- 1          # Initial variance of the states.  

v <- rnorm(T)
w <- rnorm(T)


# GENERATE TRUE STATE AND MEASUREMENTS:

y[1,1] <- c*true_x[1,1]+d*w[1] 
for (t in seq(2,T)){
  true_x[t,1] <- a*true_x[t-1,1] + b*v[t]
  y[t,1] <- c*true_x[t,1]+d*w[t] 
}

# Kalman Filter


# recursive likelihood
RecLikeMean <- matrix(0, nrow=T, ncol=1)
RecLikeVar <- RecLikeMean

# KF predictor and update means and variance
mu <- matrix(0, nrow=T, ncol=1)
Sigma <- matrix(0, nrow=T, ncol=1)
mu[1,1] <- x0
Sigma[1,1] <- initVar

mu_pred <- matrix(0, nrow=T, ncol=1)
Sigma_pred <- matrix(0, nrow=T, ncol=1)
mu_pred[1,1] <- a*x0
Sigma_pred[1,1] <- a^2*initVar + b^2

RecLikeMean[1,1] <- c*mu_pred[1,1]
RecLikeVar[1,1] <- c^2*Sigma_pred[1,1]+d^2

for (t in seq(2,T)){
  mu_pred[t,1] <- a*mu[(t-1),]
  Sigma_pred[t,1] <- a^2*Sigma[(t-1),]+b^2
  
  RecLikeMean[t,1] <- c*mu_pred[t,1]
  RecLikeVar[t,1] <- c^2*Sigma_pred[t,1]+d^2
  
  K <- c*Sigma_pred[t,1]/RecLikeVar[t,1]
  Sigma[t,1] <- (1-K*c)*Sigma_pred[t,1]
  mu[t,1]=mu_pred[t,1]+K*(y[t,1]-RecLikeMean[t,1])
}
plot(true_x, mu, xlab=TeX('True value of $X_n$'), ylab='Kalman Filter Estimation')
abline(0, 1, col = "red", lwd = 2)
dat = c(true_x, mu_pred, mu_pred +1.96*Sigma_pred, mu_pred - 1.96*Sigma_pred)
data = matrix(dat, nrow=T, ncol = 4)
matplot(data, type = c("l"), lty = c(1, 1, 2, 2), col = c("red", "blue", "green", "green"), pch=1, ylab = "Data", xlab = "Time")
dat = c(true_x, mu, mu +1.96*Sigma, mu - 1.96*Sigma)
data = matrix(dat, nrow=T, ncol = 4)
matplot(data, type = c("l"), lty = c(1, 1, 2, 2), col = c("red", "blue", "green", "green"), pch=1, ylab = "Data", xlab = "Time")