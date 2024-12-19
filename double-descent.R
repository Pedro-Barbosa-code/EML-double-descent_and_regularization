# Carregando pacotes necessários
library(corpcor) # Para pseudo-inversa
set.seed(123)    # Reprodutibilidade

# Função para treinar a ELM
train_ELM <- function(xin, yin, p) {
  # Adiciona bias
  xin <- cbind(1, xin)
  
  # Inicializa os pesos aleatórios
  n <- ncol(xin)
  Z <- matrix(runif(n * p, -0.5, 0.5), nrow = n)
  
  # Calcula a saída da camada escondida
  H <- tanh(xin %*% Z)
  
  # Adiciona bias à saída da camada escondida
  Haug <- cbind(1, H)
  
  # Ajusta os pesos usando pseudo-inversa
  W <- pseudoinverse(Haug) %*% yin
  
  return(list(W = W, H = H, Z = Z))
}

# Função para realizar previsões com a ELM
predict_ELM <- function(xin, Z, W) {
  # Adiciona bias
  xin <- cbind(1, xin)
  
  # Calcula a saída da camada escondida
  H <- tanh(xin %*% Z)
  
  # Adiciona bias à saída da camada escondida
  Haug <- cbind(1, H)
  
  # Previsões
  y_pred <- sign(Haug %*% W)
  return(y_pred)
}

# Gerando o conjunto de dados XOR com distribuições gaussianas
Ng <- 40  # Número de amostras por cluster
s <- 0.41  # Desvio padrão dos clusters

# Dados para as classes
xc11 <- matrix(rnorm(2 * Ng, sd = s), ncol = 2) + matrix(c(-1, -1), ncol = 2, nrow = Ng, byrow = TRUE)
xc12 <- matrix(rnorm(2 * Ng, sd = s), ncol = 2) + matrix(c(1, 1), ncol = 2, nrow = Ng, byrow = TRUE)
xc21 <- matrix(rnorm(2 * Ng, sd = s), ncol = 2) + matrix(c(-1, 1), ncol = 2, nrow = Ng, byrow = TRUE)
xc22 <- matrix(rnorm(2 * Ng, sd = s), ncol = 2) + matrix(c(1, -1), ncol = 2, nrow = Ng, byrow = TRUE)

# Combina os dados
X <- rbind(xc11, xc12, xc21, xc22)
y <- c(rep(1, 2 * Ng), rep(-1, 2 * Ng))

# Divisão em treinamento e teste
train_indices <- sample(1:(4 * Ng), size = 2.8 * Ng)
X_train <- X[train_indices, ]
y_train <- y[train_indices]
X_test <- X[-train_indices, ]
y_test <- y[-train_indices]

# Configurações para análise do fenômeno double descent
p_values <- seq(2, 350, by = 5)  # Número de neurônios na camada escondida
train_errors <- numeric(length(p_values))
test_errors <- numeric(length(p_values))

# Loop para treinar o modelo e calcular os erros
for (i in seq_along(p_values)) {
  p <- p_values[i]
  
  # Treina o modelo
  elm_model <- train_ELM(X_train, y_train, p)
  W <- elm_model$W
  Z <- elm_model$Z
  
  # Previsões
  y_train_pred <- predict_ELM(X_train, Z, W)
  y_test_pred <- predict_ELM(X_test, Z, W)
  
  # Calcula os erros (taxa de erro de classificação)
  train_errors[i] <- mean(y_train_pred != y_train)
  test_errors[i] <- mean(y_test_pred != y_test)
}

# Identificando o ponto de pico do erro de teste
peak_index <- which.max(test_errors[(length(train_errors) / 4):(length(train_errors) / 2)]) + (length(train_errors) / 4) - 1

# Plotando o gráfico do fenômeno double descent
plot(p_values, train_errors, type = "l", col = "blue", lwd = 2.0,
     xlab = "Neurônios na camada escondida (p)", ylab = "Erro",
     main = "Fenômeno Double Descent (Problema XOR)")

lines(p_values, test_errors, col = "red", lwd = 2.0)

# Destacando o ponto de pico do erro
abline(v = p_values[peak_index], lty = 2, lwd = 2.0, col = "gray")

# Adicionando a legenda
legend("topright", legend = c("Treinamento", "Teste"),
       col = c("blue", "red"), lwd = 1.5, cex = 0.8)

# Imprimindo os detalhes do ponto de pico
cat("Pico do erro de teste em p =", p_values[peak_index],
    "com erro de teste:", test_errors[peak_index], "\n")
