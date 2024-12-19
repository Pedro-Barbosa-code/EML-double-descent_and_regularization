set.seed(11)

# Definindo p_values
p_values <- seq(2, 350, by = 5)

# Inicializa os vetores de erros
train_errors <- numeric(length(p_values))
test_errors <- numeric(length(p_values))

# Função para treinar a ELM com Dropout ajustado
train_ELM_dropout <- function(xin, yin, p, dropout_rate = 0.1) {
  # Adiciona bias
  xin <- cbind(1, xin)
  
  # Inicializa os pesos aleatórios
  n <- ncol(xin)
  Z <- matrix(runif(n * p, -0.5, 0.5), nrow = n)
  
  # Calcula a saída da camada escondida
  H <- tanh(xin %*% Z)
  
  # Aplica Dropout e ajusta pela taxa de Dropout
  dropout_mask <- matrix(runif(nrow(H) * ncol(H)) > dropout_rate, nrow = nrow(H))
  H <- H * dropout_mask / (1 - dropout_rate)
  
  # Adiciona bias à saída da camada escondida
  Haug <- cbind(1, H)
  
  # Ajusta os pesos usando pseudo-inversa
  W <- pseudoinverse(Haug) %*% yin
  
  return(list(W = W, H = H, Z = Z))
}

# Função para realizar previsões com a ELM com Dropout ajustado
predict_ELM_dropout <- function(xin, Z, W, dropout_rate) {
  # Adiciona bias
  xin <- cbind(1, xin)
  
  # Calcula a saída da camada escondida
  H <- tanh(xin %*% Z)
  
  # Ajusta pela taxa de Dropout
  H <- H * (1 - dropout_rate)
  
  # Adiciona bias à saída da camada escondida
  Haug <- cbind(1, H)
  
  # Previsões
  y_pred <- sign(Haug %*% W)
  return(y_pred)
}

# Loop para treinar o modelo e calcular os erros
for (i in seq_along(p_values)) {
  p <- p_values[i]
  
  # Treina o modelo com Dropout
  elm_model <- train_ELM_dropout(X_train, y_train, p, dropout_rate = 0.1)
  W <- elm_model$W
  Z <- elm_model$Z
  
  # Previsões
  y_train_pred <- predict_ELM_dropout(X_train, Z, W, dropout_rate = 0.1)
  y_test_pred <- predict_ELM_dropout(X_test, Z, W, dropout_rate = 0.1)
  
  # Calcula os erros (taxa de erro de classificação)
  train_errors[i] <- mean(y_train_pred != y_train)
  test_errors[i] <- mean(y_test_pred != y_test)
}

# Plotando o gráfico do fenômeno double descent
plot(p_values, train_errors, type = "l", col = "blue", lwd = 2.0,
     xlab = "Neurônios na camada escondida (p)", ylab = "Erro",
     main = "Double Descent com Dropout (XOR)")

lines(p_values, test_errors, col = "red", lwd = 2.0)

# Adicionando a legenda
legend("topright", legend = c("Treinamento", "Teste"),
       col = c("blue", "red"), lwd = 1.5, cex = 0.8)

# Imprimindo os detalhes do ponto de pico
cat("Pico do erro de teste em p =", p_values[peak_index],
    "com erro de teste:", test_errors[peak_index], "\n")

print(p_values)
