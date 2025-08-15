# 构造 treatment 变量（以 hope >= 3 为阈值）
df$treatment <- ifelse(df$mulpain == 1, 1, 0)

# 保证 Result 是二分类因变量（若非，请转成 factor）
df$Result <- as.factor(df$Result)

# 拆分数据为 treatment group 和 control group
df_treated <- df %>% filter(treatment == 1)
df_control <- df %>% filter(treatment == 0)

# 去除无用变量，比如 hope（避免泄漏），也可去掉 treatment 本身
exclude_vars <- c( "treatment")

# 拟合模型：在 treatment=1 数据上训练预测模型
rf_treat <- randomForest(Result ~ ., data = df_treated[, !(names(df_treated) %in% exclude_vars)], ntree = 500)

# 拟合模型：在 treatment=0 数据上训练预测模型
rf_control <- randomForest(Result ~ ., data = df_control[, !(names(df_control) %in% exclude_vars)], ntree = 500)

# 用两个模型分别预测所有样本在两种情况下的潜在结果（概率）
df$pred_treat <- predict(rf_treat, newdata = df[, !(names(df) %in% exclude_vars)], type = "prob")[, "1"]
df$pred_control <- predict(rf_control, newdata = df[, !(names(df) %in% exclude_vars)], type = "prob")[, "1"]

# 计算 ITE（个体治疗效应）
df$ITE <- df$pred_treat - df$pred_control


ggplot(df, aes(x = ITE)) +
  geom_histogram(bins = 50, fill = "skyblue", color = "black") +
  geom_vline(xintercept = mean(df$ITE), linetype = "dashed", color = "red") +
  labs(title = "Distribution of Individual Treatment Effects (ITE)",
       x = "ITE (Predicted depression probability: treatment - control)",
       y = "Number of individuals") +
  theme_minimal()
table(df_treated$Result)