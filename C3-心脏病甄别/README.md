# 心脏病患者的甄别（分类）

基于机器学习的心脏病二分类预测项目，使用 13 项常规体检指标判断患者是否患有心脏病。

## 问题描述

心血管疾病难以治愈但可预防，通过常规体检数据提前筛查高危人群具有重要临床意义。本项目使用 UCI Heart Disease 风格的数据集，构建分类模型识别心脏病患者（`target=1` 表示患病，`target=0` 表示未患病），并对 61 名待测患者进行预测。

## 数据

| 文件 | 说明 |
|------|------|
| `heart_train.csv` | 训练集，242 条记录，含 13 个特征 + 标签（患病 132 人 / 健康 110 人） |
| `heart_test.csv` | 测试集，61 条记录，只有特征无标签 |
| `heart_answer.csv` | 测试集真实标签（用于离线评估） |

### 特征说明

| 特征 | 含义 | 类型 |
|------|------|------|
| age | 年龄 | 数值 |
| sex | 性别（0=女, 1=男） | 二值 |
| cp | 胸痛类型（0-3） | 分类 |
| trestbps | 静息血压 | 数值 |
| chol | 血清胆固醇 | 数值 |
| fbs | 空腹血糖 > 120（0=否, 1=是） | 二值 |
| restecg | 静息心电图结果（0-2） | 分类 |
| thalach | 最大心率 | 数值 |
| exang | 运动诱发心绞痛（0=否, 1=是） | 二值 |
| oldpeak | ST 段压低 | 数值 |
| slope | ST 段斜率（0-2） | 分类 |
| ca | 大血管数（0-3） | 分类 |
| thal | 地中海贫血（0-3） | 分类 |

## 方法

1. **Logistic Regression** + StandardScaler 标准化 → 准确率 0.7959（测试分集）/ 81.97%（最终答案）
2. **Random Forest** + RandomizedSearchCV 调参 → 最佳参数 `n_estimators=200, max_depth=5, min_samples_split=10`，准确率 0.8367（测试分集）/ 81.97%（最终答案）
3. **Soft Voting Ensemble**（LR + RF 软投票集成）→ 准确率 0.8163（测试分集）/ **83.61%**（最终答案，51/61 正确）

## 结果

| 提交文件 | 模型 | 测试集准确率 |
|----------|------|:----------:|
| `submit.csv` | Logistic Regression | 81.97% |
| `submit_rf.csv` | Random Forest (调参) | 81.97% |
| `submit_vote.csv` | Soft Voting 集成 | **83.61%** |

**最优方案**：软投票集成模型，在 61 名患者中正确分类 51 人。

## 项目结构

```
.
├── 心脏病患者的甄别（分类）.docx   # 任务文档
├── 心脏病.ipynb                   # 主代码（数据加载 → 建模 → 预测）
├── heart.ipynb                    # 空白模板笔记本
├── heart_train.csv                # 训练数据
├── heart_test.csv                 # 测试数据
├── heart_answer.csv               # 测试集真实答案
├── heart_result.csv               # 原始提交模板
├── submit.csv                     # LR 预测结果
├── submit_rf.csv                  # RF 预测结果
├── submit_vote.csv                # 集成模型预测结果
├── output.png                     # LR 混淆矩阵
├── output1.png                    # LR 特征系数图
├── output2.png                    # RF 特征重要性图
└── README.md
```

## 环境依赖

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## 运行

```bash
# 安装依赖
pip install pandas numpy scikit-learn matplotlib seaborn

# 运行主笔记本
jupyter notebook 心脏病.ipynb
```

## 改进方向

- 数据量较小（242 条），可尝试交叉验证减少过拟合
- 增加特征工程（组合特征、分箱处理）
- 尝试 XGBoost / LightGBM 等梯度提升模型
- 对类别不平衡做处理（SMOTE 过采样）
