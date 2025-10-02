# UNSW-NB15 Preprocessing Pipeline Overview

## 🎯 **6-Step Preprocessing Pipeline for Zero-Day Detection**

The system implements a comprehensive 6-step preprocessing pipeline specifically designed for blockchain federated learning with zero-day detection capabilities.

---

## 📊 **Step 1: Data Quality Assessment**

### **Purpose:**

Evaluate the quality and characteristics of the raw UNSW-NB15 dataset.

### **What it does:**

- **Memory Usage Analysis**: Calculates memory consumption in MB
- **Shape Analysis**: Records dataset dimensions
- **Data Type Analysis**: Counts different data types
- **Missing Values Detection**: Identifies null/NaN values per feature
- **Duplicate Detection**: Counts duplicate rows
- **Infinite Values Detection**: Identifies inf/-inf values
- **Outlier Detection**: Uses IQR method (Q1 - 1.5×IQR, Q3 + 1.5×IQR)

### **Output:**

```python
quality_report = {
    'memory_usage': 95.56,  # MB
    'shape': (175341, 45),
    'missing_values': 0,
    'duplicate_rows': 0,
    'infinite_values': 0,
    'outliers': {...}
}
```

---

## 🔧 **Step 2: Feature Engineering**

### **Purpose:**

Add scientifically-sound features to enhance model performance.

### **Features Added (45 → 49):**

1. **`packet_size_ratio`**: `sbytes / (dbytes + 1)`

   - Ratio of source to destination bytes
   - Indicates data flow direction and volume

2. **`packets_per_second`**: `spkts / (dur + 1)`

   - Network activity intensity
   - Helps detect flooding attacks

3. **`is_tcp`**: `(proto == 'tcp').astype(int)`

   - Binary indicator for TCP protocol
   - Important for connection-based attacks

4. **`is_http`**: `(service == 'http').astype(int)`
   - Binary indicator for HTTP service
   - Critical for web-based attacks

### **Rationale:**

- **Network Security Relevance**: All features are directly related to network intrusion detection
- **Minimal Overfitting**: Only 4 additional features to avoid curse of dimensionality
- **Scientific Soundness**: Based on network security principles

---

## 🧹 **Step 3: Data Cleaning**

### **Purpose:**

Clean and prepare data for machine learning.

### **Cleaning Operations:**

1. **Duplicate Removal**: `df.drop_duplicates()`
2. **Infinite Value Handling**: Replace `inf/-inf` with `NaN`
3. **Missing Value Imputation**:
   - **Numeric columns**: Fill with median
   - **Categorical columns**: Fill with mode (or 'unknown' if no mode)

### **Quality Metrics:**

- Tracks number of duplicates removed
- Counts infinite values converted
- Reports missing values filled per column type

---

## 🔢 **Step 4: Categorical Encoding**

### **Purpose:**

Convert categorical variables to numerical format for ML algorithms.

### **Encoding Strategy (49 → 67 features):**

#### **High-Cardinality Features (>10 unique values):**

- **`proto`**: 133 unique values → **Target Encoding**
- **`service`**: 13 unique values → **Target Encoding**

#### **Low-Cardinality Features (≤10 unique values):**

- **`state`**: 9 unique values → **One-Hot Encoding**

### **Target Encoding Process:**

```python
# For each categorical value, calculate mean target
target_mean = df.groupby(categorical_col)['label'].mean()
df[f'{categorical_col}_target_encoded'] = df[categorical_col].map(target_mean)
```

### **One-Hot Encoding Process:**

```python
# Create binary columns for each category
df_encoded = pd.get_dummies(df, columns=[categorical_col], prefix=categorical_col)
```

---

## 🎯 **Step 5: Feature Selection**

### **Purpose:**

Select the most relevant features using statistical methods.

### **Selection Process:**

1. **Pearson Correlation Analysis**: Calculate correlation between each feature and target
2. **Statistical Significance Testing**: Filter features with p-value < 0.05
3. **Top-N Selection**: Select top 30 features by absolute correlation
4. **Fallback Strategy**: If <30 significant features, take top 30 regardless

### **Selection Criteria:**

- **Statistical Significance**: p < 0.05
- **Correlation Strength**: Higher absolute correlation preferred
- **Feature Count**: Exactly 30 features selected

### **Output:**

```python
selected_features = ['id', 'sttl', 'ct_state_ttl', 'proto_target_encoded', 'packets_per_second', ...]
# Top 5 features: ['id', 'sttl', 'ct_state_ttl', 'proto_target_encoded', 'packets_per_second']
```

---

## 📏 **Step 6: Feature Scaling**

### **Purpose:**

Normalize features to prevent scale bias in machine learning.

### **Scaling Method:**

- **StandardScaler**: `(x - mean) / std`
- **Fit on Training Data Only**: Prevents data leakage
- **Transform All Datasets**: Apply same scaling to train/val/test

### **Features Excluded from Scaling:**

- **`label`**: Original multi-class labels
- **`binary_label`**: Binary labels (0=Normal, 1=Attack)

### **Scaling Process:**

```python
# Fit scaler on training data
scaler.fit(train_df[feature_cols])

# Transform all datasets
train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
```

---

## 🎯 **Zero-Day Split Strategy**

### **Purpose:**

Create realistic zero-day detection scenario.

### **Split Logic:**

1. **Training Data**: Normal + 8 attack types (exclude zero-day attack)
2. **Validation Data**: Same as training
3. **Test Data**: 50% Normal + 50% Zero-day attacks

### **Binary Label Creation:**

```python
# Convert multi-class to binary
train_df['binary_label'] = (train_df['label'] != 0).astype(int)
test_df['binary_label'] = (test_df['label'] != 0).astype(int)
```

### **Attack Types (UNSW-NB15):**

```python
attack_types = {
    'Normal': 0,
    'Fuzzers': 1,
    'Analysis': 2,
    'Backdoors': 3,
    'DoS': 4,
    'Exploits': 5,
    'Generic': 6,
    'Reconnaissance': 7,
    'Shellcode': 8,
    'Worms': 9
}
```

---

## 📈 **Data Flow Summary**

### **Input:**

- **Training CSV**: 175,341 samples × 45 features
- **Testing CSV**: 82,332 samples × 45 features

### **Processing:**

1. **Step 1**: Quality assessment
2. **Step 2**: Feature engineering (45 → 49)
3. **Step 3**: Data cleaning
4. **Step 4**: Categorical encoding (49 → 67)
5. **Step 5**: Feature selection (67 → 30)
6. **Step 6**: Feature scaling

### **Output:**

- **Training Data**: 56,000 samples × 30 features
- **Validation Data**: 56,000 samples × 30 features
- **Test Data**: 74,000 samples × 30 features
- **Binary Labels**: 0=Normal, 1=Attack
- **Feature Names**: List of 30 selected features

---

## 🔍 **Key Design Decisions**

### **1. Minimal Feature Engineering:**

- Only 4 additional features to avoid overfitting
- All features are network security relevant

### **2. Smart Categorical Encoding:**

- Target encoding for high-cardinality features
- One-hot encoding for low-cardinality features

### **3. Statistical Feature Selection:**

- Pearson correlation with significance testing
- Exactly 30 features for optimal performance

### **4. Zero-Day Holdout Strategy:**

- Realistic scenario where one attack type is unknown
- Binary classification for federated learning

### **5. Proper Data Leakage Prevention:**

- Scaler fit only on training data
- Same feature selection applied to all datasets

---

## ⚠️ **Current Issue Identified**

### **Problem:**

The zero-day split is excluding ALL attack samples from training data, resulting in:

- **Training Data**: 56,000 Normal samples, 0 Attack samples
- **Classes**: Only 1 class (Normal=0) instead of 2 classes
- **Impact**: Poor zero-day detection performance

### **Root Cause:**

The training data filtering logic needs to be fixed to include attack samples while excluding only the zero-day attack type.

---

## 🎯 **Expected Outcome After Fix**

### **Proper Distribution:**

- **Training Data**: ~28,000 Normal + ~28,000 Attack samples
- **Classes**: 2 classes (Normal=0, Attack=1)
- **Dirichlet Distribution**: Realistic non-IID across clients
- **Zero-Day Detection**: Better performance with learned attack patterns

This preprocessing pipeline provides a solid foundation for blockchain federated learning with zero-day detection capabilities, following best practices in data science and network security.
