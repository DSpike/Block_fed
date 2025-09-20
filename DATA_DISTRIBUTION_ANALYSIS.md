# Data Distribution Analysis: Normal vs Attack Samples

## 🚨 **Current Problem Identified**

### **Issue:**

The system is only seeing **1 class (Normal=0)** instead of the expected **2 classes (Normal=0, Attack=1)**.

### **Evidence from System Output:**

```
2025-09-17 13:17:18,719 - INFO - Total samples: 56,000, Classes: 1
2025-09-17 13:17:18,720 - INFO - Class 0: Dirichlet distribution = [0.41540175 0.47988724 0.10471102]

2025-09-17 13:17:18,612 - INFO -     Normal: 56000
2025-09-17 13:17:18,613 - INFO -     Attacks: 0
```

## 📊 **Expected vs Actual Distribution**

### **Expected Distribution (Binary Classification):**

| **Class** | **Label** | **Description**        | **Expected Count** |
| --------- | --------- | ---------------------- | ------------------ |
| **0**     | Normal    | Normal network traffic | ~28,000            |
| **1**     | Attack    | Various attack types   | ~28,000            |

### **Actual Distribution (Current System):**

| **Class** | **Label** | **Description**        | **Actual Count** |
| --------- | --------- | ---------------------- | ---------------- |
| **0**     | Normal    | Normal network traffic | 56,000           |
| **1**     | Attack    | Various attack types   | **0**            |

## 🔍 **Root Cause Analysis**

### **Problem Location:**

File: `blockchain_federated_unsw_preprocessor.py`
Method: `create_zero_day_split()`
Line: 356

### **Problematic Code:**

```python
# This logic is WRONG!
train_mask = (train_df['label'] == 0) | (train_df['label'] != zero_day_id)
```

### **What This Logic Does:**

1. **Include Normal samples**: `train_df['label'] == 0` ✅
2. **Include ALL other attacks**: `train_df['label'] != zero_day_id` ✅
3. **BUT**: The condition is applied to the **original multi-class labels**, not binary labels

### **The Real Issue:**

The training data filtering is working correctly, but there's a **data flow problem**:

1. **Original UNSW-NB15 data**: Has 10 classes (Normal + 9 attack types)
2. **Zero-day split**: Excludes 1 attack type (e.g., Fuzzers)
3. **Training data**: Should have Normal + 8 attack types
4. **Binary conversion**: Should convert to Normal=0, Attack=1
5. **Federated distribution**: Should distribute both Normal and Attack samples

## 🛠️ **Required Fix**

### **Step 1: Verify Training Data Content**

The training data should contain:

- **Normal samples** (label=0) → binary_label=0
- **Attack samples** (label=1,2,3,4,5,6,7,8,9) → binary_label=1

### **Step 2: Check Binary Label Conversion**

```python
# This should work correctly
train_df['binary_label'] = (train_df['label'] != 0).astype(int)
```

### **Step 3: Verify Data Flow**

1. **Preprocessing**: Creates binary labels correctly
2. **Federated Learning**: Uses binary labels for distribution
3. **Dirichlet Distribution**: Should see 2 classes (0 and 1)

## 📈 **Expected Dirichlet Distribution (α=1.0)**

### **With 2 Classes (Normal + Attack):**

```
Class 0 (Normal): Dirichlet distribution = [0.3, 0.4, 0.3]
Class 1 (Attack): Dirichlet distribution = [0.2, 0.5, 0.3]

Client 1: 30% Normal + 20% Attack = 50% of data
Client 2: 40% Normal + 50% Attack = 90% of data
Client 3: 30% Normal + 30% Attack = 60% of data
```

### **Current (Wrong) Distribution:**

```
Class 0 (Normal): Dirichlet distribution = [0.415, 0.480, 0.105]

Client 1: 41.5% of Normal samples only
Client 2: 48.0% of Normal samples only
Client 3: 10.5% of Normal samples only
```

## 🎯 **Impact on Federated Learning**

### **Current Impact:**

1. **No Attack Samples**: Models only learn to classify Normal traffic
2. **Poor Zero-Day Detection**: No attack patterns learned during training
3. **Unrealistic Scenario**: Real federated learning should have mixed data
4. **Dirichlet Distribution**: Can't create proper non-IID with 1 class

### **Expected Impact (After Fix):**

1. **Mixed Data**: Each client gets both Normal and Attack samples
2. **Realistic Non-IID**: Different clients have different attack patterns
3. **Better Zero-Day Detection**: Models learn attack patterns during training
4. **Proper Dirichlet Distribution**: Creates realistic heterogeneity

## 🔧 **Debugging Steps**

### **Step 1: Check Training Data Labels**

```python
# In preprocessor, after zero-day split
print("Training data label distribution:")
print(train_data['label'].value_counts().sort_index())
print("Training data binary label distribution:")
print(train_data['binary_label'].value_counts().sort_index())
```

### **Step 2: Verify Binary Label Creation**

```python
# Check if binary labels are created correctly
normal_count = (train_data['binary_label'] == 0).sum()
attack_count = (train_data['binary_label'] == 1).sum()
print(f"Normal samples: {normal_count}")
print(f"Attack samples: {attack_count}")
```

### **Step 3: Check Federated Learning Input**

```python
# In main.py, before distribution
print("Federated learning input:")
print(f"Train data shape: {train_data.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Unique labels: {torch.unique(train_labels)}")
```

## 📋 **Action Items**

1. **✅ Identify the problem**: Training data only contains Normal samples
2. **🔍 Debug the zero-day split logic**: Check why attack samples are excluded
3. **🛠️ Fix the data filtering**: Ensure attack samples are included in training
4. **✅ Verify binary label creation**: Ensure proper 0/1 conversion
5. **🧪 Test the fix**: Run system and verify 2-class distribution
6. **📊 Validate Dirichlet distribution**: Confirm proper non-IID creation

## 🎯 **Expected Outcome After Fix**

```
2025-09-17 XX:XX:XX,XXX - INFO - Total samples: 56,000, Classes: 2
2025-09-17 XX:XX:XX,XXX - INFO - Class 0: Dirichlet distribution = [0.3, 0.4, 0.3]
2025-09-17 XX:XX:XX,XXX - INFO - Class 1: Dirichlet distribution = [0.2, 0.5, 0.3]

Client 1: 16,800 Normal + 11,200 Attack = 28,000 samples
Client 2: 22,400 Normal + 28,000 Attack = 50,400 samples
Client 3: 16,800 Normal + 16,800 Attack = 33,600 samples
```

This would create a **realistic non-IID federated learning scenario** where:

- Each client has different amounts of data
- Each client has different ratios of Normal vs Attack samples
- The Dirichlet distribution (α=1.0) creates moderate heterogeneity
- Models learn to distinguish between Normal and Attack patterns
- Zero-day detection can leverage learned attack patterns
