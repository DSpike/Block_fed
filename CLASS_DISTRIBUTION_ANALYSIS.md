# Class Distribution Analysis in UNSW-NB15 Dataset

## 🎯 **Overview**

The UNSW-NB15 dataset contains **10 different attack categories** plus **Normal traffic**, but the `label` column is **binary** (0=Normal, 1=Attack). The detailed attack types are stored in the `attack_cat` column.

---

## 📊 **Complete Class Distribution**

### **Training Data (175,341 samples):**

| **Attack Category** | **Samples** | **Percentage** | **Label** | **Binary Label** |
| ------------------- | ----------- | -------------- | --------- | ---------------- |
| **Normal**          | 56,000      | 31.94%         | 0         | 0 (Normal)       |
| **Generic**         | 40,000      | 22.81%         | 1         | 1 (Attack)       |
| **Exploits**        | 33,393      | 19.04%         | 1         | 1 (Attack)       |
| **Fuzzers**         | 18,184      | 10.37%         | 1         | 1 (Attack)       |
| **DoS**             | 12,264      | 6.99%          | 1         | 1 (Attack)       |
| **Reconnaissance**  | 10,491      | 5.98%          | 1         | 1 (Attack)       |
| **Analysis**        | 2,000       | 1.14%          | 1         | 1 (Attack)       |
| **Backdoor**        | 1,746       | 1.00%          | 1         | 1 (Attack)       |
| **Shellcode**       | 1,133       | 0.65%          | 1         | 1 (Attack)       |
| **Worms**           | 130         | 0.07%          | 1         | 1 (Attack)       |

### **Testing Data (82,332 samples):**

| **Attack Category** | **Samples** | **Percentage** | **Label** | **Binary Label** |
| ------------------- | ----------- | -------------- | --------- | ---------------- |
| **Normal**          | 37,000      | 44.94%         | 0         | 0 (Normal)       |
| **Generic**         | 18,871      | 22.92%         | 1         | 1 (Attack)       |
| **Exploits**        | 11,132      | 13.52%         | 1         | 1 (Attack)       |
| **Fuzzers**         | 6,062       | 7.36%          | 1         | 1 (Attack)       |
| **DoS**             | 4,089       | 4.97%          | 1         | 1 (Attack)       |
| **Reconnaissance**  | 3,496       | 4.25%          | 1         | 1 (Attack)       |
| **Analysis**        | 677         | 0.82%          | 1         | 1 (Attack)       |
| **Backdoor**        | 583         | 0.71%          | 1         | 1 (Attack)       |
| **Shellcode**       | 378         | 0.46%          | 1         | 1 (Attack)       |
| **Worms**           | 44          | 0.05%          | 1         | 1 (Attack)       |

---

## 🔍 **Binary Classification Summary**

### **Training Data:**

- **Normal (Class 0)**: 56,000 samples (31.94%)
- **Attack (Class 1)**: 119,341 samples (68.06%)

### **Testing Data:**

- **Normal (Class 0)**: 37,000 samples (44.94%)
- **Attack (Class 1)**: 45,332 samples (55.06%)

---

## 🎯 **Zero-Day Detection Strategy**

### **Current Zero-Day Split Logic:**

```python
# Exclude one attack type from training (e.g., 'DoS')
# Training: Normal + 8 attack types (exclude zero-day)
# Testing: 50% Normal + 50% Zero-day attacks
```

### **Example with DoS as Zero-Day Attack:**

#### **Training Data (After Zero-Day Split):**

| **Attack Category** | **Samples** | **Included**               |
| ------------------- | ----------- | -------------------------- |
| **Normal**          | 56,000      | ✅ Yes                     |
| **Generic**         | 40,000      | ✅ Yes                     |
| **Exploits**        | 33,393      | ✅ Yes                     |
| **Fuzzers**         | 18,184      | ✅ Yes                     |
| **DoS**             | 12,264      | ❌ **EXCLUDED** (Zero-day) |
| **Reconnaissance**  | 10,491      | ✅ Yes                     |
| **Analysis**        | 2,000       | ✅ Yes                     |
| **Backdoor**        | 1,746       | ✅ Yes                     |
| **Shellcode**       | 1,133       | ✅ Yes                     |
| **Worms**           | 130         | ✅ Yes                     |

**Total Training Samples**: 56,000 + 40,000 + 33,393 + 18,184 + 10,491 + 2,000 + 1,746 + 1,133 + 130 = **163,477 samples**

#### **Binary Distribution After Zero-Day Split:**

- **Normal (Class 0)**: 56,000 samples (34.26%)
- **Attack (Class 1)**: 107,477 samples (65.74%)

---

## 📈 **Expected Client Distribution (After Fix)**

### **With Dirichlet Distribution (α = 1.0):**

#### **Class 0 (Normal) Distribution:**

```
Client 1: 30% of 56,000 = 16,800 samples
Client 2: 40% of 56,000 = 22,400 samples
Client 3: 30% of 56,000 = 16,800 samples
```

#### **Class 1 (Attack) Distribution:**

```
Client 1: 20% of 107,477 = 21,495 samples
Client 2: 50% of 107,477 = 53,739 samples
Client 3: 30% of 107,477 = 32,243 samples
```

#### **Total Client Distribution:**

```
Client 1: 16,800 Normal + 21,495 Attack = 38,295 samples
Client 2: 22,400 Normal + 53,739 Attack = 76,139 samples
Client 3: 16,800 Normal + 32,243 Attack = 49,043 samples
```

---

## 🚨 **Current Problem Analysis**

### **Why Only 1 Class in Current System:**

The zero-day split logic is **incorrectly excluding ALL attack samples** instead of just the zero-day attack type.

**Current (Wrong) Logic:**

```python
# This excludes ALL attacks, not just the zero-day attack
train_mask = (train_df['label'] == 0) | (train_df['label'] != zero_day_id)
# Result: Only Normal samples (Class 0) remain
```

**Expected (Correct) Logic:**

```python
# This should exclude only the specific zero-day attack
train_mask = (train_df['label'] == 0) | (train_df['label'] != zero_day_id)
# Result: Normal + 8 attack types (exclude only zero-day)
```

---

## 📊 **Attack Type Distribution by Severity**

### **High-Frequency Attacks (Training Data):**

1. **Generic**: 40,000 samples (22.81%) - Most common attack
2. **Exploits**: 33,393 samples (19.04%) - High-risk attacks
3. **Fuzzers**: 18,184 samples (10.37%) - Input fuzzing attacks

### **Medium-Frequency Attacks:**

4. **DoS**: 12,264 samples (6.99%) - Denial of Service
5. **Reconnaissance**: 10,491 samples (5.98%) - Information gathering

### **Low-Frequency Attacks:**

6. **Analysis**: 2,000 samples (1.14%) - Analysis attacks
7. **Backdoor**: 1,746 samples (1.00%) - Backdoor attacks
8. **Shellcode**: 1,133 samples (0.65%) - Shellcode attacks
9. **Worms**: 130 samples (0.07%) - Worm attacks

---

## 🎯 **Impact on Federated Learning**

### **Current Impact (Wrong Distribution):**

- **Training Data**: Only 56,000 Normal samples
- **Classes**: 1 class (Normal only)
- **Attack Knowledge**: None learned during training
- **Zero-Day Detection**: Poor performance (0% F1-score)

### **Expected Impact (Correct Distribution):**

- **Training Data**: 56,000 Normal + 107,477 Attack samples
- **Classes**: 2 classes (Normal + Attack)
- **Attack Knowledge**: 8 attack types learned during training
- **Zero-Day Detection**: Better performance with learned attack patterns

---

## 🔧 **How to Fix Class Distribution**

### **Step 1: Verify Zero-Day Split Logic**

```python
# Check what attack types are being excluded
zero_day_attack = 'DoS'  # or 'Fuzzers' as shown in system output
zero_day_id = self.attack_types[zero_day_attack]  # Should be 4 for DoS

# The logic should be:
train_mask = (train_df['label'] == 0) | (train_df['label'] != zero_day_id)
# This should include: Normal (0) + All attacks except DoS (4)
```

### **Step 2: Debug the Filtering**

```python
# Add debugging to see what's happening
print(f"Total samples before filtering: {len(train_df)}")
print(f"Normal samples: {len(train_df[train_df['label'] == 0])}")
print(f"DoS samples: {len(train_df[train_df['label'] == 4])}")
print(f"Other attack samples: {len(train_df[(train_df['label'] != 0) & (train_df['label'] != 4)])}")

train_data = train_df[train_mask]
print(f"Samples after filtering: {len(train_data)}")
print(f"Normal after filtering: {len(train_data[train_data['label'] == 0])}")
print(f"Attacks after filtering: {len(train_data[train_data['label'] != 0])}")
```

### **Step 3: Verify Binary Labels**

```python
# Ensure binary labels are created correctly
train_data['binary_label'] = (train_data['label'] != 0).astype(int)
print(f"Binary label distribution: {train_data['binary_label'].value_counts()}")
```

---

## 📋 **Summary**

The UNSW-NB15 dataset has **rich class diversity** with 10 attack categories, but the current system is **incorrectly filtering out all attack samples** during the zero-day split, resulting in:

1. **❌ Only Normal samples** in training data
2. **❌ 1 class instead of 2 classes**
3. **❌ No attack patterns learned**
4. **❌ Poor zero-day detection performance**

**Fix the zero-day split logic** to properly exclude only the specific zero-day attack type, and you'll get:

1. **✅ Normal + 8 attack types** in training data
2. **✅ 2 classes** (Normal + Attack)
3. **✅ Rich attack patterns learned**
4. **✅ Better zero-day detection performance**

The dataset has **excellent class distribution** - the issue is in the **data filtering logic**, not the dataset itself!
