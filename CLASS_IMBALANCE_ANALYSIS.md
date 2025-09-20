# Class Imbalance Analysis in UNSW-NB15 Dataset

## 🎯 **Overview**

The UNSW-NB15 dataset exhibits **significant class imbalance** at multiple levels:

1. **Binary Classification**: Moderate imbalance (2.13:1)
2. **Attack Categories**: Severe imbalance (430.8:1)

---

## 📊 **Binary Classification Imbalance**

### **Training Data:**

- **Normal (Class 0)**: 56,000 samples (31.94%)
- **Attack (Class 1)**: 119,341 samples (68.06%)
- **Imbalance Ratio**: **2.13:1** (Attack:Normal)

### **Testing Data:**

- **Normal (Class 0)**: 37,000 samples (44.94%)
- **Attack (Class 1)**: 45,332 samples (55.06%)
- **Imbalance Ratio**: **1.23:1** (Attack:Normal)

### **Severity Level**: ⚠️ **MODERATE IMBALANCE**

- **Training**: 2.13:1 ratio (Attack samples are 2.1x more than Normal)
- **Testing**: 1.23:1 ratio (More balanced than training)

---

## 🚨 **Attack Category Imbalance (SEVERE)**

### **Training Data Distribution:**

| **Attack Category** | **Samples** | **Percentage** | **Imbalance Ratio** |
| ------------------- | ----------- | -------------- | ------------------- |
| **Normal**          | 56,000      | 31.94%         | 1.0x (baseline)     |
| **Generic**         | 40,000      | 22.81%         | 0.71x               |
| **Exploits**        | 33,393      | 19.04%         | 0.60x               |
| **Fuzzers**         | 18,184      | 10.37%         | 0.32x               |
| **DoS**             | 12,264      | 6.99%          | 0.22x               |
| **Reconnaissance**  | 10,491      | 5.98%          | 0.19x               |
| **Analysis**        | 2,000       | 1.14%          | 0.04x               |
| **Backdoor**        | 1,746       | 1.00%          | 0.03x               |
| **Shellcode**       | 1,133       | 0.65%          | 0.02x               |
| **Worms**           | 130         | 0.07%          | 0.002x              |

### **Imbalance Statistics:**

- **Maximum**: Normal (56,000 samples)
- **Minimum**: Worms (130 samples)
- **Imbalance Ratio**: **430.8:1** (Normal:Worms)
- **Severity Level**: 🚨 **SEVERE IMBALANCE**

---

## 📈 **Imbalance Impact Analysis**

### **1. Binary Classification Impact:**

#### **Positive Aspects:**

- **Moderate imbalance** (2.13:1) is manageable
- **Attack-heavy dataset** is realistic for security applications
- **Testing data** is more balanced (1.23:1)

#### **Challenges:**

- **Model bias** toward Attack class
- **False positive risk** for Normal traffic
- **Need for balanced evaluation metrics**

### **2. Attack Category Impact:**

#### **High-Frequency Attacks (Well-represented):**

- **Generic**: 40,000 samples (22.81%)
- **Exploits**: 33,393 samples (19.04%)
- **Fuzzers**: 18,184 samples (10.37%)

#### **Medium-Frequency Attacks (Adequate):**

- **DoS**: 12,264 samples (6.99%)
- **Reconnaissance**: 10,491 samples (5.98%)

#### **Low-Frequency Attacks (Under-represented):**

- **Analysis**: 2,000 samples (1.14%)
- **Backdoor**: 1,746 samples (1.00%)
- **Shellcode**: 1,133 samples (0.65%)
- **Worms**: 130 samples (0.07%) ⚠️ **CRITICAL**

---

## 🎯 **Zero-Day Detection Impact**

### **Current Zero-Day Split (DoS as Zero-Day):**

#### **Training Data After Split:**

- **Normal**: 56,000 samples (34.26%)
- **Generic**: 40,000 samples (24.47%)
- **Exploits**: 33,393 samples (20.42%)
- **Fuzzers**: 18,184 samples (11.12%)
- **Reconnaissance**: 10,491 samples (6.42%)
- **Analysis**: 2,000 samples (1.22%)
- **Backdoor**: 1,746 samples (1.07%)
- **Shellcode**: 1,133 samples (0.69%)
- **Worms**: 130 samples (0.08%)
- **DoS**: 0 samples (EXCLUDED)

#### **Binary Distribution After Split:**

- **Normal (Class 0)**: 56,000 samples (34.26%)
- **Attack (Class 1)**: 107,477 samples (65.74%)
- **Imbalance Ratio**: **1.92:1** (Attack:Normal) - **IMPROVED**

---

## 🔧 **Imbalance Mitigation Strategies**

### **1. Data-Level Strategies:**

#### **Oversampling (SMOTE):**

```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
```

#### **Undersampling:**

```python
from imblearn.under_sampling import RandomUnderSampler

# Random undersampling
undersampler = RandomUnderSampler(random_state=42)
X_balanced, y_balanced = undersampler.fit_resample(X_train, y_train)
```

#### **Class Weighting:**

```python
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
```

### **2. Model-Level Strategies:**

#### **Cost-Sensitive Learning:**

```python
# PyTorch loss with class weights
class_weight = torch.tensor([1.0, 2.13])  # Compensate for 2.13:1 imbalance
criterion = nn.CrossEntropyLoss(weight=class_weight)
```

#### **Focal Loss:**

```python
# Focal loss for imbalanced datasets
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1-pt)**gamma * ce_loss
    return focal_loss.mean()
```

### **3. Evaluation Strategies:**

#### **Balanced Metrics:**

```python
from sklearn.metrics import balanced_accuracy_score, f1_score

# Use balanced accuracy and F1-score
balanced_acc = balanced_accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
```

#### **Confusion Matrix Analysis:**

```python
from sklearn.metrics import classification_report

# Detailed classification report
report = classification_report(y_true, y_pred, target_names=['Normal', 'Attack'])
```

---

## 📊 **Federated Learning Impact**

### **Current Client Distribution (After Fix):**

#### **With Dirichlet Distribution (α = 1.0):**

```
Client 1: 16,800 Normal + 21,495 Attack = 38,295 samples
  Imbalance: 1.28:1 (Attack:Normal) - BALANCED

Client 2: 22,400 Normal + 53,739 Attack = 76,139 samples
  Imbalance: 2.40:1 (Attack:Normal) - MODERATE

Client 3: 16,800 Normal + 32,243 Attack = 49,043 samples
  Imbalance: 1.92:1 (Attack:Normal) - MODERATE
```

### **Imbalance Benefits in Federated Learning:**

- **Realistic scenarios**: Real-world data is imbalanced
- **Diverse client distributions**: Different imbalance ratios per client
- **Robust model training**: Model learns to handle various imbalance levels

---

## 🎯 **Recommendations**

### **1. For Binary Classification:**

- **✅ Current imbalance is manageable** (2.13:1)
- **Use balanced evaluation metrics** (F1-score, balanced accuracy)
- **Consider class weighting** in loss function
- **Monitor false positive rates** for Normal traffic

### **2. For Attack Category Classification:**

- **⚠️ Severe imbalance requires attention** (430.8:1)
- **Focus on high-frequency attacks** for initial training
- **Use hierarchical classification** (Binary → Multi-class)
- **Consider data augmentation** for rare attacks

### **3. For Zero-Day Detection:**

- **✅ Zero-day split improves balance** (1.92:1)
- **Use transductive learning** to handle unseen attack types
- **Implement confidence-based detection** for low-confidence samples
- **Combine multiple detection strategies**

### **4. For Federated Learning:**

- **✅ Dirichlet distribution creates realistic imbalance**
- **Use FedAVG with class weighting**
- **Implement client-specific loss functions**
- **Monitor per-client performance metrics**

---

## 📋 **Summary**

The UNSW-NB15 dataset has **manageable binary class imbalance** but **severe attack category imbalance**:

### **Binary Classification:**

- **Training**: 2.13:1 (Moderate) - ✅ **MANAGEABLE**
- **Testing**: 1.23:1 (Balanced) - ✅ **GOOD**
- **After Zero-day Split**: 1.92:1 (Improved) - ✅ **BETTER**

### **Attack Categories:**

- **Imbalance Ratio**: 430.8:1 (Severe) - 🚨 **REQUIRES ATTENTION**
- **Worms**: Only 130 samples (0.07%) - ⚠️ **CRITICAL**
- **Generic**: 40,000 samples (22.81%) - ✅ **WELL-REPRESENTED**

### **Impact on System:**

- **Binary classification**: ✅ **Good performance expected**
- **Multi-class classification**: ⚠️ **Challenging for rare attacks**
- **Zero-day detection**: ✅ **Improved balance after split**
- **Federated learning**: ✅ **Realistic non-IID scenarios**

**The class imbalance is manageable for binary classification but requires careful handling for multi-class attack detection!**
