# Data Distribution Across Clients in Blockchain Federated Learning

## 🎯 **Overview**

The system implements **3 different data distribution strategies** for federated learning clients, with the current implementation using **Dirichlet distribution** for realistic non-IID scenarios.

---

## 📊 **Data Distribution Methods**

### **1. IID Distribution (Independent and Identically Distributed)**

```python
def distribute_data(self, train_data, train_labels, distribution_type='iid'):
    # Random split - each client gets random samples
    indices = torch.randperm(len(train_data))
    samples_per_client = len(train_data) // self.num_clients

    for i, client in enumerate(self.clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_data = train_data[indices[start_idx:end_idx]]
        client_labels = train_labels[indices[start_idx:end_idx]]
```

**Characteristics:**

- **Equal samples per client**
- **Random distribution** of all classes
- **Same class proportions** across clients
- **Unrealistic** for real federated learning

### **2. Non-IID Distribution (Simple)**

```python
def distribute_data(self, train_data, train_labels, distribution_type='non_iid'):
    # Split by class - each client gets specific classes
    unique_labels = torch.unique(train_labels)
    samples_per_class = len(train_data) // len(unique_labels)
    samples_per_client_per_class = samples_per_class // self.num_clients
```

**Characteristics:**

- **Class-based splitting**
- **Each client gets specific classes**
- **High heterogeneity**
- **Too extreme** for realistic scenarios

### **3. Dirichlet Distribution (Current Implementation)**

```python
def distribute_data_with_dirichlet(self, train_data, train_labels, alpha=1.0):
    # Realistic non-IID using Dirichlet distribution
    for label in unique_labels:
        dirichlet_dist = np.random.dirichlet([alpha] * self.num_clients)
        dirichlet_distributions[label.item()] = dirichlet_dist
```

**Characteristics:**

- **Realistic heterogeneity**
- **Configurable via α parameter**
- **Each client gets different class proportions**
- **Most realistic** for federated learning

---

## 🔍 **Current Data Distribution Analysis**

### **From System Output:**

```
2025-09-17 13:17:18,719 - INFO - Total samples: 56,000, Classes: 1
2025-09-17 13:17:18,720 - INFO - Class 0: Dirichlet distribution = [0.41540175 0.47988724 0.10471102]

Client client_1: 23,262 total samples
  Class distribution: {0: 23262}
Client client_2: 26,873 total samples
  Class distribution: {0: 26873}
Client client_3: 5,863 total samples
  Class distribution: {0: 5863}
```

### **🚨 Current Problem:**

- **Only 1 class** (Class 0 = Normal) instead of 2 classes
- **No attack samples** in training data
- **Dirichlet distribution** can't work properly with 1 class

---

## 📈 **Expected Data Distribution (After Fix)**

### **With 2 Classes (Normal + Attack):**

#### **Dirichlet Distribution (α = 1.0):**

```
Class 0 (Normal): Dirichlet distribution = [0.3, 0.4, 0.3]
Class 1 (Attack): Dirichlet distribution = [0.2, 0.5, 0.3]

Client 1: 16,800 Normal + 11,200 Attack = 28,000 samples
  Class distribution: {0: 16800, 1: 11200}
  Heterogeneity (std): 0.15

Client 2: 22,400 Normal + 28,000 Attack = 50,400 samples
  Class distribution: {0: 22400, 1: 28000}
  Heterogeneity (std): 0.25

Client 3: 16,800 Normal + 16,800 Attack = 33,600 samples
  Class distribution: {0: 16800, 1: 16800}
  Heterogeneity (std): 0.0
```

---

## 🎛️ **Dirichlet Distribution Parameters**

### **α Parameter Effects:**

| **α Value**  | **Heterogeneity** | **Description**    | **Use Case**        |
| ------------ | ----------------- | ------------------ | ------------------- |
| **α = 0.1**  | **Very High**     | Extreme non-IID    | Stress testing      |
| **α = 0.5**  | **High**          | High heterogeneity | Realistic scenarios |
| **α = 1.0**  | **Moderate**      | Balanced non-IID   | **RECOMMENDED**     |
| **α = 5.0**  | **Low**           | Low heterogeneity  | Near-IID baseline   |
| **α = 10.0** | **Very Low**      | Near-IID           | IID comparison      |

### **Current Setting: α = 1.0**

- **Moderate heterogeneity**
- **Good balance** between IID and extreme non-IID
- **Realistic** for federated learning scenarios
- **Suitable** for zero-day detection experiments

---

## 🔄 **Data Distribution Flow**

### **Step 1: Data Preparation**

```
Original UNSW-NB15 Data:
├── Training: 175,341 samples × 45 features
├── Testing: 82,332 samples × 45 features
└── Classes: 10 (Normal + 9 attack types)
```

### **Step 2: Zero-Day Split**

```
Training Data (After Zero-Day Split):
├── Normal samples: ~28,000
├── Attack samples: ~28,000 (8 attack types, exclude zero-day)
└── Binary labels: 0=Normal, 1=Attack
```

### **Step 3: Dirichlet Distribution**

```
For each class (Normal, Attack):
├── Generate Dirichlet distribution: [α, α, α] → [p1, p2, p3]
├── Calculate samples per client: p_i × class_samples
└── Randomly sample from class for each client
```

### **Step 4: Client Assignment**

```
Client 1: Gets p1_normal × normal_samples + p1_attack × attack_samples
Client 2: Gets p2_normal × normal_samples + p2_attack × attack_samples
Client 3: Gets p3_normal × normal_samples + p3_attack × attack_samples
```

---

## 📊 **Visualization of Distribution**

### **Current (Wrong) Distribution:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Class 0 (Normal)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Client 1  │  │   Client 2  │  │   Client 3  │        │
│  │   41.5%     │  │   48.0%     │  │   10.5%     │        │
│  │  23,262     │  │  26,873     │  │   5,863     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### **Expected (Correct) Distribution:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Class 0 (Normal)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Client 1  │  │   Client 2  │  │   Client 3  │        │
│  │    30%      │  │    40%      │  │    30%      │        │
│  │  16,800     │  │  22,400     │  │  16,800     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Class 1 (Attack)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Client 1  │  │   Client 2  │  │   Client 3  │        │
│  │    20%      │  │    50%      │  │    30%      │        │
│  │  11,200     │  │  28,000     │  │  16,800     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 **Impact on Federated Learning**

### **Current Impact (Wrong Distribution):**

- **No attack patterns learned** during training
- **Poor zero-day detection** performance
- **Unrealistic scenario** for security applications
- **Dirichlet distribution meaningless** with 1 class

### **Expected Impact (Correct Distribution):**

- **Mixed attack patterns** learned by each client
- **Realistic non-IID** federated learning
- **Better zero-day detection** with learned attack knowledge
- **Proper heterogeneity** for algorithm evaluation

---

## 🔧 **How to Fix Data Distribution**

### **Step 1: Fix Zero-Day Split Logic**

```python
# In blockchain_federated_unsw_preprocessor.py
# Ensure attack samples are included in training data
train_mask = (train_df['label'] == 0) | (train_df['label'] != zero_day_id)
# This should include Normal + 8 attack types (exclude only zero-day)
```

### **Step 2: Verify Binary Label Creation**

```python
# Ensure binary labels are created correctly
train_df['binary_label'] = (train_df['label'] != 0).astype(int)
# 0 = Normal, 1 = Attack (all attack types)
```

### **Step 3: Test Distribution**

```python
# Run system and verify 2-class distribution
python src/main.py
# Should show: "Classes: 2" instead of "Classes: 1"
```

---

## 📋 **Summary**

The data distribution system is **well-designed** but currently **broken** due to the zero-day split issue:

1. **✅ Dirichlet distribution** is properly implemented
2. **✅ α = 1.0** provides good heterogeneity
3. **✅ Multiple distribution methods** available
4. **❌ Zero-day split** excludes all attack samples
5. **❌ Only 1 class** in training data
6. **❌ Dirichlet distribution** can't work with 1 class

**Fix the zero-day split logic** and you'll get realistic non-IID federated learning with proper data distribution across clients!
