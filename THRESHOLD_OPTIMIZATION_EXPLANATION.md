# 🔍 **THRESHOLD OPTIMIZATION DATA LEAKAGE EXPLAINED**

## **What is Threshold Optimization?**

In binary classification, a model outputs **probabilities** (e.g., 0.3, 0.7, 0.9), but we need to make **binary decisions** (0 or 1). A **threshold** determines this:

- **If probability ≥ threshold → Predict class 1 (attack)**
- **If probability < threshold → Predict class 0 (normal)**

## **The Problem: Data Leakage in TTT Evaluation**

### **❌ WRONG WAY (Current TTT Code):**

```python
# In _evaluate_ttt_model (lines 2246-2249)
optimal_threshold, roc_auc, fpr, tpr, thresholds = find_optimal_threshold(
    query_y.cpu().numpy(),           # ← QUERY LABELS (test data!)
    attack_probabilities.cpu().numpy(),  # ← Model predictions on query data
    method='balanced'
)
```

**What happens:**

1. **Model makes predictions** on query data (400 samples)
2. **System looks at the TRUE LABELS** of query data (`query_y`)
3. **System finds the "optimal" threshold** that gives best performance on query data
4. **System uses this threshold** to make final predictions on the SAME query data
5. **Result: 96% accuracy** (because threshold was optimized for this exact data!)

### **✅ CORRECT WAY (No Data Leakage):**

```python
# Should use only support set for threshold optimization
optimal_threshold, roc_auc, fpr, tpr, thresholds = find_optimal_threshold(
    support_y.cpu().numpy(),         # ← SUPPORT LABELS (training data)
    support_probabilities.cpu().numpy(),  # ← Model predictions on support data
    method='balanced'
)
```

**What should happen:**

1. **Model makes predictions** on query data (400 samples)
2. **System uses only support set** (100 samples) to find optimal threshold
3. **System applies this threshold** to query data
4. **Result: Realistic performance** (e.g., 75-80% accuracy)

## **Visual Example:**

### **Current (Wrong) Process:**

```
Query Data: [0.3, 0.7, 0.9, 0.2, 0.8]  ← Model predictions
Query Labels: [0, 1, 1, 0, 1]          ← TRUE LABELS (test data!)

Step 1: Look at true labels
Step 2: Find threshold that maximizes accuracy on these exact samples
Step 3: Use threshold = 0.5 (gives 100% accuracy!)
Step 4: Report 100% accuracy ← CHEATING!
```

### **Correct Process:**

```
Support Data: [0.4, 0.6, 0.8, 0.3]     ← Model predictions
Support Labels: [0, 1, 1, 0]           ← TRUE LABELS (training data)

Query Data: [0.3, 0.7, 0.9, 0.2, 0.8]  ← Model predictions
Query Labels: [0, 1, 1, 0, 1]          ← TRUE LABELS (test data!)

Step 1: Use support set to find threshold = 0.5
Step 2: Apply threshold to query data
Step 3: Get realistic accuracy (e.g., 80%)
```

## **Why This Matters:**

### **The 96% TTT Result is Fake Because:**

1. **Threshold was optimized on test data** (query labels)
2. **Model "saw" the answers** before making predictions
3. **It's like taking an exam where you know the answers beforehand**

### **Real TTT Performance Should Be:**

- **Base Model**: 72.40% (genuine)
- **TTT Model**: ~75-80% (genuine improvement)
- **Improvement**: +3-8% (realistic)

## **The Fix:**

Replace line 2247-2249 in `_evaluate_ttt_model`:

```python
# WRONG (current):
optimal_threshold, roc_auc, fpr, tpr, thresholds = find_optimal_threshold(
    query_y.cpu().numpy(),  # ← Using test labels!
    attack_probabilities.cpu().numpy(),
    method='balanced'
)

# CORRECT (should be):
optimal_threshold, roc_auc, fpr, tpr, thresholds = find_optimal_threshold(
    support_y.cpu().numpy(),  # ← Using only training labels
    support_attack_probabilities.cpu().numpy(),
    method='balanced'
)
```

This is why the TTT results are inflated - the system is essentially "cheating" by using the test labels to find the perfect threshold!
