# Linear Neural Networks for Classification  
 

## 📚 Chapter 4.1: Softmax Regression

### 🎯 Classification vs Regression

**The Big Difference:**

| Regression | Classification |
|------------|----------------|
| **Question:** "How much?" | **Question:** "Which category?" |
| **Output:** Continuous number | **Output:** Discrete category |
| **Example:** House price = $320,000 | **Example:** Email = "Spam" or "Inbox" |
| **Loss:** MSE (Mean Squared Error) | **Loss:** Cross-Entropy |
| **Output layer:** Single neuron | **Output layer:** Multiple neurons (one per class) |

---

### 📧 Real-World Classification Problems

```
1. EMAIL FILTERING
   Input: Email text, sender, subject
   Output: Spam | Inbox | Social | Promotions
   
2. IMAGE CLASSIFICATION
   Input: Pixel values
   Output: Dog | Cat | Bird | Car | ...
   
3. MEDICAL DIAGNOSIS
   Input: Symptoms, test results
   Output: Disease A | Disease B | Healthy
   
4. SENTIMENT ANALYSIS
   Input: Review text
   Output: Positive | Negative | Neutral
   
5. FRAUD DETECTION
   Input: Transaction details
   Output: Fraudulent | Legitimate
```

---

### 🏷️ Two Types of Classification

**1. Hard Classification (What we usually want):**
```
Give me ONE definite answer
Example: This email IS spam (100% decision)
```

**2. Soft Classification (How models actually work):**
```
Give me probabilities for each class
Example: 
├─ P(Spam) = 0.92
├─ P(Inbox) = 0.05
└─ P(Social) = 0.03
```

**Why soft first, then hard?**
- Model outputs probabilities (soft)
- We pick highest probability (convert to hard)
- Allows us to set confidence thresholds

---

### 🔢 Label Encoding - Two Approaches

**Problem:** Computers need numbers, not words

#### **❌ Bad Approach: Integer Encoding**

```python
Labels: "cat", "chicken", "dog"
Encode: cat=0, chicken=1, dog=2

Problem:
├─ Implies ordering: cat < chicken < dog
├─ Implies distance: dog-cat = 2, chicken-cat = 1
└─ Model might think: dog = 2×cat ???

When it's OK:
✅ Ordinal data: baby < toddler < adult < geriatric
```

#### **✅ Good Approach: One-Hot Encoding**

```python
Classes: cat, chicken, dog (3 classes)

One-Hot Vectors:
cat     = [1, 0, 0]  ← 1st position = 1, rest = 0
chicken = [0, 1, 0]  ← 2nd position = 1, rest = 0
dog     = [0, 0, 1]  ← 3rd position = 1, rest = 0

Benefits:
✅ No implied ordering
✅ No implied distances
✅ Each class treated equally
✅ Works with any loss function
```

**General Pattern:**

```
q classes → q-dimensional vector
Only one element = 1, all others = 0

Position of '1' indicates the class
```

---

### 🧠 Network Architecture

**Input:** 2×2 grayscale image (4 pixels)

```
Flatten: [x₁, x₂, x₃, x₄]

Network Structure:

Input Layer (4 neurons):
    x₁ ─────┐
    x₂ ─────┤
    x₃ ─────┤
    x₄ ─────┘
            │
            ↓
    (Fully Connected)
            │
            ↓
Output Layer (3 neurons):
    o₁ → P(cat)
    o₂ → P(chicken)  
    o₃ → P(dog)
```

**Mathematical Form:**

```
o₁ = x₁w₁₁ + x₂w₂₁ + x₃w₃₁ + x₄w₄₁ + b₁
o₂ = x₁w₁₂ + x₂w₂₂ + x₃w₃₂ + x₄w₄₂ + b₂
o₃ = x₁w₁₃ + x₂w₂₃ + x₃w₃₃ + x₄w₄₃ + b₃

Compact form:
o = Xw + b

Where:
X: (batch_size × 4) - Input features
w: (4 × 3) - Weight matrix
b: (3,) - Bias vector
o: (batch_size × 3) - Output logits
```

**Parameter Count:**

```
Input features (d) = 4
Output classes (q) = 3

Weights: d × q = 4 × 3 = 12 parameters
Biases:  q = 3 parameters
Total: 15 parameters
```

---

### 🌡️ The Softmax Function - CRITICAL!

**Problem with raw outputs (logits):**

```python
o = [2.5, -1.0, 0.3]

Problems:
❌ Can be negative: -1.0
❌ Don't sum to 1: 2.5 + (-1.0) + 0.3 = 1.8 ≠ 1
❌ Can't interpret as probabilities
```

**Solution: Softmax Transformation**

```
Step 1: Exponentiate (make positive)
exp(o) = [exp(2.5), exp(-1.0), exp(0.3)]
       = [12.18, 0.37, 1.35]

Step 2: Normalize (make sum to 1)
softmax(o) = exp(o) / sum(exp(o))
           = [12.18, 0.37, 1.35] / 13.90
           = [0.876, 0.027, 0.097]
           
Final probabilities:
├─ P(cat) = 0.876 = 87.6%  ← Highest! This is our prediction
├─ P(chicken) = 0.027 = 2.7%
└─ P(dog) = 0.097 = 9.7%
```

**General Formula:**

```
For output oⱼ:

softmax(o)ⱼ = exp(oⱼ) / Σₖ exp(oₖ)

Properties:
✅ All values > 0 (exponential is always positive)
✅ All values < 1 (normalized)
✅ Sum to 1 (Σ softmax(o)ⱼ = 1)
✅ Preserves ordering (if o₁ > o₂, then softmax(o)₁ > softmax(o)₂)
```

---

### 🔥 Softmax Properties - Deep Dive

#### **1. Monotonicity:**

```python
o = [1.0, 2.0, 3.0]
softmax(o) = [0.09, 0.24, 0.67]

Increase o[2]:
o = [1.0, 2.0, 5.0]
softmax(o) = [0.02, 0.05, 0.93]  ← o₃ increased → P₃ increased
```

#### **2. Order Preservation:**

```python
o = [3.0, 1.0, 2.0]
     ↑        ↑
   Largest  Smallest

softmax(o) = [0.67, 0.09, 0.24]
               ↑         ↑
            Largest   Smallest

argmax(o) = argmax(softmax(o)) = 0
```

**Why this matters:**
```
For prediction, we only need to find argmax(o)
Don't actually need to compute softmax!
Just pick the largest logit ✅
```

#### **3. Translation Invariance:**

```python
o = [1, 2, 3]
o' = [2, 3, 4]  # Added 1 to all
o'' = [0, 1, 2]  # Subtracted 1 from all

softmax(o) = softmax(o') = softmax(o'') 
           = [0.09, 0.24, 0.67]

Why? 
exp(oⱼ - c) / Σₖ exp(oₖ - c) 
= exp(oⱼ)·exp(-c) / (Σₖ exp(oₖ)·exp(-c))
= exp(oⱼ) / Σₖ exp(oₖ)  ✅
```

**Numerical Stability Trick:**

```python
# BAD (can overflow):
o = [1000, 1001, 1002]
exp(1000) = ∞  💥

# GOOD (stable):
o_max = max(o) = 1002
o_stable = o - o_max = [-2, -1, 0]
exp(o_stable) = [0.135, 0.368, 1.0]  ✅
```

---

### 📊 Loss Function: Cross-Entropy

**Why not use Mean Squared Error?**

```
MSE for classification:
y_true = [1, 0, 0]  (cat)
y_pred = [0.7, 0.2, 0.1]

MSE = mean((y_true - y_pred)²)
    = mean([(1-0.7)², (0-0.2)², (0-0.1)²])
    = mean([0.09, 0.04, 0.01])
    = 0.047

Problems:
❌ Doesn't penalize confident wrong predictions enough
❌ Treats all wrong predictions similarly
❌ Not derived from probability theory
```

**Cross-Entropy Loss - The Right Choice:**

```
Formula:
l(ŷ, y) = -Σⱼ yⱼ log(ŷⱼ)

For one-hot y (only one yⱼ = 1):
l(ŷ, y) = -log(ŷ_true_class)
```

**Detailed Example:**

```python
# True label: cat
y = [1, 0, 0]

# Prediction 1: Confident and correct
ŷ₁ = [0.9, 0.05, 0.05]
l₁ = -log(0.9) = 0.105  ← Low loss ✅

# Prediction 2: Not confident but correct
ŷ₂ = [0.4, 0.3, 0.3]
l₂ = -log(0.4) = 0.916  ← Higher loss

# Prediction 3: Confident but WRONG
ŷ₃ = [0.1, 0.8, 0.1]
l₃ = -log(0.1) = 2.303  ← Very high loss! ❌

# Prediction 4: Extremely confident but WRONG
ŷ₄ = [0.01, 0.98, 0.01]
l₄ = -log(0.01) = 4.605  ← Massive loss! ❌❌
```

**Why Cross-Entropy Works:**

```
Good Prediction:
├─ High probability for correct class
├─ log(0.99) ≈ -0.01
└─ Loss ≈ 0 ✅

Bad Prediction:
├─ Low probability for correct class
├─ log(0.01) ≈ -4.6
└─ Loss very high ❌

Terrible Prediction:
├─ Probability → 0
├─ log(0.0001) ≈ -9.2
└─ Loss → ∞ ❌❌❌
```

---

### 📐 Cross-Entropy Derivation

**Starting Point:**

```
We have:
├─ True label: y (one-hot)
├─ Predictions: ŷ = softmax(o)
└─ Want: Loss function

Cross-entropy between distributions:
H(P, Q) = -Σⱼ P(j) log Q(j)

In our case:
├─ P = true distribution = y (one-hot)
├─ Q = predicted distribution = ŷ
└─ H(y, ŷ) = -Σⱼ yⱼ log(ŷⱼ)
```

**Simplification for One-Hot:**

```
y = [0, 0, 1, 0]  (class 3 is correct)

Full formula:
H(y, ŷ) = -[0·log(ŷ₁) + 0·log(ŷ₂) + 1·log(ŷ₃) + 0·log(ŷ₄)]
        = -log(ŷ₃)

Result: Only the probability assigned to TRUE class matters!
```

---

### 🎓 Information Theory Connection

**Key Concepts:**

#### **1. Entropy (Uncertainty):**

```
H(P) = -Σⱼ P(j) log P(j)

Intuition: How much "surprise" or "information" in distribution P

Examples:
├─ Coin flip (fair): P = [0.5, 0.5]
│  H(P) = -[0.5 log(0.5) + 0.5 log(0.5)] = 0.693
│  ↑ Maximum uncertainty
│
├─ Biased coin: P = [0.9, 0.1]
│  H(P) = -[0.9 log(0.9) + 0.1 log(0.1)] = 0.325
│  ↑ Less uncertainty
│
└─ Certain: P = [1.0, 0.0]
   H(P) = -[1.0 log(1.0) + 0 log(0)] = 0
   ↑ Zero uncertainty (no surprise)
```

#### **2. Cross-Entropy (Prediction Cost):**

```
H(P, Q) = -Σⱼ P(j) log Q(j)

Intuition: Cost of encoding P using code optimized for Q

Example:
True: P = [1, 0, 0] (definitely cat)
Pred: Q = [0.7, 0.2, 0.1]

H(P, Q) = -[1·log(0.7) + 0·log(0.2) + 0·log(0.1)]
        = -log(0.7) = 0.357

If we predicted perfectly:
Q = [1, 0, 0]
H(P, Q) = -log(1) = 0  ← Minimum possible!
```

**Key Property:**
```
H(P, Q) ≥ H(P)
Equality only when P = Q

In words: 
Cross-entropy is minimized when prediction matches truth
```

---

### 🔬 Softmax + Cross-Entropy Math

**Combined Gradient (IMPORTANT!):**

```
Loss: l = -Σⱼ yⱼ log(ŷⱼ)
where ŷⱼ = softmax(o)ⱼ = exp(oⱼ) / Σₖ exp(oₖ)

Gradient:
∂l/∂oⱼ = softmax(o)ⱼ - yⱼ
       = ŷⱼ - yⱼ

BEAUTIFUL RESULT! 
Just like linear regression: (prediction - truth)
```

**Example Calculation:**

```python
y = [1, 0, 0]  # True: cat
o = [2.0, -1.0, 0.5]  # Raw outputs

# Forward pass
ŷ = softmax(o) = [0.73, 0.04, 0.23]

# Loss
l = -log(0.73) = 0.315

# Gradient
∂l/∂o = ŷ - y 
      = [0.73-1, 0.04-0, 0.23-0]
      = [-0.27, 0.04, 0.23]

Interpretation:
├─ o₁ should DECREASE (predicted too low)
├─ o₂ should increase slightly
└─ o₃ should increase
```

---

### 📊 Multi-Class Output Layer

**Architecture:**

```
Input: d features
Output: q classes

Weight Matrix W: (d × q)
Bias Vector b: (q,)

┌─────────────────────────────────────┐
│  Input    Weights    Output         │
│                                      │
│  x₁ ─── w₁₁,w₁₂,w₁₃ ──→ o₁         │
│  x₂ ─── w₂₁,w₂₂,w₂₃ ──→ o₂         │
│  x₃ ─── w₃₁,w₃₂,w₃₃ ──→ o₃         │
│  x₄ ─── w₄₁,w₄₂,w₄₃                 │
│         └─ Each output connected    │
│            to ALL inputs             │
└─────────────────────────────────────┘

Output j receives contribution from ALL inputs:
oⱼ = Σᵢ xᵢwᵢⱼ + bⱼ
```

**Vectorized:**

```python
# For single example:
o = x @ W + b

# For batch (n examples):
O = X @ W + b

Shapes:
X: (n, d)
W: (d, q)  
b: (q,)
O: (n, q)  ← Broadcasting adds b to each row
```

---

## 📷 Chapter 4.2: Fashion-MNIST Dataset

### 🎯 Why Fashion-MNIST?

**Historical Context:**

| Dataset | Released | Size | Difficulty | Status |
|---------|----------|------|------------|--------|
| **MNIST** | 1998 | 60k train | Too easy now | Retired ❌ |
| **Fashion-MNIST** | 2017 | 60k train | Good for learning | Current ✅ |
| **ImageNet** | 2009 | 1.2M train | Too hard for tutorials | Research |

**Why not MNIST anymore?**
```
MNIST (handwritten digits):
├─ Simple models get 95%+ accuracy
├─ Can't distinguish good vs great models
├─ Not representative of real problems
└─ Too easy!

Fashion-MNIST (clothing items):
├─ Same size/format as MNIST
├─ More challenging (harder to classify)
├─ Better for learning
└─ Still manageable for tutorials ✅
```

---

### 📦 Dataset Details

**10 Categories:**

```python
0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot
```

**Dataset Split:**

```
Training Set:   60,000 images (6,000 per class)
Test Set:       10,000 images (1,000 per class)

Image Format:
├─ Grayscale (1 channel)
├─ 28×28 pixels (can resize to 32×32)
└─ Values: 0-255 (normalized to 0-1)
```

**Tensor Shapes:**

```python
Single image: (1, 28, 28)
               ↑   ↑   ↑
            channels height width

Batch: (batch_size, 1, 28, 28)
        ↑           ↑  ↑   ↑
      examples  channels H  W

Example: batch_size=64
Shape: (64, 1, 28, 28)
```

---

### 💻 Loading the Dataset

```python
class FashionMNIST(DataModule):
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        
        # Transformations
        trans = transforms.Compose([
            transforms.Resize(resize),    # Resize if needed
            transforms.ToTensor()         # Convert to tensor (0-1)
        ])
        
        # Download and load
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root,
            train=True,        # Training set
            transform=trans,
            download=True      # Auto-download if needed
        )
        
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root,
            train=False,       # Test set (we use as validation)
            transform=trans,
            download=True
        )
```

---

### 🔄 Data Loading

```python
data = FashionMNIST(batch_size=64)

# Get one batch
X, y = next(iter(data.train_dataloader()))

print(X.shape)  # torch.Size([64, 1, 28, 28])
print(y.shape)  # torch.Size([64])

X: Images (64 images, 1 channel, 28×28 pixels)
y: Labels (64 integers, each 0-9)
```

**Label Examples:**

```python
y = tensor([9, 0, 0, 3, 0, 2, 7, 2, 5, 5, ...])
           ↑  ↑  ↑  ↑  
        boot shirt shirt dress ...

# Convert to text
labels = data.text_labels(y)
# ['ankle boot', 't-shirt', 't-shirt', 'dress', ...]
```

---

## 🏗️ Chapter 4.3: Base Classification Model

### 🎯 Classifier Class

**Key Difference from Regression:**

```python
class Classifier(Module):
    """
    Base for all classification models
    
    New features vs regression:
    ├─ Accuracy metric (in addition to loss)
    ├─ Argmax for predictions
    └─ Special validation step
    """
    
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        
        # Plot BOTH loss and accuracy
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
```

---

### 🎯 Accuracy Metric - CRITICAL FOR INTERVIEWS!

**Definition:**

```
Accuracy = (Number of correct predictions) / (Total predictions)
         = (TP + TN) / (TP + TN + FP + FN)

Where:
TP = True Positives
TN = True Negatives  
FP = False Positives
FN = False Negatives
```

**Implementation:**

```python
def accuracy(self, Y_hat, Y, averaged=True):
    """
    Compute accuracy
    
    Args:
        Y_hat: Predictions (batch_size, num_classes)
        Y: True labels (batch_size,)
        averaged: Return mean or individual results
    """
    # Reshape to (batch_size, num_classes)
    Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
    
    # Get predicted class (argmax)
    preds = Y_hat.argmax(axis=1).type(Y.dtype)
    
    # Compare with truth
    compare = (preds == Y.reshape(-1)).type(torch.float32)
    
    # Return mean or individual
    return compare.mean() if averaged else compare
```

**Step-by-Step Example:**

```python
Y_hat = [[0.1, 0.7, 0.2],   # Pred: class 1
         [0.8, 0.1, 0.1],   # Pred: class 0
         [0.2, 0.3, 0.5]]   # Pred: class 2

Y = [1, 0, 1]  # True labels

# Step 1: Argmax
preds = [1, 0, 2]

# Step 2: Compare
compare = [1==1, 0==0, 2==1]
        = [True, True, False]
        = [1.0, 1.0, 0.0]

# Step 3: Average
accuracy = (1.0 + 1.0 + 0.0) / 3 = 0.667 = 66.7%
```

---

### 📊 Why Track Both Loss and Accuracy?

```
LOSS:
├─ Differentiable ✅
├─ Used for optimization
├─ Sensitive to confidence
└─ Can decrease even if accuracy stays same

ACCURACY:
├─ Not differentiable ❌
├─ Cannot optimize directly
├─ What we actually care about
└─ Easy to interpret

Example scenario:
Epoch 1: Loss=1.5, Acc=75%
Epoch 2: Loss=0.8, Acc=75%  ← Loss improved, accuracy same
Epoch 3: Loss=0.5, Acc=80%  ← Both improved!

Tracking both gives complete picture
```

---

## 🔨 Chapter 4.4: Softmax Implementation from Scratch

### 🧮 Softmax Implementation

```python
def softmax(X):
    """
    Compute softmax
    
    Args:
        X: (batch_size, num_classes)
    
    Returns:
        Probabilities: (batch_size, num_classes)
    """
    # Step 1: Exponentiate
    X_exp = torch.exp(X)
    
    # Step 2: Sum across classes (axis=1)
    partition = X_exp.sum(1, keepdims=True)
    
    # Step 3: Normalize
    return X_exp / partition  # Broadcasting
```

**Detailed Example:**

```python
X = [[1.0, 2.0, 3.0],
     [0.1, 0.2, 0.7]]

# Step 1: Exp
X_exp = [[2.72, 7.39, 20.09],
         [1.11, 1.22, 2.01]]

# Step 2: Sum per row
partition = [[30.20],  # Row 1 sum
             [4.34]]   # Row 2 sum

# Step 3: Divide
result = [[2.72/30.20, 7.39/30.20, 20.09/30.20],
          [1.11/4.34, 1.22/4.34, 2.01/4.34]]
       = [[0.09, 0.24, 0.67],
          [0.26, 0.28, 0.46]]

# Verify: Each row sums to 1
[0.09+0.24+0.67, 0.26+0.28+0.46] = [1.0, 1.0] ✅
```

---

### 🏗️ Model Architecture

```python
class SoftmaxRegressionScratch(Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize parameters
        self.W = torch.normal(
            0, sigma, 
            size=(num_inputs, num_outputs),
            requires_grad=True
        )
        self.b = torch.zeros(num_outputs, requires_grad=True)
```

**For Fashion-MNIST:**

```
num_inputs = 28 × 28 = 784  (flattened image)
num_outputs = 10             (10 classes)

W: (784, 10) → 7,840 parameters
b: (10,)     → 10 parameters
Total: 7,850 parameters
```

---

### 🔄 Forward Pass

```python
def forward(self, X):
    """
    X: (batch_size, 1, 28, 28) - Images
    Output: (batch_size, 10) - Probabilities
    """
    # Step 1: Flatten
    X = X.reshape((-1, self.W.shape[0]))
    # Now: (batch_size, 784)
    
    # Step 2: Linear transformation
    O = torch.matmul(X, self.W) + self.b
    # Now: (batch_size, 10) - Logits
    
    # Step 3: Softmax
    return softmax(O)
    # Final: (batch_size, 10) - Probabilities
```

**Visual Flow:**

```
Input: (64, 1, 28, 28)
   ↓
Flatten: (64, 784)
   ↓
X @ W: (64, 784) @ (784, 10) = (64, 10)
   ↓
Add b: (64, 10) + (10,) = (64, 10)  [broadcasting]
   ↓
Softmax: (64, 10) → (64, 10)
   ↓
Output: Each row is a probability distribution
```

---

### 💥 Cross-Entropy Implementation

```python
def cross_entropy(y_hat, y):
    """
    y_hat: (batch_size, num_classes) - Probabilities
    y: (batch_size,) - True class indices
    """
    # Select probability of true class for each example
    return -torch.log(y_hat[range(len(y_hat)), y]).mean()
```

**Indexing Trick Explained:**

```python
y_hat = [[0.1, 0.3, 0.6],   # Example 0
         [0.3, 0.2, 0.5]]   # Example 1

y = [0, 2]  # True classes

# What we want:
# Example 0: probability of class 0 = 0.1
# Example 1: probability of class 2 = 0.5

# Fancy indexing:
y_hat[[0, 1], y] = y_hat[[0, 1], [0, 2]]
                 = [y_hat[0, 0], y_hat[1, 2]]
                 = [0.1, 0.5]  ✅

# Loss
loss = -log([0.1, 0.5]).mean()
     = -[log(0.1) + log(0.5)] / 2
     = -[-2.303 + -0.693] / 2
     = 1.498
```

---

### 🏋️ Training

```python
# Setup
data = FashionMNIST(batch_size=256)
model = SoftmaxRegressionScratch(
    num_inputs=784,   # 28×28
    num_outputs=10,   # 10 classes
    lr=0.1
)
trainer = Trainer(max_epochs=10)

# Train
trainer.fit(model, data)

Results after 10 epochs:
├─ Training accuracy: ~82%
└─ Validation accuracy: ~80%
```

---

### 🎯 Making Predictions

```python
# Get test batch
X, y = next(iter(data.val_dataloader()))

# Forward pass
probs = model(X)  # (256, 10)

# Get predicted classes
preds = probs.argmax(axis=1)  # (256,)

# Example:
probs[0] = [0.01, 0.05, 0.02, 0.03, 0.01, 
            0.70, 0.10, 0.05, 0.02, 0.01]
            ↑    ↑    ↑    ↑    ↑    ↑
           0    1    2    3    4    5 ← Predicted!

preds[0] = 5  # Sandal
y[0] = 5      # True label
Correct! ✅
```

---

### 🔍 Analyzing Errors

```python
# Find wrong predictions
wrong = (preds != y)

# Get wrong examples
X_wrong = X[wrong]
y_wrong = y[wrong]
preds_wrong = preds[wrong]

# Visualize
# True: "sneaker"
# Pred: "ankle boot"  ← Easy to confuse!

# True: "shirt"
# Pred: "t-shirt"     ← Very similar!

# True: "pullover"
# Pred: "coat"        ← Reasonable mistake
```

**Common Confusions:**

```
Often Confused:
├─ Shirt ↔ T-shirt
├─ Pullover ↔ Coat
├─ Sneaker ↔ Ankle boot
└─ Dress ↔ Coat

Rarely Confused:
├─ Bag ↔ Trouser (very different!)
├─ Sandal ↔ Shirt
└─ Sneaker ↔ Bag
```

---

## 🚀 Chapter 4.5: Concise Implementation

### ⚡ Using PyTorch Built-ins

```python
class SoftmaxRegression(Classifier):
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        
        self.net = nn.Sequential(
            nn.Flatten(),              # (64,1,28,28) → (64,784)
            nn.LazyLinear(num_outputs) # (64,784) → (64,10)
        )
        
    def forward(self, X):
        return self.net(X)
```

**Key Components:**

```python
nn.Flatten():
├─ Converts (batch, channels, H, W) 
└─ To (batch, channels×H×W)

Example:
Input:  (64, 1, 28, 28)
Output: (64, 784)

nn.LazyLinear(10):
├─ Automatically determines input size
├─ Creates weight matrix (784, 10)
└─ Creates bias vector (10,)
```

---

### 🔒 Numerical Stability - IMPORTANT!

**The Problem:**

```python
# Naive softmax
o = [1000, 1001, 1002]  # Large values!

exp(1000) ≈ 10^434  💥 OVERFLOW!
Result: inf, nan

# Also problematic:
o = [-1000, -1001, -1002]  # Very negative

exp(-1000) ≈ 10^-434  💥 UNDERFLOW!
Result: 0, 0, 0
Then log(0) = -∞  💥
```

**The Solution: LogSumExp Trick**

```python
# Built-in PyTorch handles this!
loss = F.cross_entropy(logits, labels)

# What it does internally:
# Instead of: log(softmax(o))
# Computes: o - log(Σ exp(o))

# Stable version:
o_max = o.max()
log_softmax = o - o_max - log(sum(exp(o - o_max)))

Benefits:
✅ Avoids computing exp of large numbers
✅ Avoids log of very small numbers
✅ Numerically stable
```

---

### 🎯 Complete Built-in Loss

```python
@add_to_class(Classifier)
def loss(self, Y_hat, Y, averaged=True):
    """
    Y_hat: (batch_size, num_classes) - LOGITS (not probabilities!)
    Y: (batch_size,) - True class indices
    """
    Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
    Y = Y.reshape((-1,))
    
    return F.cross_entropy(
        Y_hat, Y,
        reduction='mean' if averaged else 'none'
    )
```

**Key Point:**

```
F.cross_entropy expects LOGITS, not probabilities!

DON'T do:
probs = softmax(logits)
loss = F.cross_entropy(probs, labels)  ❌

DO:
loss = F.cross_entropy(logits, labels)  ✅
(Softmax is computed internally)
```

---

### 🏃 Training Results

```python
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer.fit(model, data)

Typical results:
Epoch 1:  Train=78%, Val=77%
Epoch 5:  Train=82%, Val=81%
Epoch 10: Train=83%, Val=82%

Observations:
✅ Validation close to training (good generalization)
✅ Steady improvement
✅ Converges quickly
```

---

## 📊 Chapter 4.6: Generalization in Classification

### 🎯 The Fundamental Question

```
We train on 60,000 images
We test on 10,000 images

Question: Will our model work on the NEXT million images?

This is the generalization problem!
```

---

### 📏 Test Set Statistics

**Training Error vs Test Error:**

```
Training Error (R_emp):
├─ Computed on training data
├─ Can measure exactly
└─ Formula: (1/n) Σ loss(f(xᵢ), yᵢ)

Population Error (R):
├─ True error on entire population
├─ CANNOT measure (infinite data)
└─ Formula: E[loss(f(X), Y)]

Test Error (ε):
├─ Computed on test data
├─ ESTIMATES population error
└─ Formula: (1/m) Σ loss(f(x'ᵢ), y'ᵢ)
```

---

### 📊 How Many Test Samples Needed?

**Central Limit Theorem Application:**

```
Test error converges at rate: 1/√n

To halve the error estimate uncertainty:
├─ Need 4× more samples

To reduce by factor of 10:
├─ Need 100× more samples

Example:
Want: 95% confidence that |ε - R| < 0.01

Formula: n ≈ 10,000 samples needed!

This is why:
├─ MNIST test set = 10,000 ✅
├─ ImageNet test set = 50,000 ✅
└─ Standard practice in ML
```

---

### ⚠️ Test Set Reuse Problem

**The Danger:**

```
Round 1:
├─ Train model f₁
├─ Test on test set
└─ Accuracy: 80%

Round 2:
├─ Train model f₂ (different architecture)
├─ Test on SAME test set
└─ Accuracy: 82%  ← Is this real improvement?

Round 3:
├─ Train model f₃
├─ Test on SAME test set again
└─ Accuracy: 85%  ← Can we trust this?

Problem: TEST SET CONTAMINATION!
```

**Why This is Bad:**

```
Multiple Testing Problem:
├─ Test 1 model: 5% chance of false positive
├─ Test 20 models: 64% chance at least one is misleading!
├─ Test 100 models: 99.4% chance of contamination
└─ P(at least one false positive) = 1 - 0.95^k

Information Leakage:
├─ You saw test results
├─ You modified model based on test results
├─ Test set is no longer "unseen"
└─ Overestimate true performance
```

---

### ✅ Correct Practice

```
PROPER WORKFLOW:

1. Split Data:
   ├─ Training: 60%
   ├─ Validation: 20%
   └─ Test: 20%

2. Model Development:
   ├─ Train on training set
   ├─ Tune hyperparameters using VALIDATION set
   └─ Can use validation set 100s of times ✅

3. Model Selection:
   ├─ Try multiple architectures
   ├─ Pick best based on VALIDATION performance
   └─ Still haven't touched test set

4. Final Evaluation:
   ├─ Evaluate on test set ONCE
   ├─ Report this number
   └─ NEVER go back and modify model!

If you need more rounds:
└─ Create NEW test set (expensive!)
```

---

### 📐 VC Dimension - IMPORTANT FOR INTERVIEWS!

**Definition:**
```
VC Dimension = Maximum number of points that can be 
               perfectly classified with arbitrary labels

For binary classification:
"How many points can we shatter?"
```

**Examples:**

**1. Linear Classifier in 2D:**

```
VC Dimension = 3

Can shatter 3 points:
  •     •     •
   Any labeling → can find a line to separate

Cannot shatter 4 points:
  • XOR pattern •
     ×    ×
  Cannot separate with a line!
```

**2. Linear Model in d dimensions:**

```
VC Dimension = d + 1

Example:
├─ d=2: VC=3
├─ d=10: VC=11
└─ d=784: VC=785
```

---

### 📊 Generalization Bound

**Theoretical Result:**

```
With probability ≥ (1-δ):

|R - R_emp| ≤ ε

where ε ∝ √(VC·log(n) / n)

In words:
Test error will be close to training error
if we have enough samples
```

**What This Means:**

```
To guarantee ε=0.01 with 95% confidence:

n ≈ VC·log(n) / ε²

For VC=785 (Fashion-MNIST):
n ≈ 785·log(n) / 0.01²
n ≈ millions of samples!

But we only use 60,000! 🤔

Theory is TOO CONSERVATIVE for deep learning
(This is an active research area)
```

---

## 🌍 Chapter 4.7: Distribution Shift

### 🎯 The IID Assumption

**What We Assumed Until Now:**

```
Training data ~ P(x, y)
Test data ~ P(x, y)  ← SAME distribution!

This is called IID:
├─ Independent
├─ Identically
└─ Distributed
```

**When IID Breaks:**

```
Training ~ P_source(x, y)
Test ~ P_target(x, y)  ← DIFFERENT!

Now what? 😱
```

---

### 🔀 Types of Distribution Shift

#### **1. Covariate Shift**

```
Definition:
├─ P(x) changes  ← Input distribution shifts
└─ P(y|x) same   ← Relationship stays the same

Assumption: x causes y

Example: Cat/Dog Classification

Training Data:
├─ Professional photos
├─ Good lighting
├─ Clear backgrounds
└─ P_train(x) = distribution of pro photos

Test Data:
├─ User selfies with pets
├─ Poor lighting
├─ Cluttered backgrounds
└─ P_test(x) = distribution of amateur photos

But: P(dog | x) is same!
A dog is still a dog regardless of photo quality
```

**Real-World Examples:**

```
1. MEDICAL IMAGING:
   Train: Hospital A's scanner
   Test: Hospital B's scanner
   ↳ Different image quality, same diseases

2. AUTONOMOUS DRIVING:
   Train: California (sunny)
   Test: Seattle (rainy)
   ↳ Different weather, same road rules

3. SPAM DETECTION:
   Train: 2020 emails
   Test: 2025 emails
   ↳ Different writing styles, same spam concept
```

---

#### **2. Label Shift**

```
Definition:
├─ P(y) changes  ← Label distribution shifts
└─ P(x|y) same   ← Features given label stay same

Assumption: y causes x

Example: Medical Diagnosis

Training Data (2020):
├─ P(flu) = 0.05
├─ P(covid) = 0.01
└─ P(healthy) = 0.94

Test Data (2021 - pandemic):
├─ P(flu) = 0.02
├─ P(covid) = 0.20  ← Big change!
└─ P(healthy) = 0.78

But: P(symptoms | covid) is same!
Covid symptoms don't change
```

**More Examples:**

```
1. SEASONAL PRODUCTS:
   Train: Summer (swimsuits popular)
   Test: Winter (coats popular)
   ↳ Product demand shifts

2. ELECTION PREDICTION:
   Train: Historical elections
   Test: Current election
   ↳ Voter preferences shift

3. CUSTOMER CHURN:
   Train: Pre-pandemic
   Test: Post-pandemic
   ↳ Churn rates changed
```

---

#### **3. Concept Shift**

```
Definition:
├─ The MEANING of labels changes
└─ P(y|x) changes fundamentally

Example: Regional Terminology

"Pop" vs "Soda" vs "Coke":
├─ Northeast US: "Soda"
├─ Midwest US: "Pop"
├─ South US: "Coke" (for any soft drink!)
└─ Same product, different names!

Building translation system:
├─ Train in Northeast
├─ Deploy in South
└─ Completely different concept! ❌
```

**Other Examples:**

```
1. FASHION:
   2010: "Skinny jeans" = fashionable
   2025: "Skinny jeans" = outdated
   ↳ Concept changed

2. MENTAL HEALTH:
   Diagnostic criteria change over time
   DSM-IV → DSM-V
   ↳ Same symptoms, different diagnosis

3. JOB TITLES:
   "Data Scientist" meant different things
   2010 vs 2025
```

---

### 🛠️ Correction Methods

#### **Covariate Shift Correction:**

```
Idea: Reweight training examples

Step 1: Estimate importance weights
βᵢ = P_target(xᵢ) / P_source(xᵢ)

Step 2: Train with weighted loss
L = (1/n) Σ βᵢ · loss(f(xᵢ), yᵢ)
```

**Algorithm:**

```python
# 1. Create binary dataset
# Label=1 if from target, Label=0 if from source
combined = {
    (x from source, label=0),
    (x from target, label=1)
}

# 2. Train binary classifier
h = train_classifier(combined)
P(target | x) = sigmoid(h(x))

# 3. Compute weights
for x in training_data:
    β = P(target|x) / P(source|x)
      = sigmoid(h(x)) / (1 - sigmoid(h(x)))
      = exp(h(x))

# 4. Train with weights
for x, y, β in weighted_training_data:
    loss = β * cross_entropy(model(x), y)
    update_model(loss)
```

---

#### **Label Shift Correction:**

```
Uses: Confusion Matrix

Step 1: Train model on source data

Step 2: Compute confusion matrix C on validation
C[i,j] = P(predict i | true j)

Example (3 classes):
        True→ Cat  Dog  Bird
Pred ↓
Cat          0.8  0.1  0.1
Dog          0.1  0.7  0.2
Bird         0.1  0.2  0.7

Step 3: Get predictions on target data
μ = [0.5, 0.3, 0.2]  ← Average predictions

Step 4: Solve for target distribution
C·p = μ
p = C⁻¹·μ

Step 5: Compute weights
β_class = p_target(class) / p_source(class)
```

---

### ⚠️ Real-World Failure Cases

#### **1. Medical Diagnostics Disaster:**

```
Goal: Detect disease in older men

Training Data:
├─ Sick: Older men (hospital patients)
└─ Healthy: Young students (blood donors)

Model: 99% accuracy! 🎉

Problem:
├─ Model learned: Age discrimination
├─ Not disease detection!
└─ Failed completely in real world ❌

Covariate Shift:
├─ Age distribution completely different
├─ Hormone levels different
└─ Lifestyle factors different
```

#### **2. Self-Driving Car Failure:**

```
Goal: Detect roadside

Training Data:
├─ Synthetic images from game engine
└─ All roadsides had SAME texture

Model: Perfect on synthetic test! 🎉

Real World:
└─ Complete failure ❌

Problem:
Model learned: "That specific texture = roadside"
Not: "Object with this shape/context = roadside"
```

#### **3. Tank Detection (Famous Story):**

```
Goal: Detect tanks in forest

Training Data:
├─ Morning photos: No tanks
└─ Noon photos: With tanks

Model: 100% accuracy! 🎉

Problem:
Model learned: Shadows vs no shadows
Not: Tank vs no tank ❌

Real world: Failed completely
```

---

### 🔄 Types of Learning Problems

#### **1. Batch Learning:**

```
Training Phase:
├─ Get all data at once
├─ Train model
└─ Deploy model

Deployment:
├─ Model is FIXED
├─ No more updates
└─ Example: Shipped cat door detector

Pros: Simple, stable
Cons: Can't adapt to changes
```

#### **2. Online Learning:**

```
Continuous Process:
For each time step t:
  1. Observe xₜ
  2. Predict ŷₜ = f(xₜ)
  3. Observe true yₜ
  4. Compute loss
  5. Update model
  6. Repeat

Example: Stock Price Prediction
├─ Morning: Predict today's price
├─ Evening: See actual price
├─ Update model
└─ Next day: Repeat

Pros: Adapts to changes
Cons: Complex, can be unstable
```

#### **3. Bandits:**

```
Like online learning but:
├─ Finite set of actions (arms)
├─ Get reward for chosen action
└─ Don't see rewards for other actions

Example: Ad Selection
For each user:
  1. Choose ad to show (pull arm)
  2. User clicks or doesn't (reward)
  3. Update beliefs about ad value
  4. Repeat

Famous: Multi-Armed Bandit problem
```

#### **4. Reinforcement Learning:**

```
Environment has memory and responds:

Agent → Action → Environment
  ↑                    ↓
  ←─── Reward ────────┘

Examples:
├─ Chess: Opponent responds to your moves
├─ Self-driving: Other cars react
└─ Game playing: Environment changes

More complex than supervised learning!
```

---

### 🎯 Interview Questions on Distribution Shift

**Q1: What is covariate shift?**
```
A: Input distribution P(x) changes but 
   P(y|x) stays same.
   
   Example: Same cat detector, different camera quality
```

**Q2: What is label shift?**
```
A: Label distribution P(y) changes but
   P(x|y) stays same.
   
   Example: Disease prevalence changes, symptoms don't
```

**Q3: How to detect distribution shift?**
```
A: Train binary classifier to distinguish
   source vs target data.
   
   If accuracy >> 50%: Distributions are different
   If accuracy ≈ 50%: Distributions are similar
```

**Q4: How to fix covariate shift?**
```
A: Importance weighting
   β = P_target(x) / P_source(x)
   Weight each training example by β
```

**Q5: What is VC dimension?**
```
A: Measure of model complexity
   = Max number of points we can shatter
   
   Linear model in d dimensions: VC = d+1
```

---

## 🎯 Technical Terms for Interviews

### Must-Know Definitions

**1. Logits:**
```
Raw network outputs BEFORE softmax
o = Xw + b
Can be any real number (-∞ to +∞)
```

**2. Softmax:**
```
Converts logits to probabilities
ŷ = softmax(o)
Output: Valid probability distribution (0-1, sum=1)
```

**3. Cross-Entropy:**
```
Loss function for classification
l = -log(ŷ_true_class)
Penalizes wrong predictions
```

**4. One-Hot Encoding:**
```
Represent categories as vectors
q classes → q-dimensional vector
One element = 1, rest = 0
```

**5. Argmax:**
```
Find index of maximum value
preds = argmax(ŷ)
Converts probabilities to class prediction
```

**6. Confusion Matrix:**
```
Table showing predictions vs truth
C[i,j] = Count of (predicted=i, true=j)
Diagonal = correct predictions
```

**7. Empirical Risk:**
```
Average loss on training data
R_emp = (1/n) Σ loss(f(xᵢ), yᵢ)
What we can actually minimize
```

**8. Generalization Error:**
```
Expected loss on population
R = E[loss(f(X), Y)]
What we actually care about
```

---

## 💡 Key Interview Insights

### Common Questions & Answers

**Q: Why can't we use MSE for classification?**
```
A: Multiple reasons:
   1. Doesn't match probability interpretation
   2. Gradients can be wrong
   3. Not derived from maximum likelihood
   4. Cross-entropy has better theoretical properties
```

**Q: Why softmax and not just normalize?**
```
A: Softmax has key properties:
   1. Exponential → emphasizes differences
   2. Differentiable everywhere
   3. Preserves ordering
   4. Has probabilistic interpretation (Gibbs distribution)
```

**Q: What if two classes have similar probabilities?**
```
A: Model is uncertain!
   Example: [0.49, 0.51]
   
   Solutions:
   ├─ Collect more training data
   ├─ Add more features
   ├─ Use more complex model
   └─ Or: Return top-k predictions with confidence scores
```

**Q: Difference between accuracy and loss?**
```
A: 
LOSS:
├─ Continuous, differentiable
├─ Used for optimization
├─ Measures prediction quality

ACCURACY:
├─ Discrete (0 or 1 per example)
├─ Not differentiable
├─ What users care about
└─ Can't optimize directly
```

**Q: Why flatten images?**
```
A: Linear layers expect 1D input
   
   Image: (28, 28) → 2D
   Flatten: (784,) → 1D
   
   Later: CNNs can handle 2D directly!
```

---

## 📊 Complete Comparison Table

| Aspect | Linear Regression | Softmax Regression |
|--------|------------------|-------------------|
| **Problem Type** | Regression | Classification |
| **Output** | Single number | Probability distribution |
| **Output Activation** | None (identity) | Softmax |
| **Output Dimension** | 1 | q (num classes) |
| **Loss** | MSE | Cross-Entropy |
| **Label Format** | Continuous value | One-hot vector |
| **Prediction** | ŷ directly | argmax(ŷ) |
| **Example** | House price | Image category |

---

## 🎯 Algorithm Walkthrough - Complete

### Training Softmax Regression

```python
# SETUP
num_inputs = 784    # 28×28 flattened
num_outputs = 10    # 10 classes
batch_size = 256
lr = 0.1

# INITIALIZE
W = torch.randn(784, 10) * 0.01
b = torch.zeros(10)

# TRAINING LOOP
for epoch in range(max_epochs):
    
    for X_batch, y_batch in train_loader:
        # X_batch: (256, 1, 28, 28)
        # y_batch: (256,)
        
        # 1. FLATTEN
        X = X_batch.reshape(-1, 784)  # (256, 784)
        
        # 2. COMPUTE LOGITS
        o = X @ W + b  # (256, 10)
        
        # 3. SOFTMAX
        y_hat = softmax(o)  # (256, 10)
        
        # 4. LOSS
        loss = cross_entropy(y_hat, y_batch)
        
        # 5. BACKWARD
        optimizer.zero_grad()
        loss.backward()
        
        # 6. UPDATE
        optimizer.step()
        
    # VALIDATION
    with torch.no_grad():
        val_loss, val_acc = evaluate(model, val_loader)
        print(f'Epoch {epoch}: Val Acc = {val_acc:.2%}')
```

---

## 🔥 Common Mistakes & Fixes

### ❌ Mistake 1: Wrong Loss Function

```python
# WRONG
loss = ((y_hat - y) ** 2).mean()  # MSE for classification ❌

# RIGHT
loss = F.cross_entropy(logits, y)  # Cross-entropy ✅
```

### ❌ Mistake 2: Softmax Before Cross-Entropy

```python
# WRONG
probs = F.softmax(logits, dim=1)
loss = F.cross_entropy(probs, y)  # Expects logits! ❌

# RIGHT
loss = F.cross_entropy(logits, y)  # Give logits directly ✅
```

### ❌ Mistake 3: Wrong Label Format

```python
# WRONG for PyTorch
y = [[1, 0, 0],      # One-hot ❌
     [0, 1, 0]]

# RIGHT
y = [0, 1]           # Class indices ✅
```

### ❌ Mistake 4: Forget to Flatten

```python
# WRONG
X = (64, 1, 28, 28)
o = X @ W  # Shape mismatch! ❌

# RIGHT
X = X.reshape(64, 784)
o = X @ W  # (64, 784) @ (784, 10) = (64, 10) ✅
```

---

## 🎯 Final Summary - Must Remember!

### Core Concepts

```
1. SOFTMAX:
   ├─ Converts logits to probabilities
   ├─ Formula: exp(oⱼ) / Σexp(oₖ)
   └─ Properties: Positive, sum to 1

2. CROSS-ENTROPY:
   ├─ Loss for classification
   ├─ Formula: -log(ŷ_true_class)
   └─ Gradient: ŷ - y (simple!)

3. ONE-HOT ENCODING:
   ├─ Represent categories as vectors
   └─ [0, 0, 1, 0] for class 2

4. DISTRIBUTION SHIFT:
   ├─ Covariate: P(x) changes
   ├─ Label: P(y) changes
   └─ Concept: P(y|x) changes

5. GENERALIZATION:
   ├─ Test on held-out data
   ├─ Avoid overfitting
   └─ Never tune on test set!
```

### Key Equations

```
Softmax:        ŷⱼ = exp(oⱼ) / Σₖ exp(oₖ)
Cross-Entropy:  l = -Σⱼ yⱼ log(ŷⱼ) = -log(ŷ_true)
Gradient:       ∂l/∂o = ŷ - y
Accuracy:       (# correct) / (# total)
```

 