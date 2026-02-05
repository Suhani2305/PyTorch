## Linear Regression - THE FOUNDATION

### 🎯 What is Linear Regression?

**Simple Definition:** A method to predict **numerical values** using a straight-line relationship between inputs and outputs.

**Real-World Examples:**

| Problem | Input Features | Output (What we predict) |
|---------|---------------|-------------------------|
| 🏠 **House Pricing** | Area (sq ft), Age (years), Bedrooms | Price ($) |
| 📈 **Stock Prediction** | Previous prices, Volume, Market indicators | Tomorrow's price |
| 🏥 **Hospital Stay** | Age, Disease severity, Treatment type | Days in hospital |
| 🛒 **Retail Sales** | Season, Promotions, Weather | Number of items sold |
| 🚗 **Car Price** | Mileage, Year, Brand, Condition | Resale value |

---

### 📖 Machine Learning Terminology - MUST KNOW!

#### 1. **Dataset Components**

```
DATASET (Complete collection of data)
│
├─ Training Set (used to train the model)
│  ├─ Example/Sample/Instance/Data Point (one row)
│  │  ├─ Features/Covariates (input variables: x₁, x₂, ..., xₐ)
│  │  └─ Label/Target (what we want to predict: y)
│  │
│  └─ Example: {area=1500 sq ft, age=10 years} → price=$300,000
│
└─ Test Set (used to evaluate the model)
```

#### 2. **Detailed Example: House Price Prediction**

**Scenario:** We want to predict house prices

```
Raw Data Table:
┌─────────┬──────────┬───────────┐
│ Area    │ Age      │ Price     │
│ (sq ft) │ (years)  │ ($)       │
├─────────┼──────────┼───────────┤
│ 1500    │ 10       │ 300,000   │ ← One example/sample
│ 2000    │ 5        │ 450,000   │ ← Another example
│ 1200    │ 15       │ 250,000   │
│  ...    │ ...      │ ...       │
└─────────┴──────────┴───────────┘
    ↑         ↑           ↑
 Feature 1  Feature 2   Label (y)
   (x₁)       (x₂)
```

**Terminology mapping:**
- **n** = number of examples (3 shown above, could be 1000s)
- **d** = number of features (2: area and age)
- **Feature dimensionality** = d = 2

 

#### **Basic Intuition:**

Think of it like a recipe:
```
Final Price = (Area × price_per_sqft) + (Age × age_penalty) + base_price
```

#### **Mathematical Form:**

**Long form (explicit):**
```
price = w_area × area + w_age × age + b
```

Where:
- **w_area** = weight for area (e.g., $200/sq ft)
- **w_age** = weight for age (e.g., -$5000/year penalty)
- **b** = bias/intercept (base price when area=0, age=0)

**Example Calculation:**
```
If: w_area = 200, w_age = -5000, b = 100,000
And: area = 1500, age = 10

Then:
price = 200×1500 + (-5000)×10 + 100,000
      = 300,000 - 50,000 + 100,000
      = 350,000
```

#### **Compact Vector Form:**

```
ŷ = w^T x + b
```

**Breaking it down:**

```
w = [w₁]      x = [x₁]      w^T x = w₁×x₁ + w₂×x₂ + ... + wₐ×xₐ
    [w₂]          [x₂]
    [...]          [...]
    [wₐ]          [xₐ]
    
Weight vector  Feature vector  Dot product
```

**The "hat" symbol (ŷ):**
- ŷ = "y-hat" = **predicted** value
- y = actual/true value
- Always distinguish: ŷ (prediction) vs y (reality)

#### **Matrix Form for Multiple Examples:**

```
Design Matrix X (n × d):

       ┌─────────────────────┐
       │ x₁⁽¹⁾  x₂⁽¹⁾ ... xₐ⁽¹⁾│ ← Example 1
       │ x₁⁽²⁾  x₂⁽²⁾ ... xₐ⁽²⁾│ ← Example 2
   X = │  ...    ...  ...  ... │
       │ x₁⁽ⁿ⁾  x₂⁽ⁿ⁾ ... xₐ⁽ⁿ⁾│ ← Example n
       └─────────────────────┘
       n rows (examples) × d columns (features)

Predictions for all examples:
ŷ = Xw + b
```

---

###  Loss Function - Measuring Our Mistakes

#### **Why Do We Need a Loss Function?**

**Problem:** Our model makes predictions, but how do we know if they're good or bad?

**Solution:** Define a **loss function** that quantifies the error

#### **Squared Error Loss - The Most Common Choice**

**For a Single Example:**

```
l^(i)(w, b) = 1/2 (ŷ^(i) - y^(i))²

Example:
True price (y): $300,000
Predicted (ŷ): $320,000
Error: $20,000

Loss = 1/2 × (20,000)²
     = 1/2 × 400,000,000
     = 200,000,000
```

**Why square the error?**
1. ✅ Makes all errors positive (no cancellation)
2. ✅ Penalizes large errors more heavily
3. ✅ Mathematically convenient (smooth, differentiable)
4. ✅ Has nice statistical properties

**Why the 1/2?**
- Makes calculus cleaner
- When we differentiate: d/dx (1/2 x²) = x (the 1/2 and 2 cancel!)

**Visual Understanding:**

```
Squared Error Growth:

Error    →  $1K    $10K   $100K
Loss     →  $0.5M  $50M   $5,000M
             ↑       ↑       ↑
         Small   Medium   HUGE penalty
```

#### **Total Loss Over Entire Dataset:**

```
L(w, b) = 1/n Σᵢ₌₁ⁿ l^(i)(w, b)
        = 1/n Σᵢ₌₁ⁿ 1/2(ŷ^(i) - y^(i))²
        = 1/n Σᵢ₌₁ⁿ 1/2(w^T x^(i) + b - y^(i))²
```

**Mean Squared Error (MSE):**
- We average (÷n) to make loss independent of dataset size
- Loss of 100 on 10 examples = same average as loss of 1000 on 100 examples

#### **Goal of Training:**

```
Find: w*, b* = argmin L(w, b)
              w,b

In words: Find the weights and bias that minimize average loss
```

---

### 🔧 Optimization - Finding the Best Parameters

#### **Analytic Solution (Closed Form)**

**The Math:**

For linear regression, we can solve directly:
```
w* = (X^T X)^(-1) X^T y
```

**Requirements:**
- X^T X must be **invertible**
- Features must be **linearly independent**
- Works ONLY for linear regression

**Why we don't always use it:**
1. ❌ Matrix inversion is expensive for large d (O(d³))
2. ❌ Requires full dataset in memory
3. ❌ Only works for linear models
4. ❌ Most deep learning problems don't have closed form solutions

**When to use:**
✅ Small datasets (< 10,000 examples)
✅ Few features (< 100)
✅ Simple linear regression
✅ Exact solution needed

---

#### **Gradient Descent - The Iterative Approach**

**Core Idea:** 
- Start with random parameters
- Repeatedly take small steps in the direction that reduces loss
- Like hiking down a mountain blindfolded - always step downhill

**Full Batch Gradient Descent:**

```python
# Pseudocode
w, b = initialize_randomly()

for iteration in range(num_iterations):
    # Compute gradient using ALL examples
    grad_w = (1/n) Σᵢ₌₁ⁿ ∂l^(i)/∂w
    grad_b = (1/n) Σᵢ₌₁ⁿ ∂l^(i)/∂b
    
    # Update parameters
    w = w - learning_rate × grad_w
    b = b - learning_rate × grad_b
```

**Problems with Full Batch:**
- ❌ Must process ENTIRE dataset for one update
- ❌ Very slow for large datasets (millions of examples)
- ❌ Redundant computation if data has duplicates
- ❌ Memory intensive

---

#### **Stochastic Gradient Descent (SGD)**

**Extreme opposite:** Use only ONE example at a time

```python
# Pseudocode
w, b = initialize_randomly()

for iteration in range(num_iterations):
    # Pick ONE random example
    i = random_index()
    
    # Compute gradient on this single example
    grad_w = ∂l^(i)/∂w
    grad_b = ∂l^(i)/∂b
    
    # Update
    w = w - learning_rate × grad_w
    b = b - learning_rate × grad_b
```

**Problems with Pure SGD:**
- ❌ Very noisy updates (high variance)
- ❌ Inefficient use of modern hardware (GPUs/CPUs)
- ❌ Doesn't work well with batch normalization
- ❌ Can be unstable

---

#### **Minibatch SGD - THE GOLDILOCKS SOLUTION** ⭐

**Best of both worlds:** Use small batches of examples

```python
# Pseudocode
w, b = initialize_randomly()
batch_size = 32  # Typical: 32, 64, 128, 256

for epoch in range(num_epochs):
    shuffle(dataset)  # Important!
    
    for batch in get_batches(dataset, batch_size):
        # Compute gradient on minibatch
        grad_w = (1/batch_size) Σᵢ∈batch ∂l^(i)/∂w
        grad_b = (1/batch_size) Σᵢ∈batch ∂l^(i)/∂b
        
        # Update
        w = w - learning_rate × grad_w
        b = b - learning_rate × grad_b
```

**Why Minibatch is Best:**

| Aspect | Full Batch | Minibatch | Single Example |
|--------|-----------|-----------|----------------|
| **Speed** | ❌ Slowest | ✅ Fast | ⚠️ Medium |
| **Memory** | ❌ High | ✅ Moderate | ✅ Low |
| **Hardware efficiency** | ⚠️ OK | ✅ Excellent | ❌ Poor |
| **Gradient accuracy** | ✅ Perfect | ✅ Good | ❌ Noisy |
| **Convergence** | ⚠️ Smooth | ✅ Stable | ❌ Unstable |

**Choosing Batch Size:**

```
Factors to consider:

1. GPU Memory:
   - Larger batch = more memory
   - Typical: 32-256 for standard GPUs

2. Model Architecture:
   - Batch normalization needs batch_size > 1
   - Prefer multiples of 8, 16, 32 (hardware optimization)

3. Dataset Size:
   - Small dataset: smaller batches (32)
   - Large dataset: larger batches (256)

4. Learning Dynamics:
   - Smaller batch = more noise = better exploration
   - Larger batch = more stable = faster convergence

Common choices:
- Small models/datasets: 32
- Medium: 64-128
- Large models/datasets: 256
```

---

#### **Update Rules - The Math in Detail**

**For Squared Loss:**

```
Gradient of loss w.r.t. weights:
∂L/∂w = (1/|B|) Σᵢ∈B x^(i) (w^T x^(i) + b - y^(i))

Gradient w.r.t. bias:
∂L/∂b = (1/|B|) Σᵢ∈B (w^T x^(i) + b - y^(i))
```

**Update Step:**

```
w ← w - η × (1/|B|) Σᵢ∈B x^(i) (w^T x^(i) + b - y^(i))
b ← b - η × (1/|B|) Σᵢ∈B (w^T x^(i) + b - y^(i))
```

Where:
- **η (eta)** = learning rate
- **|B|** = minibatch size
- **Σᵢ∈B** = sum over examples in the minibatch

---

### 🎛️ Hyperparameters - What YOU Must Choose

**Definition:** Parameters that are NOT learned by the model, but set by YOU

#### **Key Hyperparameters:**

1. **Learning Rate (η)**

```
Too Small (η = 0.0001):
- Training is VERY slow
- Might not converge in reasonable time
- Safe but inefficient

Good (η = 0.01):
- Steady progress
- Stable convergence
- Sweet spot!

Too Large (η = 1.0):
- Training explodes
- Loss goes to infinity
- Model diverges
```

**Visual:**
```
Loss landscape:

        ╱\      η too large
       ╱  \     (overshoots)
      ╱    \    
─────╱      \───────
    /        \
   /    •     \    η good (reaches minimum)
  /   ↙ ↘     \
 /  ↙     ↘    \
/__________\____\__ 
            ↑
         minimum

  •          η too small (gets stuck)
```

2. **Batch Size (|B|)**
- Small (16-32): More noise, better exploration, slower
- Large (128-256): Less noise, faster, needs more memory

3. **Number of Epochs**
- Too few: Underfitting
- Too many: Overfitting
- Monitor validation loss to decide

4. **Initialization Scale (σ)**
- Weights initialized from N(0, σ²)
- Too small: slow learning
- Too large: instability
- Typical: σ = 0.01

---

### 📊 Training Process - Step by Step

**Complete Algorithm:**

```
INITIALIZATION:
├─ w ∼ N(0, 0.01²)  (small random weights)
├─ b = 0             (zero bias)
├─ η = 0.03          (learning rate)
└─ batch_size = 32

FOR EACH EPOCH (epoch = 1, 2, ..., max_epochs):
│
├─ Shuffle training data (very important!)
│
├─ FOR EACH MINIBATCH:
│  │
│  ├─ 1. FORWARD PASS:
│  │    ├─ Get batch: (X_batch, y_batch)
│  │    └─ Compute: ŷ_batch = X_batch @ w + b
│  │
│  ├─ 2. COMPUTE LOSS:
│  │    └─ L = mean((ŷ_batch - y_batch)²)
│  │
│  ├─ 3. BACKWARD PASS:
│  │    ├─ Compute: ∂L/∂w, ∂L/∂b
│  │    └─ (Automatic differentiation does this!)
│  │
│  └─ 4. UPDATE PARAMETERS:
│       ├─ w ← w - η × ∂L/∂w
│       └─ b ← b - η × ∂L/∂b
│
└─ VALIDATE (optional but recommended):
   ├─ Compute loss on validation set
   ├─ Track validation error
   └─ Check for overfitting
```

---

### 🧪 Mathematical Connection: Maximum Likelihood Estimation

**The Probabilistic View:**

Instead of "minimize squared loss", we can think:
"What's the most likely model given the data?"

#### **Assumptions:**

```
1. True relationship is linear with noise:
   y = w^T x + b + ε
   
2. Noise is Gaussian:
   ε ∼ N(0, σ²)
   
3. Therefore:
   y | x ∼ N(w^T x + b, σ²)
```

**Probability of observing y given x:**

```
p(y | x) = (1/√(2πσ²)) exp(-(y - w^T x - b)² / (2σ²))
```

#### **Maximum Likelihood:**

```
Goal: Find w, b that maximize probability of observing our data

Likelihood = Product of probabilities:
L(w, b) = ∏ᵢ₌₁ⁿ p(y^(i) | x^(i))

Take log (easier to work with):
log L(w, b) = Σᵢ₌₁ⁿ log p(y^(i) | x^(i))
            = Σᵢ₌₁ⁿ [log(1/√(2πσ²)) - (y^(i) - w^T x^(i) - b)²/(2σ²)]
            
Maximize log L ⟺ Minimize Σᵢ₌₁ⁿ (y^(i) - w^T x^(i) - b)²
```

**KEY INSIGHT:** 
```
Minimizing squared loss = Maximum likelihood estimation
(when we assume Gaussian noise)
```

---

### 🧠 Linear Regression as a Neural Network

**Network Diagram:**

```
INPUT LAYER          OUTPUT LAYER
    
    x₁ ──────w₁─────┐
                     │
    x₂ ──────w₂─────┤
                     ├──→ (+b) ──→ ŷ
    x₃ ──────w₃─────┤
                     │
    ...              │
                     │
    xₐ ──────wₐ─────┘

┌─────────────────────────────────────┐
│  All inputs connected to output     │
│  No hidden layers                   │
│  Single neuron (output)             │
│  Fully connected                    │
└─────────────────────────────────────┘
```

**Components:**

1. **Input Layer:** 
   - d neurons (one per feature)
   - No computation, just pass values

2. **Connections:**
   - Each input → output has a weight
   - Total weights = d

3. **Output Layer:**
   - 1 neuron (for regression)
   - Computes: Σ(wᵢxᵢ) + b

4. **Activation Function:**
   - None! (or identity: f(x) = x)
   - This is what makes it LINEAR

---

### 🧬 Biological Inspiration

**Real Neuron Structure:**

```
BIOLOGICAL NEURON:

Dendrites ──→ Cell Body ──→ Axon ──→ Axon Terminals
   (input)    (processing)  (output)   (to other neurons)
   
Information flow:
1. Dendrites receive signals (xᵢ)
2. Signals weighted by synapse strength (wᵢ)
3. Cell body aggregates: Σ wᵢxᵢ
4. Activation function σ(·) fires if threshold exceeded
5. Signal travels down axon
6. Axon terminals pass to next neurons
```

**Artificial Neuron (Our Model):**

```
x₁ ──┐
x₂ ──┤
x₃ ──┼──→ [Σ wᵢxᵢ + b] ──→ ŷ
... ─┤
xₐ ──┘

Same concept:
- Multiple inputs (dendrites)
- Weighted sum (cell body)
- Output (axon)
```

**Important Note:**
Modern deep learning is INSPIRED by neuroscience, but:
- ❌ Not trying to replicate brain exactly
- ✅ Using math/engineering principles
- ✅ Drawing from many fields: stats, optimization, CS, etc.

Like airplanes vs birds:
- Inspired by bird flight
- But don't flap wings!
- Use aerodynamics and engineering

---

### 📝 Summary of Chapter 3.1

**What We Learned:**

1. ✅ Linear regression predicts numerical values
2. ✅ Model: ŷ = w^T x + b
3. ✅ Loss: Mean squared error
4. ✅ Training: Minibatch SGD
5. ✅ Connection to statistics: MLE with Gaussian noise
6. ✅ Simplest neural network (1 layer, no hidden units)

**Key Equations:**

```
Model:        ŷ = w^T x + b
Loss:         L = (1/n) Σ (ŷ^(i) - y^(i))²
Update:       w ← w - η ∂L/∂w
              b ← b - η ∂L/∂b
```

---

## 🏗️ Chapter 3.2: Object-Oriented Design for Implementation

### 🎯 Why Object-Oriented Design?

**Problem:** Deep learning code can get messy fast!
- Models have many components
- Data preprocessing is complex
- Training loops are repetitive
- Hard to reuse code

**Solution:** Organize code into reusable classes!

---

### 🧩 The Three Core Classes

#### **Architecture Overview:**

```
┌─────────────────────────────────────────────────────────┐
│                      TRAINER                            │
│  ┌────────────────────────────────────────────────┐    │
│  │  fit(model, data):                             │    │
│  │    for epoch in epochs:                        │    │
│  │      train_step() ───→ MODEL.training_step()   │    │
│  │      valid_step() ───→ MODEL.validation_step() │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
           │                          │
           ↓                          ↓
┌──────────────────────┐   ┌──────────────────────┐
│       MODULE         │   │    DATAMODULE        │
│  ┌────────────────┐  │   │  ┌────────────────┐  │
│  │ forward()      │  │   │  │ train_loader() │  │
│  │ loss()         │  │   │  │ val_loader()   │  │
│  │ training_step()│  │   │  │ test_loader()  │  │
│  │ configure_opt()│  │   │  └────────────────┘  │
│  └────────────────┘  │   └──────────────────────┘
└──────────────────────┘
```

---

### 📦 Class 1: Module (The Model)

**Purpose:** Contains everything about the MODEL itself

```python
class Module(nn.Module):
    """
    Base class for all models
    
    Responsibilities:
    1. Store learnable parameters (weights, biases)
    2. Define forward pass (how to compute predictions)
    3. Define loss function
    4. Define training step (what happens per batch)
    5. Configure optimizer
    """
    
    def __init__(self):
        """Initialize model architecture and parameters"""
        super().__init__()
        self.board = ProgressBoard()  # For visualization
        
    def forward(self, X):
        """
        Compute predictions
        
        Args:
            X: Input features (batch_size × num_features)
        
        Returns:
            predictions (batch_size × output_dim)
        """
        raise NotImplementedError
        
    def loss(self, y_hat, y):
        """
        Compute loss between predictions and targets
        
        Args:
            y_hat: Predictions
            y: True labels
            
        Returns:
            scalar loss value
        """
        raise NotImplementedError
        
    def training_step(self, batch):
        """
        What happens during one training iteration
        
        Args:
            batch: (X, y) tuple from data loader
            
        Returns:
            loss value
        """
        # Unpack batch
        X, y = batch[:-1], batch[-1]
        
        # Forward pass
        y_hat = self(X)
        
        # Compute loss
        l = self.loss(y_hat, y)
        
        # Log for visualization
        self.plot('loss', l, train=True)
        
        return l
        
    def validation_step(self, batch):
        """What happens during validation"""
        X, y = batch[:-1], batch[-1]
        y_hat = self(X)
        l = self.loss(y_hat, y)
        self.plot('loss', l, train=False)
        
    def configure_optimizers(self):
        """Return optimizer(s) for training"""
        raise NotImplementedError
        
    def plot(self, key, value, train):
        """Helper for logging metrics"""
        # Implementation in full code
        pass
```

**Key Methods Explained:**

1. **`__init__`**: Constructor
   - Sets up model architecture
   - Initializes parameters
   - Creates visualization board

2. **`forward(X)`**: The prediction function
   - Takes inputs X
   - Returns predictions ŷ
   - Called automatically when you do `model(X)`

3. **`loss(y_hat, y)`**: Loss calculation
   - Compares predictions vs truth
   - Returns a scalar (single number)

4. **`training_step(batch)`**: One training iteration
   - Gets a batch of data
   - Computes predictions
   - Computes loss
   - Returns loss (for backward pass)

5. **`validation_step(batch)`**: One validation iteration
   - Same as training but no gradient updates
   - Used to monitor overfitting

6. **`configure_optimizers()`**: Setup optimizer
   - Returns optimizer object (SGD, Adam, etc.)
   - Specifies learning rate, momentum, etc.

---

### 📊 Class 2: DataModule (The Data)

**Purpose:** Contains everything about DATA handling

```python
class DataModule:
    """
    Base class for data
    
    Responsibilities:
    1. Download/prepare data
    2. Preprocess data
    3. Create train/val/test splits
    4. Provide data loaders
    """
    
    def __init__(self, root='../data', num_workers=4):
        """
        Args:
            root: Where to store/load data
            num_workers: Parallel data loading threads
        """
        self.save_hyperparameters()
        
    def get_dataloader(self, train):
        """
        Create data loader
        
        Args:
            train: If True, return training loader
                   If False, return validation loader
        """
        raise NotImplementedError
        
    def train_dataloader(self):
        """Return training data loader"""
        return self.get_dataloader(train=True)
        
    def val_dataloader(self):
        """Return validation data loader"""
        return self.get_dataloader(train=False)
        
    def test_dataloader(self):
        """Return test data loader"""
        # Optional: for final evaluation
        pass
```

**Data Loader Concept:**

```python
# A data loader is a GENERATOR that yields batches

dataloader = data.train_dataloader()

for batch in dataloader:  # Iterates through dataset
    X, y = batch
    # X shape: (batch_size, num_features)
    # y shape: (batch_size, 1)
    
    # Do something with batch
    predictions = model(X)
    loss = loss_fn(predictions, y)
```

**Why Use Data Loaders?**

```
WITHOUT Data Loader:
├─ Must manually batch data
├─ Must manually shuffle
├─ Must handle last batch (might be smaller)
├─ Hard to parallelize
└─ Lots of boilerplate code

WITH Data Loader:
├─ ✅ Automatic batching
├─ ✅ Automatic shuffling
├─ ✅ Handles edge cases
├─ ✅ Parallel data loading
└─ ✅ Clean, simple code
```

---

### 🎮 Class 3: Trainer (The Training Loop)

**Purpose:** Orchestrates the training process

```python
class Trainer:
    """
    Base class for training models
    
    Responsibilities:
    1. Run training loop
    2. Handle epochs and batches
    3. Call model's training/validation steps
    4. Track progress
    5. Handle multi-GPU training (advanced)
    """
    
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        """
        Args:
            max_epochs: How many times to iterate through data
            num_gpus: Number of GPUs to use
            gradient_clip_val: Clip gradients (prevent explosion)
        """
        self.save_hyperparameters()
        
    def prepare_data(self, data):
        """Get data loaders from DataModule"""
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = len(self.val_dataloader)
        
    def prepare_model(self, model):
        """Setup model for training"""
        model.trainer = self  # Give model access to trainer
        model.board.xlim = [0, self.max_epochs]  # Set plot limits
        self.model = model
        
    def fit(self, model, data):
        """
        Main training loop
        
        Args:
            model: Module instance
            data: DataModule instance
        """
        # Setup
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        
        # Training loop
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
            
    def fit_epoch(self):
        """Train for one epoch"""
        # Will be implemented in detail later
        raise NotImplementedError
```

**Complete Training Flow:**

```
trainer.fit(model, data)
    │
    ├─→ prepare_data(data)
    │     └─ Get train/val loaders
    │
    ├─→ prepare_model(model)
    │     └─ Link model ↔ trainer
    │
    ├─→ Get optimizer from model
    │
    └─→ FOR epoch in range(max_epochs):
          │
          ├─→ fit_epoch()
          │     │
          │     ├─ TRAINING PHASE:
          │     │   FOR batch in train_dataloader:
          │     │     ├─ loss = model.training_step(batch)
          │     │     ├─ optimizer.zero_grad()
          │     │     ├─ loss.backward()
          │     │     └─ optimizer.step()
          │     │
          │     └─ VALIDATION PHASE:
          │         FOR batch in val_dataloader:
          │           └─ model.validation_step(batch)
          │
          └─ Plot/log results
```

---

### 🛠️ Utility Functions

#### **1. `@add_to_class` Decorator**

**Problem:** In Jupyter notebooks, we want to split class definitions across cells

**Solution:** Add methods AFTER class is created

```python
def add_to_class(Class):
    """
    Decorator to register function as method in existing class
    
    Usage:
        class A:
            def __init__(self):
                self.x = 1
        
        @add_to_class(A)
        def new_method(self):
            return self.x + 1
        
        a = A()
        a.new_method()  # Returns 2
    """
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
        return obj
    return wrapper
```

**Detailed Example:**

```python
# Step 1: Define basic class
class Calculator:
    def __init__(self, value):
        self.value = value

# Step 2: Create instance
calc = Calculator(10)

# Step 3: Add method AFTER creation
@add_to_class(Calculator)
def add(self, x):
    return self.value + x

@add_to_class(Calculator)
def multiply(self, x):
    return self.value * x

# Step 4: Use new methods
print(calc.add(5))       # 15
print(calc.multiply(3))  # 30

# Even new instances have these methods!
calc2 = Calculator(20)
print(calc2.add(5))      # 25
```

**Why This is Useful:**

```
Jupyter Notebook Flow:

Cell 1:
  class Model:
      def __init__(self): ...

Cell 2 (explanation of forward pass):
  @add_to_class(Model)
  def forward(self, X): ...

Cell 3 (explanation of loss):
  @add_to_class(Model)
  def loss(self, y_hat, y): ...

Benefits:
├─ ✅ Each cell focuses on one concept
├─ ✅ Can add explanatory text between methods
├─ ✅ More readable notebook
└─ ✅ Methods still part of class
```

---

#### **2. `HyperParameters` Class**

**Problem:** Lots of boilerplate saving constructor arguments

**Bad Way (Manual):**

```python
class Model:
    def __init__(self, lr, batch_size, num_layers, hidden_dim):
        self.lr = lr                    # Repetitive!
        self.batch_size = batch_size    # Annoying!
        self.num_layers = num_layers    # Error-prone!
        self.hidden_dim = hidden_dim    # So much typing!
```

**Good Way (HyperParameters):**

```python
class Model(HyperParameters):
    def __init__(self, lr, batch_size, num_layers, hidden_dim):
        self.save_hyperparameters()  # That's it!
        # Now self.lr, self.batch_size, etc. are automatically saved
```

**Implementation:**

```python
class HyperParameters:
    """Automatically save constructor arguments as attributes"""
    
    def save_hyperparameters(self, ignore=[]):
        """
        Save all __init__ arguments as instance attributes
        
        Args:
            ignore: List of argument names to NOT save
        """
        import inspect
        
        # Get the calling function's frame
        frame = inspect.currentframe().f_back
        
        # Get argument names and values
        args = inspect.getargvalues(frame)
        
        # Save each argument as attribute
        for arg in args.locals:
            if arg != 'self' and arg not in ignore:
                setattr(self, arg, args.locals[arg])
```

**Detailed Example:**

```python
class MyModel(HyperParameters):
    def __init__(self, lr, batch_size, dropout, secret_key):
        # Save everything except secret_key
        self.save_hyperparameters(ignore=['secret_key'])
        
        print(f"lr: {self.lr}")              # ✅ Saved
        print(f"batch_size: {self.batch_size}")  # ✅ Saved
        print(f"dropout: {self.dropout}")    # ✅ Saved
        # print(f"secret_key: {self.secret_key}")  # ❌ Not saved!

model = MyModel(lr=0.01, batch_size=32, dropout=0.5, secret_key="xyz")
```

**Use Cases:**

```
When to use save_hyperparameters():
✅ Model hyperparameters (lr, layers, etc.)
✅ Data parameters (batch_size, shuffle, etc.)
✅ Training config (epochs, patience, etc.)

When NOT to use:
❌ Large objects (datasets, models)
❌ Sensitive information (passwords, keys)
❌ Temporary variables
```

---

#### **3. `ProgressBoard` Class**

**Purpose:** Visualize training progress in real-time

```python
class ProgressBoard(HyperParameters):
    """
    Plot metrics during training
    
    Features:
    - Real-time updates
    - Multiple curves (train/val loss, accuracy, etc.)
    - Smoothing for noisy metrics
    - Customizable appearance
    """
    
    def __init__(self, xlabel=None, ylabel=None, 
                 xlim=None, ylim=None,
                 xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'],
                 colors=['C0', 'C1', 'C2', 'C3'],
                 figsize=(3.5, 2.5), display=True):
        """
        Args:
            xlabel, ylabel: Axis labels
            xlim, ylim: Axis limits
            xscale, yscale: 'linear' or 'log'
            ls: Line styles for different curves
            colors: Colors for different curves
            figsize: Figure size
            display: Whether to show plot
        """
        self.save_hyperparameters()
        
    def draw(self, x, y, label, every_n=1):
        """
        Add point to plot
        
        Args:
            x: X-coordinate
            y: Y-coordinate  
            label: Curve name (e.g., 'train_loss')
            every_n: Plot every n-th point (for smoothing)
        """
        # Implementation details...
        pass
```

**Usage Example:**

```python
# Create board
board = ProgressBoard(xlabel='epoch', ylabel='loss')

# During training
for epoch in range(100):
    # Training
    for batch in train_loader:
        train_loss = compute_loss(batch)
        board.draw(epoch, train_loss, 'train_loss', every_n=5)
    
    # Validation
    val_loss = validate()
    board.draw(epoch, val_loss, 'val_loss', every_n=1)
```

**Result:**

```
Visualization:

Loss ↑
    │
 1.0│ ●●●●
    │      ●●●
    │         ●●●
 0.5│            ●●●  train_loss (smooth)
    │               ●●●
    │                  ●●
 0.0│ ─ ─ ─ ─ ─ ─ ─ ─ ─●─  val_loss (jumpy)
    └──────────────────────→ epoch
    0                    100
```

**Parameters Explained:**

```python
every_n=1:   Plot every point (jumpy for noisy data)
every_n=5:   Average every 5 points (smoother)
every_n=10:  Average every 10 points (very smooth)

Example:
Raw data:  [1.0, 0.9, 1.1, 0.8, 1.0, 0.7, ...]
every_n=1: [1.0, 0.9, 1.1, 0.8, 1.0, 0.7, ...]
every_n=3: [1.0, 0.93, 0.90, 0.83, ...]  (averaged)
```

---

### 🔄 How Everything Fits Together

**Complete Example:**

```python
# ============================================
# 1. DEFINE MODEL
# ============================================
class LinearRegression(Module):
    def __init__(self, num_inputs, lr):
        super().__init__()
        self.save_hyperparameters()
        
        # Parameters
        self.w = torch.randn(num_inputs, 1, requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        
    def forward(self, X):
        return X @ self.w + self.b
        
    def loss(self, y_hat, y):
        return ((y_hat - y) ** 2).mean()
        
    def configure_optimizers(self):
        return SGD([self.w, self.b], lr=self.lr)

# ============================================
# 2. DEFINE DATA
# ============================================
class RegressionData(DataModule):
    def __init__(self, w_true, b_true, num_train, num_val, batch_size):
        super().__init__()
        self.save_hyperparameters()
        
        # Generate synthetic data
        n = num_train + num_val
        self.X = torch.randn(n, len(w_true))
        self.y = self.X @ w_true + b_true + torch.randn(n, 1) * 0.01
        
    def get_dataloader(self, train):
        if train:
            indices = range(self.num_train)
        else:
            indices = range(self.num_train, self.num_train + self.num_val)
            
        # Create data loader
        dataset = TensorDataset(self.X[indices], self.y[indices])
        return DataLoader(dataset, self.batch_size, shuffle=train)

# ============================================
# 3. CREATE INSTANCES
# ============================================
model = LinearRegression(num_inputs=2, lr=0.03)
data = RegressionData(
    w_true=torch.tensor([2.0, -3.4]), 
    b_true=4.2,
    num_train=1000,
    num_val=200,
    batch_size=32
)
trainer = Trainer(max_epochs=10)

# ============================================
# 4. TRAIN
# ============================================
trainer.fit(model, data)

# Behind the scenes:
# - Trainer gets data loaders from data
# - Trainer gets optimizer from model
# - For each epoch:
#     - For each batch in train_loader:
#         - Call model.training_step(batch)
#         - Compute gradients
#         - Update parameters
#     - For each batch in val_loader:
#         - Call model.validation_step(batch)
# - ProgressBoard updates plots
```

---

### 📚 Benefits of This Design

**1. Separation of Concerns:**

```
MODULE       → Model architecture, loss, forward pass
DATAMODULE   → Data loading, preprocessing
TRAINER      → Training loop, optimization

Each class has ONE job ✅
```

**2. Reusability:**

```python
# Same Trainer works for ANY model!
trainer = Trainer(max_epochs=10)

# Linear regression
trainer.fit(linear_model, regression_data)

# Neural network (later)
trainer.fit(neural_net, image_data)

# Transformer (much later)
trainer.fit(transformer, text_data)
```

**3. Testability:**

```python
# Test model independently
model = MyModel()
y_hat = model(test_input)
assert y_hat.shape == expected_shape

# Test data independently  
data = MyData()
batch = next(iter(data.train_dataloader()))
assert len(batch) == 2  # (X, y)
```

**4. Flexibility:**

```python
# Easy to customize specific parts

class CustomModel(Module):
    def loss(self, y_hat, y):
        # Custom loss function!
        return my_special_loss(y_hat, y)

# Everything else stays the same
trainer.fit(CustomModel(), data)
```

---

### 💡 Design Patterns Used

**1. Template Method Pattern:**

```python
# Base class defines structure
class Module:
    def training_step(self, batch):
        # Same for all models
        y_hat = self(batch[0])
        loss = self.loss(y_hat, batch[1])
        return loss
    
    def forward(self, X):
        # Subclass implements
        raise NotImplementedError
```

**2. Strategy Pattern:**

```python
# Different optimizers (strategies)
def configure_optimizers(self):
    if self.optimizer == 'sgd':
        return SGD(self.parameters(), lr=self.lr)
    elif self.optimizer == 'adam':
        return Adam(self.parameters(), lr=self.lr)
```

**3. Builder Pattern:**

```python
# Trainer builds the training process step by step
trainer.prepare_data(data)
trainer.prepare_model(model)
trainer.fit_epoch()
```

---

### 🎯 Summary of Chapter 3.2

**What We Learned:**

1. ✅ How to organize ML code into classes
2. ✅ Module: Contains model logic
3. ✅ DataModule: Contains data logic
4. ✅ Trainer: Contains training logic
5. ✅ Utility decorators and classes
6. ✅ How to split class definitions across notebook cells
7. ✅ Benefits of separation of concerns

**Key Takeaways:**

```
Good Code Organization:
├─ ✅ Separate concerns (model, data, training)
├─ ✅ Reusable components
├─ ✅ Easy to test
├─ ✅ Easy to extend
└─ ✅ Readable and maintainable

This design will be used throughout the entire book!
```

---

## 📊 Chapter 3.3: Synthetic Regression Data

### 🎯 Why Synthetic Data?

**The Testing Problem:**

```
When building ML models, we need to know:
├─ ❓ Does our code work correctly?
├─ ❓ Is our math implementation right?
├─ ❓ Does the optimizer converge?
└─ ❓ Can we trust our results?

With REAL data:
├─ ❌ Don't know the true parameters
├─ ❌ Can't verify if we found the "right" answer
└─ ❌ Hard to debug

With SYNTHETIC data:
├─ ✅ We KNOW the true parameters
├─ ✅ Can check if our model recovers them
├─ ✅ Perfect for testing and debugging
└─ ✅ Controlled experimentation
```

### 🔬 Generating Synthetic Data

#### **The Data Generation Process:**

**Step 1: Choose True Parameters**

```python
# These are the "ground truth" we want to recover
w_true = torch.tensor([2.0, -3.4])  # True weights
b_true = 4.2                         # True bias
```

**Step 2: Generate Random Features**

```python
n = 1000  # Number of examples
d = 2     # Number of features

# Features from standard normal distribution
X = torch.randn(n, d)

Shape:
X[0] = [x₁⁽¹⁾, x₂⁽¹⁾]  ← First example
X[1] = [x₁⁽²⁾, x₂⁽²⁾]  ← Second example
...
X[999] = [x₁⁽¹⁰⁰⁰⁾, x₂⁽¹⁰⁰⁰⁾]  ← Last example
```

**Step 3: Generate Labels with Noise**

```python
# True linear relationship
y_true = X @ w_true + b_true

# Add Gaussian noise
noise = torch.randn(n, 1) * 0.01  # σ = 0.01
y = y_true + noise

Mathematical formula:
y⁽ⁱ⁾ = w_true^T x⁽ⁱ⁾ + b_true + ε⁽ⁱ⁾
where ε⁽ⁱ⁾ ∼ N(0, 0.01²)
```

**Visualizing One Example:**

```python
Example i=0:
X[0] = [2.2793, -0.2246]

Calculation:
y[0] = 2.0 × 2.2793 + (-3.4) × (-0.2246) + 4.2 + ε
     = 4.5586 + 0.7636 + 4.2 + 0.0056  (small noise)
     = 9.5278

Actually generated:
y[0] = 9.5014  (slightly different due to noise)
```

---

### 💾 Complete Data Class Implementation

```python
class SyntheticRegressionData(DataModule):
    """
    Synthetic data for testing linear regression
    
    Generates data from:
    y = X @ w + b + noise
    """
    
    def __init__(self, w, b, noise=0.01, 
                 num_train=1000, num_val=1000, 
                 batch_size=32):
        """
        Args:
            w: True weight vector (d-dimensional)
            b: True bias (scalar)
            noise: Standard deviation of Gaussian noise
            num_train: Number of training examples
            num_val: Number of validation examples
            batch_size: Minibatch size
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Total number of examples
        n = num_train + num_val
        
        # Generate features: X ∼ N(0, I)
        self.X = torch.randn(n, len(w))
        
        # Generate noise: ε ∼ N(0, noise²)
        noise_vec = torch.randn(n, 1) * noise
        
        # Generate labels: y = Xw + b + ε
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise_vec
```

**Parameter Breakdown:**

```python
w = torch.tensor([2.0, -3.4])
└─> d = 2 features
    First feature contributes:  +2.0 × x₁
    Second feature contributes: -3.4 × x₂

b = 4.2
└─> Intercept: base value when all features = 0

noise = 0.01
└─> Small random variation
    95% of noise is between ±0.02
    Simulates measurement error

num_train = 1000
└─> Examples for training the model

num_val = 1000
└─> Examples for evaluating the model

batch_size = 32
└─> Process 32 examples at a time
```

---

### 🔄 Data Loading - Manual Implementation

**Goal:** Split data into batches and iterate through them

#### **Manual Iterator (From Scratch):**

```python
@add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    """
    Create a data loader
    
    Args:
        train: If True, shuffle and return train data
               If False, return validation data in order
    """
    if train:
        # Training indices
        indices = list(range(0, self.num_train))
        # Shuffle for better training
        random.shuffle(indices)
    else:
        # Validation indices
        indices = list(range(self.num_train, 
                           self.num_train + self.num_val))
    
    # Yield batches
    for i in range(0, len(indices), self.batch_size):
        # Get batch indices
        batch_indices = torch.tensor(
            indices[i : i + self.batch_size]
        )
        
        # Yield corresponding data
        yield self.X[batch_indices], self.y[batch_indices]
```

**How It Works:**

```
Training Mode (train=True):
├─ indices = [0, 1, 2, ..., 999]
├─ Shuffle: [342, 12, 891, ...]
└─ Batches:
   ├─ Batch 0: indices [342, 12, 891, ..., 156]  (32 examples)
   ├─ Batch 1: indices [789, 23, 445, ..., 901]  (32 examples)
   ...
   └─ Batch 31: indices [234, 567, ...]          (8 examples)
   
Validation Mode (train=False):
├─ indices = [1000, 1001, ..., 1999]
├─ NO shuffle (keep same order each time)
└─ Batches:
   ├─ Batch 0: indices [1000, 1001, ..., 1031]
   ...
```

**Using the Data Loader:**

```python
# Create data
data = SyntheticRegressionData(
    w=torch.tensor([2.0, -3.4]), 
    b=4.2
)

# Get first batch
X, y = next(iter(data.train_dataloader()))

print("X shape:", X.shape)  # torch.Size([32, 2])
print("y shape:", y.shape)  # torch.Size([32, 1])

Explanation:
├─ 32 examples in batch
├─ 2 features per example
└─ 1 label per example
```

**Full Iteration:**

```python
# Iterate through all batches
for batch_idx, (X, y) in enumerate(data.train_dataloader()):
    print(f"Batch {batch_idx}:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    
Output:
Batch 0:
  X shape: torch.Size([32, 2])
  y shape: torch.Size([32, 1])
Batch 1:
  X shape: torch.Size([32, 2])
  y shape: torch.Size([32, 1])
...
Batch 31:
  X shape: torch.Size([8, 2])   ← Last batch (partial)
  y shape: torch.Size([8, 1])
```

---

### ⚡ Concise Implementation with PyTorch

**Why Use Built-in Loaders?**

```
Manual Implementation:
├─ ❌ Slow (Python loops)
├─ ❌ No parallelization
├─ ❌ More code to maintain
├─ ❌ Must handle edge cases manually
└─ ❌ Not memory efficient

PyTorch DataLoader:
├─ ✅ Fast (C++ backend)
├─ ✅ Parallel data loading
├─ ✅ Minimal code
├─ ✅ Handles all edge cases
└─ ✅ Memory efficient
```

**Implementation:**

```python
@add_to_class(DataModule)
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    """
    Create PyTorch data loader
    
    Args:
        tensors: Tuple of (X, y)
        train: Whether to shuffle
        indices: Which indices to use
    """
    # Select subset of data
    tensors = tuple(a[indices] for a in tensors)
    
    # Create PyTorch dataset
    dataset = torch.utils.data.TensorDataset(*tensors)
    
    # Create data loader
    return torch.utils.data.DataLoader(
        dataset, 
        self.batch_size,
        shuffle=train
    )

@add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    """Use built-in PyTorch loader"""
    # Determine which slice to use
    if train:
        i = slice(0, self.num_train)
    else:
        i = slice(self.num_train, None)
    
    return self.get_tensorloader((self.X, self.y), train, i)
```

**Comparison:**

```python
# Both give same results!

# Manual
manual_loader = data.get_dataloader_manual(train=True)

# Built-in
builtin_loader = data.get_dataloader(train=True)

# Same batches
for (X1, y1), (X2, y2) in zip(manual_loader, builtin_loader):
    assert torch.allclose(X1, X2)
    assert torch.allclose(y1, y2)
```

---

### 📏 Data Loader Features

#### **1. Length Support:**

```python
loader = data.train_dataloader()

print(len(loader))  # Number of batches

Calculation:
num_train = 1000
batch_size = 32
num_batches = ceil(1000 / 32) = 32
```

#### **2. Shuffling:**

```python
# First epoch
for X, y in data.train_dataloader():
    # Batch order: random

# Second epoch  
for X, y in data.train_dataloader():
    # DIFFERENT random order

Why shuffle?
├─ Prevents model from learning batch order
├─ Reduces overfitting
└─ Better gradient estimates
```

#### **3. Parallel Loading:**

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4  # 4 parallel processes
)

Benefits:
├─ Load next batch while training current batch
├─ Faster overall training
└─ Better GPU utilization
```

#### **4. Automatic Batching:**

```python
# Handles last batch automatically

If num_train = 1000, batch_size = 32:
├─ Batches 0-30: 32 examples each
└─ Batch 31: 8 examples (remainder)

No special code needed!
```

---

### 🔍 Inspecting the Data

**Individual Examples:**

```python
# First example
print("First example:")
print(f"  Features: {data.X[0]}")
print(f"  Label: {data.y[0]}")

Output:
First example:
  Features: tensor([ 2.2793, -0.2246])
  Label: tensor([9.5014])
  
Interpretation:
├─ Feature 1 = 2.2793
├─ Feature 2 = -0.2246
└─ Label (price) = $9.5014
```

**Statistics:**

```python
print("Feature statistics:")
print(f"  Mean: {data.X.mean(dim=0)}")
print(f"  Std: {data.X.std(dim=0)}")

Output:
Mean: tensor([-0.0123,  0.0089])  ← Close to 0 ✅
Std:  tensor([0.9987, 1.0012])    ← Close to 1 ✅

print("Label statistics:")
print(f"  Mean: {data.y.mean()}")
print(f"  Std: {data.y.std()}")

Interpretation:
├─ X generated from N(0, 1) → mean≈0, std≈1
└─ y depends on w, b, and noise
```

**Verifying Ground Truth:**

```python
# Manually compute expected values
w_true = torch.tensor([2.0, -3.4])
b_true = 4.2

# Predict for first example
x0 = data.X[0]
y_pred = (w_true * x0).sum() + b_true

print(f"Predicted (no noise): {y_pred:.4f}")
print(f"Actual (with noise): {data.y[0].item():.4f}")
print(f"Noise: {(data.y[0] - y_pred).item():.4f}")

Output:
Predicted (no noise): 9.5222
Actual (with noise): 9.5014
Noise: -0.0208  ← Small random variation ✅
```

---

### 🎲 Why Random Data Generation Matters

**Advantages:**

```
1. KNOWN GROUND TRUTH:
   ├─ We set w_true, b_true
   ├─ Can verify if model recovers them
   └─ Objective measure of success

2. CONTROLLED EXPERIMENTS:
   ├─ Vary noise level → test robustness
   ├─ Vary num_features → test scaling
   ├─ Vary num_samples → test sample efficiency
   └─ Perfect for ablation studies

3. DEBUGGING:
   ├─ If model can't fit synthetic data → bug in code
   ├─ If model fits synthetic but not real → data issue
   └─ Isolate problems systematically

4. REPRODUCIBILITY:
   ├─ Set random seed → same data every time
   ├─ Others can replicate experiments
   └─ Fair comparison across methods
```

**Setting Random Seeds:**

```python
import torch
import random
import numpy as np

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
# Use it
set_seed(42)
data1 = SyntheticRegressionData(w=torch.tensor([2.0, -3.4]), b=4.2)

set_seed(42)
data2 = SyntheticRegressionData(w=torch.tensor([2.0, -3.4]), b=4.2)

# Same data!
assert torch.allclose(data1.X, data2.X)
assert torch.allclose(data1.y, data2.y)
```

---

### 📊 Data Splits

**Why Split Data?**

```
Full Dataset (2000 examples)
│
├─ Training Set (1000 examples)
│  └─ Used to FIT the model
│
└─ Validation Set (1000 examples)
   └─ Used to EVALUATE the model

Critical: Model never sees validation data during training!
```

**Accessing Splits:**

```python
# Training data
train_loader = data.train_dataloader()
print(f"Training batches: {len(train_loader)}")  # 32

# Validation data
val_loader = data.val_dataloader()
print(f"Validation batches: {len(val_loader)}")  # 32

# Different data!
train_X, train_y = next(iter(train_loader))
val_X, val_y = next(iter(val_loader))

# Training uses examples 0-999
# Validation uses examples 1000-1999
```

---

### 🎯 Summary of Chapter 3.3

**What We Learned:**

1. ✅ Why synthetic data is useful (known ground truth)
2. ✅ How to generate linear regression data
3. ✅ Manual data loader implementation
4. ✅ PyTorch built-in data loaders
5. ✅ Train/validation splits
6. ✅ Batching and shuffling

**Key Code:**

```python
# Generate data
data = SyntheticRegressionData(
    w=torch.tensor([2.0, -3.4]),
    b=4.2,
    noise=0.01,
    num_train=1000,
    num_val=1000,
    batch_size=32
)

# Iterate
for X, y in data.train_dataloader():
    # X: (32, 2)
    # y: (32, 1)
    pass
```
# Linear Regression - COMPLETE NOTES (Continued)

---

## 🛠️ Chapter 3.4: Linear Regression Implementation from Scratch

### 🎯 Complete Training Flow

```
STEP 1: Initialize Parameters
STEP 2: Define Model (Forward Pass)
STEP 3: Define Loss Function
STEP 4: Define Optimizer
STEP 5: Training Loop
```

---

### **STEP 1: Initialize Parameters**

```python
class LinearRegressionScratch(Module):
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize weights: w ~ N(0, σ²)
        self.w = torch.normal(0, sigma, (num_inputs, 1), 
                             requires_grad=True)
        
        # Initialize bias: b = 0
        self.b = torch.zeros(1, requires_grad=True)
```

**Why these initializations?**

| Parameter | Initialization | Reason |
|-----------|---------------|---------|
| **Weights (w)** | N(0, 0.01²) | ✅ Small random values break symmetry<br>✅ Not too large (unstable)<br>✅ Not zero (no learning) |
| **Bias (b)** | 0 | ✅ Common practice<br>✅ Will be learned anyway |
| **requires_grad** | True | ✅ Enable automatic differentiation<br>✅ PyTorch tracks gradients |

---

### **STEP 2: Define Model (Forward Pass)**

```python
@add_to_class(LinearRegressionScratch)
def forward(self, X):
    """Compute predictions: ŷ = Xw + b"""
    return torch.matmul(X, self.w) + self.b
```

**Matrix multiplication details:**

```
X shape:    (batch_size, num_inputs) = (32, 2)
w shape:    (num_inputs, 1) = (2, 1)
Result:     (32, 1)

Example:
X = [[x₁⁽¹⁾, x₂⁽¹⁾],    w = [[w₁],    Xw = [[ŷ⁽¹⁾],
     [x₁⁽²⁾, x₂⁽²⁾],         [w₂]]          [ŷ⁽²⁾],
     ...]                                     ...]

Then add b (broadcasts to all rows):
Xw + b = [[ŷ⁽¹⁾ + b],
          [ŷ⁽²⁾ + b],
          ...]
```

---

### **STEP 3: Define Loss Function**

```python
@add_to_class(LinearRegressionScratch)
def loss(self, y_hat, y):
    """Mean Squared Error: (1/n)Σ(ŷ - y)²/2"""
    l = (y_hat - y) ** 2 / 2
    return l.mean()
```

**Step-by-step:**

```python
y_hat = [[320000],    y = [[300000],
         [450000],         [450000],
         [250000]]         [260000]]

# Element-wise difference
diff = [[20000],      # Error for example 1
        [0],          # Perfect prediction!
        [-10000]]     # Error for example 3

# Square
squared = [[400000000],
           [0],
           [100000000]]

# Divide by 2
halved = [[200000000],
          [0],
          [50000000]]

# Mean
loss = (200000000 + 0 + 50000000) / 3 = 83333333
```

---

### **STEP 4: Define Optimizer (SGD)**

```python
class SGD(HyperParameters):
    def __init__(self, params, lr):
        self.save_hyperparameters()
        
    def step(self):
        """Update parameters: θ ← θ - η∇θ"""
        for param in self.params:
            param -= self.lr * param.grad
            
    def zero_grad(self):
        """Clear gradients (must do before backward!)"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

@add_to_class(LinearRegressionScratch)
def configure_optimizers(self):
    return SGD([self.w, self.b], self.lr)
```

**Why zero_grad()?**

```python
# PyTorch ACCUMULATES gradients by default!

Iteration 1:
├─ loss.backward() → w.grad = [0.5, 0.3]
└─ optimizer.step() → w -= lr * grad ✅

Iteration 2 (WITHOUT zero_grad):
├─ loss.backward() → w.grad = [0.5, 0.3] + [0.4, 0.2] = [0.9, 0.5]
└─ Wrong! Using accumulated gradients ❌

Iteration 2 (WITH zero_grad):
├─ optimizer.zero_grad() → w.grad = [0, 0]
├─ loss.backward() → w.grad = [0.4, 0.2]
└─ optimizer.step() → Correct! ✅
```

---

### **STEP 5: Training Loop**

```python
@add_to_class(Trainer)
def fit_epoch(self):
    """Train for one epoch"""
    
    # TRAINING PHASE
    self.model.train()  # Set to training mode
    
    for batch in self.train_dataloader:
        # 1. Forward pass
        loss = self.model.training_step(
            self.prepare_batch(batch)
        )
        
        # 2. Backward pass
        self.optim.zero_grad()      # Clear old gradients
        loss.backward()             # Compute new gradients
        
        # 3. Update parameters
        self.optim.step()
        
        self.train_batch_idx += 1
    
    # VALIDATION PHASE
    if self.val_dataloader is None:
        return
        
    self.model.eval()  # Set to evaluation mode
    
    for batch in self.val_dataloader:
        with torch.no_grad():  # Don't compute gradients
            self.model.validation_step(
                self.prepare_batch(batch)
            )
        self.val_batch_idx += 1
```

**Complete Training:**

```python
# Create components
model = LinearRegressionScratch(num_inputs=2, lr=0.03)
data = SyntheticRegressionData(
    w=torch.tensor([2, -3.4]), 
    b=4.2
)
trainer = Trainer(max_epochs=3)

# Train!
trainer.fit(model, data)

# What happens:
# Epoch 1:
#   32 training batches → update parameters 32 times
#   32 validation batches → check performance
# Epoch 2:
#   32 training batches
#   32 validation batches
# Epoch 3:
#   32 training batches
#   32 validation batches
```

---

### 🎯 Training Process Visualization

```
EPOCH 1:
────────────────────────────────────────
Training:
  Batch 0:  loss = 10.5  →  update w, b
  Batch 1:  loss = 9.8   →  update w, b
  ...
  Batch 31: loss = 1.2   →  update w, b
  
Validation:
  Batch 0:  loss = 1.5   (just measure)
  Batch 1:  loss = 1.4   (just measure)
  ...
  Batch 31: loss = 1.1   (just measure)
  
  Avg train loss: 5.2
  Avg val loss:   1.3
────────────────────────────────────────

EPOCH 2:
  Avg train loss: 0.8  ← Getting better!
  Avg val loss:   0.6
────────────────────────────────────────

EPOCH 3:
  Avg train loss: 0.3  ← Even better!
  Avg val loss:   0.2
────────────────────────────────────────
```

---

### ✅ Checking Results

```python
# After training, compare with ground truth
w_true = torch.tensor([2.0, -3.4])
b_true = 4.2

w_learned = model.w.reshape(w_true.shape)
b_learned = model.b

print(f'True w: {w_true}')
print(f'Learned w: {w_learned}')
print(f'Error: {w_true - w_learned}')

print(f'True b: {b_true}')
print(f'Learned b: {b_learned}')
print(f'Error: {b_true - b_learned}')

# Output:
# True w: tensor([ 2.0000, -3.4000])
# Learned w: tensor([ 1.8837, -3.1980])
# Error: tensor([ 0.1163, -0.2020])  ← Close! ✅
```

---

## 🚀 Chapter 3.5: Concise Implementation with PyTorch

### 🎯 Using Built-in Components

**Comparison:**

| Component | From Scratch | Built-in PyTorch |
|-----------|-------------|------------------|
| **Model** | Manual w, b | `nn.Linear()` |
| **Loss** | Manual MSE | `nn.MSELoss()` |
| **Optimizer** | Custom SGD | `torch.optim.SGD()` |
| **Lines of code** | ~50 | ~15 |

---

### **Model Definition**

```python
class LinearRegression(Module):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        
        # PyTorch layer (automatically creates w and b!)
        self.net = nn.LazyLinear(1)
        
        # Initialize (same as before)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)
        
    def forward(self, X):
        return self.net(X)
```

**What is `LazyLinear`?**

```python
# Regular Linear:
net = nn.Linear(in_features=2, out_features=1)
# ❌ Must specify input dimension

# LazyLinear:
net = nn.LazyLinear(out_features=1)
# ✅ Input dimension inferred on first forward pass
# ✅ More flexible!

# First forward pass:
X = torch.randn(32, 2)  # 2 input features
output = net(X)          # Now net knows: in_features=2
```

---

### **Loss Function**

```python
@add_to_class(LinearRegression)
def loss(self, y_hat, y):
    fn = nn.MSELoss()
    return fn(y_hat, y)
```

**Built-in vs Manual:**

```python
# Manual:
loss = ((y_hat - y) ** 2 / 2).mean()

# Built-in (no /2 factor):
loss = nn.MSELoss()(y_hat, y)

# Mathematically equivalent for optimization!
# (constant factors don't affect argmin)
```

---

### **Optimizer**

```python
@add_to_class(LinearRegression)
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), self.lr)
```

**`self.parameters()`:**

```python
# Automatically finds ALL learnable parameters!

for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Output:
# net.weight: torch.Size([1, 2])
# net.bias: torch.Size([1])

# No need to manually track [self.w, self.b] ✅
```

---

### **Training (Same as Before!)**

```python
model = LinearRegression(lr=0.03)
data = SyntheticRegressionData(
    w=torch.tensor([2, -3.4]), 
    b=4.2
)
trainer = Trainer(max_epochs=3)

trainer.fit(model, data)

# Results:
# Error in w: tensor([ 0.0031, -0.0099])  ← Even better!
# Error in b: tensor([0.0127])
```

---

### 🔍 Accessing Parameters

```python
@add_to_class(LinearRegression)
def get_w_b(self):
    return (self.net.weight.data, self.net.bias.data)

w, b = model.get_w_b()

# w.shape: (1, 2) - Note: transposed!
# In nn.Linear: output = X @ w^T + b
# So weight matrix is (out_features, in_features)
```

---

## 🎓 Chapter 3.6: Generalization

### 🎯 The Core Problem

**Two Students Example:**

```
ELLIE (Memorization):
├─ Strategy: Memorize all past exam answers
├─ Past exams: 100% ✅
└─ New exam: 0% ❌ (never seen these questions!)

IRENE (Pattern Learning):
├─ Strategy: Understand underlying patterns
├─ Past exams: 90% 
└─ New exam: 90% ✅ (patterns still apply!)

We want to be like IRENE!
```

---

### 📊 Key Definitions

**1. Training Error (Empirical Error):**

```python
# Loss on data we TRAINED on
train_loss = (1/n_train) Σ loss(model(X_train[i]), y_train[i])

Can measure this! ✅
```

**2. Generalization Error (True Error):**

```python
# Expected loss on ALL possible data
gen_error = E[loss(model(X), y)]  # X,y ~ true distribution

Cannot measure this exactly! ❌
(would need infinite data)
```

**3. Validation Error:**

```python
# Loss on data we held out for testing
val_loss = (1/n_val) Σ loss(model(X_val[i]), y_val[i])

Can measure this! ✅
Used to ESTIMATE generalization error
```

---

### 🎭 Underfitting vs Overfitting

**Underfitting:**

```
Training loss:   HIGH ❌
Validation loss: HIGH ❌
Gap:             SMALL

Problem: Model too simple
Solution: 
├─ More complex model
├─ More features
└─ Train longer
```

**Good Fit:**

```
Training loss:   LOW ✅
Validation loss: LOW ✅
Gap:             SMALL ✅

Sweet spot! This is what we want.
```

**Overfitting:**

```
Training loss:   VERY LOW
Validation loss: HIGH ❌
Gap:             LARGE ❌

Problem: Model memorizing training data
Solution:
├─ More training data
├─ Regularization
├─ Simpler model
└─ Early stopping
```

---

### 📈 Polynomial Example

```python
# Fit polynomials of different degrees

Degree 1: y = w₀ + w₁x
├─ 2 parameters
├─ Training error: 5.0
├─ Validation error: 5.2
└─ Status: Underfit (too simple)

Degree 3: y = w₀ + w₁x + w₂x² + w₃x³
├─ 4 parameters
├─ Training error: 0.5
├─ Validation error: 0.6
└─ Status: Good fit ✅

Degree 10: y = w₀ + w₁x + ... + w₁₀x¹⁰
├─ 11 parameters
├─ Training error: 0.001
├─ Validation error: 100.0
└─ Status: Overfit (memorized training data)
```

**Visual:**

```
Complexity →
Low        Medium        High
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Underfit  |  Good  |  Overfit
  
High error  | Sweet  | High val error
both sets   | spot   | (memorization)
```

---

### 📏 Model Selection

**The Three-Way Split:**

```
Full Dataset (10,000 examples)
│
├─ Training Set (8,000) - 80%
│  └─ FIT model parameters
│
├─ Validation Set (1,000) - 10%
│  └─ SELECT hyperparameters
│
└─ Test Set (1,000) - 10%
   └─ FINAL evaluation (touch ONCE!)
```

**Critical Rules:**

```
❌ NEVER use test set for:
   ├─ Choosing model architecture
   ├─ Tuning hyperparameters
   ├─ Deciding when to stop training
   └─ ANY decision making!

✅ ONLY use test set for:
   └─ Final performance report
```

**Why?**

```
If you tune on test set:
├─ Model sees test data indirectly
├─ Test error becomes optimistic
├─ You're overfitting to test set!
└─ Results won't generalize to real world ❌
```

---

### 🔄 K-Fold Cross-Validation

**When to use:** Not enough data for separate validation set

```
5-Fold Cross-Validation:

Dataset: [1][2][3][4][5]

Fold 1: [T][T][T][T][V]  Train on 1-4, validate on 5
Fold 2: [T][T][T][V][T]  Train on 1-3,5, validate on 4
Fold 3: [T][T][V][T][T]  Train on 1-2,4-5, validate on 3
Fold 4: [T][V][T][T][T]  Train on 1,3-5, validate on 2
Fold 5: [V][T][T][T][T]  Train on 2-5, validate on 1

Final validation error = average of 5 folds
```

**Trade-offs:**

```
Pros:
├─ ✅ Uses all data for both training and validation
├─ ✅ More reliable estimate with limited data
└─ ✅ Reduces variance in performance estimate

Cons:
├─ ❌ K times more expensive (train K models)
├─ ❌ Takes K times longer
└─ ❌ More complex implementation
```

---

## ⚖️ Chapter 3.7: Weight Decay (L2 Regularization)

### 🎯 The Problem

```python
# High-dimensional, low-sample scenario
num_features = 200
num_training = 20  # Only 20 examples!

Problem:
├─ More parameters (200) than data (20)
├─ Model can perfectly fit training data
├─ But performance on new data is terrible
└─ Extreme overfitting!
```

---

### 💡 The Solution: Weight Decay

**Intuition:** Add penalty for large weights

```
Modified Loss Function:
L(w, b) = MSE(w, b) + (λ/2)||w||²
          ︸━━━━━━━━━︸   ︸━━━━━━━━︸
          Original       Penalty
          loss          term
```

**Components:**

| Symbol | Name | Meaning |
|--------|------|---------|
| **λ (lambda)** | Regularization strength | How much to penalize large weights |
| **‖w‖²** | L2 norm squared | w₁² + w₂² + ... + wₐ² |
| **/2** | Constant | Makes derivative cleaner |

---

### 🔢 Why L2 Norm?

**L2 vs L1:**

```python
# L2 Regularization (Ridge):
penalty = λ * (w₁² + w₂² + ... + wₐ²)
Effect: All weights shrink proportionally

# L1 Regularization (Lasso):
penalty = λ * (|w₁| + |w₂| + ... + |wₐ|)
Effect: Many weights become exactly 0 (sparse)
```

**L2 Benefits:**

```
✅ Smooth (differentiable everywhere)
✅ Unique solution
✅ Distributes weight evenly across features
✅ More robust to noise
✅ Easier to optimize
```

---

### 🔄 Updated Training

**Gradient with Weight Decay:**

```
Without weight decay:
∂L/∂w = (1/|B|) Σ x^(i)(ŷ^(i) - y^(i))

With weight decay:
∂L/∂w = (1/|B|) Σ x^(i)(ŷ^(i) - y^(i)) + λw
        ︸━━━━━━━━━━━━━━━━━━━━━━━━━━━━︸   ︸━︸
        Original gradient               Extra term
```

**Update Rule:**

```python
# Standard SGD:
w ← w - η∇L

# With weight decay:
w ← w - η(∇L + λw)
  = w - η∇L - ηλw
  = (1 - ηλ)w - η∇L
    ︸━━━━━━━︸
    Decay factor
```

**Why "Weight Decay"?**

```
Factor (1 - ηλ) < 1 always shrinks weights:

Example: η = 0.01, λ = 0.1
├─ (1 - ηλ) = (1 - 0.001) = 0.999
└─ Each step: w ← 0.999w - η∇L
              Shrink by 0.1% each iteration
```

---

### 💻 Implementation

**From Scratch:**

```python
def l2_penalty(w):
    return (w ** 2).sum() / 2

class WeightDecayScratch(LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()
        
    def loss(self, y_hat, y):
        # Original loss + penalty
        return (super().loss(y_hat, y) + 
                self.lambd * l2_penalty(self.w))
```

**Using PyTorch:**

```python
class WeightDecay(LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.wd = wd
        
    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 
             'weight_decay': self.wd},  # Apply to weights
            {'params': self.net.bias}    # NOT to bias
        ], lr=self.lr)
```

---

### 📊 Effect of Lambda

**λ = 0 (No Regularization):**

```
Training loss:   0.01  ← Very low
Validation loss: 5.00  ← High!
Gap:             4.99  ← Overfitting!
||w||²:          10.5  ← Large weights
```

**λ = 3 (With Regularization):**

```
Training loss:   0.50  ← Higher
Validation loss: 0.60  ← Much lower!
Gap:             0.10  ← Better generalization ✅
||w||²:          0.15  ← Small weights ✅
```

---

### 🎛️ Choosing Lambda

```
Lambda = 0:
├─ No regularization
└─ Risk of overfitting

Lambda small (0.001):
├─ Weak regularization
└─ Good if model isn't overfitting

Lambda medium (0.1 - 1):
├─ Moderate regularization
└─ Good default choice

Lambda large (10+):
├─ Strong regularization
├─ Prevents overfitting
└─ Risk of underfitting

Rule: Tune on validation set!
```

---

## 📝 FINAL SUMMARY - Everything Together

### The Complete Linear Regression Pipeline

```
1. DATA PREPARATION
   ├─ Generate/load data
   ├─ Split: train/val/test
   └─ Create data loaders

2. MODEL DEFINITION
   ├─ Initialize parameters (w, b)
   ├─ Define forward pass
   └─ Define loss function

3. OPTIMIZATION
   ├─ Choose optimizer (SGD)
   ├─ Set learning rate
   └─ Add regularization (optional)

4. TRAINING LOOP
   For each epoch:
      ├─ For each batch:
      │  ├─ Forward pass
      │  ├─ Compute loss
      │  ├─ Backward pass (gradients)
      │  └─ Update parameters
      └─ Validate on validation set

5. EVALUATION
   ├─ Check on validation set
   ├─ Tune hyperparameters
   └─ Final test on test set (once!)
```

---

### Key Formulas

```
Model:     ŷ = w^T x + b

Loss:      L = (1/n)Σ(ŷ - y)² + (λ/2)||w||²

Update:    w ← w - η(∇L + λw)
           b ← b - η∇L
```

---

### Important Concepts

| Concept | Simple Explanation |
|---------|-------------------|
| **Overfitting** | Model memorizes training data, fails on new data |
| **Underfitting** | Model too simple, fails on everything |
| **Regularization** | Add penalty to prevent overfitting |
| **Weight Decay** | Shrink weights toward zero |
| **Learning Rate** | How big each update step is |
| **Batch Size** | How many examples to process together |
| **Validation** | Separate data to check generalization |

---

### Critical Rules ⚠️

```
✅ DO:
├─ Always split train/val/test
├─ Shuffle training data each epoch
├─ Monitor validation loss
├─ Use regularization if overfitting
├─ Normalize/standardize features
└─ Set random seeds for reproducibility

❌ DON'T:
├─ Train on test set
├─ Tune hyperparameters on test set
├─ Forget to zero gradients
├─ Use learning rate too large
├─ Ignore validation error
└─ Skip data shuffling
 