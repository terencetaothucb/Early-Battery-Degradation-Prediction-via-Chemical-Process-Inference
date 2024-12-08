# Degradation-pattern-informed Non-destructive Verification Enables Sustainable Manufacturing and Recycling of Battery Prototypes
Battery research and development (R&D) faces fierce technology path competition for future energy use scenarios, generating endless battery prototypes. Performance mismatches between theoretical designs and as-manufactured battery prototypes remain a significant bottleneck, due to time-consuming, costly, and often destructive verification. Although machine learning has been well-documented in battery R&D, it still fails to interpret electrochemistry degradation patterns and, is hardly supportive of prototype verification. Here, distinct from purely statistical modeling, we explore thermodynamic and kinetic degradation patterns decoupling from initial manufacturing differences among normally identical batteries, and inferring multidimensional chemical process signals for the entire lifetime performance verification at minimal physical sensory data retrieval cost. This physics-informed method shows great promise in integrating material science priors into machine learning models, demonstrating sustainable and economic superiorities by rationally managing massive prototype batteries based on statistical degradation status.

# 1. Setup
## 1.1 Enviroments
* Python (Jupyter notebook) 
## 1.2 Python requirements
* python=3.11.5
* numpy=1.26.4
* tensorflow=2.15.0
* keras=2.15.0
* matplotlib=3.9.0
* scipy=1.13.1
* scikit-learn=1.3.1
* pandas=2.2.2

# 2. Datasets

* Raw and processed datasets have been deposited in TBSI-Sunwoda-Battery-Dataset, which can be accessed at [TBSI-Sunwoda-Battery-Dataset](https://github.com/terencetaothucb/TBSI-Sunwoda-Battery-Dataset).
* Please refer to __Data generation__ part of the paper for detailed dataset explanation.


# 3. Experiment
## 3.1 Overview
The entire experiment consists of three steps as well as three models: 
* ChemicalProcessModel
* DomainAdaptModel
* DegradationTrajectoryModel

First, we model multi-dimensional chemical processes using early cycle and guiding sample data; second, we adapt these predictions to specific temperatures; and third, we use adapted chemical processes to avoid the need for physical measures in later cycles. The extent of early data used is tailored to meet the desired accuracy, assessed by mean absolute percentage error for consistent cross-stage comparisons.

## 3.2 Chemical process prediction model considering initial manufacturing variability (ChemicalProcessModel)
The **ChemicalProcessModel** predicts chemical process variations by using input voltage matrix $U$. Given a feature matrix $\mathbf{F} \in \mathbb{R}^{(C \times m) \times N}$ (see paper for more details on the featurization taxonomy), where $N$ is the number of features, the model learns a composition of $L$ intermediate layers of a neural network:

$$
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\ \quad\quad\quad\quad\\quad\quad\quad 
\hat{\mathbf{F}} = f_\theta(U) = \left(f_\sigma^{(L)} \left(f_\theta^{(L)} \circ \cdots \circ f_\sigma^{(1)} \left(f_\theta^{(1)}\right)\right)\right)(U) quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\\quad\quad\quad\quad\ \text{(1)}
$$


where $L = 3$ in this work. $\hat{\mathbf{F}}$ is the output feature matrix, i.e., $\hat{\mathbf{F}} \in \mathbb{R}^{(C \times m) \times N}$, $\theta = \{\theta^{(1)}, \theta^{(2)}, \theta^{(3)}\}$ is the collection of network parameters for each layer, $U \in \mathbb{R}^{(C \times m) \times 10}$ is the broadcasted input voltage matrix, and $f_\theta(U)$ is a neural network predictor. All layers are fully connected. The activation function used is Leaky ReLU (leaky rectified linear unit), denoted as $f_\sigma$. 

Here is the implementation:
```python
 class ChemicalProcessModel(nn.Module):
    def __init__(self):
        super(ChemicalProcessModel, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 42)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
```
In each selected temperature, we split the data into 75% and 25% for training and testing, respectively. We train the chemical process prediction model using the Adam optimizer, with 30 epochs and a learning rate of $10^{-4}$. The loss function of the chemical process prediction model is implementated as:
```python
criterion = nn.MSELoss().to(device)
def loss_fn(outputs, labels, model, l1_strength):
        loss = criterion(outputs, labels)
        l1_regularization = add_l1_regularization(model, l1_strength)
        loss += l1_regularization
        return loss
```
See the __Methods__ section of the paper for more details.
### Settings
* In the code of Chemical Process Model, there are options to change parameters at the very beginning. The following parameters can be modified to adjust the training process.
```python

# List of learning rates to be used for training.
learning_rates = [3e-4, 1e-4]  
lr_losses = {}  
best_lr = None  
best_loss = float('inf')  
best_model_state = None  

# Total number of training epochs.
train_epochs = 100  
# Read raw data from csv file.
raw_data = pd.read_csv("./raw_data_0920.csv")  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

# Create training dataset and its data loader with batch size 1 and shuffle enabled.
train_dataset = BattDataset(raw_data, train=True)  
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  
# Create validation dataset and its data loader
valid_dataset = BattDataset(raw_data, train=True)  
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)  
# Create test dataset and its data loader
test_dataset = BattDataset(raw_data, train=False)  
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  

# Define MSE loss function
criterion = nn.MSELoss().to(device) 
```
### Run
* After changing the experiment settings, __run `ChemicalProcessModel.py` directly for training and testing.__  


## 3.3 Multi-domain adaptation
After predicting the chemical process of source domain, it is necessary to transfer it to the target domain. This section will specifically explain how to perform multi-domain adaptation by utilizing our proposed physics-informed transferability metric.

### 3.3.1 Physics-informed transferability metric
It is time-consuming and cost-intensive to enumerate continuous temperature verifications, we therefore formulate a knowledge transfer from existing measured data (source domain) to arbitrary intermediate temperatures (target domain). The transfer is compatible with multi- and uni-source domain adaptation cases for tailored verification purposes. Here we use a multi-source domain adaptation to elucidate the core idea. For instance, we take 25, 55℃ as source domains and 35, 45℃ as target domains. 

We propose a physics-informed transferability metric to quantitatively evaluate the effort in the knowledge transfer. The proposed transferability metric integrates prior physics knowledge inspired by the Arrhenius equation:

$$
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\ \quad\quad\quad\quad\\quad\quad\quad\quad\\quad\quad\quad\quad\  \quad\quad\quad  \hspace{0.55em} 
r = A e^{-\frac{E_a}{k_B T}}
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\ \quad\quad\quad\quad\\quad\quad\quad\quad\\quad\quad\quad\quad\  \quad\quad\quad (2)
$$

where: $A$ is a constant, $r$ is the aging rate of the battery, $E_a$ is the activation energy, $k_B$ is the Boltzmann constant, and $T$ is the Kelvin temperature.

The Arrhenius equation provides us with important information that the aging rate of batteries is directly related to the temperature. Therefore, the Arrhenius equation offers valuable insights into translating the aging rate between different temperatures. We observe the domain-invariant representation of the aging rate ratio, consequently, the proposed Arrhenius equation-based transferability metric $AT_{score}$ is defined as:

$$
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\ \quad\quad\quad\quad\\quad\quad\quad\quad\\quad\quad\quad\hspace{0.7em}
AT_{score}  = \frac{r_{target}}{r_{target}} = \frac{e^{-\frac{E_a^s}{k_B T_s}}}{e^{-\frac{E_a^t}{k_B T_t}}}
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\ \quad\quad\quad\quad\\quad\quad\quad\quad\\quad\quad\quad\quad(3)
$$

where $E_a^s$ is the activation energy of the source domain, $E_a^t$ is the activation energy of the target domain, $T_s$ and $T_t$ are the Kelvin temperatures of the source domain and the target domain, respectively.

The closer the $AT_{score}$ is to 1, the more similar the source domain and target domain are, so the better the knowledge transfer is expected. Since the dominating aging mechanism is unknown (characterized by $E_a$) as a posterior, we alternatively determine the aging rate by calculating the first derivative concerning the variations on the predicted chemical process curve:

$$
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\ \quad\quad\quad\quad\\quad\quad\quad\quad\\quad\quad\quad\quad\  \quad\quad\quad\quad
r = \frac{d\hat{F}}{dC}
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\ \quad\quad\quad\quad\\quad\quad\quad\quad\\quad\quad\quad\quad\  \quad\quad\quad\quad\\ (4)
$$

where $\hat{F}$ is the predicted chemical process feature matrix. We linearize the calculation in adjacent cycles by sampling the point pairs on the predicted chemical process:

$$
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\ \quad\quad\quad\quad\\quad\quad\quad\quad\\quad\quad\quad\quad
r = \frac{\sum_{i=0}^n \left(F_{\text{end}+i} - F_{\text{start}+i}\right)}{n \cdot (\text{end} - \text{start})}
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\ \quad\quad\quad\quad\\quad\quad\quad\quad\\quad\quad\quad\hspace{0.35em}(5)
$$

where $n$ is the number of point pairs, $start$ and $end$ are the cycle indices where we start and end the sampling, respectively, $F_{end+i}$ and $F_{start+i}$ are the feature values for the $(start+i)th$ and $(end+i)th$ cycles, respectively.

This calculation mitigates the noise-induced errors, resulting in a more robust aging rate computation. For domains where the aging mechanism is already known (different domains share the same \( E_a \)), the $AT_{score}$ can be expressed in the following form:

$$
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\ \quad\quad\quad\quad\\quad\quad\quad\quad\\quad\quad\quad\hspace{0.5em}
\text{AT}_{\text{score}}^{source \to target} = e^{\frac{E_a}{k_B} \left( \frac{1}{T_t} - \frac{1}{T_s} \right)}
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\ \quad\quad\quad\quad\\quad\quad\quad\quad\\quad\quad\quad\(6)
$$

where $\frac{E_a}{k_B}$ is a constant value. This formula ensures that, in cases where the aging mechanism is known, we can calculate transferability between different domains using only the temperatures of the source and target domains. This allows the model for continuous temperature generalization without any target data.

### 3.3.2 Multi-domain adaptation using the physics-informed transferability metric

The multi-source transfer based on $AT_{\text{score}}$ includes the following three steps. 

* First, we calculate aging rates $r$ for all target and source domains by using early-stage data, i.e., we set $\text{start} = 100$, $\text{end} = 200$, $n = 50$. After calculating aging rates for all features or aging curves, we obtain a target domain aging rate vector $r_{\text{target } 1\times N}$ and a source domain aging rate matrix $r_{\text{source } K\times N}$, where $K$ and $N$ are the number of source domains and the number of features, respectively.
```python
def gradient(feature_list,start,end,sample_size):
    mean_start = 0
    mean_end = 0
    for i in range(sample_size):
      mean_start = mean_start + feature_list[start+i]
    for i in range(sample_size):
      mean_end = mean_end + feature_list[end+i]
    mean_start = mean_start/sample_size
    mean_end = mean_end/sample_size
    grad = mean_end - mean_start
    rang = end-start
    return np.log(abs(grad/rang))

start45 = early_cycle_start#100
end45 = early_cycle_end#200
g25.append(gradient(pre25[i][:],start,end,sample_size))
g55.append(gradient(pre55[i][:],start,end,sample_size))
g35.append(gradient(real35[i][:],start,end,sample_size))
g45.append(gradient(real45[i][:],start45,end45,sample_size))
```

* Second, we calculate the transferability metric $AT_{\text{score}}$ and weight vector $W_{1\times K} = \{W_i\}$.

```python
#ATscore calculation
at25.append(gradient(real45[i][:],start45,end45,sample_size)/gradient(pre25[i][:],start,end,sample_size))
at55.append(gradient(real45[i][:],start45,end45,sample_size)/gradient(pre55[i][:],start,end,sample_size))

#weight calculation
w25 = abs(step25-1)+abs(step55-1)
w25 = abs(step55-1)/w25
w_at_25.append(w25)
```
* Third, we predict the late stage (cycles after 200) aging rate of the target domain ($r_{\text{target}}$) (shown in 4.2.3). Note that $AT_{\text{score}}^{\text{source } i \to \text{target}}$ and $W_i$ are obtained by both target and source domain early-stage data, which are used to measure the transferability from source domain to target domain based on their aging rate similarity. $r_{\text{source } i}$ is obtained from all accessible data in the source domain, consistent with our definition of the early-stage estimate problem.

Using the physics-informed transferability metric, we assign a weight vector $W_{1\times K} = \{W_i\}$ (where $K$ is the number of source domains, $W_i$ is the ensemble weight for the $i$-th source domain) to source domains to quantify the contributions when predicting the chemical process of the target domain. The $W_i$ is defined as:

$$
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\ \quad\quad\quad\quad
W_i = \left( \left| AT_{\text{score}}^{\text{source } i \to \text{target}} - 1 \right| \cdot \left( \sum_{j=1}^K \frac{1}{\left| AT_{\text{score}}^{\text{source } j \to \text{target}} - 1 \right|} \right)^{-1} \right)
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\ \quad\quad\quad\quad \hspace{1.1em}(7)
$$

where, $AT_{\text{score}}^{\text{source } i \to \text{target}}$ is the $AT_{\text{score}}$ from the $i$-th source domain to the target domain. This mechanism ensures the source domain with better transferability has a higher weight, effectively quantifying the contribution of each source domain to the prediction of the target domain. From Equation (3) and Equation (7), we can obtain the aging rate of the target domain:

$$
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\ \quad\quad\quad\quad\\quad\quad\quad\quad\quad
r_{\text{target}} = \sum_{i=1}^K W_i \cdot AT_{\text{score}}^{\text{source } i \to \text{target}} \cdot r_{\text{source } i}
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\ \quad\quad\quad\quad\\quad\quad\quad\quad\\quad\ \hspace{0.35em}(8)
$$



See the __Methods__ section of the paper for more details.


### 3.3.3 Chain of degradation

Battery chemical process degradation is continuous, which we call the "Chain of Degradation". We have predicted the $r_{\text{target}}$ aging rates of each feature in the target domain, which can be further used to predict the chemical process. Therefore, when using aging rates $r_{\text{target}}$ to calculate each target feature vector $F_{\text{(C×m)×1}}$ in the feature matrix $F_{\text{(C×m)×N}}$, the $i$-th cycle target feature vector $F_{\text{target}}^i$ should be based on $F_{\text{target}}^{i-1}$ and $r^{i-1}$:

$$
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\ \quad\quad\quad\quad\quad\quad\quad
F_{\text{target}}^i = F_{\text{target}}^{i-1} + \sum_{j=i}^K W_j \cdot A_{\text{score}}^{\text{source } j \to \text{target}} \cdot r_{\text{source } j}^{i-1}
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\ \quad\quad\quad\quad\quad\quad\quad\quad\(9)
$$

where $F_{\text{target}}^i$ is the feature value of the target domain in the $i$-th cycle, $r_{\text{source } j}^{i-1}$ is the aging rate of source domain $j$ at the $(i-1)$-th cycle.

We concatenate the $N$ feature vectors $F_{\text{(C×m)×1}}$ to get the feature matrix $F_{\text{(C×m)×N}}$. Since our chemical process prediction for each step is based on the result of the previous step, we can track the accumulation of degradation in the aging process and thus it is robust against noise.
Here is the implementation:
```python

      #################################################################################################
      # AT method:
      # Input: at25_to_45, at55_to_45, w_at_25, pre25, pre55 (these are from this round), last45, last55, last25 (these are from the previous round).
      # Output: pre45, stored in feature_1 (replaces the original model1 output).
      # Renew: last45 = pre45, last25 = pre25, last55 = pre55
      #################################################################################################

      if(batch>early_cycle_end):
        if(batch==early_cycle_end):
            pred_feature.append(last45)
        for i in range(42):

          step1 = w_at_25[i]*(pre25[i][0]-last25[i][0])*at25[i]
          step2 = (1-w_at_25[i])*(pre55[i][0]-last55[i][0])*at55[i]
          feature_45[i] =  last45[i] + w_at_25[i]*(pre25[i][0]-last25[i][0])*at25[i] + (1-w_at_25[i])*(pre55[i][0]-last55[i][0])*at55[i]
          last45[i] = feature_1[i]
          last25[i][0] = pre25[i][0]
          last55[i][0] = pre55[i][0]
        pred_feature.append(feature_45)

```

### Settings
* In the code of DomainAdaptModel, there are options to change parameters at the very beginning. The following parameters can be modified to adjust the training process.
```python
#load raw data from csv
raw_data = pd.read_csv("./raw_data_0920.csv")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load chemical process model under different temperatures
model_1_25 = MyNetwork1()
model_path_1 = "./0921_best_model_1_T25.pt"
model_1_25.load_state_dict(torch.load(model_path_1, map_location=torch.device('cpu')))

model_1_55 = MyNetwork1()
model_path_1 = "./0921_best_model_1_T55.pt"
model_1_55.load_state_dict(torch.load(model_path_1, map_location=torch.device('cpu')))

# load the previous chemical process model for compare。
model_1 = MyNetwork3()
model_path_1 = "./best_model_1.pt"
model_1.load_state_dict(torch.load(model_path_1, map_location=torch.device('cpu')))

# load different types of degradation trajectory model 
model_2 = MyNetwork2()
model_2_old = MyNetwork2()
model_path_2_old = "./best_model_2_previous.pt"
model_path_2 = "./best_model_2_now.pt"
model_2.load_state_dict(torch.load(model_path_2, map_location=torch.device('cpu')))
model_2_old.load_state_dict(torch.load(model_path_2_old, map_location=torch.device('cpu')))

# the battery_dict is used to filter the specific battery data from the raw data
battery_dict = {
    "T25": [1.362,1.336,1.3,1.351,1.318,1.329,1.286,1.442,1.478],
    "T35": [-0.0816,-0.121,-0.157,-0.092,-0.15,-0.128,-0.244],
    "T45": [-0.854,-0.941,-0.912,-0.937,-0.988,-0.865,-1.006],
    "T55": [-1.101,-1.304,-1.242,-1.220,-1.144,-1.224,-1.090],
}
# set the target temperature: T25/T35/T45/T55
test_tmp = "T35"
# set the early cycle numbers, for now, it means using 0-200 cycles of target domain to train the DomainAdaptModel
early_cycle_start = 0
early_cycle_end = 200
sample_size= 20

# the lifetime of battery under different temperature is not equal, which causes the test size is different
test_size = 1295
if test_tmp == "T45":
  test_size = 1095
if test_tmp == "T55":
  test_size = 895
```
### Run
* After changing the experiment settings, __run `DomainAdaptModel.py` directly for training and testing.__

## 3.4 Battery degradation trajectory model
We have successfully predicted the battery chemical process. It is assumed that the chemical process of the battery deterministically affects the aging process, we therefore use the predicted chemical process to predict the battery degradation curve. The battery degradation trajectory model learns a composition of $L$ intermediate mappings:

$$
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\ \quad\quad\quad\quad\\quad\quad\quad 
\hat{\mathbf{D}} = f_\theta(\hat{\mathbf{F}}) = \left(f_\sigma^{(L)} \left(f_\theta^{(L)} \circ \cdots \circ f_\sigma^{(1)} \left(f_\theta^{(1)}\right)\right)\right)(\hat{\mathbf{F}})
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\ \quad\quad\quad\quad\\quad\quad\quad (10)
$$

where $L = 3$ in this work. $\hat{\mathbf{D}}$ is predicted battery degradation trajectories, $\theta = \{\theta^{(1)}, \theta^{(2)}, \theta^{(3)}\}$ is the collection of network parameters for each layer, $\hat{\mathbf{F}}$ is the predicted battery chemical process feature matrix, and $f_\theta(\hat{\mathbf{F}})$ is a neural network predictor. All layers are fully connected. The activation function used is Leaky ReLU (leaky rectified linear unit), denoted as $f_\sigma$. 

Here is the implementation of DegradationTrajectoryModel:
```python
   class DegradationTrajectoryModel(nn.Module):
    def __init__(self):
        super(DegradationTrajectoryModel, self).__init__()
        self.fc1 = nn.Linear(53, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
```
### Settings
* In the code of Degradation Trajectory Modell, there are options to change parameters at the very beginning. The following parameters can be modified to adjust the training process.
```python
# a set of learning rate
learning_rates = [1e-3, 2e-3, 3e-3]
lr_losses = {}
# The information of the best model
best_lr = None
best_loss = float('inf')
best_model_state = None

train_epochs = 100
raw_data = pd.read_csv("./raw_data_0920.csv")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create training dataset and its data loader with batch size 1 and shuffle enabled.
train_dataset = BattDataset(raw_data, train=True)  
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  
# Create validation dataset and its data loader with batch size 1 and no shuffle.
valid_dataset = BattDataset(raw_data, train=True)  
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)  
# Create test dataset and its data loader with batch size 1 and no shuffle.
test_dataset = BattDataset(raw_data, train=False)  
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  
# Define MSE loss function and move it to the device.
criterion = nn.MSELoss().to(device) 
```
### Run
* After changing the experiment settings, __run `DegradationTrajectoryModel.py` directly for training and testing.__  
# 4. Access
Access the raw data and processed features [here]((https://github.com/terencetaothucb/TBSI-Sunwoda-Battery-Dataset)) under the [MIT licence](https://github.com/terencetaothucb/Pulse-Voltage-Response-Generation/blob/main/LICENSE). Correspondence to [Terence (Shengyu) Tao](terencetaotbsi@gmail.com) and CC Prof. [Xuan Zhang](xuanzhang@sz.tsinghua.edu.cn) and [Guangmin Zhou](guangminzhou@sz.tsinghua.edu.cn) when you use, or have any inquiries.
# 5. Acknowledgements
[Terence (Shengyu) Tao](mailto:terencetaotbsi@gmail.com) and [Zixi Zhao](zhaozx23@mails.tsinghua.edu.cn)  at Tsinghua Berkeley Shenzhen Institute designed the model and algorithms, developed and tested the experiments, uploaded the model and experimental code, revised the testing experiment plan, and wrote this instruction document based on supplementary materials.  

