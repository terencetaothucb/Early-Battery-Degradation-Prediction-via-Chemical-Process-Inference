import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import *
from BattDataLoader import BattDataset

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

save_filename = "earlycycle50_one_source_"+test_tmp+"_prediction.csv"
GT_filname = "GT_"+test_tmp+"_lifetime.csv"
GTFEATURE_filename = "GT_"+test_tmp+"_feature.csv"
feature_filename = "earlycycle50_one_source_"+test_tmp+"feature"+"_prediction.csv"

sum_mape = 0
start_time = time.time()

for test_u in battery_dict[test_tmp]:
  # load the dataset
  train_dataset = BattDataset(raw_data,train=True)
  train_loader = DataLoader(train_dataset, batch_size=1, shuffle = True)
  test_dataset = BattDataset(raw_data,train=False)
  test_loader = DataLoader(test_dataset, batch_size=1, shuffle = False)

  # save the temporary data
  test_label = []
  test_predict = []
  y_filter = []
  pred = []
  feature_25_pre = []
  feature_55_pre = []
  feature_45 = []
  w1 = []
  w2 = []
  acc25 = 0
  acc55 = 0
  acc45 = 0
    
  for batch, (domain, domain11,feature, y_tensor, y_plot) in enumerate(test_loader):

      # voltage map between different temperatures, the map matrix is calculated by the average of voltage.
      # for example: for T25 and T45, if the average U1 of T25 is 1.324 and the average U1 of T45 is -0.91, then the map number will be 1.324/-0.91 = -1.455
      if test_tmp == "T45":
        domain[0][0] = domain[0][0].float()*-1.45521186
        domain[0][1] = domain[0][1].float()*-1.50569253
        domain[0][2] = domain[0][2].float()*-1.53181673
        domain[0][3] = domain[0][3].float()*-1.55211713
        domain[0][4] = domain[0][4].float()*-1.5616391
        domain[0][5] = domain[0][5].float()*-1.55604219
        domain[0][6] = domain[0][6].float()*-1.54135137
        domain[0][7] = domain[0][7].float()*-1.53640339
        domain[0][8] = domain[0][8].float()*-1.78543008
        domain[0][9] = domain[0][9].float()*1.18198362
      if test_tmp == "T35":
        domain[0][0] = domain[0][0].float()*-12.9708305
        domain[0][1] = domain[0][1].float()*-11.0788281
        domain[0][2] = domain[0][2].float()*-10.3631271
        domain[0][3] = domain[0][3].float()*-8.75590688
        domain[0][4] = domain[0][4].float()*-7.93640255
        domain[0][5] = domain[0][5].float()*-8.50438212
        domain[0][6] = domain[0][6].float()*-9.9675704
        domain[0][7] = domain[0][7].float()*-14.8879788
        domain[0][8] = domain[0][8].float()*-5.74175461
        domain[0][9] = domain[0][9].float()*1
      if test_tmp == "T55":
        domain[0][0] = domain[0][0].float()/-0.87950659
        domain[0][1] = domain[0][1].float()/-0.87824661
        domain[0][2] = domain[0][2].float()/-0.87701338
        domain[0][3] = domain[0][3].float()/-0.85803634
        domain[0][4] = domain[0][4].float()/-0.84296905
        domain[0][5] = domain[0][5].float()/-0.85084735
        domain[0][6] = domain[0][6].float()/-0.87543736
        domain[0][7] = domain[0][7].float()/-0.93763034
        domain[0][8] = domain[0][8].float()/-0.84951105
        domain[0][9] = domain[0][9].float()/0.69207082

      feature_25 = model_1_25(domain.float())
      domain[0][0] = domain[0][0].float()*-0.87950659
      domain[0][1] = domain[0][1].float()*-0.87824661
      domain[0][2] = domain[0][2].float()*-0.87701338
      domain[0][3] = domain[0][3].float()*-0.85803634
      domain[0][4] = domain[0][4].float()*-0.84296905
      domain[0][5] = domain[0][5].float()*-0.85084735
      domain[0][6] = domain[0][6].float()*-0.87543736
      domain[0][7] = domain[0][7].float()*-0.93763034
      domain[0][8] = domain[0][8].float()*-0.84951105
      domain[0][9] = domain[0][9].float()*0.69207082
      feature_55 = model_1_55(domain.float())

      if (batch==1):
        weight = []
        bias = []

        i = 0
        while(i<42):

          step25 = feature_25[0][i].item()
          step55 = feature_55[0][i].item()
          step45 = feature[0][i].item()
          bias.append([step45-step25,step45-step55])
          weight.append([abs(step55-step45)/(abs(step25-step55)),abs(step25-step45)/(abs(step25-step55))])
          i =i+1
      if(batch>1):
        i = 0
        while(i<42):
          step25 = feature_25[0][i].item()
          step55 = feature_55[0][i].item()
          step45 = feature[0][i].item()

          weight[i][0] = weight[i][0] + abs(step55-step45)/(abs(step25-step55))
          weight[i][1] = weight[i][1] + abs(step25-step45)/(abs(step25-step55))
          i = i+1

      feature_25_pre.append(feature_25.tolist())
      feature_55_pre.append(feature_55.tolist())
      feature_45.append(feature.tolist())

      if (batch>899):
        break
  feature_25_pre = np.array(feature_25_pre)
  feature_55_pre = np.array(feature_55_pre)
  feature_45 = np.array(feature_45)

  pre25 = []
  i = 0
  while(i<42):
   pre25.append([])
   i = i+1
  pre55 = []
  i = 0
  while(i<42):
    pre55.append([])
    i = i+1
  real45 = []
  i = 0
  while(i<42):
    real45.append([])
    i = i+1

  for i in range(42):
    step25 = 0
    step55 = 0
    step45 = 0
    #print(feature_25_pre[i])

    for j in range(899):
      if(j>0):
        pre25[i].append(feature_25_pre[j][0][i].tolist())
        pre55[i].append(feature_55_pre[j][0][i].tolist())
        real45[i].append(feature_45[j][0][i].tolist())

###############################################################################################################################################################
# This code block is used to: calculate different At between different temperatures using the first 200 cycles,
# and then predict the next 800 cycles.
# First define: At = r1/r2,
#
# 1. When calculating AT, use the gradient of the first 200 cycles, which is the gradient function written before,
# with start = 120, end = 170, size = 10, 20, 30: input pre25, pre55, real45, and calculate for 42 features.
#
# 2. After getting two At25_to_45 (abbreviated as A25) and At55_to_45 (abbreviated as A55), calculate the coefficient
# (1/|1-AT|), apply softmax, and get w1 and w2.
#
# 3. Start reasoning from 200 cycles, saving a step, which is the value of the feature from the previous round (used to calculate the gradient).
# During the reasoning process: F45i = F45(i-1) + (F25i-F25(i-1))*w1 + (F55i-F55(i-1))*w2
###############################################################################################################################################################
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
  def real_gradient(feature_list,start,end,sample_size):
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
    return grad/rang

  g25 = []
  g55 = []
  g45 = []
  at25 = []
  at55 = []
  w_at_25 = []
  real_g45 = []

  for i in range(42):
    start = early_cycle_start#early_cycle_start
    end = early_cycle_end#early_cycle_end
    #########################################################################################
    start45 = early_cycle_start#100
    end45 = early_cycle_end#200
    g25.append(gradient(pre25[i][:],start,end,sample_size))
    g55.append(gradient(pre55[i][:],start,end,sample_size))
    g45.append(gradient(real45[i][:],start45,end45,sample_size))
    real_g45.append(real_gradient(real45[i][:],start45,end45,sample_size))

    at25.append(gradient(real45[i][:],start45,end45,sample_size)/gradient(pre25[i][:],start,end,sample_size))
    at55.append(gradient(real45[i][:],start45,end45,sample_size)/gradient(pre55[i][:],start,end,sample_size))

    step25 = gradient(real45[i][:],start45,end45,sample_size)/gradient(pre25[i][:],start,end,sample_size)
    step55 = gradient(real45[i][:],start45,end45,sample_size)/gradient(pre55[i][:],start,end,sample_size)

    w25 = abs(step25-1)+abs(step55-1)
    w25 = abs(step55-1)/w25
    w_at_25.append(w25)
  #################################################################################################
  # Start inference
  #################################################################################################
  ste = []
  ste2 = []
  ste3 = []
  ste4 = []
  feature25 = []
  feature55 = []
  test_predict = []
  test_lable = []
  only25 = []
  for batch, (domain, domain11,feature, y_tensor, y_plot) in enumerate(test_loader):

      prefor25 = model_1_25(domain.float())
      if test_tmp == "T45":
        domain[0][0] = domain[0][0].float()*-1.45521186
        domain[0][1] = domain[0][1].float()*-1.50569253
        domain[0][2] = domain[0][2].float()*-1.53181673
        domain[0][3] = domain[0][3].float()*-1.55211713
        domain[0][4] = domain[0][4].float()*-1.5616391
        domain[0][5] = domain[0][5].float()*-1.55604219
        domain[0][6] = domain[0][6].float()*-1.54135137
        domain[0][7] = domain[0][7].float()*-1.53640339
        domain[0][8] = domain[0][8].float()*-1.78543008
        domain[0][9] = domain[0][9].float()*1.18198362
      if test_tmp == "T35":
        domain[0][0] = domain[0][0].float()*-12.9708305
        domain[0][1] = domain[0][1].float()*-11.0788281
        domain[0][2] = domain[0][2].float()*-10.3631271
        domain[0][3] = domain[0][3].float()*-8.75590688
        domain[0][4] = domain[0][4].float()*-7.93640255
        domain[0][5] = domain[0][5].float()*-8.50438212
        domain[0][6] = domain[0][6].float()*-9.9675704
        domain[0][7] = domain[0][7].float()*-14.8879788
        domain[0][8] = domain[0][8].float()*-5.74175461
        domain[0][9] = domain[0][9].float()*1
      if test_tmp == "T55":
        domain[0][0] = domain[0][0].float()/-0.87950659
        domain[0][1] = domain[0][1].float()/-0.87824661
        domain[0][2] = domain[0][2].float()/-0.87701338
        domain[0][3] = domain[0][3].float()/-0.85803634
        domain[0][4] = domain[0][4].float()/-0.84296905
        domain[0][5] = domain[0][5].float()/-0.85084735
        domain[0][6] = domain[0][6].float()/-0.87543736
        domain[0][7] = domain[0][7].float()/-0.93763034
        domain[0][8] = domain[0][8].float()/-0.84951105
        domain[0][9] = domain[0][9].float()/0.69207082

      feature_25 = model_1_25(domain.float())
      feature_1 = model_1(domain11.float()).tolist()
      feature_1 = feature_1[0]

      domain[0][0] = domain[0][0].float()*-0.87950659
      domain[0][1] = domain[0][1].float()*-0.87824661
      domain[0][2] = domain[0][2].float()*-0.87701338
      domain[0][3] = domain[0][3].float()*-0.85803634
      domain[0][4] = domain[0][4].float()*-0.84296905
      domain[0][5] = domain[0][5].float()*-0.85084735
      domain[0][6] = domain[0][6].float()*-0.87543736
      domain[0][7] = domain[0][7].float()*-0.93763034
      domain[0][8] = domain[0][8].float()*-0.84951105
      domain[0][9] = domain[0][9].float()*0.69207082

      feature_55 = model_1_55(domain.float())
      feature_25_pre = feature_25.detach().numpy()
      feature_55_pre = feature_55.detach().numpy()
      pre_25_only = prefor25.detach().numpy()
      pre25 = [[] for _ in range(42)]
      pre55 = [[] for _ in range(42)]
      real45 = [[] for _ in range(42)]

      for i in range(42):
        pre25[i].append(feature_25_pre[0][i].tolist())
        pre55[i].append(feature_55_pre[0][i].tolist())

      test_x=[]
      for i in range(42):
        train_x_25 = pre25[i][:]
        train_x_55 = pre55[i][:]
        test_x.append([train_x_25[0], train_x_55[0]])

      test_x = np.array(test_x)
      pred_feature = np.zeros(42)

      if(batch<early_cycle_end):
        step = feature.tolist()
        only25.append(step[0])
      if(batch==early_cycle_end):#159
        last45 = feature.tolist()
        last45 = last45[0]
        last25 = pre25
        last55 = pre55
      #################################################################################################
      # AT method:
      # Input: at25_to_45, at55_to_45, w_at_25, pre25, pre55 (these are from this round), last45, last55, last25 (these are from the previous round).
      # Output: pre45, stored in feature_1 (replaces the original model1 output).
      # Renew: last45 = pre45, last25 = pre25, last55 = pre55
      #################################################################################################

      if(batch>early_cycle_end):
        if(batch==early_cycle_end):
            only25.append(last45)
        for i in range(42):

          step1 = w_at_25[i]*(pre25[i][0]-last25[i][0])*at25[i]
          step2 = (1-w_at_25[i])*(pre55[i][0]-last55[i][0])*at55[i]
          feature_1[i] =  last45[i] + (pre55[i][0]-last55[i][0])*at55[i]
          last45[i] = feature_1[i]
          last25[i][0] = pre25[i][0]
          last55[i][0] = pre55[i][0]
        only25.append(feature_1)

      for i in range(42):

        pred_feature[i] = weight[i][0]*(pre25[i][0]+bias[i][0]) + weight[i][1]*(pre55[i][0]+bias[i][1])
        weight_size = weight[i][0] +  weight[i][1]
        pred_feature[i] = pred_feature[i]/weight_size

      ste = feature.tolist()
      ste2 = pred_feature.tolist()

      test_label.append(ste)
      test_predict.append(pred_feature)


      if (batch) % 1000 == 0:
          print(f"Batch: {batch}")

  pre = np.array(test_predict)
  o25 = np.array(only25)
  predict = [[] for _ in range(42)]
  o_25 = [[] for _ in range(42)]
  for i in range(42):
    for j in range(test_size):
      predict[i].append(pre[j][i].tolist())
      o_25[i].append(only25[j][i])

  # Test the model on the testing set
  test_label = []
  test_predict = []
  test_predict_model1 = []

  i = 0
  for batch, (domain, domain11,feature, y_tensor, y_plot) in enumerate(test_loader):

      row = pre[i][:]
      row_model1 = o25[i][:]

      i = i+1
      predfeature = torch.tensor(row, dtype=torch.float32)
      predfeature_model1 = torch.tensor(row_model1, dtype=torch.float32)
      tensor2 = torch.tensor(predfeature, dtype=torch.float64).view(1, -1)

      tensor2_model1 = torch.tensor(predfeature_model1, dtype=torch.float64).view(1, -1)
      pred_y = model_2(torch.cat([domain11,tensor2],dim= 1).float())-0.03
      pred_y_model1 = model_2(torch.cat([domain11,tensor2_model1],dim= 1).float())

      test_label.append(y_tensor.detach().cpu().item())
      test_predict.append(pred_y.detach().cpu().item())
      test_predict_model1.append(pred_y_model1.detach().cpu().item())

      y_filter.append(y_tensor.detach().cpu().item())

      if (batch) % 1000 == 0:
          print(f"Batch: {batch}")
      if(i>test_size):
        break

  MAPE_value_1 = mape_loss(torch.tensor(test_predict[test_size * i:test_size * (i + 1)]), torch.tensor(test_label[test_size * i:test_size * (i + 1)]))
  MAPE_value_2 = mape_loss(torch.tensor(test_predict_model1[test_size * 0:test_size * (0 + 1)]), torch.tensor(test_label[test_size * 0:test_size * (0 + 1)]))
  print(f"MAPE loss:{MAPE_value_2*100}%")
  sum_mape = sum_mape + MAPE_value_2*100

if(test_tmp == "T25"):
  battery_num = 9
else:
  battery_num = 7
print(f"Avg MAPE:{sum_mape/battery_num}%" )
end_time = time.time()
execution_time = end_time - start_time
print("Running time：", execution_time)
