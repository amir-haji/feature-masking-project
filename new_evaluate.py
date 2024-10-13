import json
import os


# waterbirds or celeba
dataset = 'civilcomments'


path = f'/home/f_hosseini/hajimohammadrezaei/pseudo-grad-unmasked-logs/{dataset}/not_free/'
  
total_res = {}



for sample in range(8):
  if sample == 0:
    s = ''
  else:
    s = str(sample)
      
  partial_res = {"val": {"avg": 0, "worst": 0}, "test": {"avg": 0, "worst": 0}}
  count = 0
  key = f'{dataset}_LR0.0005_step85_gamma0.5_samples20_l10.0{s}'
    
  for seed in range(1, 4):
    partial = f'{dataset}_LR0.0005_step85_gamma0.5_seed{seed}_samples20_l10.0{s}'
    result_path = path + f'seed{seed}/' + f'loss__' + partial + '/results.json'
      
    if os.path.exists(result_path):
      count += 1
      with open(result_path, 'r') as f:
        data = json.loads(f.read())
        f.close()
          
      for a in ['val', 'test']:
        for b in ['avg', 'worst']:
          partial_res[a][b] += data[a][b]
        
    else:
      print(result_path + ' doesn\'t exist')
          
          
  for a in ['val', 'test']:
    for b in ['avg', 'worst']:
      partial_res[a][b] /= count
        
  total_res[key] = partial_res
    

final = sorted(total_res.items(), key=lambda x: x[1]['test']['worst'], reverse=True)

for k, v in final:
  print(k, ' ', v)


          
          
          
          
