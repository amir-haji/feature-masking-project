import json
import numpy as np
import re


if __name__=='__main__':
    all_results = {}

    for seed in range(1, 4):
        path = f"/home/f_hosseini/spurious_project/celeba_misc_balanced_gradmask_seed{seed}.txt"
        file = open(path, "r")
        content = file.read()
        file.close()

        content = content.replace("\n", "")

        pattern = r'\{.*?\}'
        dictionaries = re.findall(pattern, content)

        final_dictionaries = []

        for i in range(len(dictionaries)):
            if i % 3 == 0:
                final_dictionaries.append(dictionaries[i].replace('\'', '\"'))
            if i % 3 == 1:
                s = dictionaries[i] + ', \'test\': ' + dictionaries[i + 1] + '}'
                final_dictionaries.append(s.replace('\'', '\"'))

        converted = []
        for i in range(0, len(final_dictionaries), 2):
            hp = json.loads(final_dictionaries[i])
            print (hp['l1'])
            key = i//2
            res = json.loads(final_dictionaries[i + 1])

            if key in all_results.keys():
                all_results[key]['val_mean'] += res['val']['avg']/3
                all_results[key]['val_worst'] += res['val']['worst']/3
                all_results[key]['test_mean'] += res['test']['avg']/3
                all_results[key]['test_worst'] += res['test']['worst']/3

            else:
                all_results[key] = {}
                all_results[key]['val_mean'] = res['val']['avg']/3
                all_results[key]['val_worst'] = res['val']['worst']/3
                all_results[key]['test_mean'] = res['test']['avg']/3
                all_results[key]['test_worst'] = res['test']['worst']/3

    data = all_results.items()

    print (sorted(data, key=lambda x: x[1]['val_worst'], reverse=True)[:10])



