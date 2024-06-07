import json
import pyarrow as pa
import pyarrow.parquet as pq
path2 = 'data/raw_data/bell_open_source/train_2M_CN.json'
count = 0
data = {'prompt': [], 'response': []}
with open(path2, 'r', errors='ignore') as f:
    for line in f:
        line = json.loads(line.strip())
        prompt = line['instruction']
        response = line['output']
        data['prompt'].append(prompt)
        data['response'].append(response)
        count += 1
        if count > 200000:
            break
        if count % 10000 == 0:
            print(count)
print(count)
table = pa.table(data)
pq.write_table(table, 'data/sft_test.parquet')

