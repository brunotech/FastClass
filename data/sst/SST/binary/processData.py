

fc = open('label_names.txt', 'w', encoding='utf-8')
fc.write('negative\npositive\n')

fw = open('test.txt', 'w', encoding='utf-8')
fl = open('test_labels.txt', 'w', encoding='utf-8')
with open('sentiment-test', 'r') as f:
    for line in f:
        data, label = line.strip().split('\t')
        fw.write(f'{data}\n')
        fl.write(f'{label}\n')

fw = open('train.txt', 'w', encoding='utf-8')
fl = open('train_labels.txt', 'w', encoding='utf-8')
with open('sentiment-train', 'r') as f:
    for line in f:
        data, label = line.strip().split('\t')
        fw.write(f'{data}\n')
        fl.write(f'{label}\n')

