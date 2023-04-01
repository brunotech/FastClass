
fw = open('./test.txt', 'w', encoding='utf-8')
data, label = [], []
with open('originalData/test.txt', 'r') as f:
    data.extend(line.strip() for line in f)
with open('originalData/test_labels.txt', 'r') as f:
    label.extend(line.strip() for line in f)
for i in range(len(label)):
    fw.write(label[i]+'\t'+data[i]+'\n')

fw = open('./train.txt', 'w', encoding='utf-8')
data, label = [], []
with open('originalData/train.txt', 'r') as f:
    data.extend(line.strip() for line in f)
with open('originalData/train_labels.txt', 'r') as f:
    label.extend(line.strip() for line in f)
for i in range(len(label)):
    fw.write(label[i]+'\t'+data[i]+'\n')