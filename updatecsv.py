import csv
with open('data/IMDBDataset.csv', 'rt', encoding="ascii") as inp, open('data/IMDBDataset_edit.csv', 'wt') as out:
    writer = csv.writer(out)
    count = 0
    for row in csv.reader(inp):
        if count < 5000:
            writer.writerow(row)
        count += 1
    print(count)
