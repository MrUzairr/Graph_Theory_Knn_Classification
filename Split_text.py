import pandas as pd

def process_csv(input_file, output_file):
    df = pd.read_csv(input_file, encoding='latin1')

    processed_data = []
    for index, row in df.iterrows():
        category = row['category']
        content = row['content']
        if pd.notnull(content):
            words = content.split()
            for word in words:
                processed_data.append([word, category])
        else:
            continue
        
    processed_df = pd.DataFrame(processed_data, columns=['text', 'label'])
    processed_df.to_csv(output_file, index=False)

for i in range(16):
    input_file = "Travel {0}.csv".format(i+1)
    output_file = "Processed_data/Travel_{0}_processed.csv".format(i+1)
    process_csv(input_file, output_file)
    print('done {0}'.format(i+1))
