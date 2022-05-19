import numpy as np
import pandas as pd
from matplotlib import pyplot as plt  
from matplotlib import cm


three_ind_filenames_in = ["Three_Indicator_0_in_Results.csv", "Three_Indicator_1_in_Results.csv", "Three_Indicator_2_in_Results.csv", "Three_Indicator_3_in_Results.csv", "Three_Indicator_4_in_Results.csv"]
three_ind_filenames_out = ["Three_Indicator_0_out_Results.csv", "Three_Indicator_1_out_Results.csv", "Three_Indicator_2_out_Results.csv", "Three_Indicator_3_out_Results.csv", "Three_Indicator_4_out_Results.csv"]

four_ind_filenames_in = ["Four_Indicator_0_in_Results.csv", "Four_Indicator_1_in_Results.csv", "Four_Indicator_2_in_Results.csv", "Four_Indicator_3_in_Results.csv"]
four_ind_filenames_out = ["Four_Indicator_0_out_Results.csv", "Four_Indicator_1_out_Results.csv", "Four_Indicator_2_out_Results.csv", "Four_Indicator_3_out_Results.csv"]

five_ind_filenames_in = ["Five_Indicator_0_in_Results.csv", "Five_Indicator_1_in_Results.csv", "Five_Indicator_2_in_Results.csv"]
five_ind_filenames_out = ["Five_Indicator_0_out_Results.csv", "Five_Indicator_1_out_Results.csv", "Five_Indicator_2_out_Results.csv"]

six_ind_filenames_in = ["Six_Indicator_in_Results.csv"]
six_ind_filenames_out = ["Six_Indicator_out_Results.csv"]

seven_ind_filenames_in = ["Seven_Indicator_in_Results.csv"]
seven_ind_filenames_out = ["Seven_Indicator_out_Results.csv"]

in_samples = [three_ind_filenames_in, four_ind_filenames_in, five_ind_filenames_in, six_ind_filenames_in, seven_ind_filenames_in]
out_samples = [three_ind_filenames_out, four_ind_filenames_out, five_ind_filenames_out, six_ind_filenames_out, seven_ind_filenames_out]
in_sample_graph_names = ['Three Indicator In-Sample Results', 'Four Indicator In-Sample Results', 'Five Indicator In-sample Results', 'Six Indicator In-sample Results', 'Seven Indicator In-sample Results']
out_of_sample_graph_names = ['Three Indicator Out-of-Sample Results', 'Four Indicator Out-of-Sample Results', 'Five Indicator Out-of-Sample Results', 'Six Indicator Out-of-sample Results', 'Seven Indicator Out-of-sample Results']

print(in_samples)

same_as_base = []

"""
#Data Cleaning, Five indicators out baseline had in-sample baseline
df = pd.read_csv("Four_Indicator_1_out_Results.csv")
df.Date = pd.to_datetime(df.Date)
df.set_index('Date', inplace=True)
baseline = pd.DataFrame(df['Baseline'])
print(baseline)

for filename in five_ind_filenames_out:
    df = pd.read_csv(filename)
    df.Date = pd.to_datetime(df.Date)
    df.set_index('Date', inplace=True)    
    df = df.drop(columns='Baseline')
    df = pd.concat([df, baseline], axis=1)
    df = df.dropna()
    df.to_csv(filename, index=True)
    
"""

for idx, n_indicators in enumerate(in_samples):
    #merge all the seperate n_ind csvs
    merge_n_ind = []
    for index, filename in enumerate(n_indicators): 
        df = pd.read_csv(filename)
        #avoid repeats
        if index != 0:
            df = df.drop(columns='Baseline')
        df.Date = pd.to_datetime(df.Date)
        df.set_index('Date', inplace=True)
        merge_n_ind.append(df)
    merged_df = pd.concat(merge_n_ind, axis = 1)
    print(merged_df.columns)
    
    
    baseline = pd.DataFrame(merged_df['Baseline'])
    merged_df = merged_df.drop(columns='Baseline')

    #indicator combinations that doesn't equal baseline
    mask = merged_df.iloc[-1] != baseline.values[-1][0]

    same_as_baseline_ind = []
    for index, value in mask.items():
        if not value:
            same_as_baseline_ind.append(index)
    print(same_as_baseline_ind)
    same_as_base.append(same_as_baseline_ind)

    #plot top 3 ind combinations that doesn't equal baseline
    merged_df = merged_df.loc[:,mask]
    merged_df = pd.concat([baseline, merged_df], axis=1)
    #top_three = merged_df.iloc[-1, np.argsort(-merged_df.values[0])[:3]]
    #for i in merged_df.index:
    #    compare = pd.concat([compare, merged_df[i]], axis=1)
    
    ax = merged_df.plot.line(title = in_sample_graph_names[idx], colormap=cm.Accent, alpha=1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)    
    ax.grid()
    plt.tight_layout()
    plt.show()



for idx, n_indicators in enumerate(out_samples):
    #merge all the seperate n_ind csvs
    merge_n_ind = []
    for index, filename in enumerate(n_indicators): 
        df = pd.read_csv(filename)
        #avoid repeats
        if index != 0:
            df = df.drop(columns='Baseline')
        df.Date = pd.to_datetime(df.Date)
        df.set_index('Date', inplace=True)
        merge_n_ind.append(df)
    merged_df = pd.concat(merge_n_ind, axis = 1)
    print(merged_df.columns)
    
    
    baseline = pd.DataFrame(merged_df['Baseline'])
    merged_df = merged_df.drop(columns='Baseline')

    #indicator combinations that doesn't equal baseline
    mask = merged_df.iloc[-1] != baseline.values[-1][0]

    same_as_baseline_ind = []
    for index, value in mask.items():
        if not value:
            same_as_baseline_ind.append(index)
    print(same_as_baseline_ind)
    same_as_base.append(same_as_baseline_ind)

    #plot top 3 ind combinations that doesn't equal baseline
    merged_df = merged_df.loc[:,mask]
    merged_df = pd.concat([baseline, merged_df], axis=1)
    #top_three = merged_df.iloc[-1, np.argsort(-merged_df.values[0])[:3]]
    #for i in merged_df.index:
    #    compare = pd.concat([compare, merged_df[i]], axis=1)
    
    ax = merged_df.plot.line(title=out_of_sample_graph_names[idx], colormap=cm.Accent, alpha=1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)    
    ax.grid()
    plt.tight_layout()
    plt.show()