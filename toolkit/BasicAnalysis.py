import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 






def collectNames(dataframe):
        import numpy as np
        import pandas as pd

        columns = list(dataframe.columns)
        return columns 
       

def collectgraph(dataframe, columns):
    
    #future feature - ask for a Y variable to drop

    #necessary Packages 
    import matplotlib.pyplot as plt 
    import seaborn as sns
    import numpy as np
    import pandas as pd
    #Scaling is needed correctly show all variables distribution at the same time -> Scaling prep
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    scaler = StandardScaler()
    minmax = MinMaxScaler()


    tempdf = dataframe.select_dtypes(exclude=['object'])
    o_length = len(columns)
    new_length = len(list(tempdf.columns))
    dropped_columns = o_length - new_length

    #start one map before scaling

    fig, (ax1) = plt.subplots(ncols=2 , figsize=(20, 8))
   


    #scale data
    tempdf = pd.DataFrame(scaler.fit_transform(tempdf),index=tempdf.index, columns=tempdf.columns)

    #warn of dropped columns
    if (dropped_columns > 0):
        print(dropped_columns,"columns were dropped")
        print("try to convert the dropped columns")

    #Print preliminary info of original data frame
    print(dataframe.info())
   
    #graphing
   
    ax1[0].set_title('Original Distributions')
    ax1[1].set_title('Heatmap Correlation')
    

    pp_map = sns.pairplot(dataframe, diag_kind='kde',hue='Survived')

   
    ds_map
    ns_heatmap
    pp_map

    
def buildgraph(dataframe):
    import numpy as np
    import pandas as pd
    columns = collectNames(dataframe)
    collectgraph(dataframe, columns)