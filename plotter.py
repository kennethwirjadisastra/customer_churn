import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

def plot_hist(df, figsize, nrows=None, ncols=None, hue=None, palette=None, showmeans=False):
    n_features = len(df.columns) - 1 if 'Churn' in df.columns else len(df.columns)    # exclude 'Churn'
    n_features = n_features - 1 if 'Cluster' in df.columns else n_features            # exclude 'cluster'

    # set nrows and ncols accordingly
    if nrows==ncols==None:
        ncols = 3
        nrows = nrows = (n_features//ncols) + (n_features%ncols!=0)
    elif nrows!=None and ncols!=None:
        if nrows*ncols < n_features:
            raise ValueError("nrows*ncols less than n_features")
    elif nrows==None and ncols!=None:
        nrows = (n_features//ncols) + (n_features%ncols!=0)
    elif nrows!=None and ncols==None:
        ncols = (n_features//nrows) + (n_features%nrows!=0)
    
    # plot
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)

    ax = ax.flatten()

    for idx in range(len(ax)):
        col = df.columns[idx] if idx < len(df.columns) else None
        if col != None and col != 'Churn' and col != 'Cluster': # only plot if col exists and is not 'Churn' or 'cluster'
            kde = True if col[:3]=='Avg' else False
            sns.histplot(data=df, x=col, hue=hue, kde=kde, palette=palette, ax=ax[idx])
            
            if showmeans:
                col_mean = df[col].mean()
                top, bottom = ax[idx].get_ylim()
                ax[idx].plot(2*[col_mean], [top, bottom], '--r', label='Feature Mean')
            ax[idx].set_title(f'{col} Distribution', fontsize=14)
            ax[idx].set_xlabel(f'{col}', fontsize=12)
            ax[idx].set_ylabel(None)
            ax[idx].grid()
            if idx%ncols==0:
                ax[idx].set_ylabel('Count', fontsize=12)
        else:
            ax[idx].text(0.5, 0.5, 'None', fontsize=20,
                         horizontalalignment='center', verticalalignment='center', transform=ax[idx].transAxes)
            
    if showmeans:
        handle, label = ax[0].get_legend_handles_labels()
        fig.legend(handle, label, fontsize=16, bbox_to_anchor=(0.95,0.95))

    return (fig, ax)


def plot_box(df, figsize, nrows=None, ncols=None, hue=None, palette=None, showmeans=False):
    n_features = len(df.columns) - 1 if 'Churn' in df.columns else len(df.columns)    # exclude 'Churn'
    n_features = n_features - 1 if 'Cluster' in df.columns else n_features            # exclude 'cluster'

    # set nrows and ncols accordingly
    if nrows==ncols==None:
        ncols = 3
        nrows = nrows = (n_features//ncols) + (n_features%ncols!=0)
    elif nrows!=None and ncols!=None:
        if nrows*ncols < n_features:
            raise ValueError("nrows*ncols less than n_features")
    elif nrows==None and ncols!=None:
        nrows = (n_features//ncols) + (n_features%ncols!=0)
    elif nrows!=None and ncols==None:
        ncols = (n_features//nrows) + (n_features%nrows!=0)
    
    # plot
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)

    ax = ax.flatten()

    for idx in range(len(ax)):
        col = df.columns[idx] if idx < len(df.columns) else None
        if col != None and col != 'Churn' and col != 'Cluster': # only plot if col exists and is not 'Churn' or 'cluster'
        
            sns.boxplot(data=df, y=col, hue=hue, palette=palette, showmeans=showmeans, ax=ax[idx]) 
            ax[idx].set_title(f'{col} Boxplot', fontsize=14)
            ax[idx].set_ylabel(f'{col}', fontsize=12)
            ax[idx].grid()
        else:
            ax[idx].text(0.5, 0.5, 'None', fontsize=20,
                         horizontalalignment='center', verticalalignment='center', transform=ax[idx].transAxes)
            
    if showmeans:
        mean_marker = mlines.Line2D([], [], color='green', marker='^', linestyle='None',
                                    markersize=10, label='Mean') 
        fig.legend(handles=[mean_marker], labels=['Feature Mean'], fontsize=16, bbox_to_anchor=(0.95,0.95)) # only one label
        
    return (fig, ax)


def plot_bar(df, figsize, nrows=None, ncols=None, hue=None, palette=None, raw_magnitude=True):
    n_features = len(df.columns) - 1 if 'Churn' in df.columns else len(df.columns)    # exclude 'Churn'
    n_features = n_features - 1 if 'Cluster' in df.columns else n_features            # exclude 'cluster'

    # set nrows and ncols accordingly
    if nrows==ncols==None:
        ncols = 3
        nrows = nrows = (n_features//ncols) + (n_features%ncols!=0)
    elif nrows!=None and ncols!=None:
        if nrows*ncols < n_features:
            raise ValueError("nrows*ncols less than n_features")
    elif nrows==None and ncols!=None:
        nrows = (n_features//ncols) + (n_features%ncols!=0)
    elif nrows!=None and ncols==None:
        ncols = (n_features//nrows) + (n_features%nrows!=0)
    
    # plot
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)

    ax = ax.flatten()

    for idx in range(len(ax)):
        col = df.columns[idx] if idx < len(df.columns) else None
        if col != None and col != 'Churn' and col != 'Cluster': # only plot if col exists and is not 'Churn' or 'cluster'

            if raw_magnitude:
                sns.countplot(data=df, x=col, hue=hue,palette=palette, ax=ax[idx])
                ax[idx].set_title(f'{hue} Count by {col}', fontsize=14)
                ax[idx].set_ylabel(None)
                
            else:
                sns.barplot(data=df.groupby(col).mean(), x=col, y=hue, ax=ax[idx])    
                ax[idx].set_title(f'Relative {hue} by {col}', fontsize=14)
                ax[idx].set_ylabel(None)
                    
            ax[idx].set_xlabel(f'{col}', fontsize=12)
            
            if idx%ncols==0: # set y label on leftmost column only
                ax[idx].set_ylabel('Count', fontsize=12) if raw_magnitude else ax[idx].set_ylabel('Churn rate (%)', fontsize=12)
            else:
                ax[idx].set_ylabel(None)
            
        else: # fill remaining plots with None text
            ax[idx].text(0.5, 0.5, 'None', fontsize=20,
                         horizontalalignment='center', verticalalignment='center', 
                         transform=ax[idx].transAxes)
    return (fig, ax)


def plot_pie(df, figsize, nrows=None, ncols=None, n_clusters=3):
    n_features = len(df.columns) - 1 if 'Cluster' in df.columns else len(df.columns)    # exclude 'Cluster'

    on_proportion = df.groupby('Cluster').mean()

    # set nrows and ncols accordingly
    if nrows==ncols==None:
        ncols = 3
        nrows = nrows = (n_features//ncols) + (n_features%ncols!=0)
    elif nrows!=None and ncols!=None:
        if nrows*ncols < n_features:
            raise ValueError("nrows*ncols less than n_features")
    elif nrows==None and ncols!=None:
        nrows = (n_features//ncols) + (n_features%ncols!=0)
    elif nrows!=None and ncols==None:
        ncols = (n_features//nrows) + (n_features%nrows!=0)


    # plot
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)

    ax = ax.flatten()

    for idx in range(len(ax)):
        col = on_proportion.columns[idx] if idx < len(on_proportion.columns) else None
        super_ax = ax[idx]
        super_ax.set_xticks([])
        super_ax.set_yticks([])

        if col is not None:
            super_ax.set_title(f'{col} by Cluster', fontweight='bold')

            for n in range(n_clusters):
                sub_ax = super_ax.inset_axes([n*0.3, 0.1, 0.28, 0.8])

                if idx < len(on_proportion.columns):
                    on = on_proportion.iloc[n,idx]
                    off = 1 - on

                    sub_ax.pie([on, off], labels=['1','0'], autopct='%1.2f%%')
                    sub_ax.set_title(f'Cluster {n}')
        else:
            super_ax.text(0.5, 0.5, 'None', fontsize=20,
                                horizontalalignment='center', verticalalignment='center', 
                                transform=super_ax.transAxes)


    plt.tight_layout()
    
    return (fig, ax)