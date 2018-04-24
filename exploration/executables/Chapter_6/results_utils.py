def create_dict(groups_k):
    return {k[0] + '&' + k[1]: [] for k in groups_k}

def incremental_mean(arr_):
    n_samples = len(arr_)
    sum_ = 0
    out = []
    for i in range(n_samples):
        sum_ += arr_[i]
        out += [sum_/(i + 1.)]
    out = np.array(out).flatten()
    return out

def moving_av(arr_, win_sz):
    n_samples = len(arr_)
    sum_ = 0
    out = []
    for i in range(win_sz):
        sum_ += arr_[i]
        out += [sum_/(i + 1.)]
    for i in range(win_sz,n_samples):
        sum_ = sum_ + arr_[i] - arr_[i-win_sz]
        out += [sum_/win_sz]
    out = np.array(out).flatten()
    return out

def std_markers(ax,y,m,sz,color='b'):
    plt.sca(ax)
    for i,(y_,ms_) in enumerate(zip(y,sz)):
        plt.plot(i, y_, marker = m, ms = ms_, color = color)
        
def get_stat_test(dict_):
    thresh = []
    values  = []
    for key in dict_.keys():
        if 'autonomous' in key:
            continue
        values += [dict_[key]]
        key = key.split('&')
        key[0] = key[0].replace('_','.')
        thresh += [float(key[0])]

    return pearsonr(thresh, values)