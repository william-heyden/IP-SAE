from sklearn.manifold import TSNE

X_tr_ip, X_te_ip, S_tr_ip, S_te_ip, Y_te_ip, test_cls_ip, Y_tr_ip, S_te_all_ip = awa2(int_proj=True)
X_tr_og, X_te_og, S_tr_og, S_te_og, Y_te_og, test_cls_og, Y_tr_og, S_te_all_og = awa2(int_proj=False)

# Solve for W
lamb  = 500000;
W_og = SAE(X_tr_og.T,S_tr_og.T,lamb).T
W_ip = SAE(X_tr_ip.T,S_tr_ip.T,lamb).T

#|~~~~~~~~~~~~~~~~~~~~|
#| CONFUSION MATRIX   |
#|~~~~~~~~~~~~~~~~~~~~|

top_hit = 1
X_pred_og = S_te_og.dot(W_og.T)
X_pred_ip = S_te_ip.dot(W_ip.T)

dist_og = 1 - spatial.distance.cdist(X_te_og,X_pred_og,'cosine')
dist_ip = 1 - spatial.distance.cdist(X_te_ip,X_pred_ip,'cosine')

Y_pred_og =np.zeros((dist_og.shape[0],top_hit))
for i in range(dist_og.shape[0]):
    I=np.argsort(dist_og[i])[::-1]
    Y_pred_og[i,:]=test_cls_og[I[0:top_hit]]
Y_pred_ip =np.zeros((dist_ip.shape[0],top_hit))
for i in range(dist_ip.shape[0]):
    I=np.argsort(dist_ip[i])[::-1]
    Y_pred_ip[i,:]=test_cls_ip[I[0:top_hit]]

confmat_og = confusion_matrix(y_true=Y_te_og, y_pred=Y_pred_og)
confmat_ip = confusion_matrix(y_true=Y_te_ip, y_pred=Y_pred_ip)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,10),sharey=False)
ax2.set_title('Original space', weight='bold', color='C0', fontsize=25)
ax2.matshow(confmat_og, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat_og.shape[0]):
    for j in range(confmat_og.shape[1]):
        ax2.text(x=j, y=i, s=confmat_og[i,j], va='center', ha='center')

ax1.set_title('Enriched space', weight='bold', color='C0', fontsize=25)
ax1.matshow(confmat_ip, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat_ip.shape[0]):
    for j in range(confmat_ip.shape[1]):
        ax1.text(x=j, y=i, s=confmat_ip[i,j], va='center', ha='center')

ax1.set_xlabel('Predicted label', weight='bold', fontsize=15)
ax1.set_ylabel('True label', weight='bold', fontsize=15)
ax2.set_xlabel('Predicted label', weight='bold', fontsize=15)
ax2.set_ylabel('True label', weight='bold', fontsize=15)

ax1.xaxis.set_ticks([i for i in range(len(test_labels))])
ax1.xaxis.set_ticklabels(test_labels, horizontalalignment='right', rotation=45)
ax1.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_ticks([i for i in range(len(test_labels))])
ax2.xaxis.set_ticklabels(test_labels, horizontalalignment='right', rotation=45)
ax2.xaxis.set_ticks_position('bottom')

ax1.yaxis.set_ticks([i for i in range(len(test_labels))])
ax1.yaxis.set_ticklabels(test_labels, horizontalalignment='right', ma='right', fontsize=13)
ax2.yaxis.set_ticks([i for i in range(len(test_labels))])
ax2.yaxis.set_ticklabels(test_labels, horizontalalignment='right', ma='right', fontsize=13)

plt.subplots_adjust(wspace=.3)
plt.show()

#|~~~~~~~~~~~~~~|
#| T-SNE PLOT   |
#|~~~~~~~~~~~~~~|

from sklearn.manifold import TSNE
def get_data(data, lbls, tsne_obj):

    n_cls = len(np.unique(lbls))
    z = tsne_obj.fit_transform(data)
    df_temp = pd.DataFrame()
    df_temp['y'] = lbls
    df_temp['t-SNE1'] = z[:,0]
    df_temp['t-SNE2'] = z[:,1]

    return df_temp, n_cls

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
og, n_cls_og = get_data(X_tr_og, Y_tr_og, tsne)
ip, n_cls_ip = get_data(X_tr_ip, Y_tr_ip, tsne)
cls_nm = './data/AWA2/trainvalclasses.txt'
with open(cls_nm) as file:
    n_unique_cls = [line.rstrip() for line in file]

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))

g1 = sns.scatterplot(x=ip['t-SNE1'], y=ip['t-SNE2'], hue=ip.y.tolist(), palette=sns.color_palette("hls", n_cls_ip), data=ip, legend=True, ax=ax1)
g1.set_title('Enhanced space', fontdict={'weight':'bold', 'size': 20, 'color':'C0'})
g1.set(xticklabels=[],yticklabels=[])#, xlabel=None, ylabel=None)
g1.tick_params(bottom=False, left=False)

hands, labs = ax1.get_legend_handles_labels()
#g1.legend(handles=hands, labels=n_unique_cls[:10], loc='upper left', fontsize=13zx45, fancybox=True, shadow=True, ncol=1, bbox_to_anchor=(1.0, .8))
g1.legend(handles=hands, labels=n_unique_cls[:15], loc='upper center', fontsize=13, fancybox=True, shadow=True, ncol=5, bbox_to_anchor=(1.0, -.06))

g2 = sns.scatterplot(x=og['t-SNE1'], y=og['t-SNE2'], hue=og.y.tolist(), palette=sns.color_palette("hls", n_cls_og), data=og, legend=False, ax=ax2)
g2.set_title('Original space', fontdict={'weight':'bold', 'size': 20, 'color':'C0'})
g2.set(xticklabels=[],yticklabels=[])#, xlabel=None, ylabel=None)
g2.tick_params(bottom=False, left=False)

plt.subplots_adjust(wspace=.1)
plt.show()


#|~~~~~~~~~~~~~~~~~~~~|
#| PRECISION/RECALL   |
#|~~~~~~~~~~~~~~~~~~~~|

def get_y_pred(x,s,w, tst_cls):
    top_hit = 1
    X_pred = s.dot(w.T)

    dist = 1 - spatial.distance.cdist(x,X_pred,'cosine')

    Y_pred =np.zeros((dist.shape[0],top_hit))
    for i in range(dist.shape[0]):
        I=np.argsort(dist[i])[::-1]
        Y_pred[i,:]=tst_cls[I[0:top_hit]]

    return Y_pred

df_names = ['AWA2', 'CUB', 'SUN']
ip = False
pr, rc, f1 = [],[],[]
for n in df_names:
    if n=='AWA2':
        X_tr, X_te, S_tr, S_te, Y_te, test_cls, Y_tr, S_te_all = awa2(int_proj=ip)
        lamb = 200000
    if n=='CUB':
        X_tr, X_te, S_tr, S_te, Y_te, test_cls, Y_tr, S_te_all = cub(int_proj=ip)
        lamb = 500000
    if n=='SUN':
        X_tr, X_te, S_tr, S_te, Y_te, test_cls, Y_tr, S_te_all = sun(int_proj=ip)
        lamb = 500000
    W = SAE(X_tr.T,S_tr.T,lamb).T
    y_pred = get_y_pred(X_te, S_te, W, test_cls)
    pr.append(precision_score(y_true=Y_te, y_pred=y_pred, average='weighted'))
    rc.append(recall_score(y_true=Y_te, y_pred=y_pred, average='weighted'))
    f1.append(f1_score(y_true=Y_te, y_pred=y_pred, average='weighted'))


#|~~~~~~~~~~~~~~~~~~~~~~~~~~|
#| IRREGULARITY OF LAMBDA   |
#|~~~~~~~~~~~~~~~~~~~~~~~~~~|

def acc_w(w, s_test, x_test, y_test, cls_test):
    X_pred = s_test.dot(w.T)
    dist = 1 - spatial.distance.cdist(x_test,X_pred,'cosine')
    return acc_zsl(dist, cls_test, y_test)

w_ip_awa = np.load('./data/W_lambda/W_list_lambda_AWA2_IPtrue.npy')
w_og_awa = np.load('./data/W_lambda_NoIP/W_list_lambda_AWA2.npy')
w_ip_cub = np.load('./data/W_lambda/W_list_lambda_CUB_IPtrue.npy')
w_og_cub = np.load('./data/W_lambda_NoIP/W_list_lambda_CUB.npy')
w_lambda = [i for i in range(0,1000000, 10000)]# = np.load('data/W_lambda_NoIP/W_lambda_AWA2.npy')
X_tr_ip, X_te_ip, S_tr_ip, S_te_ip, Y_te_ip, test_cls_ip, Y_tr_ip, S_te_all_ip = ld.awa2(int_proj=True)
X_tr_og, X_te_og, S_tr_og, S_te_og, Y_te_og, test_cls_og, Y_tr_og, S_te_all_og = ld.awa2(int_proj=False)
X_tr_ip_c, X_te_ip_c, S_tr_ip_c, S_te_ip_c, Y_te_ip_c, test_cls_ip_c, Y_tr_ip_c, S_te_all_ip_c = ld.cub(int_proj=True)
X_tr_og_c, X_te_og_c, S_tr_og_c, S_te_og_c, Y_te_og_c, test_cls_og_c, Y_tr_og_c, S_te_all_og_c = ld.cub(int_proj=False)

acc_ip_awa = [acc_w(w_ip_awa[i], S_te_ip, X_te_ip, Y_te_ip, test_cls_ip) for i in range(len(w_lambda))]
acc_og_awa = [acc_w(w_og_awa[i], S_te_og, X_te_og, Y_te_og, test_cls_og) for i in range(len(w_lambda))]
acc_ip_cub = [acc_w(w_ip_cub[i], S_te_ip_c, X_te_ip_c, Y_te_ip_c, test_cls_ip_c) for i in range(len(w_lambda))]
acc_og_cub = [acc_w(w_og_cub[i], S_te_og_c, X_te_og_c, Y_te_og_c, test_cls_og_c) for i in range(len(w_lambda))]

fig, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=(5,2))
fig.suptitle('')
ax1.set_title('AWA2', weight='bold', color='C0')
ax1.plot(w_lambda, acc_ip_awa, label='Enriched space', color='b')
ax1.plot(w_lambda, acc_og_awa, label='Original space', color='y')
ax1.set_yticks((.2,.5,.8))
ax1.set_xticks((w_lambda[2], w_lambda[-2]))
ax1.set_xlabel('Lambda (λ)', fontsize=10, labelpad=.5)
ax1.set_ylabel('Accuracy (%)', fontsize=10)

ax2.set_title('CUB', weight='bold', color='C0')
ax2.plot(w_lambda, acc_ip_cub, label='Enriched space', color='b')
ax2.plot(w_lambda, acc_og_cub, label='Original space', color='y')
ax2.set_yticks((.2,.5,.8))
ax2.set_xticks((w_lambda[2], w_lambda[-2]))
ax2.set_xlabel('Lambda (λ)', fontsize=10, labelpad=.5)
ax1.legend(loc='upper center', fontsize=10, fancybox=True, shadow=True, ncol=2, bbox_to_anchor=(.99, -.25))
plt.show()
