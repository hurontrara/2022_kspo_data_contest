import pandas as pd

총인구=pd.read_csv('/content/주민등록인구_20221105170557.csv')
장애인구=pd.read_csv('/content/장애인+현황(장애유형별_동별)_20221105173400.csv')
총인구.columns = ['합계', '시군구', '행정동','총인구']
장애인구.columns = ['합계', '시군구', '행정동','장애인구']
총인구 = 총인구.drop(['합계'], axis = 1)
총인구=총인구[총인구.총인구 != '계']
총인구=총인구[총인구.시군구 != '소계']
총인구=총인구[총인구.행정동 != '소계']

장애인구 = 장애인구.drop(['합계'], axis = 1)
장애인구 = 장애인구[장애인구.장애인구 != '계']
장애인구 = 장애인구[장애인구.장애인구 != '합계']
장애인구 = 장애인구[장애인구.시군구 != '소계']
장애인구 = 장애인구[장애인구.행정동 != '소계']
장애인구 = 장애인구[장애인구.행정동 != '기타']

인구데이터=pd.merge(총인구, 장애인구, on=['시군구','행정동'])
인구데이터 = 인구데이터.astype({'총인구':'int','장애인구':'int'})
인구데이터['비장애인구'] = 인구데이터['총인구'] - 인구데이터['장애인구']
pd.set_option('display.max_rows', None)
인구데이터=인구데이터.append({'시군구': '강동구','행정동':'상일동','총인구':39031,'장애인구':1580,'비장애인구':37451},ignore_index=True)
인구데이터 = 인구데이터[인구데이터.행정동 != '상일1동']
인구데이터 = 인구데이터[인구데이터.행정동 != '상일2동']
인구데이터.to_csv('./인구데이터.csv', encoding="cp949")

#상일동이 2021년 7월 기준, 상일1동과 상일2동으로 분리
#공공체육시설 데이터와 결합하기 위하여 상일1동과 상일2동을 결합

공공체육시설=pd.read_csv('/content/전국공공체육시설 (3).csv')
공공체육시설=공공체육시설[['FACI_NM','FACI_POINT_X','FACI_POINT_Y','FMNG_CP_NM','ADDR_CP_NM','FACI_STAT','DEL_YN']]
공공체육시설.columns = ['시설명', '경도', '위도','소유주체 시도명','도로명 시도명','시설상태','삭제여부']
공공체육시설 = 공공체육시설[(공공체육시설['삭제여부']=='N') & (공공체육시설['시설상태']==0)]

#정상운영하며 삭제되지 않은 공공체육시설 추출
공공체육시설 = 공공체육시설[(공공체육시설['소유주체 시도명']=='서울특별시') & (공공체육시설['도로명 시도명']=='서울특별시')]

#소유주체와 도로명 시도명이 상이한 경우에는 오류이며 소유주체 기준으로 서울특별시 추출
위도경도=공공체육시설[공공체육시설['경도'].isnull()]

#위도경도 제거하면 공공체육시설의 84.51% 정도 데이터 존재, 즉 15.49% 결측치 존재
#위도경도를 하나하나 입력하는 방향보다 결측치는 모두 제거 후, 본 프로젝트 한계점으로 기재

공공체육시설=공공체육시설.dropna(axis=0)
공공체육시설.isnull().sum()
공공체육시설.describe()
공공체육시설.to_csv('./공공체육시설.csv', encoding="cp949")

#########################################################################

#서울시 행정동별 공공체육시설 개수는 QGIS 프로그램을 활용하여 추출하였기에 코드 X

import pandas as pd

d1 = pd.read_csv('./인구데이터(최종).csv', encoding='cp949')
d2 = pd.read_csv('./행정동 별 시설 개수(최종).csv')

b = pd.merge(d2, d1, how = 'outer', left_on = 'ADM_DR_NM', right_on = '행정동')

b.to_csv('./인구+시설 합집합.csv', encoding="cp949")

# 이후 동 표기법이 다르거나 또는 같은 이름의 행정동으로 인해
# 병합이 잘 안 된 데이터들은 엑셀에서 마무리 작업 함

# ----------------------------------
인구시설=pd.read_csv('/content/인구+시설 병합.csv',encoding='cp949')
장애인구시설=인구시설[['시군구','행정동','NUMPOINTS','장애인구']]
비장애인구시설=인구시설[['시군구','행정동','NUMPOINTS','비장애인구']]

#장애인구와 비장애인구 분리

#########################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid', font_scale=1.5)
sns.set_palette('Set2', n_colors=10)
plt.rc('font', family='malgun gothic')
plt.rc('axes', unicode_minus=False)

d1 = pd.read_csv('인구+시설 병합.csv', encoding='cp949')
d1.info()

d2 = d1.drop(['행정동 코드', 'fid'], axis = 1)

gu = d2.groupby('시군구').sum()
gu = gu.reset_index()

#자치구별 시설 개수 시각화
pop = gu.sort_values(by = ['시설 개수'], ascending=False)

plt.figure(figsize = (15,8))
ax = sns.barplot(data = pop,
           x = '시군구',
           y = '시설 개수')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40)

#자치구별 총인구수 시각화
pop = gu.sort_values(by = ['총인구'], ascending=False)

plt.figure(figsize = (15,8))
ax = sns.barplot(data = pop,
           x = '총인구',
           y = '시설 개수')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40)

#자치구별 장애인구수 시각화
pop = gu.sort_values(by = ['장애인구'], ascending=False)

plt.figure(figsize = (15,8))
ax = sns.barplot(data = pop,
           x = '장애인구',
           y = '시설 개수')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40)

#자치구별 비장애인구수 시각화
pop = gu.sort_values(by = ['비장애인구'], ascending=False)

plt.figure(figsize = (15,8))
ax = sns.barplot(data = pop,
           x = '비장애인구',
           y = '시설 개수')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40)

#########################################################################

# (1) k-means
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 장애인구
pop_o = pd.read_csv('장애인구시설.csv', encoding='cp949')

pop_o = pop_o.drop('Unnamed: 0', axis=1)

장애인구체육=pop_o[['NUMPOINTS','장애인구']]
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
print(standardScaler.fit(장애인구체육))
장애인구체육_standardScaled = standardScaler.transform(장애인구체육)
장애인구체육_standardScaled

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer

model = KMeans()
visualizer = KElbowVisualizer(model,k=(1,10))
visualizer.fit(장애인구체육_standardScaled)

optimal_k_df1 = 3

kmeans_1 = KMeans(n_clusters=optimal_k_df1, random_state=111,init='random').fit(장애인구체육_standardScaled)

pop_o['K-means 군집'] = kmeans_1.predict(장애인구체육_standardScaled)

pop_o.groupby('K-means 군집').mean()

pop_o.to_csv('./군집 결과(장애인구).csv', encoding="cp949")

# 비장애인구
pop_x = pd.read_csv('비장애인구시설.csv', encoding='cp949')

pop_x = pop_x.drop('Unnamed: 0', axis=1)

pop=pop_x[['NUMPOINTS','비장애인구']]
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
print(standardScaler.fit(pop))
pop_sc = standardScaler.transform(pop)
pop_sc

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer

model = KMeans()
visualizer = KElbowVisualizer(model,k=(1,10))
visualizer.fit(pop_sc)

optimal_k_df1 = 3
kmeans_1 = KMeans(n_clusters=optimal_k_df1, random_state=111,init='random').fit(pop_sc)

pop_x['K-means 군집'] = kmeans_1.predict(pop_sc)

pop_x.groupby('K-means 군집').mean()

pop_x.to_csv('./군집 결과(비장애인구).csv', encoding="cp949")

###################################################
# (2) k-medoids -> coded by R
# # 장애인구
# install.packages('cluster')
# install.packages('factoextra')
# library(factoextra)
# library(cluster)
# 장애인구시설=read.csv('장애인구시설.csv')
# 장애별=장애인구시설[,4:5]
# 정규화장애별=scale(장애별)
#
# fviz_nbclust(정규화장애별, pam, method = "wss")
# medoids <- pam(정규화장애별, k = 3)
# fviz_cluster(medoids, data = 장애인구시설,geom = "point")
# 장애인구시설$medoid<-medoids$clustering
# aggregate(cbind(NUMPOINTS,장애인구)~medoid,장애인구시설,mean)
#
#
# # 비장애인구
# 비장애인구시설=read.csv('비장애인구시설.csv')
# 비장애별=비장애인구시설[,4:5]
# 정규화비장애별=scale(비장애별)
#
# fviz_nbclust(정규화비장애별, pam, method = "wss")
# medoids2 <- pam(정규화비장애별, k = 3)
# fviz_cluster(medoids2, data = 비장애인구시설,geom = "point")
# 비장애인구시설$medoid<-medoids2$clustering
# aggregate(cbind(NUMPOINTS,비장애인구)~medoid,비장애인구시설,mean)


################################################
# (3) Gaussian
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 장애인구
pop_o2 = pop_o.drop(['시군구', '행정동', '군집'], axis = 1)

def visualize_silhouette(cluster_lists, X_features):

    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import math

    n_cols = len(cluster_lists)

    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)

    for ind, n_cluster in enumerate(cluster_lists):

        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(X_features)

        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)

        y_lower = 10
        axs[ind].set_title('Number of Cluster : '+ str(n_cluster)+'\n' \
                          'Silhouette Score :' + str(round(sil_avg,3)) )
        axs[ind].set_xlabel("The silhouette coefficient values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels==i]
            ith_cluster_sil_values.sort()

            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.Pastel2(float(i) / n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")

visualize_silhouette([2, 3, 4, 5], pop_o2)

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3, random_state=0)
gmm_label = gmm.fit_predict(pop_o2)

pop_o["GMM_군집"] = gmm_label

pop_o.groupby('GMM_군집').mean()

pop_o.to_csv('./군집 결과(장애인구).csv', encoding="cp949")

# 비장애인구
pop_x2 = pop_x.drop(['시군구', '행정동', 'K-means 군집'], axis = 1)

def visualize_silhouette(cluster_lists, X_features):

    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import math

    n_cols = len(cluster_lists)

    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)

    for ind, n_cluster in enumerate(cluster_lists):

        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(X_features)

        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)

        y_lower = 10
        axs[ind].set_title('Number of Cluster : '+ str(n_cluster)+'\n' \
                          'Silhouette Score :' + str(round(sil_avg,3)) )
        axs[ind].set_xlabel("The silhouette coefficient values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels==i]
            ith_cluster_sil_values.sort()

            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.Pastel2(float(i) / n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")

visualize_silhouette([2, 3, 4, 5], pop_x2)

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3, random_state=0)
gmm_label = gmm.fit_predict(pop_x2)

pop_x["GMM_군집"] = gmm_label

pop_x.groupby('GMM_군집').mean()

pop_x.to_csv('./군집 결과(비장애인구).csv', encoding="cp949")

########################################################################

# (4) HKmeans -> coded by R
# # 장애인구
# hkmeans <-hkmeans(정규화장애별, 4)
# names(hkmeans)
# fviz_dend(hkmeans, cex = 0.6)
# fviz_cluster(hkmeans,data = 장애인구시설,geom = "point")
# 장애인구시설$HKmeans<-hkmeans$cluster
# aggregate(cbind(NUMPOINTS,장애인구)~HKmeans,장애인구시설,mean)
#
#
# # 비장애인구
# hkmeans2 <-hkmeans(정규화비장애별, 4)
# names(hkmeans2)
# fviz_dend(hkmeans2, cex = 0.6)
# fviz_cluster(hkmeans2,data = 비장애인구시설,geom = "point")
# 비장애인구시설$HKmeans<-hkmeans2$cluster
# aggregate(cbind(NUMPOINTS,비장애인구)~HKmeans,비장애인구시설,mean)
#
#
# write.csv(비장애인구시설,file="비장애인구시설.csv")
# write.csv(장애인구시설,file="장애인구시설.csv")

################################################################

장애인구1=pd.read_csv('/content/군집 결과.csv',encoding='cp949')
장애인구2=pd.read_csv('/content/장애인구시설.csv',encoding='cp949')
최종장애데이터=pd.merge(장애인구1, 장애인구2, on=['시군구','행정동'])
최종장애데이터=최종장애데이터[['시군구','행정동','NUMPOINTS_x','장애인구_x','K-means 군집','medoid','GMM_군집','HKmeans']]
최종장애데이터.columns = ['시군구','행정동','공공체육시설 수','장애인구','K-means 군집','k-medoids 군집','GMM 군집','HKmeans 군집']

비장애인구1=pd.read_csv('/content/군집 결과(비장애인구).csv',encoding='cp949')
비장애인구2=pd.read_csv('/content/비장애인구시설.csv',encoding='cp949')
비장애인구2.head()
최종비장애데이터=pd.merge(비장애인구1, 비장애인구2, on=['시군구','행정동'])
최종비장애데이터=최종비장애데이터[['시군구','행정동','NUMPOINTS_x','비장애인구_x','K-means 군집','medoid','GMM_군집','HKmeans']]
최종비장애데이터.columns = ['시군구','행정동','공공체육시설 수','비장애인구','K-means 군집','k-medoids 군집','GMM 군집','HKmeans 군집']


# -------------------수정된 부분(여기부터)-------------------------------------------------------------------------------------
# 최적의 군집이란 수요가 많고 공급이 적은 군집 즉, 수요가 많으나 공급이 적은 지역일수록 시급한 지역임을 시사

# 군집분석 시작번호는 Python의 경우 0, R의 경우 1입니다. 그렇기 때문에 PPT에 시각화 자료를 첨부하기 위해서는 R 기준으로
# 작성할 것이므로 Python으로 분석을 진행한 K-means와 GMM 군집번호에 +1를 더해야 합니다.
# 장애인구 K-means 군집: 1, k-medoids 군집: 3, GMM 군집: 1, HKmeans 군집: 3
# 비장애인구 K-means 군집: 0, k-medoids 군집: 3, GMM 군집: 1, HKmeans 군집: 3

# 위의 최적 군집번호는 Python 군집 번호를 조정하기 전이며, 시각화를 진행할 때만 수정할 부분이라 위의 번호(ppt 번호 아닙니다)로 분석을 진행해야 합니다
# 긴급 지수(=인구/공공체육시설)가 높을수록 수요가 많고, 공급이 적은 것을 만족하기에 최적 군집을 선택하는 기준으로 도입

최종데이터 = 최종데이터[(최종데이터['K-means 군집_x']==1) & (최종데이터['k-medoids 군집_x']==3) & (최종데이터['GMM 군집_x']==1) & (최종데이터['HKmeans 군집_x']==3)]
최종데이터 = 최종데이터[(최종데이터['K-means 군집_y']==0) & (최종데이터['k-medoids 군집_y']==3) & (최종데이터['GMM 군집_y']==1) & (최종데이터['HKmeans 군집_y']==3)]

#(수요/공급) 변수: 1개 체육시설에 얼마나 많은 수요가 있을 것인가를 나타내는 지표
#그러나 선정된 17개의 행정동 모두 공급이 없기 때문에, 수요로 우선순위를 판단해야 한다

최종데이터=최종데이터[['시군구','행정동','공공체육시설 수_x','장애인구','비장애인구']]

최종데이터['장애인구 순위']=최종데이터['장애인구'].rank(method='dense',ascending=False)
최종데이터['비장애인구 순위']=최종데이터['비장애인구'].rank(method='dense',ascending=False)
최종데이터['우선 순위']=최종데이터['장애인구 순위']+최종데이터['비장애인구 순위']
우선지역=최종데이터.sort_values('우선 순위')
우선지역
우선지역.to_csv('./우선지역.csv', encoding="cp949")

# 공급(공공체육시설)과 수요(총인구)는 수준 차이가 있을 수 밖에 없으며, 특히 수요 내 장애인과 비장애인에도 차이가 존재
# 국내에서는 장애인구가 증가하는 추세라도 비장애인구가 상대적으로 많은 상황이기에 단순 수치를 이용한다며 상대적으로 적은 장애인의 수요를 반영하기에 한계가 있을 것
# 이에 모두 공급이 없는 지역이므로 수요가 높을수록 설치가 시급한 지역이니 높은 순위를 부여 (ex: 1순위가 가장 설치가 시급)

# 우선 지역 1곳: 상일동(강동구)
# 우선 지역 3곳: 상일동(강동구), 용신동(동대문구), 전농1동(동대문구)
# 우선 지역 4곳: 상일동(강동구), 용신동(동대문구), 전농1동(동대문구), 송중동(강북구)
# 우선 지역 5곳: 상일동(강동구), 용신동(동대문구), 전농1동(동대문구), 송중동(강북구), 장안2동(동대문구)

# 시군구 중복 허용 X 우선 지역 3곳: 상일동(강동구), 용신동(동대문구), 송중동(강북구)
# 시군구 중복 허용 X 우선 지역 4곳: 상일동(강동구), 용신동(동대문구), 송중동(강북구), 성현동(관악구)