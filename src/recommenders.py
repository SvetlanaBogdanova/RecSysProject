import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, weighting=None, fake_id=999999):
        
        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        if fake_id is not None:
            self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != fake_id]
        
        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        if fake_id is not None:
            self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != fake_id]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()
        #
        self.fake_id = fake_id
        self.user_item_matrix = self._prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, \
            self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)
        self.user_item_matrix = csr_matrix(self.user_item_matrix).tocsr()
        self.user_item_matrix_for_pred = self.user_item_matrix
        if weighting == "bm25":
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T.tocsr()
        elif weighting == "tfidf":
            self.user_item_matrix = tfidf_weight(self.user_item_matrix.T).T.tocsr()
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
     
    @staticmethod
    def _prepare_matrix(data):
        
        user_item_matrix = pd.pivot_table(data, 
                                          index='user_id', columns='item_id', 
                                          values='quantity',
                                          aggfunc='count', 
                                          fill_value=0
                                         )

        return user_item_matrix.astype(float)
    
    @staticmethod
    def _prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(user_item_matrix)
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(user_item_matrix)
        
        return model
    
    def _update_dict(self, user_id):
        """Если появился новый user / item, то нужно обновить словари"""
        
        if user_id not in self.userid_to_id.keys():
            
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1
            
            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})
    
    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)  # Товар похож на себя -> рекомендуем 2 товара
        top_rec = recs[0][1]  # И берем второй (не товар из аргумента метода)
        return self.id_to_itemid[top_rec]   
    
    def _get_similar_users(self, user_id, N=5):
        """Находит пользователей, похожих на user_id"""
        recs = self.model.similar_users(self.userid_to_id[user_id], N=N+1)
        top_rec = recs[0][1:]
        return [self.id_to_userid[rec] for rec in top_rec]
    
    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то полняем их топ-популярными"""
        
        if len(recommendations) < N:
            top_popular = [rec for rec in self.overall_top_purchases[:N] if rec not in recommendations]
            recommendations.extend(top_popular)
            recommendations = recommendations[:N]
        
        return recommendations
    
    def _get_recommendations(self, user, model, N=5):
        """Рекомендации через стандартные библиотеки implicit"""
        
        self._update_dict(user_id=user)
        filter_items = [] if self.fake_id is None else [self.itemid_to_id[self.fake_id]]
        userIndex = self.userid_to_id[user]
        user_items = csr_matrix([]).tocsr() if userIndex >= self.user_item_matrix_for_pred.shape[0] else self.user_item_matrix_for_pred[userIndex]
        res = model.recommend(userid=userIndex,
                              user_items=user_items,
                              N=N,
                              filter_already_liked_items=False,
                              filter_items=filter_items,
                              recalculate_user=True)
        mask = res[1].argsort()[::-1]
        res = [self.id_to_itemid[rec] for rec in res[0][mask]]
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_als_recommendations(self, user, N=5):
        """Рекомендации через стандартные библиотеки implicit"""
        
        return self._get_recommendations(user, model=self.model, N=N)

    def get_own_recommendations(self, user, N=5):
        """Рекомендации товары среди тех, которые юзер уже купил"""
        
        return self._get_recommendations(user, model=self.own_recommender, N=N)

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        
        top_purchases = self.top_purchases[self.top_purchases['user_id'] == user]['item_id'][:N]
        res = [self._get_similar_item(item_id=item_id) for item_id in top_purchases]
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    
        top_users = self._get_similar_users(user_id=user, N=N)
        fakeFilter = self.top_purchases['item_id'] != 999999
        res = []
        for user in top_users:
            items = self.top_purchases[(self.top_purchases['user_id'] == user) & fakeFilter]['item_id'].values
            for item in items:
                if item not in res:
                    res.append(item)
                    break

        res = self._extend_with_top_popular(res, N=N)
        
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
