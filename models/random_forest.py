# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 01:37:55 2026

@author: AZROULMAHLI Chaimae
"""

from sklearn.ensemble import RandomForestRegressor

class RandomForestModel:

    def __init__(self):
        self.model = RandomForestRegressor(
          n_estimators=50,
          max_depth=10,
          n_jobs=-1,
          random_state=42
        )

    def train(self,X,Y):
        Xf=X.reshape(len(X),-1)
        Yf=Y.reshape(len(Y),-1)
        self.model.fit(Xf,Yf)

    def predict(self,X):
        Xf=X.reshape(len(X),-1)
        return self.model.predict(Xf)