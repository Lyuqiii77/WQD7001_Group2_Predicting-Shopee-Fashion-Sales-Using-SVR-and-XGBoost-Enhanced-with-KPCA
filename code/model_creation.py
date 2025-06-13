from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import sklearn as sk
# from collections import deque
# from graphviz import Digraph
import pandas as pd
# import numpy as np
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

# Create NODE FOR XGBOOST AND TREE OF XGBOOST


class Node:

    def __init__(self, model=None, value=None):
        self.model = model
        self.value = value
        self.left = None
        self.right = None


class TreeDecision_XGBOOST:
    def __init__(self):
        self.root = None

    def insert(self, data, model=None):
        if self.root is None:
            self.root = Node(model=model, value=data)
        else:
            self._insert_recursive(self.root, data, model)

    def _insert_recursive(self, current_node, data, model):
        if data < current_node.value:
            if current_node.left is None:
                current_node.left = Node(model=model, value=data)
            else:
                self._insert_recursive(current_node.left, data, model)
        else:
            if current_node.right is None:
                current_node.right = Node(model=model, value=data)
            else:
                self._insert_recursive(current_node.right, data, model)

    def insert_leaf_model(self, model, value):

        def _insert_leaf(node, model, value):
            if value < node.value:
                if node.left is None:
                    node.left = Node(model=model, value='<' + str(node.value))
                else:
                    _insert_leaf(node.left, model, value)
            else:
                if node.right is None:
                    node.right = Node(model=model, value='>=' + str(node.value)
                                      )
                else:
                    _insert_leaf(node.right, model, value)

        if self.root is not None:
            _insert_leaf(self.root, model, value)

    def train_structure_decomposition(self, data_frame):
        features = ['price_ori', 'item_rating',
                    'price_actual', 'total_rating',
                    'favorite', 'discount'
                    ]

        def train_leaf_model(node, df):
            if node is None:
                return
            if node.left is None and node.right is None:
                if len(df) == 0:
                    return
                X = df[features]
                y = df['total_sold']
                liability = sk.model_selection.train_test_split(
                    X, y, test_size=0.33
                    )
                X_train, X_test, y_train, y_test = liability
                model = XGBRegressor(n_estimators=100)
                model.fit(X_train, y_train)
                node.model = model
            else:
                # Internal node: split and continue
                try:
                    threshold = float(node.value)
                    left_df = df[df['price_ori'] < threshold]
                    right_df = df[df['price_ori'] >= threshold]
                    train_leaf_model(node.left, left_df)
                    train_leaf_model(node.right, right_df)
                except ValueError:
                    return

        if self.root is not None:
            train_leaf_model(self.root, data_frame)

    def prediction(self, df):
        predictions = []

        def choosing_model(node, df):
            if node is None or len(df) == 0:
                return
            if node.left is None and node.right is None:
                #  Leaf node: make predictions and store with index
                """X_to_predict = df[['price_ori', 
                'item_rating', 'price_actual'
                , 'total_rating', 'favorite', 'discount']]
                y_pred = node.model.predict(X_to_predict)"""
                y_pred = node.model.predict(df)
                result_df = pd.DataFrame({
                    'index': df.index,
                    'prediction': y_pred
                })
                predictions.append(result_df)
            else:
                # Internal node: split and continue
                try:
                    threshold = float(node.value)
                    left_df = df[df['price_ori'] < threshold]
                    right_df = df[df['price_ori'] >= threshold]
                    choosing_model(node.left, left_df)
                    choosing_model(node.right, right_df)
                except ValueError:
                    return
        if self.root is not None:
            choosing_model(self.root, df)
        if predictions:
            # Combine all predictions and sort by original index
            final_df = pd.concat(predictions).set_index('index')
            return final_df
        else:
            return pd.DataFrame(columns=['prediction'])


# Create NODE FOR XGBOOST AND TREE OF XGBOOST COUPLED WITH PCA
class Node_PCA:
    def __init__(self, model=None, value=None):
        self.left = None
        self.right = None
        self.model = model
        self.value = value
        self.scaler = None
        self.pca = None


class TreeDecision_XGBOOST_PCA:
    def __init__(self):
        self.root = None

    def insert(self, data, model=None):
        if self.root is None:
            self.root = Node_PCA(model=model, value=data)
        else:
            self._insert_recursive(self.root, data, model)

    def _insert_recursive(self, current_node, data, model):
        if data < current_node.value:
            if current_node.left is None:
                current_node.left = Node_PCA(model=model, value=data)
            else:
                self._insert_recursive(current_node.left, data, model)
        else:
            if current_node.right is None:
                current_node.right = Node_PCA(model=model, value=data)
            else:
                self._insert_recursive(current_node.right, data, model)

    def insert_leaf_model(self, model, value):

        def _insert_leaf(node, model, value):
            if value < node.value:
                if node.left is None:
                    node.left = Node_PCA(
                        model=model, value='<' + str(node.value)
                        )
                else:
                    _insert_leaf(node.left, model, value)
            else:
                if node.right is None:
                    node.right = Node_PCA(
                        model=model, value='>=' + str(node.value)
                        )
                else:
                    _insert_leaf(node.right, model, value)

        if self.root is not None:
            _insert_leaf(self.root, model, value)

    def train_structure_decomposition(self, data_frame):
        features = ['price_ori', 'item_rating',
                    'price_actual', 'total_rating',
                    'favorite', 'discount']

        def train_leaf_model(node, df):
            if node is None:
                return
            if node.left is None and node.right is None:
                if len(df) == 0:
                    return
                X = df[features]
                y = df['total_sold']
                liability = sk.model_selection.train_test_split(X,
                                                                y,
                                                                test_size=0.33)
                scaler = StandardScaler()
                X_train, X_test, y_train, y_test = liability
                df_scaled = scaler.fit_transform(X_train)
                pca = PCA(n_components=5)  # Choose number of components
                pca_result = pca.fit_transform(df_scaled)

                # 3. Convert to DataFrame
                df_pca = pd.DataFrame(pca_result)
                df_pca.index = X_train.index
                X_train = df_pca
                model = XGBRegressor(n_estimators=100)
                model.fit(X_train, y_train)
                node.model = model
                node.scaler = scaler
                node.pca = pca
            else:
                # Internal node: split and continue
                try:
                    threshold = float(node.value)
                    left_df = df[df['price_ori'] < threshold]
                    right_df = df[df['price_ori'] >= threshold]
                    train_leaf_model(node.left, left_df)
                    train_leaf_model(node.right, right_df)
                except ValueError:
                    return

        if self.root is not None:
            train_leaf_model(self.root, data_frame)

    def prediction(self, df):
        predictions = []

        def choosing_model(node, df):
            if node is None or len(df) == 0:
                return
            if node.left is None and node.right is None:
                df_scaled = node.scaler.transform(df)
                df_pca = node.pca.transform(df_scaled)
                df_pca = pd.DataFrame(df_pca, index=df.index)

                y_pred = node.model.predict(df_pca)
                result_df = pd.DataFrame({
                    'index': df.index,
                    'prediction': y_pred
                })
                predictions.append(result_df)
            else:
                # Internal node: split and continue
                try:
                    threshold = float(node.value)
                    left_df = df[df['price_ori'] < threshold]
                    right_df = df[df['price_ori'] >= threshold]
                    choosing_model(node.left, left_df)
                    choosing_model(node.right, right_df)
                except ValueError:
                    return
        if self.root is not None:
            choosing_model(self.root, df)
        if predictions:
            # Combine all predictions and sort by original index
            final_df = pd.concat(predictions).set_index('index')
            return final_df
        else:
            return pd.DataFrame(columns=['prediction'])


# Create NODE FOR XGBOOST AND TREE OF XGBOOST WITH KPCA
class Node_KPCA:
    def __init__(self, model=None, value=None):
        self.left = None
        self.right = None
        self.model = model
        self.value = value
        self.scaler = None
        self.pca = None


class TreeDecision_XGBOOST_KPCA:
    def __init__(self):
        self.root = None

    def insert(self, data, model=None):
        if self.root is None:
            self.root = Node_KPCA(model=model, value=data)
        else:
            self._insert_recursive(self.root, data, model)

    def _insert_recursive(self, current_node, data, model):
        if data < current_node.value:
            if current_node.left is None:
                current_node.left = Node_KPCA(model=model, value=data)
            else:
                self._insert_recursive(current_node.left, data, model)
        else:
            if current_node.right is None:
                current_node.right = Node_KPCA(model=model, value=data)
            else:
                self._insert_recursive(current_node.right, data, model)

    def insert_leaf_model(self, model, value):

        def _insert_leaf(node, model, value):
            if value < node.value:
                if node.left is None:
                    node.left = Node_KPCA(
                        model=model, value='<' + str(node.value)
                        )
                else:
                    _insert_leaf(node.left, model, value)
            else:
                if node.right is None:
                    node.right = Node_KPCA(
                        model=model, value='>=' + str(node.value)
                        )
                else:
                    _insert_leaf(node.right, model, value)

        if self.root is not None:
            _insert_leaf(self.root, model, value)

    def train_structure_decomposition(self, data_frame):
        features = ['price_ori', 'item_rating',
                    'price_actual', 'total_rating',
                    'favorite', 'discount']

        def train_leaf_model(node, df):
            if node is None:
                return
            if node.left is None and node.right is None:
                if len(df) == 0:
                    return
                X = df[features]
                y = df['total_sold']
                liability = sk.model_selection.train_test_split(X,
                                                                y,
                                                                test_size=0.33)
                scaler = StandardScaler()
                X_train, X_test, y_train, y_test = liability
                df_scaled = scaler.fit_transform(X_train)
                pca = KernelPCA(n_components=5)  # Choose number of components
                pca_result = pca.fit_transform(df_scaled)

                # 3. Convert to DataFrame
                df_pca = pd.DataFrame(pca_result)
                df_pca.index = X_train.index
                X_train = df_pca
                model = XGBRegressor(n_estimators=100)
                model.fit(X_train, y_train)
                node.model = model
                node.scaler = scaler
                node.pca = pca
            else:
                # Internal node: split and continue
                try:
                    threshold = float(node.value)
                    left_df = df[df['price_ori'] < threshold]
                    right_df = df[df['price_ori'] >= threshold]
                    train_leaf_model(node.left, left_df)
                    train_leaf_model(node.right, right_df)
                except ValueError:
                    return

        if self.root is not None:
            train_leaf_model(self.root, data_frame)

    def prediction(self, df):
        predictions = []

        def choosing_model(node, df):
            if node is None or len(df) == 0:
                return
            if node.left is None and node.right is None:
                df_scaled = node.scaler.transform(df)
                df_pca = node.pca.transform(df_scaled)
                df_pca = pd.DataFrame(df_pca, index=df.index)

                y_pred = node.model.predict(df_pca)
                result_df = pd.DataFrame({
                    'index': df.index,
                    'prediction': y_pred
                })
                predictions.append(result_df)
            else:
                # Internal node: split and continue
                try:
                    threshold = float(node.value)
                    left_df = df[df['price_ori'] < threshold]
                    right_df = df[df['price_ori'] >= threshold]
                    choosing_model(node.left, left_df)
                    choosing_model(node.right, right_df)
                except ValueError:
                    return
        if self.root is not None:
            choosing_model(self.root, df)
        if predictions:
            # Combine all predictions and sort by original index
            final_df = pd.concat(predictions).set_index('index')
            return final_df
        else:
            return pd.DataFrame(columns=['prediction'])


# #Create NODE FOR SVR AND TREE OF SVR
class Node_SVR:
    def __init__(self, model_bundle=None, value=None):
        self.model_bundle = model_bundle
        self.value = value
        self.left = None
        self.right = None


class TreeDecision_SVR:
    def __init__(self):
        self.root = None

    def insert(self, data, model_bundle=None):
        if self.root is None:
            self.root = Node_SVR(model_bundle=model_bundle, value=data)
        else:
            self._insert_recursive(self.root, data, model_bundle)

    def _insert_recursive(self, current_node, data, model_bundle):
        if data < current_node.value:
            if current_node.left is None:
                current_node.left = Node_SVR(model_bundle=model_bundle,
                                             value=data)
            else:
                self._insert_recursive(current_node.left, data, model_bundle)
        else:
            if current_node.right is None:
                current_node.right = Node_SVR(model_bundle=model_bundle,
                                              value=data
                                              )
            else:
                self._insert_recursive(current_node.right, data, model_bundle)

    def insert_leaf_model(self, model_bundle, value):
        def _insert_leaf(node, mb, val):
            if val < node.value:
                if node.left is None:
                    node.left = Node_SVR(
                        model_bundle=mb, value='<' + str(node.value)
                        )
                else:
                    _insert_leaf(node.left, mb, val)
            else:
                if node.right is None:
                    node.right = Node_SVR(
                        model_bundle=mb, value='>=' + str(node.value)
                        )
                else:
                    _insert_leaf(node.right, mb, val)

        if self.root is not None:
            _insert_leaf(self.root, model_bundle, value)

    def train_structure_decomposition(self, data_frame):
        features = ['price_ori', 'item_rating',
                    'price_actual', 'total_rating',
                    'favorite', 'discount']
        MIN_SAMPLES_FOR_SVR_LEAF = 5  # Minimum samples to attempt SVR training

        def train_leaf_model(node, df):
            if node is None:
                return

            if node.left is None and node.right is None:  # Leaf node
                if len(df) < MIN_SAMPLES_FOR_SVR_LEAF:
                    # Check for minimum samples
                    node.model_bundle = None  # Not enough data to train
                    return

                X = df[features]
                y = df['total_sold']

                if X.empty or y.empty:
                    # Should be caught by len(df) check already
                    node.model_bundle = None
                    return

                scaler_X = StandardScaler()
                scaler_y = StandardScaler()

                try:
                    X_scaled = scaler_X.fit_transform(X)
                    y_scaled = scaler_y.fit_transform(
                        y.values.reshape(-1, 1)
                        ).ravel()
                except ValueError:
                    node.model_bundle = None
                    return

                svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
                try:
                    svr_model.fit(X_scaled, y_scaled)
                    node.model_bundle = {
                        'model': svr_model,
                        'scaler_X': scaler_X,
                        'scaler_y': scaler_y
                    }
                except Exception:
                    node.model_bundle = None
                return

            # Internal node: split and continue
            try:
                threshold = float(node.value)
                left_df = df[df['price_ori'] < threshold]
                right_df = df[df['price_ori'] >= threshold]
                train_leaf_model(node.left, left_df)
                train_leaf_model(node.right, right_df)
            except ValueError:
                return

        if self.root is not None:
            train_leaf_model(self.root, data_frame)

    def prediction(self, df_input):
        predictions = []
        features = ['price_ori',
                    'item_rating', 'price_actual',
                    'total_rating', 'favorite',
                    'discount']

        def choosing_model(node, current_df_subset):
            if node is None or len(current_df_subset) == 0:
                return

            if node.left is None and node.right is None:
                # Leaf node: make predictions and store with index
                if node.model_bundle and node.model_bundle.get('model'):
                    svr_model = node.model_bundle['model']
                    scaler_X_leaf = node.model_bundle['scaler_X']
                    scaler_y_leaf = node.model_bundle['scaler_y']

                    X_to_predict = current_df_subset[features]
                    if X_to_predict.empty:
                        # Should be caught by len(current_df_subset)
                        return

                    try:
                        X_to_predict_scaled = scaler_X_leaf.transform(
                            X_to_predict
                            )
                        y_pred_scaled = svr_model.predict(X_to_predict_scaled)
                        y_pred = scaler_y_leaf.inverse_transform(
                            y_pred_scaled.reshape(-1, 1)
                            )

                        result_df = pd.DataFrame({
                            'index': current_df_subset.index,
                            'prediction': y_pred.ravel()
                        })
                        predictions.append(result_df)
                    except Exception:
                        pass
                return

            # Internal node: split and continue
            try:
                threshold = float(node.value)
                left_df_subset = current_df_subset[
                    current_df_subset['price_ori'] < threshold
                    ]
                right_df_subset = current_df_subset[
                    current_df_subset['price_ori'] >= threshold
                    ]
                choosing_model(node.left, left_df_subset)
                choosing_model(node.right, right_df_subset)
            except ValueError:
                return

        if self.root is not None:
            choosing_model(self.root, df_input)

        if predictions:
            final_df = pd.concat(predictions).set_index('index').sort_index()
            return final_df
        else:
            return pd.DataFrame(columns=['prediction'])
