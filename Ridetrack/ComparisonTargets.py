import time
import numpy as np
import pandas as pd
from sklearn import svm, tree, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

# Try importing Keras components, but make them optional or at least handle missing imports
try:
    from tensorflow.keras.layers import Input, Conv2D, Lambda, LSTM, Dropout, Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from tensorflow.keras.optimizers import Adam
    import tensorflow.keras.backend as K
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False

class ComparisonTargets:

    def Introduction(self):
        """
        Function: Introduction to how to use this class and its methods.

        Note: This function will print out the guide to console.
        """
        intro = """

       ╔═╗ ╔╦═══╦╗ ╔╦╗ ╔╗╔═══╦═══╗╔╗╔═╦═══╦═══╦═══╦═══╗╔╗
       ║║╚╗║║╔═╗║║ ║║║ ║║║╔══╣╔══╝║║║╔╩╗╔╗╠╗╔╗║╔═╗║╔═╗╠╝║
       ║╔╗╚╝║║ ╚╣╚═╝║║ ║║║╚══╣╚══╗║╚╝╝ ║║║║║║║╠╝╔╝╠╝╔╝╠╗║
       ║║╚╗║║║ ╔╣╔═╗║║ ║║║╔══╣╔══╝║╔╗║ ║║║║║║║║ ║╔╬═╝╔╝║║
       ║║ ║║║╚═╝║║ ║║╚═╝║║╚══╣╚══╗║║║╚╦╝╚╝╠╝╚╝║ ║║║║╚═╦╝╚╗
       ╚╝ ╚═╩═══╩╝ ╚╩═══╝╚═══╩═══╝╚╝╚═╩═══╩═══╝ ╚╝╚═══╩══╝
                  ╔═══╗  ╔╗  ╔════╗       ╔╗
                  ║╔═╗║  ║║  ║╔╗╔╗║       ║║
                  ║╚═╝╠╦═╝╠══╬╝║║╚╬═╦══╦══╣║╔╗
                  ║╔╗╔╬╣╔╗║║═╣ ║║ ║╔╣╔╗║╔═╣╚╝╝
                  ║║║╚╣║╚╝║║═╣ ║║ ║║║╔╗║╚═╣╔╗╗
                  ╚╝╚═╩╩══╩══╝ ╚╝ ╚╝╚╝╚╩══╩╝╚╝
        ╭━━━╮                      ╭━━━━╮         ╭╮
        ┃╭━╮┃                      ┃╭╮╭╮┃        ╭╯╰╮
        ┃┃ ╰╋━━┳╮╭┳━━┳━━┳━┳┳━━┳━━┳━╋╯┃┃┣┻━┳━┳━━┳━┻╮╭╋━━╮
        ┃┃ ╭┫╭╮┃╰╯┃╭╮┃╭╮┃╭╋┫━━┫╭╮┃╭╮╮┃┃┃╭╮┃╭┫╭╮┃┃━┫┃┃━━┫
        ┃╰━╯┃╰╯┃┃┃┃╰╯┃╭╮┃┃┃┣━━┃╰╯┃┃┃┃┃┃┃╭╮┃┃┃╰╯┃┃━┫╰╋━━┃
        ╰━━━┻━━┻┻┻┫╭━┻╯╰┻╯╰┻━━┻━━┻╯╰╯╰╯╰╯╰┻╯╰━╮┣━━┻━┻━━╯
                  ┃┃                        ╭━╯┃
                  ╰╯                        ╰━━╯

        歡迎使用 RideTrack ComparisonTargets 功能！
        
        這個Class包含以下功能：
        
        1. compare_ml_models: 傳統機器學習演算法(SnapShot)。
           用法：compare_ml_models(self, train_dataset, test_dataset, feature, save_path=None)
        
        2. compare_m2_models: 傳統機器學習演算法(Window)。
           用法：compare_m2_models(self, train_dataset, test_dataset, feature, window_size, save_path=None)
        
        3. compare_m3_models: DeepConvLSTM   ( Python 3.6 )
           用法：compute_accuracy(self, dataset, frequency, save_path=None)
       
        """
        print(intro)


    # 儲存檔案使用
    @staticmethod
    def _save_dataframe(df, path):
        """
        Function: Save dataframe to csv.

        Parameters:
            df: The dataframe to be saved.
            path: The path to save the dataframe.
        """
        df.to_csv(path, index=False)

    # 計算執行時間
    @staticmethod
    def _print_execution_time(start_time):
        """
        Function: Print the execution time from start_time to now.

        Parameters:
            start_time: The start time of execution.
        """
        # Compute and print the execution time
        execution_time = time.time() - start_time
        hours, rem = divmod(execution_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Execution time: {hours} hours {minutes} minutes {seconds} seconds")


    def __init__(self):
        self.models = [
            ('Support Vector Machines', svm.SVC()), 
            ('Nearest Neighbors', KNeighborsClassifier(n_neighbors=6)), 
            ('Decision Trees', tree.DecisionTreeClassifier()), 
            ('Forests of randomized trees', RandomForestClassifier(n_estimators=10)), 
            ('Neural Network models', MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100,), random_state=42, activation='relu')),
            ('GaussianProcess', GaussianProcessClassifier())
        ]
        
        self.EPOCH = 10
        self.BATCH_SIZE = 16
        self.LSTM_UNITS = 32
        self.CNN_FILTERS = 3
        self.LEARNING_RATE = 0.001
        self.PATIENCE = 20
        self.SEED = 0
        self.DROPOUT = 0.1
        
        # Mapping for Keras components if available
        if HAS_KERAS:
            self.ModelCheckpoint = ModelCheckpoint
            self.EarlyStopping = EarlyStopping
            self.optimizers = type('obj', (object,), {'Adam': Adam})

    def _train_and_evaluate(self, model, train_dataset, test_dataset, feature):
        model_name, model_instance = model
        model_instance.fit(train_dataset[feature], train_dataset['Action'])
        test_predict = model_instance.predict(test_dataset[feature])
        acc = metrics.accuracy_score(test_dataset['Action'], test_predict)
        f1 = f1_score(test_dataset['Action'], test_predict, average='weighted')
        recall = recall_score(test_dataset['Action'], test_predict, average='weighted')
        confusion_mat = confusion_matrix(test_dataset['Action'], test_predict)
        return model_name, acc, f1, recall, confusion_mat

    def compare_ml_models(self, train_dataset, test_dataset, feature, save_path=None):
        start_time = time.time()
        results = []
        confusion_matrices = {}
        for model in self.models:
            model_name, acc, f1, recall, confusion_mat = self._train_and_evaluate(model, train_dataset, test_dataset, feature)
            results.append((model_name, acc, f1, recall))
            confusion_matrices[model_name] = confusion_mat

        results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1_Score', 'Recall'])
        if save_path:
            self._save_dataframe(results_df, save_path)

        self._print_execution_time(start_time)
        return results_df, confusion_matrices

    def create_windows(self, data, feature, window_size):
        windows = []
        labels = []
        for i in range(window_size, len(data)):
            windows.append(data[feature][i-window_size:i].values)
            labels.append(data['Action'].iloc[i])
        return np.array(windows), np.array(labels)

    def compare_m2_models(self, train_dataset, test_dataset, feature, window_size, save_path=None):
        start_time = time.time()
        train_data, train_labels = self.create_windows(train_dataset, feature, window_size)
        test_data, test_labels = self.create_windows(test_dataset, feature, window_size)

        train_data = train_data.reshape((train_data.shape[0], -1))
        test_data = test_data.reshape((test_data.shape[0], -1))
 
        results = []
        confusion_matrices = {}
        for model in self.models:
            model_name, model_instance = model
            model_instance.fit(train_data, train_labels)
            test_predict = model_instance.predict(test_data)
            acc = metrics.accuracy_score(test_labels, test_predict)
            f1 = f1_score(test_labels, test_predict, average='weighted')
            recall = recall_score(test_labels, test_predict, average='weighted')
            confusion_mat = confusion_matrix(test_labels, test_predict)
            results.append((model_name, acc, f1, recall))
            confusion_matrices[model_name] = confusion_mat

        results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1_Score', 'Recall'])
        if save_path:
            self._save_dataframe(results_df, save_path)

        self._print_execution_time(start_time)
        return results_df, confusion_matrices

    def model(self, x_train, num_labels, LSTM_units, dropout, num_conv_filters, batch_size):
        if not HAS_KERAS:
            raise ImportError("Tensorflow/Keras is required for this model.")
            
        cnn_inputs = Input(batch_shape=(batch_size, x_train.shape[1], x_train.shape[2], 1), name='rnn_inputs')
        cnn_layer = Conv2D(num_conv_filters, kernel_size = (1, x_train.shape[2]), strides=(1, 1), padding='valid', data_format="channels_last")
        cnn_out = cnn_layer(cnn_inputs)

        sq_layer = Lambda(lambda x: K.squeeze(x, axis = 2))
        sq_layer_out = sq_layer(cnn_out)

        rnn_layer = LSTM(LSTM_units, return_sequences=False, name='lstm')
        rnn_layer_output = rnn_layer(sq_layer_out)

        dropout_layer = Dropout(rate = dropout)
        dropout_layer_output = dropout_layer(rnn_layer_output)

        dense_layer = Dense(num_labels, activation = 'softmax')
        dense_layer_output = dense_layer(dropout_layer_output)

        model = Model(inputs=cnn_inputs, outputs=dense_layer_output)

        print (model.summary())

        return model

    def compare_m3_models(self, train_data ,model_path=None, save_path=None):
        if not HAS_KERAS:
            raise ImportError("Tensorflow/Keras is required for compare_m3_models.")
            
        tmp = np.load(train_data, allow_pickle=True)
        X = tmp['X']
        X = np.squeeze(X, axis = 1)
        y_one_hot = tmp['y']
        folds = tmp['folds']

        print (y_one_hot.shape)

        NUM_LABELS = y_one_hot.shape[1]

        y = np.argmax(y_one_hot, axis=1)

        results = {
            'Seed': [],
            'DataFile': [],
            'Fold': [],
            'EarlyStoppingEpoch': [],
            'AllTrainableCount': [],
            'Accuracy': [],
            'MAE': [],
            'Recall': [],
            'F1': []
        }


        for i in range(0, len(folds)):
            train_idx = folds[i][0]
            test_idx = folds[i][1]

            X_train, y_train, y_train_one_hot = X[train_idx], y[train_idx], y_one_hot[train_idx]
            X_test, y_test, y_test_one_hot = X[test_idx], y[test_idx], y_one_hot[test_idx]

            X_train_ = np.expand_dims(X_train, axis = 3)
            X_test_ = np.expand_dims(X_test, axis = 3)
            
            train_trailing_samples =  X_train_.shape[0]%self.BATCH_SIZE
            test_trailing_samples =  X_test_.shape[0]%self.BATCH_SIZE

            if train_trailing_samples!= 0:
                X_train_ = X_train_[0:-train_trailing_samples]
                y_train_one_hot = y_train_one_hot[0:-train_trailing_samples]
                y_train = y_train[0:-train_trailing_samples]
            if test_trailing_samples!= 0:
                X_test_ = X_test_[0:-test_trailing_samples]
                y_test_one_hot = y_test_one_hot[0:-test_trailing_samples]
                y_test = y_test[0:-test_trailing_samples]

            print (y_train.shape, y_test.shape)   

            rnn_model = self.model(x_train = X_train_, num_labels = NUM_LABELS, LSTM_units = self.LSTM_UNITS, \
                dropout = self.DROPOUT, num_conv_filters = self.CNN_FILTERS, batch_size = self.BATCH_SIZE)
            
            model_filename = (model_path or "") + 'model_baseline_'+ str(i) + '.h5'
            callbacks = [self.ModelCheckpoint(filepath=model_filename, monitor = 'val_acc', save_weights_only=True, save_best_only=True), self.EarlyStopping(monitor='val_acc', patience=self.PATIENCE)]

            opt = self.optimizers.Adam(clipnorm=1.)

            rnn_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            history = rnn_model.fit(X_train_, y_train_one_hot, epochs=self.EPOCH, batch_size=self.BATCH_SIZE, verbose=1, callbacks=callbacks, validation_data=(X_test_, y_test_one_hot))

            early_stopping_epoch = callbacks[1].stopped_epoch - self.PATIENCE + 1 
            print('Early stopping epoch: ' + str(early_stopping_epoch))

            if early_stopping_epoch < 0:
                early_stopping_epoch = -100

            # Evaluate model and predict data on TEST 
            print("******Evaluating TEST set*********")
            rnn_model.load_weights(model_filename)
            y_test_predict = rnn_model.predict(X_test_, batch_size = self.BATCH_SIZE)
            y_test_predict = np.array(y_test_predict)
            y_test_predict = np.argmax(y_test_predict, axis=1)

            all_trainable_count = int(np.sum([K.count_params(p) for p in set(rnn_model.trainable_weights)]))
            
            acc_fold = accuracy_score(y_test, y_test_predict)
            recall_fold = recall_score(y_test, y_test_predict, average='macro')
            f1_fold  = f1_score(y_test, y_test_predict, average='macro')
            
            MAE = metrics.mean_absolute_error(y_test, y_test_predict, sample_weight=None, multioutput='uniform_average')

            results['Fold'].append(i)
            results['EarlyStoppingEpoch'].append(early_stopping_epoch)
            results['AllTrainableCount'].append(all_trainable_count)
            results['Accuracy'].append(acc_fold)
            results['MAE'].append(MAE)
            results['Recall'].append(recall_fold)
            results['F1'].append(f1_fold)
        

        results['Accuracy'].append(np.mean(results['Accuracy']))
        results['Recall'].append(np.mean(results['Recall']))
        results['F1'].append(np.mean(results['F1']))

        return results
