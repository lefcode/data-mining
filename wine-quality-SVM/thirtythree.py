import pandas as pd
import numpy as np

data = pd.read_csv('winequality-red.csv')
ph = data['pH'].values #retrieve ph column
ph = np.random.choice(ph.astype(int),size =int(ph.size*0.66),replace=False,p = None)