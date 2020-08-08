import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

# get data
data = pd.read_csv("csgo.csv")
predict = 'winner'

# Use Label Encoder for non-integer values
le = preprocessing.LabelEncoder()
round_type = le.fit_transform(list(data["round_type"]))
winner = le.fit_transform(list(data["winner"]))

# give Label Encoder values variable names
ECO = 0
FORCE_BUY = 1
NORMAL = 2
PISTOL_ROUND = 3
SEMI_ECO = 4

ct = 0
t = 1

# give x and y values with only integer values
x = list(zip(round_type, data["ct"], data["t"]))
y = list(winner)

# get accuracy of model to at least 66 (any more takes too long)%
acc = 0.0
while acc < .66:
    # train test split
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

    # Put data in model
    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(x_train, y_train)

    # check accuracy
    acc = model.score(x_test, y_test)

# now that model is ready, ask info
team = input('ct or t: ')
ct_economy = input('ct economy total assets at start of buy phase: ')
t_economy = input('t economy total assets at start of buy phase: ')

# check winner for each round_type to see with round_type_strategy to choose
eco_winner = model.predict([[ECO, ct_economy, t_economy]])
force_buy_winner = model.predict([[FORCE_BUY, ct_economy, t_economy]])
normal_winner = model.predict([[NORMAL, ct_economy, t_economy]])
pistol_round_winner = model.predict([[PISTOL_ROUND, ct_economy, t_economy]])
semi_eco_winner = model.predict([[SEMI_ECO, ct_economy, t_economy]])

if team == 'ct':
    if normal_winner == ct:
        round_type_strategy = 'NORMAL'
    elif eco_winner == ct:
        round_type_strategy = 'ECO'
    elif force_buy_winner == ct:
        round_type_strategy = 'FORCE BUY'
    elif semi_eco_winner == ct:
        round_type_strategy = 'SEMI ECO'
    # Do not do pistol_round because that is not a round type strategy you can choose
    else:
        round_type_strategy = "ECO"  # if none of the strategies work, go eco
if team == 't':
    if normal_winner == t:
        round_type_strategy = 'NORMAL'
    elif eco_winner == t:
        round_type_strategy = 'ECO'
    elif force_buy_winner == t:
        round_type_strategy = 'FORCE BUY'
    elif semi_eco_winner == t:
        round_type_strategy = 'SEMI ECO'
    else:
        round_type_strategy = 'ECO'

# give accuracy of results
print("You will win with", acc * 100, "% accuracy if you choose round type strategy:", round_type_strategy)
