import pandas as pd
import numpy as np
import math
from scipy import stats
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import sys

year = 2018
password = sys.argv[1]
engine = create_engine('mysql+pymysql://shank_evan:' + str(password) + '@marchmadness.czez3i9mtzt9.us-east-1.rds.amazonaws.com:3306/marchmadness')
conn = engine.connect()

# TODO model_data should just be the entire table, if I pull that in, I can create a dataframe off of a restricted columnset. If this is established early
# on, that same columnset can be applied to the round-to round prediction data. This would make feature inclusion/exclusion way easier than editing sql queries
model_data = pd.read_sql("""SELECT (RPI - Opponent_RPI) AS RPI, (Seed-Opponent_Seed) AS Seed, (team_luck - opp_luck) as Luck,
                       (team_kp_rank - opp_kp_rank) AS Rank, (team_adj_o - opp_adj_o) as AdjO, (team_adj_d - opp_adj_d) as AdjD,
                       (team_adj_t - opp_adj_t) as AdjT, (team_opp_o - opp_team_o) as OppO, (team_opp_d - opp_team_d) as OppD,
                       (team_ncsos_adj_em - opp_ncsos_adj_em) as NCSOSAdjEM,(predictor_sag1 - predictor_sag2) as SagPred, 
                       Wins as WinLoss from model_data WHERE Year <""" + str(year), con=conn)

# TODO - oversample 'upsets'. First define what an upset is based on seeding, then oversample that group in model training.
# TODO - train a separate model only on the 'middling' matchups (eliminate 1-16, 2-15, 3-14, 4-13) and provide potential upset picks?

predictor = LogisticRegression(solver='lbfgs')

#randomly assigns data into a test and training set
model_data.loc[:,'is_train'] = np.random.uniform(0, 1, len(model_data)) <= .75
train = model_data.query('is_train == True')
test = model_data.query('is_train == False')

#removes the test & training column from data so it's not included in model features
del model_data['is_train']
del train['is_train']
del test['is_train']

#establishes the set of features to be used in the model, takes all columns except the response (WinLoss)
features = model_data.loc[:, model_data.columns != 'WinLoss'].columns.copy()
x_train, y_train = train[features].copy(), train['WinLoss'].copy()

# TODO - given lack of tournament games, it may be worth looking into oversampling here to see if performance improves
# TODO - normalize features so the feature importance is clearer for interpretability (takes different weights out of picture)
predictor.fit(x_train, y_train)
predictions = predictor.predict(test[features])
y_test = test['WinLoss'].copy()
print(metrics.classification_report(y_test, predictions))

features_values = []
for x in range(len(features)):
    features_values.append([features[x],predictor.coef_[0,x]])
features_values.sort(reverse=True,key=lambda x: x[1])
print(features_values)
print('\n')

#SEED DATA FOR THE MODEL RUN
rounds= {1:"Round of 64", 2:"Round of 32", 3:"Sweet Sixteen", 4:"Elite Eight", 5:"National Semifinals", 6:"National Championship"}
games = {1:32, 2:48, 3:56, 4:60, 5:62, 6:0}
matchups = pd.read_csv('data/' + str(year) + "_matchups.csv")

for i in range(1,7):

    conn.execute("""DROP TABLE IF EXISTS i_matchups""")
    conn.execute("""DROP TABLE IF EXISTS pred_data""")

    i_matchups = matchups.loc[matchups['Round'] == rounds[i]].copy()
    i_matchups.to_sql('i_matchups', conn)

    # TODO tie this into the consolidation in the query used to train the model
    pred_data_create = """CREATE TABLE pred_data AS SELECT * FROM i_matchups a 
                            JOIN team_data b USING(Team) 
                            JOIN opponent_data c USING(Opponent)
                            JOIN kenpom_team d on a.Team = d.team_kp AND a.Year = d.team_year
                            JOIN kenpom_opponent e on a.Opponent = e.opp_kp AND a.Year = e.opp_year
                            JOIN sagarin1 f ON f.team_sag1 = a.Team AND f.year_sag1 = a.Year
                            JOIN sagarin2 g ON g.team_sag2 = a.Opponent AND g.year_sag2 = a.Year"""

    conn.execute(pred_data_create) 

    # TODO tie this into the consolidation in the query used to train the model
    pred_data = pd.read_sql("""SELECT matchup_id, (Team_RPI - Opp_RPI) AS RPI, (Team_Seed-Opp_Seed) AS Seed, (team_luck - opp_luck) as Luck,
                       (team_kp_rank - opp_kp_rank) AS Rank,(team_adj_o - opp_adj_o) as AdjO, (team_adj_d - opp_adj_d) as AdjD,
                       (team_adj_t - opp_adj_t) as AdjT, (team_opp_o - opp_team_o) as OppO, (team_opp_d - opp_team_d) as OppD,
                       (team_ncsos_adj_em - opp_ncsos_adj_em) as NCSOSAdjEM, (predictor_sag1 - predictor_sag2) as SagPred 
                       from pred_data ORDER BY matchup_id ASC;""", con=conn)

    preds = matchups.loc[matchups['Round']==rounds[i]].copy()

    preds.loc[:,'WinProb'] = predictor.predict_proba(pred_data[features])[:,1]
    preds.reset_index(inplace=True)

    win_threshold = 0.5


# TODO - this chunk of code feels hideous. Could definitely use more 'pythonic' string construction at the very least, but could probably be trimmed
# down to a much larger degree if I take a harder look at the logic flow.

    for index, row in preds.iterrows():
        if row['WinProb'] > win_threshold and row['matchup_id']%2 == 1 and len(preds.index)>1:
            print(row['Team'] + " defeats " + row['Opponent'] + " in the " + rounds[i] + " | " + str(round(row['WinProb'],2)))
            matchups.at[(math.floor(index/2)+games[i]), 'Team'] = row['Team']
        elif row['WinProb'] > win_threshold and row['matchup_id']%2 == 0 and len(preds.index)>1:
            print(row['Team'] + " defeats " + row['Opponent'] + " in the " + rounds[i] + " | " + str(round(row['WinProb'],2)))
            matchups.at[(math.floor(index/2)+games[i]), 'Opponent'] = row['Team']
        elif row['WinProb'] <= win_threshold and row['matchup_id']%2 == 1 and len(preds.index)>1:
            print(row['Opponent'] + " defeats " + row['Team'] + " in the " + rounds[i] + " | " + str(round(1-row['WinProb'],2)))
            matchups.at[(math.floor(index/2)+games[i]), 'Team'] = row['Opponent']
        elif row['WinProb'] <= win_threshold and row['matchup_id']%2 == 0 and len(preds.index)>1:
            print(row['Opponent'] + " defeats "  + row['Team'] + " in the " + rounds[i] + " | " + str(round(1-row['WinProb'],2)))
            matchups.at[(math.floor(index/2)+games[i]), 'Opponent'] = row['Opponent']
        elif row['WinProb'] > win_threshold:
            print(row['Team'] + " defeats "  + row['Opponent'] + " in the " + rounds[i] + " | " + str(round(row['WinProb'],2)))
        else:
            print(row['Opponent'] + " defeats "  + row['Team'] + " in the " + rounds[i] + " | Prob: " + str(round(1-row['WinProb'],2)))

    print('\n')