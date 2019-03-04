# Imports - these should be in requirements.txt of virtualenv
from lxml import html
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
import sys

# set flags to run data imports
kprun = True
sagrun = True
aggregate = True
run_year = 2018

# connect to RDS instance
password = sys.argv[1]
engine = create_engine('mysql+pymysql://shank_evan:' + str(password) + '@marchmadness.czez3i9mtzt9.us-east-1.rds.amazonaws.com:3306/marchmadness')
conn = engine.connect()

#############################################################
##################### Scrape Kenpom #########################
#############################################################

if(kprun == True):
    conn.execute("""DROP TABLE IF EXISTS kenpom_team""")
    conn.execute("""DROP TABLE IF EXISTS kenpom_opponent""")
    print('looping kenpom')
    year = 2003
    page_base = 'https://kenpom.com/index.php?y='

    # creates the dictionary that will clean up name mismatches with the game by game file
    kp_name_map = pd.read_csv('data/kenpom_name_map.csv')
    kp_name_dict = dict(zip(kp_name_map['Kenpom'], kp_name_map['Matchups']))

    while (year <= run_year) and (kprun == True):

        page_year = page_base + str(year)
        page = requests.get(page_year)

        tree = html.fromstring(page.content)

        stats = tree.xpath('//td/text()')
        trimstats = [i for i in stats if i != ' ']

        stats_df = pd.DataFrame(np.array(trimstats).reshape(len(trimstats)//11, 11),
        columns = ['team_kp_rank', 'winloss_kp', 'team_adj_em', 'team_adj_o', 'team_adj_d', 'team_adj_t', 'team_luck', 'team_opp_adj_em',	'team_opp_o',
                'team_opp_d', 'team_ncsos_adj_em'])

        teams_conf = tree.xpath('//td/a/text()')
        teams_conf_df = pd.DataFrame(np.array(teams_conf).reshape(len(teams_conf)//2, 2), columns = ['team_kp', 'team_conf'])

        insert_data1 = pd.concat([stats_df, teams_conf_df], axis=1) 
        insert_data1['team_year'] = year

        insert_data1['team_win'], insert_data1['team_loss'] = zip(*insert_data1['winloss_kp'].apply(lambda x: x.split('-')))
        insert_data1.drop(['winloss_kp'], axis=1, inplace=True)

        insert_data1 = insert_data1.apply(pd.to_numeric, errors='ignore')
        insert_data1['team_kp'] = insert_data1['team_kp'].replace(kp_name_dict)

        insert_data2 = insert_data1.rename(columns={'team_kp_rank':'opp_kp_rank', 'team_win':'opp_win', 'team_adj_em':'opp_adj_em', 'team_adj_o':'opp_adj_o', 
                                                    'team_adj_d':'opp_adj_d', 'team_adj_t':'opp_adj_t', 'team_luck':'opp_luck', 'team_opp_adj_em':'opp_team_adj_em',
                                                    'team_opp_o':'opp_team_o', 'team_opp_d':'opp_team_d', 'team_ncsos_adj_em':'opp_ncsos_adj_em', 'team_loss':'opp_loss',
                                                    'team_year':'opp_year', 'team_kp':'opp_kp', 'team_conf':'opp_conf'})

        insert_data1.to_sql('kenpom_team', conn, if_exists='append', index=False)
        insert_data2.to_sql('kenpom_opponent', conn, if_exists='append', index=False)
        
        year += 1

    print('kenpom tables added')

#############################################################
##################### Scrape Sagarin ########################
#############################################################

# TODO Remove parentheses from the schedule and schedule rank data
# TODO Consolidate looping structure here. The 'chunks' are treated the same for most of the data, too much copy/paste going on here.

if(sagrun == True):
    year = 2003
    page_base = 'https://www.usatoday.com/sports/ncaab/sagarin/'
    print('looping sagarin')
    conn.execute("""DROP TABLE IF EXISTS sagarin1""")
    conn.execute("""DROP TABLE IF EXISTS sagarin2""")

    # creates the dictionary that will clean up name mismatches with the game by game file
    sag_name_map = pd.read_csv('data/sagarin_name_map.csv')
    sag_name_dict = dict(zip(sag_name_map['Sagarin'], sag_name_map['Matchups']))

    while (year <= run_year) and (sagrun == True):
        
        clean_sagarin = pd.DataFrame()
            
        page_year = page_base + str(year) + '/team'
        page = requests.get(page_year)

        tree = html.fromstring(page.content)

        if(year <= 2013):
            stats = tree.xpath('//div[@class="sagarin-page"]//pre[2]//text()')
            trimmed_sagarin = []
            for index, val in enumerate(stats):
                if('&nbsp' in val and '=' in val):
                    i = 0
                    while i < 6:
                        trimmed_sagarin.append(stats[index+i])
                        i+=1
            
            # Initial dataframe constructed, still will require some 
            sagarin_df = pd.DataFrame(np.array(trimmed_sagarin).reshape(len(trimmed_sagarin)//6, 6),
                columns = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6'])

            ## Chunk 1: Name & Rank ##
            clean_sagarin['team_sag1'] = sagarin_df['Col1'].str.replace('=','')
            clean_sagarin['team_sag1'] = clean_sagarin['team_sag1'].str.replace('&nbsp','')
            clean_sagarin['team_sag1'] = clean_sagarin['team_sag1'].str.replace('$','')
            clean_sagarin['team_sag1'] = clean_sagarin['team_sag1'].str.replace('*','')
            clean_sagarin['team_sag1'] = clean_sagarin['team_sag1'].str.strip()
            clean_sagarin[['rank', 'team_sag1']] = clean_sagarin['team_sag1'].str.split(" ", 1, expand=True)
            clean_sagarin['team_sag1'] = clean_sagarin['team_sag1'].str.strip()
            del(clean_sagarin['rank'])

            ## Chunk 2 ##
            clean_sagarin[['blank', 'rating_sag1']]  = sagarin_df['Col2'].str.split(expand=True)
            clean_sagarin.drop(['blank'], axis=1, inplace=True)

            ## Chunk 3 ##
            clean_sagarin[['blank', 'win_sag1', 'loss_sag1', 'sched_sag1', 'sched_rank_sag1', 'win_tier1_sag1', 'loss_tier1_sag1', 'pipe1', 'win_tier2_sag1', 'loss_tier2_sag1', 'pipe2']] = sagarin_df['Col3'].str.split(expand=True)
            clean_sagarin.drop(['blank', 'pipe1', 'pipe2'], axis=1, inplace=True)

            ## Chunk 6 ##
            clean_sagarin[['blank', 'predictor_sag1', 'predictor_rank_sag1']]  = sagarin_df['Col4'].str.split(expand=True)
            clean_sagarin.drop(['blank'], axis=1, inplace=True)

            clean_sagarin['year_sag1'] = year

            insert_sag1 = clean_sagarin.apply(pd.to_numeric, errors='ignore')
            insert_sag1['team_sag1'] = insert_sag1['team_sag1'].replace(sag_name_dict)


            insert_sag2 = insert_sag1.rename(columns={'team_sag1':'team_sag2','rating_sag1':'rating_sag2', 'win_sag1':'win_sag2', 
                                                    'loss_sag1':'loss_sag2', 'sched_sag1':'sched_sag2','sched_rank_sag1':'sched_rank_sag2', 'win_tier1_sag1':'win_tier1_sag2', 'loss_tier1_sag1':'loss_tier1_sag2', 
                                                    'win_tier2_sag1':'win_tier2_sag2','loss_tier2_sag1':'loss_tier2_sag2', 'predictor_sag1':'predictor_sag2',
                                                    'predictor_rank_sag1':'predictor_rank_sag2','year_sag1':'year_sag2'})

            insert_sag1.to_sql('sagarin1', conn, if_exists='append', index=False)
            insert_sag2.to_sql('sagarin2', conn, if_exists='append', index=False)
            
            year += 1
    
        elif(year >= 2014):

            stats = tree.xpath('//div[@class="sagarin-page"]//font/text()')

            trimmed_sagarin = []
            for index, val in enumerate(stats):
                if('&nbsp' in val and '=' in val):
                    i = 0
                    while i < 8:
                        trimmed_sagarin.append(stats[index+i])
                        i+=1

            # Initial dataframe constructed, still will require some 
            sagarin_df = pd.DataFrame(np.array(trimmed_sagarin).reshape(len(trimmed_sagarin)//8, 8),
                columns = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8'])

            ## Chunk 1: Name & Rank ##
            clean_sagarin['team_sag1'] = sagarin_df['Col1'].str.replace('=','')
            clean_sagarin['team_sag1'] = clean_sagarin['team_sag1'].str.replace('&nbsp','')
            clean_sagarin['team_sag1'] = clean_sagarin['team_sag1'].str.strip()
            clean_sagarin[['rank', 'team_sag1']] = clean_sagarin['team_sag1'].str.split(" ", 1, expand=True)
            clean_sagarin['team_sag1'] = clean_sagarin['team_sag1'].str.strip()
            del(clean_sagarin['rank'])

            ## Chunk 2 ##
            clean_sagarin[['blank', 'rating_sag1']]  = sagarin_df['Col2'].str.split(expand=True)
            clean_sagarin.drop(['blank'], axis=1, inplace=True)

            ## Chunk 3 ##
            clean_sagarin[['col1', 'win_sag1', 'loss_sag1', 'sched_sag1', 'sched_rank_sag1', 'win_tier1_sag1', 'loss_tier1_sag1', 'col2', 'win_tier2_sag1', 'loss_tier2_sag1']]  = sagarin_df['Col3'].str.split(expand=True)
            clean_sagarin.drop(['col1', 'col2'], axis=1, inplace=True)

            ## Chunk 7 ##
            if (year <= 2015):
                clean_sagarin[['blank','predictor_sag1', 'predictor_rank_sag1', 'pipe']]  = sagarin_df['Col7'].str.split(expand=True)
                clean_sagarin.drop(['blank','pipe'], axis=1, inplace=True)
            else:
                clean_sagarin[['blank','predictor_sag1', 'predictor_rank_sag1']]  = sagarin_df['Col5'].str.split(expand=True)
                clean_sagarin.drop(['blank'], axis=1, inplace=True)

            clean_sagarin['year_sag1'] = year

            insert_sag1 = clean_sagarin.apply(pd.to_numeric, errors='ignore')
            insert_sag1['team_sag1'] = insert_sag1['team_sag1'].replace(sag_name_dict)

            insert_sag2 = insert_sag1.rename(columns={'team_sag1':'team_sag2','rating_sag1':'rating_sag2', 'win_sag1':'win_sag2', 
                                                    'loss_sag1':'loss_sag2', 'sched_sag1':'sched_sag2','sched_rank_sag1':'sched_rank_sag2', 'win_tier1_sag1':'win_tier1_sag2', 'loss_tier1_sag1':'loss_tier1_sag2', 
                                                    'win_tier2_sag1':'win_tier2_sag2','loss_tier2_sag1':'loss_tier2_sag2','predictor_sag1':'predictor_sag2',
                                                    'predictor_rank_sag1':'predictor_rank_sag2','year_sag1':'year_sag2'})

            insert_sag1.to_sql('sagarin1', conn, if_exists='append', index=False)
            insert_sag2.to_sql('sagarin2', conn, if_exists='append', index=False)

            year += 1

    print('sagarin tables added')

#############################################################
############ Create Aggregated Model Data ###################
#############################################################

# mdb file retrieved from http://www.hoopstournament.net/Database.html
# TODO automate retrieval and conversion to MySQL table (mdb odbc drivers don't play nice with linux). 
# For now, use online tool to convert to csv, save to data directory as 'game_by_game.csv'

if(aggregate == True):
    game_data = pd.read_csv('data/game_by_game.csv')
    game_data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)
    game_data['identifier'] = np.where(game_data['Wins'] == 1, game_data['Team']+game_data['Date'].astype(str)+game_data['Year'].astype(str), game_data['Opponent']+game_data['Date'].astype(str)+game_data['Year'].astype(str)) 
    sampled_games = game_data.groupby('identifier').apply(lambda x:x.sample(1))

    # Help with some common name mismatches here
    # 1. Create mapping of name mismatches according to some other common identifier? And then replace those names in the sampled game data?

    
    sampled_games.to_sql('tourny_games', conn, if_exists='replace', index=False)

    # I think this is the current bracket data that I manually input. Not a huge deal, but could be cool to automagically pull from espn
    team_data = pd.read_csv('data/' + str(run_year) + '_team_data.csv')
    team_data.to_sql('team_data', conn, if_exists='replace', index=False)

    opponent_data = pd.read_csv('data/' + str(run_year) + '_opponent_data.csv')
    opponent_data.to_sql('opponent_data', conn, if_exists='replace', index=False)

    # Create the mass model data table that the model script pulls from to run the model

    model_data_create = """CREATE TABLE model_data AS
                        SELECT * FROM tourny_games tg 
                        LEFT JOIN kenpom_team kp1 ON kp1.team_kp = tg.Team AND kp1.team_year = tg.Year 
                        LEFT JOIN kenpom_opponent kp2 ON kp2.opp_kp = tg.Opponent AND kp2.opp_year = tg.Year
                        LEFT JOIN sagarin1 sg1 ON sg1.team_sag1 = tg.Team AND sg1.year_sag1 = tg.Year
                        LEFT JOIN sagarin2 sg2 ON sg2.team_sag2 = tg.Opponent AND sg2.year_sag2 = tg.Year
                        WHERE tg.Year > 2002"""

    print('creating model data')
    conn.execute("""DROP TABLE IF EXISTS model_data""")
    conn.execute(model_data_create) 
    print('model master table created')

    print('checking for team name mismatches')
    mismatches = pd.read_sql("""SELECT Team, Opponent,team_kp, opp_kp, team_sag1, team_sag2, Year  FROM model_data 
                                WHERE team_kp IS NULL OR opp_kp IS NULL OR team_sag1 IS NULL OR team_sag2 IS NULL""", conn)
    if(mismatches.empty):
        print('no missing data, import is successful')
    else:
        print('missing team data, adjust mapping before running model')

    conn.close()


