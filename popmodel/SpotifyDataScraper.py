#Imports
import spotipy
from credentials import cid, secret, username, password
import pandas as pd
import pymongo
import tqdm
import time

#Access Spotify API with spotipy library
client_credentials_manager = spotipy.oauth2.SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager
                     = client_credentials_manager)

# Setting up MongoDB connection with the pymongo library
client = pymongo.MongoClient("mongodb://localhost/")

# Connect to the different databases
db_mongo = client.song_data
songs_info = db_mongo.popmusic_info
songs_features = db_mongo.popmusic_features
songs_analysis = db_mongo.popmusic_analysis
analysis_bad_ids = db_mongo.popmusic_bad_ids

# #Get Song Ids from csv
# track_import = pd.read_csv("models/track_id.csv")
# song_ids = track_import["track_id"].tolist()
# track_names = track_import["track_name"].tolist()

#Get artists by genre
artists = pd.DataFrame()
search_string = 'genre:"rock"'
for i in tqdm.tqdm(range(0,2000,50)):
    artists = artists.append(sp.search(search_string, limit=50, offset=i, type='artist', market="DE"), ignore_index=True)

#Get all album IDs from Artists
albums = pd.DataFrame()
for z in tqdm.tqdm(range(len(artists))):
    for i in range(len(artists.iloc[z]["artists"]["items"])):
        albums = albums.append(sp.artist_albums(artists.iloc[z]["artists"]["items"][i]["id"], limit=50), ignore_index=True)


#Get track IDs from each album
song_ids = pd.DataFrame()
for z in tqdm.tqdm(range(len(albums))):
    for i in range(len(albums.iloc[z]["items"])):
        song_ids = song_ids.append(sp.album_tracks(albums.iloc[z]["items"][i]["id"], limit=100), ignore_index=True)

#Get song infos and write it in a mongoDB collection
for i in range(0, len(song_ids), 50):
    info_df = pd.DataFrame()
    temp_ids = []
    #temp_names = []
    if i % 10000 == 0 and i != 0:
        print(str(i) + " tracks scraped")
    for id in song_ids[i:i+50]:
        temp_ids += [id]
    # for name in track_names[i:i+50]:
    #     temp_names += [name]
    info_df = info_df.append(sp.tracks(temp_ids)["tracks"], ignore_index=True)
    info_df["track_id"] = temp_ids
    #info_df["track_name"] = temp_names
    songs_info.insert_many(info_df.to_dict('records'))

print("All song info data collected and written in database")

#Get song features and write it in a mongoDB collection
features_dict = []
features_df = pd.DataFrame()

for i in range(0, len(song_ids), 100):
    temp_ids = []
    if i % 1000 == 0:
        print(i)
    for id in song_ids[i:i+100]:
        temp_ids += [id]
    features_dict.append(sp.audio_features(temp_ids))

print("All features data collected")

for i in range(len(features_dict)):
    for z in features_dict[i]:
        if z == None:
            features_dict[i].remove(z)

for i in features_dict:
    features_df = features_df.append(i)

songs_features.insert_many(features_df.to_dict('records'))

print("All features data written in database")

# Remove duplicates from Database
feature_df = pd.DataFrame()
feature_df = pd.DataFrame(list(songs_features.find()))
feature_df.drop_duplicates("id", inplace=True)
songs_features.insert_many(feature_df.to_dict('records'))

#Get song analysis and write it in a mongoDB collection
analysis_df = pd.DataFrame()
i = 0
for id in tqdm.tqdm(song_ids):
    try:
        analysis_df = analysis_df.append(sp.audio_analysis(id)["track"], ignore_index=True)
        analysis_df.drop(['analysis_channels', 'analysis_sample_rate', 'code_version',
           'codestring', 'echoprint_version', 'echoprintstring', 'rhythm_version',
           'rhythmstring', 'sample_md5', 'synch_version',
           'synchstring'], axis=1, inplace=True)
        analysis_df.loc[i ,"id"] = id
        if i == 0:
            songs_analysis.insert_many(analysis_df.to_dict('records'))
            analysis_df = pd.DataFrame()
            i = 0
        else:
            i += 1
    # in case analysis for a song doesn't exist, skip and remember song id
    except:
        analysis_bad_ids.insert_one({"id":id, "time":time.time()})
        pass

print("All analysis data collected and written in database")

# Write collections to external MongoDB
client_ext = pymongo.MongoClient(f"mongodb+srv://{username}:{password}@cluster0.9vhde.mongodb.net/?retryWrites=true")
db_mongo_ext = client_ext.song_data
popmusic_info_db_ext = db_mongo_ext.popmusic_info
popmusic_features_db_ext = db_mongo_ext.popmusic_features
popmusic_analysis_db_ext = db_mongo_ext.popmusic_analysis

info_df = pd.DataFrame(list(songs_info.find()))
popmusic_info_db_ext.insert_many(info_df.to_dict('records'))

features_df = pd.DataFrame(list(songs_features.find()))
popmusic_features_db_ext.insert_many(features_df.to_dict('records'))

analysis_df = pd.DataFrame(list(songs_analysis.find()))
popmusic_analysis_db_ext.insert_many(analysis_df.to_dict('records'))