import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
import logging

# --- CONFIG ---

SPOTIPY_CLIENT_ID = '260e2fd648cf42dbab5157bb9857e07e'
SPOTIPY_CLIENT_SECRET = 'aabb18f4c93440f7b115cfe963189e9f'
SPOTIPY_REDIRECT_URI = 'http://127.0.0.1:8888/callback'
CSV_FILE = "recommended_tracks.csv"
PLAYLIST_NAME = "🎵 AI Recommended Playlist"

# --- LOGGING ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- AUTHENTICATION ---

def get_spotify_clients():
    try:
        user_sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=SPOTIPY_CLIENT_ID,
            client_secret=SPOTIPY_CLIENT_SECRET,
            redirect_uri=SPOTIPY_REDIRECT_URI,
            scope="user-top-read playlist-modify-public",
            cache_path=".cache"
        ))

        client_sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=SPOTIPY_CLIENT_ID,
            client_secret=SPOTIPY_CLIENT_SECRET
        ))
        # Log user info for debugging
        user_info = user_sp.me()
        logging.info(f"Authenticated as: {user_info['id']}")
        return user_sp, client_sp
    except Exception as e:
        logging.error(f"Failed to authenticate with Spotify: {e}")
        return None, None


# --- FETCHING ---

def get_user_top_tracks(user_sp, limit=20, time_range='medium_term'):
    """Fetch user's top tracks with full metadata."""
    try:
        results = user_sp.current_user_top_tracks(limit=limit, time_range=time_range)
        track_data = [{
            "id": track['id'],
            "name": track['name'],
            "artist": track['artists'][0]['name'],
            "preview_url": track['preview_url']
        } for track in results['items'] if track['preview_url']]  # Filter out tracks without previews
        return track_data
    except Exception as e:
        logging.error(f"Failed to fetch user's top tracks: {e}")
        return []



def get_audio_features(client_sp, track_ids):
    """Fetch audio features for a list of track IDs."""
    features = []
    for i in range(0, len(track_ids), 20):
        chunk = track_ids[i:i+20]
        try:
            chunk_features = client_sp.audio_features(chunk)
            if chunk_features:
                features.extend([f for f in chunk_features if f is not None])
        except Exception as e:
            logging.error(f"Failed to fetch audio features: {e}")

    return pd.DataFrame(features)


# --- CLUSTERING + RECOMMENDATION ---

def cluster_and_recommend(audio_df, track_metadata, n_clusters=3):
    """Cluster audio features and return metadata of similar tracks."""
    features_df = audio_df[['danceability', 'energy', 'valence', 'tempo']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled)
    audio_df['cluster'] = labels

    target_cluster = labels[0]
    clustered_df = audio_df[audio_df['cluster'] == target_cluster]

    # Join with metadata
    final_df = clustered_df.merge(pd.DataFrame(track_metadata), left_on='id', right_on='id')
    return final_df[['id', 'name', 'artist', 'preview_url', 'danceability', 'energy', 'valence', 'tempo', 'cluster']]


# --- PLAYLIST CREATION ---

def create_playlist(user_sp, track_ids, playlist_name, image_url=None):
    """Create and populate a playlist with recommended tracks, optionally adding a cover image."""
    try:
        user_id = user_sp.me()['id']
        playlist = user_sp.user_playlist_create(user=user_id, name=playlist_name, public=True)

        # Add tracks to the playlist
        user_sp.playlist_add_items(playlist_id=playlist['id'], items=track_ids)
        logging.info(f"Playlist '{playlist_name}' created with {len(track_ids)} songs!")

        # Add a cover image (if provided)
        if image_url:
            user_sp.playlist_upload_cover_image(playlist_id=playlist['id'], image_url=image_url)
            logging.info(f"Cover image added to the playlist.")
    except Exception as e:
        logging.error(f"Failed to create playlist: {e}")



# --- MAIN ---

def main():
    user_sp, client_sp = get_spotify_clients()
    if user_sp is None or client_sp is None:
        return

    top_tracks = get_user_top_tracks(user_sp)
    print(f"Top tracks: {top_tracks}")
    if not top_tracks:
        logging.info("No top tracks found.")
        return

    track_ids = [t['id'] for t in top_tracks]
    audio_df = get_audio_features(client_sp, track_ids)
    if audio_df.empty:
        logging.info("Could not fetch audio features.")
        return

    recommended_df = cluster_and_recommend(audio_df, top_tracks)
    if recommended_df.empty:
        logging.info("No recommended tracks found.")
        return
    # Shuffle the recommended tracks
    recommended_df = recommended_df.sample(frac=1).reset_index(drop=True)

    # Now create the playlist with shuffled tracks
    recommended_ids = recommended_df['id'].tolist()
    create_playlist(user_sp, recommended_ids, PLAYLIST_NAME)
    logging.info(f"Playlist '{PLAYLIST_NAME}' created with {len(recommended_ids)} songs")
    return recommended_df, recommended_ids

if __name__ == "__main__":
    main()