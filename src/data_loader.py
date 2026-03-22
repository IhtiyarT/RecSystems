import pandas as pd


class DataLoader:

    def __init__(self, path):
        self.path = path
    
    def load_ratings(self):
        ratings = pd.read_csv(
            f"{self.path}/ratings.dat",
            sep="::",
            engine="python",
            names=["userId", "movieId", "rating", "timestamp"]
        )
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
        
        return ratings
    
    def load_movies(self):
        movies = pd.read_csv(
            f"{self.path}/movies.dat",
            sep="::",
            engine="python",
            names=["movieId", "title", "genres"],
            encoding="latin-1"
        )

        return movies
    
    def load_users(self):
        users = pd.read_csv(
            f"{self.path}/users.dat",
            sep="::",
            engine="python",
            names=["userId", "gender", "age", "occupation", "zip"]
        )
        return users

    def load_all(self):
        ratings = self.load_ratings()
        movies = self.load_movies()
        users = self.load_users()

        return ratings, movies, users