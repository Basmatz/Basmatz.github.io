from flask import Flask, render_template, request
from model import get_combined_scores

app = Flask(__name__)
app.secret_key = b'message_flashing'

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/recommender')
def recommender():
    return render_template('recommender.html')

@app.route('/results')
def results():
    user_input = request.args.to_dict()

    movies = []
    for n in range(1, 6):
        if user_input["movie" + str(n)] != "":
            movies.append(int(user_input["movie" + str(n)]))

    movie_list = get_combined_scores(movies)

    return render_template('results.html', movies=movie_list)

if __name__ == '__main__':
    app.run(debug=True)