import psycopg2
import numpy as np
from flask import Flask, request
from flask_restful import Resource, Api
from statsmodels.discrete.discrete_model import MNLogit
import statsmodels.api as sm
import threading
from time import sleep

def connect():
    conn = psycopg2.connect(
       database="postgres", user='postgres', password='', host='', port= '5432'
    )
    return conn

def score_matrix(conn, question_list = []):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM result")
    results = cursor.fetchall()

    if question_list == []:
        cursor.execute("SELECT id FROM question WHERE \"isDeleted\"=FALSE")
        question_list = cursor.fetchall()
        question_list = [q[0] for q in question_list]

    cursor.execute("SELECT id, value FROM answer")
    answer_scores = cursor.fetchall()
    answer_scores = dict(answer_scores)

    user_list = list(set([x[3] for x in results]))

    scores = []
    for user in user_list:
        results_user = [x for x in results if x[3] == user]
        question_answer = sorted([(x[1], answer_scores[x[2]]) for x in results_user if x[1] in question_list], key=lambda y: y[0])
        if len(question_answer) == len(question_list):
            scores.append([x[1] for x in question_answer])

    return scores

def user_list(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM \"user\"")
    tab = cursor.fetchall()

    return [{"id": user[0], "age": user[1], "sex": user[2], "created": user[3], "email": user[4], "porientation": user[5]} for user in tab]

def cov_matrix(scores):
    return np.cov(np.transpose(scores))

def tot_scores(scores):
    return [np.sum(scores[k]) for k in range(np.shape(scores)[0])]

def tot_factor_scores(scores, v):
    return [np.dot(scores[k], v) for k in range(np.shape(scores)[0])]

def compute_model(conn):
    m = score_matrix(conn)
    users = user_list(conn)
    
    c = cov_matrix(m)
    w, v = np.linalg.eig(c)

    sr = [-1, 1, 0]
    or_list = [sr[int(users[k]["porientation"])-2] for k in range(len(users)) if users[k]["porientation"] in [2, 3, 4]]
    tot_factor_arr = np.transpose(np.array([tot_factor_scores([m[k] for k in range(len(m)) if users[k]["porientation"] in [2, 3, 4]], v[:, k]) for k in range(2)]))

    model = MNLogit(endog=or_list, exog=sm.add_constant(tot_factor_arr))
    res = model.fit()

    return m, users, w, v, res

def thread_func():
    global gm, gusers, gw, gv, gres
    while True:
        with lock:
            conn = connect()
            gm, gusers, gw, gv, gres = compute_model(conn)
            conn.close()
        sleep(7200)

def user_odds(uid, m, v, res):
    f1 = np.dot(m[uid], v[:, 0])
    f2 = np.dot(m[uid], v[:, 1])

    return res.predict(exog=[1, f1, f2])

app = Flask(__name__)
api = Api(app)

lock = threading.Lock()

# todos = {}

class Odds(Resource):
    def __init__(self):
        super(Odds, self).__init__()
        
    def get(self, user_id):
        if 0 <= user_id and user_id < len(gm):
            odds = user_odds(user_id, gm, gv, gres)
            return {"left": odds[0][0], "center": odds[0][1], "right": odds[0][2]}
        else:
            return {}

api.add_resource(Odds, '/odds/<int:user_id>')

if __name__ == '__main__':
    th = threading.Thread(target=thread_func)
    th.start()

    th_main = threading.Thread(target=app.run, kwargs={"debug": False})
    th_main.start()
